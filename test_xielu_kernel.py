"""Sanity + perf check for the fused xIELU MLP kernel.

Run on a single GPU:
    python3 test_xielu_kernel.py

Correctness: compares FusedLinearXIELUFunction against an unfused PyTorch
reference (the path that exp/xielu shipped with) for forward output and all
five gradients (dx, dW1, dW2, dαp_raw, dαn_raw).

Perf: times the fused xIELU vs fused ReLU² on the MLP shapes the model uses,
to verify the kernel recovers the ~11.6% wallclock regression seen in the
unfused experiment.
"""
import time

import torch
import torch.nn.functional as F

from triton_kernels import FusedLinearReLUSquareFunction, FusedLinearXIELUFunction


def xielu_ref(x, W1, W2, alpha_p_raw, alpha_n_raw):
    """Unfused reference (the original path on exp/xielu before the kernel landed)."""
    h = x @ W1.type_as(x).T
    ap = F.softplus(alpha_p_raw.float()).type_as(h)
    an = F.softplus(alpha_n_raw.float()).type_as(h)
    pos = ap * h * h
    neg = an * torch.expm1(h.clamp(max=0.0))
    h_act = torch.where(h > 0, pos, neg)
    return h_act @ W2.type_as(x)


def make_inputs(M, K, N, device, seed=0):
    g = torch.Generator(device=device).manual_seed(seed)
    x = torch.randn(M, K, device=device, dtype=torch.bfloat16, generator=g) * 0.5
    W1 = torch.randn(N, K, device=device, dtype=torch.bfloat16, generator=g) * (K ** -0.5)
    W2 = torch.randn(N, K, device=device, dtype=torch.bfloat16, generator=g) * (N ** -0.5)
    # init α_raw at log(e−1) ≈ 0.5413 to match train_gpt.py
    alpha_p_raw = torch.tensor(0.5413, device=device, dtype=torch.bfloat16)
    alpha_n_raw = torch.tensor(0.5413, device=device, dtype=torch.bfloat16)
    return x, W1, W2, alpha_p_raw, alpha_n_raw


def run_with_grads(fn, x, W1, W2, ap_raw, an_raw):
    x = x.detach().clone().requires_grad_(True)
    W1 = W1.detach().clone().requires_grad_(True)
    W2 = W2.detach().clone().requires_grad_(True)
    ap_raw = ap_raw.detach().clone().requires_grad_(True)
    an_raw = an_raw.detach().clone().requires_grad_(True)
    out = fn(x, W1, W2, ap_raw, an_raw)
    g = torch.randn_like(out)
    out.backward(g)
    return out.detach(), x.grad, W1.grad, W2.grad, ap_raw.grad, an_raw.grad, g


def correctness(M, K, N, device):
    print(f"\n=== correctness (M={M}, K={K}, N={N}) ===")
    inp = make_inputs(M, K, N, device)

    # Drive both with the SAME upstream gradient by seeding identical clones.
    torch.manual_seed(0)
    out_ref, dx_ref, dW1_ref, dW2_ref, dap_ref, dan_ref, g_ref = run_with_grads(xielu_ref, *inp)
    torch.manual_seed(0)
    out_fus, dx_fus, dW1_fus, dW2_fus, dap_fus, dan_fus, g_fus = run_with_grads(
        FusedLinearXIELUFunction.apply, *inp,
    )
    assert torch.equal(g_ref, g_fus), "upstream grads diverged — seeding bug"

    def cmp(name, a, b, atol, rtol):
        a_f = a.float()
        b_f = b.float()
        max_abs = (a_f - b_f).abs().max().item()
        rel = (a_f - b_f).abs().max().item() / (b_f.abs().max().item() + 1e-9)
        ok = torch.allclose(a_f, b_f, atol=atol, rtol=rtol)
        flag = "OK " if ok else "FAIL"
        print(f"  [{flag}] {name:>10}  max|Δ|={max_abs:.3e}  max_rel={rel:.3e}")
        return ok

    # bf16 matmul tolerances are loose; tighten if you want to chase precision.
    all_ok = True
    all_ok &= cmp("out",    out_fus, out_ref, atol=1e-2, rtol=1e-2)
    all_ok &= cmp("dx",     dx_fus,  dx_ref,  atol=1e-2, rtol=1e-2)
    all_ok &= cmp("dW1",    dW1_fus, dW1_ref, atol=2e-2, rtol=2e-2)
    all_ok &= cmp("dW2",    dW2_fus, dW2_ref, atol=2e-2, rtol=2e-2)
    # α gradients are scalar reductions over (M, N); compare with looser rtol.
    all_ok &= cmp("dap",    dap_fus, dap_ref, atol=5e-2, rtol=5e-2)
    all_ok &= cmp("dan",    dan_fus, dan_ref, atol=5e-2, rtol=5e-2)
    return all_ok


def bench_one(fn, args, n_warmup=10, n_iter=50):
    for _ in range(n_warmup):
        out = fn(*args)
        out.sum().backward()
        for a in args:
            if a.grad is not None:
                a.grad = None
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(n_iter):
        out = fn(*args)
        out.sum().backward()
        for a in args:
            if a.grad is not None:
                a.grad = None
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / n_iter  # ms / iter


def perf(M, K, N, device):
    print(f"\n=== perf (M={M}, K={K}, N={N}, fwd+bwd) ===")
    x, W1, W2, ap_raw, an_raw = make_inputs(M, K, N, device)
    x = x.requires_grad_(True)
    W1 = W1.requires_grad_(True)
    W2 = W2.requires_grad_(True)
    ap_raw = ap_raw.requires_grad_(True)
    an_raw = an_raw.requires_grad_(True)

    # ReLU² takes 3 args; xIELU takes 5. Pack two arg-lists.
    relu2_args  = (x, W1, W2)
    xielu_args  = (x, W1, W2, ap_raw, an_raw)

    t_relu2 = bench_one(FusedLinearReLUSquareFunction.apply, relu2_args)
    t_xielu = bench_one(FusedLinearXIELUFunction.apply,      xielu_args)

    delta_pct = 100.0 * (t_xielu - t_relu2) / t_relu2
    print(f"  ReLU²  : {t_relu2:.3f} ms/iter")
    print(f"  xIELU  : {t_xielu:.3f} ms/iter   ({delta_pct:+.2f}% vs ReLU²)")


def main():
    assert torch.cuda.is_available(), "CUDA required"
    device = "cuda"
    torch.manual_seed(0)

    # Model shapes: 12 layers × MLP with model_dim=768, mlp_hdim=3072.
    # M = micro-batch tokens; pick a value typical of training (32k tokens/microbatch).
    K, N = 768, 3072
    ok_small = correctness(M=2048,  K=K, N=N, device=device)
    ok_full  = correctness(M=32768, K=K, N=N, device=device)
    perf(M=32768, K=K, N=N, device=device)

    print("\n" + ("ALL CORRECTNESS PASS" if (ok_small and ok_full) else "SOME CORRECTNESS FAILED"))


if __name__ == "__main__":
    main()
