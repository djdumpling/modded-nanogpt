# Thirteen underexplored levers for the modded-nanogpt speedrun

**Bottom line.** The current record (#78, ~86 s wall-clock on 8×H100 to FineWeb val ≤3.28) has already absorbed nearly every obvious efficiency trick: Muon with Polar-Express Newton–Schulz, NorMuon, batched/reshaped orthogonalization, QK-Norm, half-RoPE + YaRN, long-short FlexAttention via FA3, ReLU², asymmetric softcap, FP8 LM head with fused softcapped cross-entropy, untied embeddings, a bigram hash embedding, 5× value embeddings, U-net skip + backout, smear module, multi-token prediction, paired-head attention, cautious weight decay, trapezoidal LR, and an explicit kernel-warmup region. The **real frontier** is now a narrow set of signal-shaping, optimizer-variance, kernel, and schedule-shape tweaks that *compose* with all of the above. Notably absent from every PR, issue, or discussion thread: Scion-style unified LMO optimizers, variance-adapted Muon (AdaMuon), manifold optimizers (Mano), scalable softmax, xIELU, Dynamic Tanh, Forgetting-Transformer gating, LAWA/WSM checkpoint merging, Schedule-Free on the AdamW legs, CUDA-graphed Muon Newton–Schulz, stochastic-rounding BF16 optimizer states, DeepSeek-style block-scaled FP8 on MLP linears, actual sequence-length curriculum (only attention window is warmed), and the new FA4 CuTeDSL backend for FlexAttention. Each is a plausible 0.3–5 s lever on an 86 s budget. The report below details 13 specific, implementable directions ordered by expected value × probability-of-working at this exact scale.

## What is already exhausted

To calibrate novelty, the following should be treated as **occupied territory** and not re-proposed: Polar-Express and batched Newton–Schulz, Cautious Weight Decay, NorMuon's per-neuron rescale, MuonClip-style interactions with softcap, Triton symmetric `X·Xᵀ` for NS (this already captures much of Tri Dao's Gram-NS idea), Gloeckle-style MTP, value embeddings and value residual, asymmetric logit softcap, FP8 LM-head + fused CE kernel, document-level `cu_seqlens` in FA3, BOS alignment, YaRN dynamic window extension, paired-head attention, smear module, bigram hash embed, cautious Adam, Adam-every-other-step heterogeneous batching, `reduce_scatter`-based Muon gradient comms, scalar smoothing across transitions, and WSD/trapezoidal LR with linear cooldown. Rejected or rule-blocked: FP8 on hidden linears via per-tensor scaling (matmuls at d=768 are too small), nGPT (conflicts with Muon's orthogonality), SOAP (fork exists, never landed), tokenizer changes, test-time training, FineWeb-Edu filtering, reordering tokens. The ideas below deliberately route around all of these.

---

## 1. Scion / ScionLight as a unified LMO optimizer replacing the Muon + AdamW split

**Technique.** Pethick et al. (arXiv:2502.07529, ICML 2025) and Clipped-Scion (arXiv:2506.01913) generalize Muon to every parameter group via a per-group Linear Minimization Oracle over a dual norm ball: spectral (RMS→RMS) for hidden matrices, sign-LMO (ℓ∞ dual) for the LM head, RMS-LMO for embeddings. **ScionLight** re-uses `p.grad` in place of a momentum buffer, **cutting optimizer state memory to zero** relative to Muon/AdamW. Gluon (arXiv:2505.13416) provides the convergence theory.

**Why it might win.** The speedrun already segregates parameters across two optimizers with ad-hoc per-group learning rates, betas, weight decay, and comms schemes (DistAdam vs sharded Muon). Scion gives a principled unification: embedding and head get a Muon-analog rather than an independent adaptive method, which eliminates the "river-valley" instability AdamW-on-the-head causes at high LR. Filatov et al. (arXiv:2510.03871) show the output-layer operator norm is the scaling invariant, so a spectral-dual LMO on the head has a stronger theoretical grounding than AdamW. ScionLight's zero optimizer state could plausibly shave a few ms of comms per step (the current DistAdam has two-tier 1024-element branching logic that is complex and partly serial).

**Why it might not.** The existing split is hand-tuned over 78 records; new per-group LR scales would need a fresh sweep. Muon already covers the hidden layers, so the delta is in the embedding/head/bigram/value-embeddings groups — each with distinct shape characteristics the sign-LMO may not respect. The FP8 head's asymmetric scaling interacts with the sign update in ways not tested. Ruling out a regression requires ~5 full runs at minimum.

**Effort.** Medium (1–2 days). Reference implementation at github.com/LIONS-EPFL/scion; must wrap the FP8 head path.

**Suggestions.** Start by replacing **only the AdamW legs** (embed, lm_head, value_embeds, bigram, gates) with sign-LMO Scion; keep Muon untouched. Initial LR scales: hidden unchanged, head `0.003`, embeddings `0.01`, gates/scalars `0.02`. Ablate against current `betas=(0.5,0.95)` AdamW head.

---

## 2. Variance-adapted (AdaMuon) or cautious (C-Muon) overlays on top of NorMuon

**Technique.** AdaMuon (Si et al., arXiv:2507.11005) adds an element-wise second-moment EMA over the *orthogonalized* update `O_t`, rescaling by `1/√V_t`. C-Muon applies Liang et al.'s cautious mask `sign(O_t)==sign(g_t)` (arXiv:2411.16085) to zero out components of the spectral update that disagree with the raw gradient sign — the matrix analog of the 1.47×-speedup one-line trick that has been shown to help AdamW. Frans's matrix-whitening benchmark (arXiv:2510.25000) shows **variance adaptation is the single differentiator of SOAP/AdaMuon over sign-style Muon/SPlus**.

**Why it might win.** NorMuon is a *per-neuron* LR (Adafactor-like, reduction over one axis); AdaMuon is element-wise *per-entry* variance. They are complementary: NorMuon handles coordinate-rescale, AdaMuon handles per-entry noise. C-Muon is nearly free (one bitmask) and theoretically preserves Muon's Hamiltonian.

**Why it might not.** NorMuon already captured most of the variance-adaptation low-hanging fruit at this scale; stacking AdaMuon adds another m×n second-moment buffer and may double-count the adaptation. The cautious mask at the matrix level has been studied on AdamW, not on a matrix-whitening optimizer — there is some probability the mask zeros so much of the orthogonalized update that the spectral norm is no longer unit-bounded, which is the property Muon exploits for LR transfer.

**Effort.** Very low (~30 LoC each).

**Suggestions.** Try C-Muon first (one-line change; zero state overhead). If variance adaptation is desired, try **NorMuon's axis-0 variance replaced by AdaMuon's full element-wise variance** using half-precision storage (the buffer is ~20 MB, compatible with DistAdam's comm path). Sweep β₂ ∈ {0.95, 0.98, 0.99}.

---

## 3. Manifold-style optimizer (Mano) or Dion-style power iteration as a Muon replacement

**Technique.** Mano (arXiv:2601.23000) replaces Muon's flatten-to-{±1} singular-value profile with a monotone manifold transformation that *preserves singular-value ordering*. Reports 1.75× wall-clock win over Muon on LLaMA-350M at 10k steps — closer in regime to this speedrun than most LM benchmarks. Dion (arXiv:2504.05295, microsoft/dion) replaces Newton–Schulz with amortized power iteration plus error feedback; at full rank it matches or beats Muon due to more accurate orthonormalization than NS's approximate polynomial.

**Why it might win.** The speedrun's NS is already extremely well-tuned (Polar Express + symmetric-matmul + bf16), so further wallclock gains in this primitive are hard. Mano attacks a different dimension — *information content* of the update, not compute — and is explicitly validated in the few-thousand-step LM-loss regime. Dion's error-feedback property may let it survive more aggressive precision reduction (FP8 momentum) than Muon.

**Why it might not.** Mano is extremely recent (late 2025) and not yet reproduced. Dion's wins are reported primarily on larger FSDP setups where NS's communication cost matters; at 8 GPUs with Muon's owner-per-param sharding, NS is already <1 % of step FLOPs per Keller Jordan's analysis. Several 2025–2026 "beats-Muon" claims have failed to set modded-nanogpt records.

**Effort.** Medium-high for Mano (custom polynomial + manifold operator). Medium for Dion (reference impl exists).

**Suggestions.** Treat Mano as a research side-bet, not a safe lever. Dion is worth a back-to-back A/B replacing the current NS routine in the Muon optimizer only; keep all else fixed.

---

## 4. Schedule-Free AdamW (or AdEMAMix) applied *only* to the AdamW-optimized parameter groups

**Technique.** Defazio et al.'s Schedule-Free (arXiv:2405.15682, NeurIPS 2024 oral) replaces momentum + LR decay with a Polyak–primal interpolation. Through-the-River (arXiv:2507.09846) demonstrates SF-AdamW implicitly performs river-valley averaging *without* a separate decay phase. AdEMAMix (arXiv:2409.03137) uses two EMAs, one slow (β₃≈0.9999), and in Semenov et al.'s benchmark (arXiv:2509.01440) ranks as the single best scalar-based optimizer at ~130M, occasionally beating Muon at small batch.

**Why it might win.** The modded-nanogpt WR has long been dominated by Muon on hidden weights, and the AdamW legs (lm_head, embed, value_embeds, bigram, gates) are where scalar-state optimizer innovations still have unclaimed upside. Replacing AdamW there with SF-AdamW opens two wins: (a) no LR-decay commitment on those specific parameters (they can continue receiving peak-LR updates even as Muon decays); (b) a natural averaging effect on the FP8 LM head that may stabilize the end-of-training val loss.

**Why it might not.** Short-horizon benefits of SF and AdEMAMix are partially documented — AdEMAMix's slow β₃ needs ~1000 steps to warm. At ~1750 total steps this is marginal. The `.train()/.eval()` toggle (required for SF) is an extra complexity against CUDA-graph capture (#12 below). At 130M scale, LR-schedule choice is one of the top-three factors; replacing it with a schedule-free variant is a high-variance bet.

**Effort.** Low (drop-in libraries: `schedule_free`, `AdEMAMix-Optimizer-Pytorch`). ~50 LoC each.

**Suggestions.** Run SF-AdamW on the lm_head and embedding groups only, keeping Muon+momentum-warmup on everything else. Try β₁∈{0.95,0.98}. Then try AdEMAMix with α=2–4 (lower than the paper's default 8) so the slow EMA warms in 1/3 of the run.

---

## 5. Scalable Softmax (SSMax) to replace the hand-tuned attention-temperature scalar

**Technique.** Nakanishi (arXiv:2501.19399) multiplies pre-softmax attention logits by `s · log(n)` with a learnable per-head scalar `s`, preventing the softmax-flattening failure mode as sequence length grows. Adopted in Llama-4. **Near-zero FLOP cost.**

**Why it might win.** The current record uses a hand-picked `attn_scale=0.1` and a window-warmup schedule that also effectively shifts softmax sharpness via YaRN. SSMax generalizes this: it replaces the constant with a length-aware learnable term that should also reduce the need for YaRN's `log(ws)/log(ws_trained)` rescale on each window transition, potentially simplifying the 3-phase schedule. Paper explicitly reports faster *pretraining* loss descent (not just long-context inference).

**Why it might not.** The existing YaRN+tuned-scalar combination was tuned to 1–2 % of the step budget; SSMax may underperform once that tuning is re-done from scratch. SSMax interacts with asymmetric logit softcap — the softcap's clip value would likely need re-tuning.

**Effort.** Very low (~20 LoC inside the FlexAttention `score_mod`).

**Suggestions.** Replace `attn_scale` constant with `s_h · log(seq_len_effective)` where `s_h` is per-head (6 scalars total), initialized to `0.1/log(window_size_trained)`. Keep YaRN in place initially; after confirming a gain, try removing YaRN transitions entirely.

---

## 6. xIELU activation replacing ReLU² with a fused Triton kernel

**Technique.** NVIDIA's xIELU (Huang et al., arXiv:2411.13010, v3 Jan 2025): piecewise trainable activation `αp·x² + αn·(exp(x)−1)` with learnable `αp, αn` per layer. In NVIDIA's own FineWeb-Edu pretraining at 1.1B/126B tokens, xIELU **beat both SwiGLU and ReLU²** on val loss.

**Why it might win.** It is the only published activation with a *better-than-ReLU²* result on the most relevant data distribution — FineWeb-Edu is effectively a subset of the speedrun's training data. xIELU's adaptive behavior (it reduces non-linearity with depth) could be a better match for the asymmetric 10-attn-/12-MLP block structure with U-net skips than ReLU² (which is fixed).

**Why it might not.** Reference implementation is *slower* than ReLU² without a fused kernel; the speedrun already has a fused `linear → ReLU² → linear` kernel (PR #197), so the baseline is extremely tight. Adding two learnable scalars per MLP layer introduces new Adam params and a new comms pattern that must be folded into DistAdam. Published gains are reported at 1.1B scale and may not transfer to 124M.

**Effort.** Medium-high. Activation itself is 10 LoC; a Hopper-class fused kernel matching the current ReLU² primitive is several days.

**Suggestions.** First validate quality-per-token gain in an uncompiled fork; only if ≥0.01 val-loss improvement materializes, invest in fusing (αp·x² + αn·expm1(x)) into the up-projection + down-projection kernel path.

---

## 7. Dynamic Tanh (DyT) as a drop-in for RMSNorm

**Technique.** Zhu, Chen, He, LeCun, Liu (arXiv:2503.10622, CVPR 2025): replace every LayerNorm/RMSNorm with `γ · tanh(α · x) + β`, eliminating reductions and matching or exceeding LN across vision/speech/LLaMA.

**Why it might win.** Each transformer block currently fires two RMSNorms, each of which is a bandwidth-bound reduction. Replacing them with an element-wise DyT saves reduction latency (estimated 1–3 % of step time) and frees shared memory for matmul stages in Inductor's fused epilogue patterns. Meshes with Muon's spectral-descent story: Muon already controls the spectral norm of weights, so the activation-side variance stabilization that RMSNorm currently handles can plausibly be replaced by tanh's saturation.

**Why it might not.** The speedrun already uses QK-Norm (RMSNorm applied to Q and K), which means a pure DyT replacement may not mesh well in the attention path without redesign — α and β would need to be applied inside FlexAttention's `score_mod`. The logit softcap is *also* a tanh; composing two tanhs in series (DyT after the head, then softcap) risks gradient saturation. Several 2025 reproductions (HoloNorm, arXiv:2511.10504) suggest DyT's tanh distorts representations.

**Effort.** Medium. Straightforward for the MLP path; careful for QK-Norm interaction.

**Suggestions.** Try a hybrid first: DyT only at the pre-MLP and pre-attention input normalization positions; keep QK-Norm as RMSNorm. Initialize α=0.8, β=0. Anneal α over warmup. Combine with **z-loss** (see #9) so softcap can be relaxed or removed.

---

## 8. Forgetting-Transformer (FoX) gate on the long-window attention layers

**Technique.** Lin et al. (arXiv:2503.02130, ICLR 2025) add a data-dependent learnable forget gate multiplying unnormalized attention logits, removing the need for RoPE. Adaptive Computation Pruning (arXiv:2504.06949) on top gives 10–40 % throughput improvement at longer contexts. The "Pro" block variant adds a pre/post depthwise conv + output gate for extra quality.

**Why it might win.** The two long-window attention layers (indices 3 and 10 in the current record) are the most expensive in the attention path. A FoX gate is essentially a learnable, input-conditioned ALiBi that may make the tuned YaRN + half-RoPE scheme redundant on those two layers, simplifying the 3-phase window schedule. ACP is most effective at longer contexts — directly relevant to the long-window layers. The Pro block variant has a small depthwise conv (Primer-style) that was part of the Primer package but is not explicitly in the current record.

**Why it might not.** FoX's reported wins are concentrated on long-context retrieval, not 2 048-token LM. Interaction with value-embeddings mixing (currently in layers 1, 2, last 3) is untested. Rewriting FlexAttention `score_mod` to include a learnable per-head forget-gate scalar multiplied into the logits is straightforward but breaks the existing blockmask caching.

**Effort.** Medium. ~200 LoC in the attention module + FlexAttention score_mod; separate paper has Triton reference.

**Suggestions.** Apply FoX gating **only to the two long-window layers** initially. Init forget-gate bias so exp(bias) ≈ 0.99 per step (near-no-forgetting). If the simplification of YaRN materializes, remove YaRN from the long layers.

---

## 9. Register tokens + z-loss for output-distribution stabilization

**Technique.** Two composable signal-shaping ideas. Register tokens (Darcet et al., arXiv:2309.16588) prepend 4–16 learnable tokens to the sequence to absorb attention-sink / high-norm artifacts. Z-loss (`α · log²(Z)`, PaLM/ST-MoE/OLMo-2) penalizes the log partition function of the softmax, explicitly stabilizing output logit magnitude.

**Why it might win.** The current record uses asymmetric soft-capping on logits and a sparse-attention gate on BOS specifically to suppress sink dynamics. Register tokens provide a **dedicated** sink absorber per head that may dominate BOS, and the 4 extra tokens are nearly free. Z-loss (α ≈ 1 e−4) is well-known to stabilize FP8 heads — exactly the regime of the FP8 LM-head in the record. Together they may permit *removing* the asymmetric softcap constants and simplifying the head path.

**Why it might not.** Register tokens may double-count with BOS — the BOS-alignment recipe already absorbs much of the sink mass. Z-loss adds a second regularizer to an already-regularized head (softcap is there partly for the same reason); tuning α without destabilizing FP8 requires a sweep.

**Effort.** Low. Register tokens: ~30 LoC. Z-loss: ~5 LoC.

**Suggestions.** Start with **4 register tokens** (one per two attention heads) appended to each document, masked so registers can attend only within their own document group. Z-loss α ∈ {1e-5, 1e-4, 1e-3}. If the combination works, test softcap removal; if not, treat as additive to softcap.

---

## 10. LAWA + WSM: eliminate LR decay, recover it via checkpoint merging

**Technique.** Cross-pollinate two closely related ideas. LAWA (Sanyal et al., arXiv:2306.03241): sliding-window uniform average of k checkpoints sampled at wide interval ν. WSM (Tian et al., arXiv:2507.17634): *delete* LR decay entirely; replace with weighted merge of the last N checkpoints using cosine or 1-sqrt weights — Ling 2.0 reports +1–2 % average on benchmarks vs WSD. Key insight: the LR-decay phase of the speedrun is ~10–15 % of wallclock wasted at suboptimal LR; if the decayed solution can be synthesized offline from peak-LR checkpoints, those steps become full-LR productive steps.

**Why it might win.** LAWA was shown by Sanyal et al. to give *largest* gains early in training at high LR — the speedrun regime. WSM lets you convert the 200-step linear cooldown into additional peak-LR steps plus a free checkpoint merge. The two compose: use LAWA mid-run for variance reduction, WSM at the end for decay emulation. 124M × 4B × 8 checkpoints ≈ 4 GB, trivially fits in HBM.

**Why it might not.** Removing the cooldown requires re-tuning peak LR — too-aggressive LR without decay diverges. The WSM paper's reported gains (+1–2 %) are on downstream benchmarks, not val loss; on a fixed val metric they may shrink. Variance of checkpoint-averaged weights interacts with scalar smoothing (record #52) — may need to disable scalar freezing during the would-be cooldown window.

**Effort.** Low-medium. ~100 LoC. Must re-tune peak LR.

**Suggestions.** Instead of linear cooldown over last 45 % of steps, run peak LR for 100 % of steps, then merge the last 10 checkpoints with 1-sqrt weights at eval time only. Checkpoint interval: every 50 steps from 70 % of run onward.

---

## 11. Sequence-length curriculum (distinct from the existing attention-window warmup)

**Technique.** Li et al. (arXiv:2108.06084, "Sequence-Length Warmup"), extended with OpenLM variable-sequence curriculum. Train with actual shorter sequences in early steps (say 512 for the first 30 %, 1024 for middle, 2048 for final), not just a shorter attention window over a 2048-token sequence.

**Why it might win.** The current record warms only the FlexAttention *window*; the underlying sequence length stays at ~2048. Attention FLOPs scale with L², so halving L in the first third frees non-trivial FLOPs. At 124M, roughly 15–20 % of step time is attention-bound; a 512-token first phase recovers ~10 % of that phase's step cost. Document-packing with `cu_seqlens` already naturally supports variable L.

**Why it might not.** Breaks torch.compile caching: three graph shapes instead of one, adding compile-warmup time that may exceed the ~1 s saved. The intra-document masking's `max_doc_len=2048` was tightened in record #78 specifically to match FineWeb — shrinking further may hurt convergence. Partially redundant with the existing attention-window schedule.

**Effort.** Medium. Requires three AOT-compiled shapes and careful warmup section update.

**Suggestions.** Use shapes L∈{1024, 1536, 2048} transitioning at 30 % and 60 % of steps, co-scheduled with the existing attention-window schedule. Pre-compile all three shapes in the explicit warmup region.

---

## 12. Full-step CUDA Graph capture including Muon Newton–Schulz and NCCL ops

**Technique.** PyTorch's CUDA Graph Trees (since 2.2) can capture fwd + bwd + NCCL all-reduce/reduce-scatter + optimizer step as a single replayable graph. Kernel launch overhead (~3–6 µs × ~150 kernels ≈ 0.5–1 ms per step) is eliminated, plus Python-side overhead (~4–5 ms of the current ~50 ms step).

**Why it might win.** Modded-nanogpt already uses torch.compile and has reached the wall where Python and launch overhead are meaningful. The three phase transitions (attention-window changes at 1/3 and 2/3, FP8 head retying at 2/3) force graph re-captures — but within each phase, a single fixed-shape step can be captured. Muon's Newton–Schulz is iterative with static shapes, so it is graph-capturable. NCCL ops are graph-compatible since 2.2. Expected savings at this scale are 1–3 %.

**Why it might not.** Data-dependent control flow in the current code (scalar smoothing, per-transition scalar freezing, YaRN rescaling) may not be static; those branches force graph variants. The explicit kernel-warmup region would need to seed graphs for every distinct shape combination.

**Effort.** Medium. ~1–2 days to thread `cudagraph_trees` correctly through DistAdam's two-tier path.

**Suggestions.** Capture **three graphs** (one per attention-window phase) and toggle between them. Move scalar smoothing and freezing into graph-external code paths. Use `torch._inductor.config.triton.cudagraphs=True` with `reduce-overhead` mode (default under `max-autotune`).

---

## 13. Stochastic-rounding BF16 AdamW states + DeepSeek-style block-scaled FP8 on MLP linears

**Technique.** Combine two precision tricks that go beyond the current FP8-head-only regime. (a) Amazon's "Stochastic Rounding for LLM Training" (arXiv:2502.20566) and llm.c PR #772 show BF16 first/second moments with stochastic rounding *match FP32 master-weight AdamW* at 7B scale with higher throughput. (b) DeepSeek-V3 block-scaled FP8 (arXiv:2412.19437): weights in 128×128 blocks, activations in 1×128 tiles, with CUDA-core promotion to FP32 every Nc=128 partial products. The 768×768 matmul size that killed per-tensor FP8 in the current record is *not* a blocker for block-scaled FP8 because scaling is finer-grained.

**Why it might win.** (a) eliminates FP32 master weights for the AdamW-optimized legs, reducing memory traffic in DistAdam's reduce-scatter/all-gather path (embeddings and head are ~40 M params; removing the FP32 mirror shaves comms by ~4 bytes × 40 M). (b) The two MLP linears per layer (up-proj 768→3072, down-proj 3072→768) are the largest compute chunks in the step and are *not yet FP8* — the earlier FP8-everywhere attempt failed on per-tensor scaling, but 128×128 block scaling was not tried. DeepSeek reports <0.25 % loss gap vs BF16 at 670B/14.8T tokens, well within the ±0.005 val-loss margin.

**Why it might not.** (b) needs a CUTLASS-built FP8 block-scaled GEMM extension (CUTLASS example 67) — substantial engineering. At 124M the MLP matmuls may still be too small to achieve the FP8 throughput advantage, similar to why the original FP8 experiment failed. Stacking block-scaled FP8 activations with the current Smear module and value-embedding mixing inside attention has not been tested and may produce outlier distributions that even block scaling can't contain. Stability on the tight ±0.005 loss margin is the principal risk.

**Effort.** (a) low (~50 LoC in the optimizer step). (b) high (~3–5 days for a clean CUTLASS/TE Float8BlockScaling integration).

**Suggestions.** Do (a) first as a quick win on the AdamW legs. For (b), start with just the up-projection (768→3072) where FP8 headroom is largest; keep down-projection BF16; measure val loss across 5 seeds before committing. The *activation* tile of 1×128 aligns well with the 3072 inner dimension.

---

## Honorable mentions not expanded into full sections

- **Scalable-Softmax + Forgetting-Transformer stack**: if both #5 and #8 land, the YaRN transition schedule can likely be simplified or removed entirely — a secondary ~0.5 s win and a large complexity reduction.
- **FlexAttention's new FA4 CuTeDSL backend** (PyTorch blog, Nov 2025): reports 1.2–3.2× forward / 1.85–2.3× backward over Triton on GB200, consistent gains on Hopper. Drop-in backend flag if the score_mod/mask_mod patterns (value-embedding mixing inside attention) are CuTeDSL-lowerable. Worth verifying compatibility before counting on it.
- **Per-group weight-decay sweep** specifically on norms, gate parameters, and the FP8 head: current `wd_mul=150.` on lm_head is aggressive; systematic `wd_mul ∈ {50, 75, 100, 150, 200}` sweep is cheap.
- **Power scheduler** (arXiv:2408.13359): LR schedule provably invariant to batch and token count; may give a slightly better peak-LR choice than the current tuned value, with zero additional state.
- **u-µP** (Graphcore, arXiv:2407.17465): unit-scaled µP may tighten FP8 numerics and provide principled per-param LR scaling — useful if #13(b) is attempted.

## Synthesis

The highest-probability, highest-expected-value bets at this point are the ones that **compose with Muon + FP8 head without redesigning them**: C-Muon (1 line), SSMax (20 LoC), register tokens + z-loss (35 LoC), LAWA/WSM (100 LoC), stochastic-rounding BF16 AdamW state (50 LoC), and CUDA-graph capture (1–2 days). Taken together these occupy the bottom-left corner of the effort × probability plane and plausibly compose for a 5–10 s reduction on the current ~86 s budget. The medium-probability, medium-effort bets — Scion-style unified optimizer, DyT, xIELU, FoX on long layers, FA4 backend — each carry individual 2–5 s upside but with higher regression risk and require more ablation rigor. The high-variance bets — Mano/Dion for Muon replacement, DeepSeek-style block-scaled FP8 on MLP linears, sequence-length curriculum — are research-grade and should be staged behind the cheaper wins. The pattern in the record history (optimizer/scalar micro-tweaks dominating recent PRs) strongly suggests the next 5 s will be captured by three-to-four of the category-1 ideas in stacked combination, not by any single heroic architectural swap.