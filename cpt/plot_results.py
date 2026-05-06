"""Aggregate the 12 result JSONs and produce the headline figure + summary table."""

import argparse
import json
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


COND_LABEL = {
    ("constant_low", 0.00): "(a) constant low LR",
    ("constant_low", 0.05): "(c) constant low + 5% replay",
    ("rewarm_cosine", 0.00): "(b) re-warmed cosine",
    ("rewarm_cosine", 0.05): "(d) re-warm + 5% replay",
}

COND_COLOR = {
    ("constant_low", 0.00): "#888888",
    ("constant_low", 0.05): "#3b7dd8",
    ("rewarm_cosine", 0.00): "#d83b3b",
    ("rewarm_cosine", 0.05): "#33aa55",
}

COND_ORDER = [
    ("constant_low", 0.00),
    ("constant_low", 0.05),
    ("rewarm_cosine", 0.00),
    ("rewarm_cosine", 0.05),
]


def load_runs(results_dir: Path) -> dict:
    grouped: dict[tuple[str, float], list[dict]] = defaultdict(list)
    for f in sorted(results_dir.glob("*.json")):
        d = json.loads(f.read_text())
        cfg = d["config"]
        key = (cfg["schedule"], round(cfg["replay"], 2))
        grouped[key].append(d)
    return grouped


def stack_curves(runs: list[dict], field: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    steps = np.array([h["step"] for h in runs[0]["history"]])
    matrix = np.array([[h[field] for h in r["history"]] for r in runs])
    return steps, matrix.mean(axis=0), matrix.std(axis=0)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="cpt/results")
    ap.add_argument("--out", default="cpt/figure_main.png")
    ap.add_argument("--summary", default="cpt/summary.txt")
    args = ap.parse_args()

    grouped = load_runs(Path(args.results))
    if not grouped:
        raise SystemExit(f"no result JSONs in {args.results}")

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))

    for key in COND_ORDER:
        if key not in grouped:
            continue
        runs = grouped[key]
        label = COND_LABEL[key] + f"  (n={len(runs)})"
        color = COND_COLOR[key]

        steps, web_mean, web_std = stack_curves(runs, "web_nll")
        steps, code_mean, code_std = stack_curves(runs, "code_nll")

        axes[0].plot(steps, web_mean, label=label, color=color, lw=1.8)
        axes[0].fill_between(steps, web_mean - web_std, web_mean + web_std, color=color, alpha=0.15)

        axes[1].plot(steps, code_mean, label=label, color=color, lw=1.8)
        axes[1].fill_between(steps, code_mean - code_std, code_mean + code_std, color=color, alpha=0.15)

        # Pareto scatter at final step: forgetting (Δweb) vs adaptation (Δcode)
        web0 = np.array([r["baseline_web_nll"] for r in runs])
        code0 = np.array([r["baseline_code_nll"] for r in runs])
        webT = np.array([r["history"][-1]["web_nll"] for r in runs])
        codeT = np.array([r["history"][-1]["code_nll"] for r in runs])
        d_web = webT - web0
        d_code = codeT - code0
        axes[2].errorbar(d_web.mean(), d_code.mean(),
                          xerr=d_web.std(), yerr=d_code.std(),
                          fmt="o", color=color, markersize=10,
                          ecolor=color, elinewidth=1.5, capsize=4, label=label)
        axes[2].annotate(COND_LABEL[key].split(" ")[0],
                          (d_web.mean(), d_code.mean()),
                          textcoords="offset points", xytext=(8, 4), fontsize=10, color=color)

    axes[0].set_title("Old domain: FineWeb val NLL\n(lower = less forgetting)")
    axes[0].set_xlabel("CPT step")
    axes[0].set_ylabel("nats / token")
    axes[0].grid(alpha=0.3)
    axes[0].legend(loc="lower right", fontsize=9)

    axes[1].set_title("New domain: Python val NLL\n(lower = better adaptation)")
    axes[1].set_xlabel("CPT step")
    axes[1].set_ylabel("nats / token")
    axes[1].grid(alpha=0.3)

    axes[2].set_title("Pareto: forgetting vs adaptation\n(bottom-left is best)")
    axes[2].set_xlabel("Δ FineWeb NLL  (forgetting)")
    axes[2].set_ylabel("Δ Python NLL  (adaptation)")
    axes[2].axhline(0, color="k", lw=0.5, alpha=0.4)
    axes[2].axvline(0, color="k", lw=0.5, alpha=0.4)
    axes[2].grid(alpha=0.3)
    axes[2].invert_yaxis()  # so "more adaptation" is up

    plt.tight_layout()
    plt.savefig(args.out, dpi=140)
    print(f"wrote {args.out}")

    # ------------------------------------------------------------------ summary
    lines = []
    lines.append(f"{'condition':<32}  {'n':>2}  {'web0':>6}  {'webT':>6}  {'Δweb':>7}  {'code0':>6}  {'codeT':>6}  {'Δcode':>7}")
    lines.append("-" * 92)
    for key in COND_ORDER:
        if key not in grouped:
            continue
        runs = grouped[key]
        web0 = np.array([r["baseline_web_nll"] for r in runs])
        code0 = np.array([r["baseline_code_nll"] for r in runs])
        webT = np.array([r["history"][-1]["web_nll"] for r in runs])
        codeT = np.array([r["history"][-1]["code_nll"] for r in runs])
        d_web = webT - web0
        d_code = codeT - code0
        label = COND_LABEL[key]
        lines.append(
            f"{label:<32}  {len(runs):>2}  "
            f"{web0.mean():>6.3f}  {webT.mean():>6.3f}  {d_web.mean():>+7.3f}  "
            f"{code0.mean():>6.3f}  {codeT.mean():>6.3f}  {d_code.mean():>+7.3f}"
        )
        lines.append(
            f"{'  std':<32}  {'':>2}  {web0.std():>6.3f}  {webT.std():>6.3f}  {d_web.std():>7.3f}  "
            f"{code0.std():>6.3f}  {codeT.std():>6.3f}  {d_code.std():>7.3f}"
        )

    summary = "\n".join(lines)
    print("\n" + summary)
    Path(args.summary).write_text(summary + "\n")
    print(f"\nwrote {args.summary}")


if __name__ == "__main__":
    main()
