"""Pairwise significance tests across the four conditions.

We compute Welch's t-test on the per-seed Δweb and Δcode values for each
relevant pair. With n=3 per group this is a low-power test, but here the
std is at the third-decimal level while the effect sizes are 0.05–0.25
nats — so the question is mostly about putting numbers on what the figure
already shows.
"""

import json
import math
from pathlib import Path

import numpy as np


COND_LABEL = {
    ("constant_low", 0.00): "(a) constant low LR",
    ("constant_low", 0.05): "(c) constant low + 5% replay",
    ("rewarm_cosine", 0.00): "(b) re-warmed cosine",
    ("rewarm_cosine", 0.05): "(d) re-warm + 5% replay",
}


def welch_t(a: np.ndarray, b: np.ndarray) -> tuple[float, float, float]:
    """Welch's t-test: returns (t, df, two-sided p)."""
    na, nb = len(a), len(b)
    va, vb = a.var(ddof=1), b.var(ddof=1)
    diff = a.mean() - b.mean()
    se = math.sqrt(va / na + vb / nb)
    t = diff / se if se > 0 else float("inf")
    # Welch–Satterthwaite df
    df = (va / na + vb / nb) ** 2 / ((va / na) ** 2 / (na - 1) + (vb / nb) ** 2 / (nb - 1)) if (va > 0 and vb > 0) else max(na + nb - 2, 1)
    # two-sided p via t-distribution survival; small-sample so use survival fn
    # approximation via Cornish–Fisher would be overkill; use scipy if available
    try:
        from scipy.stats import t as tdist
        p = 2 * (1 - tdist.cdf(abs(t), df))
    except ImportError:
        # crude normal approximation -> overestimates p for low df, but with
        # |t| > 30 here it doesn't matter
        from math import erf
        p = 2 * (1 - 0.5 * (1 + erf(abs(t) / math.sqrt(2))))
    return t, df, p


def deltas(runs: list[dict]) -> tuple[np.ndarray, np.ndarray]:
    web0 = np.array([r["baseline_web_nll"] for r in runs])
    code0 = np.array([r["baseline_code_nll"] for r in runs])
    webT = np.array([r["history"][-1]["web_nll"] for r in runs])
    codeT = np.array([r["history"][-1]["code_nll"] for r in runs])
    return webT - web0, codeT - code0


def main() -> None:
    grouped = {}
    for f in sorted(Path("cpt/results").glob("*.json")):
        d = json.loads(f.read_text())
        cfg = d["config"]
        key = (cfg["schedule"], round(cfg["replay"], 2))
        grouped.setdefault(key, []).append(d)

    d_web = {k: deltas(v)[0] for k, v in grouped.items()}
    d_code = {k: deltas(v)[1] for k, v in grouped.items()}

    # Per-condition stats
    print(f"{'condition':<32}  {'Δweb mean':>10}  {'Δweb std':>10}  {'Δcode mean':>11}  {'Δcode std':>10}")
    print("-" * 80)
    for key, label in COND_LABEL.items():
        if key not in d_web:
            continue
        w, c = d_web[key], d_code[key]
        print(f"{label:<32}  {w.mean():>+10.4f}  {w.std(ddof=1):>10.4f}  {c.mean():>+11.4f}  {c.std(ddof=1):>10.4f}")

    print()

    # Pairwise comparisons that matter
    pairs = [
        ("(a) vs (b)  schedule effect (no replay)",     ("constant_low", 0.0), ("rewarm_cosine", 0.0)),
        ("(a) vs (c)  replay effect (constant LR)",      ("constant_low", 0.0), ("constant_low", 0.05)),
        ("(b) vs (d)  replay effect (cosine LR)",        ("rewarm_cosine", 0.0), ("rewarm_cosine", 0.05)),
        ("(c) vs (d)  schedule effect (with replay)",    ("constant_low", 0.05), ("rewarm_cosine", 0.05)),
    ]
    print(f"{'pairwise comparison':<48}  {'metric':<6}  {'mean diff':>9}  {'t':>8}  {'df':>6}  {'p':>10}  {'sig?'}")
    print("-" * 105)
    for desc, k1, k2 in pairs:
        for metric_name, dct in [("Δweb", d_web), ("Δcode", d_code)]:
            a = dct[k1]
            b = dct[k2]
            t, df, p = welch_t(a, b)
            sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "ns"
            print(f"{desc:<48}  {metric_name:<6}  {a.mean()-b.mean():>+9.4f}  {t:>8.2f}  {df:>6.2f}  {p:>10.2e}  {sig}")


if __name__ == "__main__":
    main()
