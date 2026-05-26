"""
Bootstrap 95% CI + figure regeneration for BPC EMNLP paper.
Reads per-example JSONL files, computes bootstrap CIs for F1 and Hits@1,
then regenerates three PDF figures with error bars.

Usage:  python analysis/bootstrap_and_figures.py
Output: paper_emnlp/figures/qwen35_k_sweep.pdf
        paper_emnlp/figures/qwen35_tradeoff.pdf
        paper_emnlp/figures/qwen35_random_gap.pdf
        + CI table printed to stdout
"""

import json, pathlib, sys
import numpy as np

ROOT = pathlib.Path(__file__).resolve().parent.parent
RES  = ROOT / "results"
FIG  = ROOT / "paper_emnlp" / "figures"
FIG.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# 1. File registry  (model, dataset, method_K, path)
# ---------------------------------------------------------------------------
FILES = {
    # --- 9B BPC ---
    ("9B", "WebQSP", "BPC/0"):    RES / "qwen35_awq_tp2_nothink_9h_probe"        / "bpc_webqsp_n999999_d5_s42_K0.jsonl",
    ("9B", "WebQSP", "BPC/1"):    RES / "qwen35_9b_awq_bpc_k1_k2_cuda02"         / "bpc_webqsp_n999999_d5_s42_K1.jsonl",
    ("9B", "WebQSP", "BPC/2"):    RES / "qwen35_9b_awq_bpc_k1_k2_cuda02"         / "bpc_webqsp_n999999_d5_s42_K2.jsonl",
    ("9B", "WebQSP", "BPC/full"): RES / "qwen35_9b_awq_bpc_kfull_webqsp_cuda01"   / "bpc_webqsp_n999999_d5_s42_Kfull.jsonl",
    ("9B", "CWQ",    "BPC/0"):    RES / "qwen35_awq_tp2_nothink_9h_probe"        / "bpc_cwq_n999999_d5_s42_K0.jsonl",
    ("9B", "CWQ",    "BPC/1"):    RES / "qwen35_9b_awq_bpc_k1_k2_cuda02"         / "bpc_cwq_n999999_d5_s42_K1.jsonl",
    ("9B", "CWQ",    "BPC/2"):    RES / "qwen35_9b_awq_bpc_k1_k2_cuda02"         / "bpc_cwq_n999999_d5_s42_K2.jsonl",
    ("9B", "CWQ",    "BPC/full"): RES / "qwen35_9b_awq_bpc_kfull_cwq_cuda23_retry4096_mem078" / "bpc_cwq_n999999_d5_s42_Kfull.jsonl",
    # --- 9B Controls ---
    ("9B", "WebQSP", "CoT"):      RES / "rev_p0" / "cot_webqsp_n999999_d5_s42_K0.jsonl",
    ("9B", "WebQSP", "Random"):   RES / "rev_cuda01_random_9b" / "random_webqsp_n999999_d5_s42_K0.jsonl",
    ("9B", "WebQSP", "ToG"):      RES / "rev_tog_9b" / "tog_webqsp_n999999_d5_s42_K0.jsonl",
    ("9B", "CWQ",    "CoT"):      RES / "rev_p0" / "cot_cwq_n999999_d5_s42_K0.jsonl",
    ("9B", "CWQ",    "Random"):   RES / "rev_cuda01_random_9b" / "random_cwq_n999999_d5_s42_K0.jsonl",
    ("9B", "CWQ",    "ToG"):      RES / "rev_tog_9b" / "tog_cwq_n999999_d5_s42_K0.jsonl",
    # --- 4B BPC ---
    ("4B", "WebQSP", "BPC/0"):    RES / "qwen35_4b_awq_bpc_k0_full_cuda3"  / "bpc_webqsp_n999999_d5_s42_K0.jsonl",
    ("4B", "WebQSP", "BPC/1"):    RES / "rev_p0" / "bpc_webqsp_n999999_d5_s42_K1.jsonl",
    ("4B", "WebQSP", "BPC/2"):    RES / "rev_4b_k2_cuda0" / "bpc_webqsp_n999999_d5_s42_K2.jsonl",
    ("4B", "WebQSP", "BPC/full"): RES / "qwen35_4b_awq_bpc_kfull_webqsp_cuda0" / "bpc_webqsp_n999999_d5_s42_Kfull.jsonl",
    ("4B", "CWQ",    "BPC/0"):    RES / "qwen35_4b_awq_bpc_k0_full_cuda3"  / "bpc_cwq_n999999_d5_s42_K0.jsonl",
    ("4B", "CWQ",    "BPC/1"):    RES / "rev_p0" / "bpc_cwq_n999999_d5_s42_K1.jsonl",
    ("4B", "CWQ",    "BPC/2"):    RES / "rev_4b_k2_cuda0" / "bpc_cwq_n999999_d5_s42_K2.jsonl",
    ("4B", "CWQ",    "BPC/full"): RES / "qwen35_4b_awq_bpc_kfull_cwq_cuda1" / "bpc_cwq_n999999_d5_s42_Kfull.jsonl",
    # --- 4B Controls ---
    ("4B", "WebQSP", "CoT"):      RES / "rev_4b_cot_tog_cuda1" / "cot_webqsp_n999999_d5_s42_K0.jsonl",
    ("4B", "WebQSP", "Random"):   RES / "qwen35_4b_awq_random_cuda3" / "random_webqsp_n999999_d5_s42_K0.jsonl",
    ("4B", "WebQSP", "ToG"):      RES / "rev_4b_cot_tog_cuda1" / "tog_webqsp_n999999_d5_s42_K0.jsonl",
    ("4B", "CWQ",    "CoT"):      RES / "rev_4b_cot_tog_cuda1" / "cot_cwq_n999999_d5_s42_K0.jsonl",
    ("4B", "CWQ",    "Random"):   RES / "qwen35_4b_awq_random_cuda3" / "random_cwq_n999999_d5_s42_K0.jsonl",
    ("4B", "CWQ",    "ToG"):      RES / "rev_4b_cot_tog_cuda1" / "tog_cwq_n999999_d5_s42_K0.jsonl",
}

# Cost data from paper manifest (input tokens in millions)
COST = {
    ("9B", "WebQSP", "BPC/0"):    13.92,
    ("9B", "WebQSP", "BPC/1"):    13.43,
    ("9B", "WebQSP", "BPC/2"):    14.27,
    ("9B", "WebQSP", "BPC/full"): 14.87,
    ("9B", "WebQSP", "CoT"):       0.09,
    ("9B", "WebQSP", "Random"):    1.18,
    ("9B", "CWQ",    "BPC/0"):    36.69,
    ("9B", "CWQ",    "BPC/1"):    38.09,
    ("9B", "CWQ",    "BPC/2"):    39.98,
    ("9B", "CWQ",    "BPC/full"): 41.74,
    ("9B", "CWQ",    "CoT"):       0.23,
    ("9B", "CWQ",    "Random"):    2.52,
    ("9B", "WebQSP", "ToG"):       5.84,
    ("9B", "CWQ",    "ToG"):      28.95,
    ("4B", "WebQSP", "BPC/0"):    12.94,
    ("4B", "WebQSP", "BPC/1"):    11.45,
    ("4B", "WebQSP", "BPC/2"):    11.59,
    ("4B", "WebQSP", "BPC/full"): 12.08,
    ("4B", "WebQSP", "CoT"):       0.09,
    ("4B", "WebQSP", "Random"):    1.18,
    ("4B", "WebQSP", "ToG"):       9.50,
    ("4B", "CWQ",    "BPC/0"):    33.39,
    ("4B", "CWQ",    "BPC/1"):    31.58,
    ("4B", "CWQ",    "BPC/2"):    29.32,
    ("4B", "CWQ",    "BPC/full"): 17.32,
    ("4B", "CWQ",    "CoT"):       0.23,
    ("4B", "CWQ",    "Random"):    2.52,
    ("4B", "CWQ",    "ToG"):      44.67,
}

# ---------------------------------------------------------------------------
# 2. Load per-example metrics
# ---------------------------------------------------------------------------
def load_jsonl(path):
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            rows.append(json.loads(line))
    return rows

data = {}  # key -> {"f1": array, "hits1": array, "n": int}
missing = []
for key, path in FILES.items():
    if not path.exists():
        missing.append((key, path))
        continue
    rows = load_jsonl(path)
    data[key] = {
        "f1":    np.array([r["f1"] for r in rows]),
        "hits1": np.array([r["hits1"] for r in rows]),
        "n":     len(rows),
    }

if missing:
    print("WARNING: missing files:")
    for k, p in missing:
        print(f"  {k} -> {p}")
    print()

# ---------------------------------------------------------------------------
# 3. Bootstrap CI
# ---------------------------------------------------------------------------
B = 10_000
rng = np.random.default_rng(42)

ci_results = {}  # key -> {metric: (mean, lo, hi)}
for key, d in data.items():
    ci_results[key] = {}
    for metric_name in ("f1", "hits1"):
        arr = d[metric_name]
        n = len(arr)
        # resample indices
        idx = rng.integers(0, n, size=(B, n))
        means = arr[idx].mean(axis=1)
        lo, hi = np.percentile(means, [2.5, 97.5])
        mu = arr.mean()
        ci_results[key][metric_name] = (mu, lo, hi)

# Print CI table
print("=" * 90)
print(f"{'Model':<5} {'Dataset':<8} {'Method':<10}  {'N':>5}  "
      f"{'F1':>6} {'CI_lo':>6} {'CI_hi':>6}   "
      f"{'H@1':>6} {'CI_lo':>6} {'CI_hi':>6}")
print("-" * 90)
for key in sorted(ci_results.keys()):
    model, dataset, method = key
    n = data[key]["n"]
    f1_mu, f1_lo, f1_hi = ci_results[key]["f1"]
    h1_mu, h1_lo, h1_hi = ci_results[key]["hits1"]
    print(f"{model:<5} {dataset:<8} {method:<10}  {n:>5}  "
          f"{f1_mu:6.3f} {f1_lo:6.3f} {f1_hi:6.3f}   "
          f"{h1_mu:6.3f} {h1_lo:6.3f} {h1_hi:6.3f}")
print("=" * 90)

# Also print LaTeX-friendly CI strings
print("\n--- LaTeX-ready CI (F1) ---")
for key in sorted(ci_results.keys()):
    model, dataset, method = key
    f1_mu, f1_lo, f1_hi = ci_results[key]["f1"]
    hw = (f1_hi - f1_lo) / 2
    print(f"  {model} {dataset} {method}: "
          f"${f1_mu:.3f} \\pm {hw:.3f}$ "
          f"  [{f1_lo:.3f}, {f1_hi:.3f}]")

# ---------------------------------------------------------------------------
# 4. Figures
# ---------------------------------------------------------------------------
try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import FixedLocator
except ImportError:
    print("\nERROR: matplotlib not installed; skipping figures.", file=sys.stderr)
    sys.exit(0)

# ---- Global style ----
plt.rcParams.update({
    "font.family": "serif",
    "font.size": 9,
    "axes.labelsize": 10,
    "axes.titlesize": 11,
    "legend.fontsize": 8,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "figure.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.05,
})

COLORS = {
    "BPC/0":  "#2176AE",
    "BPC/1":  "#57B8FF",
    "BPC/2":  "#B0D0E8",
    "BPC/full": "#F76C5E",
    "CoT":    "#F5A623",
    "Random": "#8B8B8B",
    "ToG":    "#6A0DAD",
}
MARKERS = {
    "BPC/0": "o", "BPC/1": "s", "BPC/2": "D",
    "BPC/full": "^", "CoT": "X", "Random": "v",
    "ToG": "P",
}

K_ORDER = ["BPC/0", "BPC/1", "BPC/2", "BPC/full"]
K_LABELS = ["$K{=}0$", "$K{=}1$", "$K{=}2$", "Full"]

# ====== Figure 1: K-sweep with CI error bars (2-panel) ======
fig, axes = plt.subplots(1, 2, figsize=(6.5, 2.6), sharey=False)

for ax_i, dataset in enumerate(["WebQSP", "CWQ"]):
    ax = axes[ax_i]
    ax2 = ax.twinx()

    f1_vals, f1_errs, tok_vals = [], [], []
    for method in K_ORDER:
        key = ("9B", dataset, method)
        mu, lo, hi = ci_results[key]["f1"]
        f1_vals.append(mu)
        f1_errs.append([mu - lo, hi - mu])
        tok_vals.append(COST[key])

    x = np.arange(len(K_ORDER))
    err_arr = np.array(f1_errs).T  # shape (2, 4)

    # F1 bars with error bars
    bars = ax.bar(x - 0.18, f1_vals, 0.36,
                  color=[COLORS[m] for m in K_ORDER],
                  edgecolor="white", linewidth=0.5,
                  yerr=err_arr, capsize=3,
                  error_kw={"linewidth": 0.8, "capthick": 0.8},
                  label="F1", zorder=3)

    # Token line on right axis
    ax2.plot(x, tok_vals, "k--o", markersize=4, linewidth=1.2,
             label="Input tokens (M)", zorder=4)

    ax.set_xticks(x)
    ax.set_xticklabels(K_LABELS)
    ax.set_ylabel("Answer F1")
    ax.set_title(dataset, fontweight="bold")

    if dataset == "WebQSP":
        ax.set_ylim(0.44, 0.52)
    else:
        ax.set_ylim(0.24, 0.32)

    ax2.set_ylabel("Input tokens (M)", color="gray")
    ax2.tick_params(axis="y", labelcolor="gray")

    # Grid
    ax.set_axisbelow(True)
    ax.yaxis.grid(True, linewidth=0.3, alpha=0.5)
    ax.set_xlabel("Visible history bound")

    # Annotate best
    best_idx = int(np.argmax(f1_vals))
    ax.annotate(f"{f1_vals[best_idx]:.3f}",
                xy=(x[best_idx] - 0.18, f1_vals[best_idx] + err_arr[1, best_idx]),
                ha="center", va="bottom", fontsize=7, fontweight="bold",
                color=COLORS[K_ORDER[best_idx]])

fig.tight_layout(w_pad=2.5)
out1 = FIG / "qwen35_k_sweep.pdf"
fig.savefig(out1)
print(f"\nSaved: {out1}")
plt.close(fig)

# ====== Figure 2: Accuracy-cost tradeoff (9B) ======
fig, axes = plt.subplots(1, 2, figsize=(6.5, 2.8), sharey=False)

all_methods_9b = ["BPC/0", "BPC/1", "BPC/2", "BPC/full", "CoT", "Random", "ToG"]

for ax_i, dataset in enumerate(["WebQSP", "CWQ"]):
    ax = axes[ax_i]
    for method in all_methods_9b:
        key = ("9B", dataset, method)
        if key not in ci_results:
            continue
        mu, lo, hi = ci_results[key]["f1"]
        cost = COST[key]
        ax.errorbar(cost, mu, yerr=[[mu - lo], [hi - mu]],
                     fmt=MARKERS.get(method, "o"),
                     color=COLORS.get(method, "gray"),
                     markersize=7, capsize=3, linewidth=1,
                     label=method, zorder=5)
        # Label
        label_text = method.replace("BPC/", "$K{=}$").replace("full", "full")
        if method.startswith("BPC/"):
            label_text = method.split("/")[1]
            if label_text != "full":
                label_text = f"$K{{=}}{label_text}$"
            else:
                label_text = "Full"
        offset_x, offset_y = 5, 3
        if method == "CoT":
            offset_x, offset_y = 5, -8
        elif method == "Random":
            offset_x, offset_y = 5, -8
        elif method == "ToG":
            offset_x, offset_y = 5, -8
        ax.annotate(label_text,
                    xy=(cost, mu), xytext=(offset_x, offset_y),
                    textcoords="offset points", fontsize=7,
                    color=COLORS.get(method, "gray"))

    ax.set_xlabel("Input tokens (M)")
    ax.set_ylabel("Answer F1")
    ax.set_title(dataset, fontweight="bold")
    ax.set_axisbelow(True)
    ax.grid(True, linewidth=0.3, alpha=0.5)

    # Arrow annotation for "better" direction
    ax.annotate("", xy=(0.15, 0.85), xytext=(0.35, 0.65),
                xycoords="axes fraction",
                arrowprops=dict(arrowstyle="->", color="gray", lw=0.8))
    ax.text(0.12, 0.88, "better", transform=ax.transAxes,
            fontsize=7, color="gray", fontstyle="italic")

fig.tight_layout(w_pad=2.5)
out2 = FIG / "qwen35_tradeoff.pdf"
fig.savefig(out2)
print(f"Saved: {out2}")
plt.close(fig)

# ====== Figure 3: Random gap (both 9B and 4B) ======
fig, axes = plt.subplots(1, 2, figsize=(6.5, 2.8))

for ax_i, dataset in enumerate(["WebQSP", "CWQ"]):
    ax = axes[ax_i]

    # Group: [9B BPC, 9B Random, 9B CoT, 4B BPC, 4B Random]
    groups = []
    labels = []
    colors = []
    errs_lo, errs_hi = [], []

    configs = [
        ("9B", "BPC/0",  "9B BPC\n$K{=}0$",   "#2176AE"),
        ("9B", "CoT",    "9B CoT",              "#F5A623"),
        ("9B", "Random", "9B\nRandom",           "#8B8B8B"),
        ("4B", "BPC/0",  "4B BPC\n$K{=}0$",   "#2176AE"),
        ("4B", "CoT",    "4B CoT",              "#F5A623"),
        ("4B", "Random", "4B\nRandom",           "#8B8B8B"),
    ]

    for model, method, lbl, col in configs:
        key = (model, dataset, method)
        if key not in ci_results:
            continue
        mu, lo, hi = ci_results[key]["f1"]
        groups.append(mu)
        labels.append(lbl)
        colors.append(col)
        errs_lo.append(mu - lo)
        errs_hi.append(hi - mu)

    x = np.arange(len(groups))
    err_arr = np.array([errs_lo, errs_hi])

    bars = ax.bar(x, groups, 0.55, color=colors,
                  edgecolor="white", linewidth=0.5,
                  yerr=err_arr, capsize=3,
                  error_kw={"linewidth": 0.8, "capthick": 0.8},
                  zorder=3)

    # Value labels on top of each bar
    for i, (val, elo, ehi) in enumerate(zip(groups, errs_lo, errs_hi)):
        ax.text(i, val + ehi + 0.008, f".{int(val*1000):03d}",
                ha="center", va="bottom", fontsize=7, fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=7)
    ax.set_ylabel("Answer F1")
    ax.set_title(dataset, fontweight="bold")
    ax.set_axisbelow(True)
    ax.yaxis.grid(True, linewidth=0.3, alpha=0.5)
    ax.set_ylim(0, max(groups) * 1.25)

    # Separator between 9B and 4B groups
    n_9b = sum(1 for m, meth, _, _ in configs if m == "9B" and (m, dataset, meth) in ci_results)
    sep_x = n_9b - 0.5
    ax.axvline(sep_x, color="gray", linestyle=":", linewidth=0.8)
    ax.text((n_9b - 1) / 2, max(groups) * 1.18, "9B", ha="center",
            fontsize=8, color="gray", fontstyle="italic")
    n_4b = len(groups) - n_9b
    ax.text(n_9b + (n_4b - 1) / 2, max(groups) * 1.18, "4B", ha="center",
            fontsize=8, color="gray", fontstyle="italic")

fig.tight_layout(w_pad=2.0)
out3 = FIG / "qwen35_random_gap.pdf"
fig.savefig(out3)
print(f"Saved: {out3}")
plt.close(fig)

print("\nDone. All figures saved to", FIG)
