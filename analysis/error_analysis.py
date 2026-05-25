"""
Error analysis for BPC EMNLP paper.
Compares K=0 vs K=1 vs K=full per example:
  1. Win/Loss/Tie counts
  2. F1 delta by max_depth_reached
  3. F1 delta by answer cardinality (single vs multi)
  4. Over-prediction vs under-prediction patterns
  5. Top qualitative examples (biggest K=0 wins and K=full wins)
  6. Generates a depth-analysis figure

Usage:  python analysis/error_analysis.py
"""

import json, pathlib, sys, textwrap
from collections import defaultdict
import numpy as np

ROOT = pathlib.Path(__file__).resolve().parent.parent
RES  = ROOT / "results"
FIG  = ROOT / "paper_emnlp" / "figures"

# ── File registry ──
PAIRS = {
    # (model, dataset): {K_label: path}
    ("9B", "WebQSP"): {
        "K0":   RES / "qwen35_awq_tp2_nothink_9h_probe"      / "bpc_webqsp_n999999_d5_s42_K0.jsonl",
        "K1":   RES / "qwen35_9b_awq_bpc_k1_k2_cuda02"       / "bpc_webqsp_n999999_d5_s42_K1.jsonl",
        "Kfull": RES / "qwen35_9b_awq_bpc_kfull_webqsp_cuda01" / "bpc_webqsp_n999999_d5_s42_Kfull.jsonl",
    },
    ("9B", "CWQ"): {
        "K0":   RES / "qwen35_awq_tp2_nothink_9h_probe"      / "bpc_cwq_n999999_d5_s42_K0.jsonl",
        "K1":   RES / "qwen35_9b_awq_bpc_k1_k2_cuda02"       / "bpc_cwq_n999999_d5_s42_K1.jsonl",
        "Kfull": RES / "qwen35_9b_awq_bpc_kfull_cwq_cuda23_retry4096_mem078" / "bpc_cwq_n999999_d5_s42_Kfull.jsonl",
    },
    ("4B", "WebQSP"): {
        "K0":   RES / "qwen35_4b_awq_bpc_k0_full_cuda3"      / "bpc_webqsp_n999999_d5_s42_K0.jsonl",
        "K1":   RES / "rev_p0"                                / "bpc_webqsp_n999999_d5_s42_K1.jsonl",
        "Kfull": RES / "qwen35_4b_awq_bpc_kfull_webqsp_cuda0" / "bpc_webqsp_n999999_d5_s42_Kfull.jsonl",
    },
    ("4B", "CWQ"): {
        "K0":   RES / "qwen35_4b_awq_bpc_k0_full_cuda3"      / "bpc_cwq_n999999_d5_s42_K0.jsonl",
        "K1":   RES / "rev_p0"                                / "bpc_cwq_n999999_d5_s42_K1.jsonl",
        "Kfull": RES / "qwen35_4b_awq_bpc_kfull_cwq_cuda1"    / "bpc_cwq_n999999_d5_s42_Kfull.jsonl",
    },
}

def load_jsonl(path):
    rows = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            rows[r["id"]] = r
    return rows

# ── Load all data ──
data = {}
for key, kpaths in PAIRS.items():
    data[key] = {}
    for klabel, path in kpaths.items():
        if path.exists():
            data[key][klabel] = load_jsonl(path)
        else:
            print(f"WARNING: missing {key} {klabel}: {path}")

# ═══════════════════════════════════════════════════════════
# 1. Win / Loss / Tie analysis
# ═══════════════════════════════════════════════════════════
print("=" * 80)
print("1. WIN / LOSS / TIE  (K=0 vs Full, K=1 vs Full)")
print("=" * 80)

def wlt(rows_a, rows_b, ids):
    """Count examples where a > b, a < b, a == b in F1."""
    win, loss, tie = 0, 0, 0
    delta_sum = 0.0
    for eid in ids:
        fa = rows_a[eid]["f1"]
        fb = rows_b[eid]["f1"]
        if fa > fb + 1e-9:
            win += 1
        elif fb > fa + 1e-9:
            loss += 1
        else:
            tie += 1
        delta_sum += fa - fb
    return win, loss, tie, delta_sum / len(ids)

for (model, dataset), kdata in data.items():
    if "K0" not in kdata or "Kfull" not in kdata:
        continue
    ids = sorted(set(kdata["K0"].keys()) & set(kdata["Kfull"].keys()))

    print(f"\n  {model} {dataset} (n={len(ids)})")

    for label_a, label_b in [("K0", "Kfull"), ("K1", "Kfull")]:
        if label_a not in kdata:
            continue
        ids_ab = sorted(set(kdata[label_a].keys()) & set(kdata[label_b].keys()))
        w, l, t, avg_d = wlt(kdata[label_a], kdata[label_b], ids_ab)
        a_name = "$K{=}0$" if label_a == "K0" else "$K{=}1$"
        print(f"    {label_a} vs Full:  {label_a} wins={w}  Full wins={l}  Tie={t}  "
              f"mean dF1={avg_d:+.4f}")

# ═══════════════════════════════════════════════════════════
# 2. F1 delta by max_depth_reached
# ═══════════════════════════════════════════════════════════
print("\n" + "=" * 80)
print("2. F1 DELTA BY MAX DEPTH REACHED  (K=0 - Full)")
print("=" * 80)

depth_data_all = {}  # for figure

for (model, dataset), kdata in data.items():
    if "K0" not in kdata or "Kfull" not in kdata:
        continue
    ids = sorted(set(kdata["K0"].keys()) & set(kdata["Kfull"].keys()))

    by_depth = defaultdict(list)
    for eid in ids:
        d = kdata["K0"][eid].get("max_depth_reached", 0)
        delta = kdata["K0"][eid]["f1"] - kdata["Kfull"][eid]["f1"]
        by_depth[d].append(delta)

    print(f"\n  {model} {dataset}")
    print(f"  {'Depth':>5}  {'N':>5}  {'mean dF1':>9}  {'K0 wins':>7}  {'Full wins':>9}  {'Tie':>5}")

    depth_rows = []
    for d in sorted(by_depth.keys()):
        deltas = by_depth[d]
        n = len(deltas)
        mean_d = np.mean(deltas)
        wins = sum(1 for x in deltas if x > 1e-9)
        losses = sum(1 for x in deltas if x < -1e-9)
        ties = n - wins - losses
        print(f"  {d:>5}  {n:>5}  {mean_d:>+9.4f}  {wins:>7}  {losses:>9}  {ties:>5}")
        depth_rows.append((d, n, mean_d, wins, losses, ties))

    depth_data_all[(model, dataset)] = depth_rows

# Also do K=1 vs Full by depth
print("\n" + "=" * 80)
print("2b. F1 DELTA BY MAX DEPTH REACHED  (K=1 - Full)")
print("=" * 80)

for (model, dataset), kdata in data.items():
    if "K1" not in kdata or "Kfull" not in kdata:
        continue
    ids = sorted(set(kdata["K1"].keys()) & set(kdata["Kfull"].keys()))

    by_depth = defaultdict(list)
    for eid in ids:
        d = kdata["K1"][eid].get("max_depth_reached", 0)
        delta = kdata["K1"][eid]["f1"] - kdata["Kfull"][eid]["f1"]
        by_depth[d].append(delta)

    print(f"\n  {model} {dataset}")
    print(f"  {'Depth':>5}  {'N':>5}  {'mean dF1':>9}  {'K1 wins':>7}  {'Full wins':>9}  {'Tie':>5}")

    for d in sorted(by_depth.keys()):
        deltas = by_depth[d]
        n = len(deltas)
        mean_d = np.mean(deltas)
        wins = sum(1 for x in deltas if x > 1e-9)
        losses = sum(1 for x in deltas if x < -1e-9)
        ties = n - wins - losses
        print(f"  {d:>5}  {n:>5}  {mean_d:>+9.4f}  {wins:>7}  {losses:>9}  {ties:>5}")

# ═══════════════════════════════════════════════════════════
# 3. F1 delta by answer cardinality
# ═══════════════════════════════════════════════════════════
print("\n" + "=" * 80)
print("3. F1 DELTA BY ANSWER CARDINALITY  (K=0 - Full)")
print("=" * 80)

for (model, dataset), kdata in data.items():
    if "K0" not in kdata or "Kfull" not in kdata:
        continue
    ids = sorted(set(kdata["K0"].keys()) & set(kdata["Kfull"].keys()))

    single, multi = [], []
    for eid in ids:
        gold = kdata["K0"][eid]["gold"]
        delta = kdata["K0"][eid]["f1"] - kdata["Kfull"][eid]["f1"]
        if len(gold) == 1:
            single.append(delta)
        else:
            multi.append(delta)

    print(f"\n  {model} {dataset}")
    print(f"    Single-answer (n={len(single)}): mean dF1 = {np.mean(single):+.4f}")
    print(f"    Multi-answer  (n={len(multi)}):  mean dF1 = {np.mean(multi):+.4f}")

# ═══════════════════════════════════════════════════════════
# 4. Prediction behavior: over-prediction, under-prediction, no-path
# ═══════════════════════════════════════════════════════════
print("\n" + "=" * 80)
print("4. PREDICTION BEHAVIOR PATTERNS")
print("=" * 80)

def pred_type(pred, gold):
    pred_set = set(p.lower().strip() for p in pred if p and "no relevant" not in p.lower() and "no answer" not in p.lower())
    if len(pred_set) == 0:
        return "no_path"
    elif len(pred) > len(gold):
        return "over"
    elif len(pred) < len(gold):
        return "under"
    else:
        return "exact_count"

for (model, dataset), kdata in data.items():
    if "K0" not in kdata or "Kfull" not in kdata:
        continue
    ids = sorted(set(kdata["K0"].keys()) & set(kdata["Kfull"].keys()))

    patterns = {"K0": defaultdict(int), "Kfull": defaultdict(int)}
    for eid in ids:
        for klabel in ["K0", "Kfull"]:
            r = kdata[klabel][eid]
            pt = pred_type(r["pred"], r["gold"])
            patterns[klabel][pt] += 1

    print(f"\n  {model} {dataset} (n={len(ids)})")
    for pt in ["no_path", "under", "exact_count", "over"]:
        c0 = patterns["K0"].get(pt, 0)
        cf = patterns["Kfull"].get(pt, 0)
        print(f"    {pt:<12}  K=0: {c0:>5}  Full: {cf:>5}  diff: {c0-cf:>+5}")

# ═══════════════════════════════════════════════════════════
# 5. Top qualitative examples
# ═══════════════════════════════════════════════════════════
print("\n" + "=" * 80)
print("5. TOP QUALITATIVE EXAMPLES  (9B only)")
print("=" * 80)

for dataset in ["WebQSP", "CWQ"]:
    key = ("9B", dataset)
    if key not in data or "K0" not in data[key] or "Kfull" not in data[key]:
        continue
    kdata = data[key]
    ids = sorted(set(kdata["K0"].keys()) & set(kdata["Kfull"].keys()))

    # Compute per-example delta
    examples = []
    for eid in ids:
        r0 = kdata["K0"][eid]
        rf = kdata["Kfull"][eid]
        delta = r0["f1"] - rf["f1"]
        examples.append((delta, eid, r0, rf))

    examples.sort(key=lambda x: x[0], reverse=True)

    print(f"\n  === {dataset}: Top 5 where K=0 WINS over Full ===")
    for delta, eid, r0, rf in examples[:5]:
        q = r0["q"]
        print(f"  [{eid}] dF1={delta:+.3f} depth={r0.get('max_depth_reached','?')}")
        print(f"    Q: {q}")
        print(f"    Gold: {r0['gold']}")
        print(f"    K=0 pred:  {r0['pred']}  (F1={r0['f1']:.3f})")
        print(f"    Full pred: {rf['pred']}  (F1={rf['f1']:.3f})")
        print()

    print(f"  === {dataset}: Top 5 where Full WINS over K=0 ===")
    for delta, eid, r0, rf in examples[-5:]:
        q = r0["q"]
        print(f"  [{eid}] dF1={delta:+.3f} depth={r0.get('max_depth_reached','?')}")
        print(f"    Q: {q}")
        print(f"    Gold: {r0['gold']}")
        print(f"    K=0 pred:  {r0['pred']}  (F1={r0['f1']:.3f})")
        print(f"    Full pred: {rf['pred']}  (F1={rf['f1']:.3f})")
        print()

# ═══════════════════════════════════════════════════════════
# 6. Depth-analysis figure
# ═══════════════════════════════════════════════════════════
try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({
        "font.family": "serif", "font.size": 9, "axes.labelsize": 10,
        "axes.titlesize": 11, "figure.dpi": 300,
        "savefig.bbox": "tight", "savefig.pad_inches": 0.05,
    })

    fig, axes = plt.subplots(1, 2, figsize=(6.5, 2.8), sharey=True)

    for ax_i, dataset in enumerate(["WebQSP", "CWQ"]):
        ax = axes[ax_i]
        key = ("9B", dataset)
        if key not in depth_data_all:
            continue
        rows = depth_data_all[key]

        depths = [r[0] for r in rows]
        ns = [r[1] for r in rows]
        means = [r[2] for r in rows]

        colors = ["#2176AE" if m >= 0 else "#F76C5E" for m in means]
        bars = ax.bar(depths, means, color=colors, edgecolor="white", linewidth=0.5, zorder=3)

        # Annotate N on top of each bar
        for d, n, m in zip(depths, ns, means):
            y_pos = m + 0.002 if m >= 0 else m - 0.008
            ax.text(d, y_pos, f"n={n}", ha="center", va="bottom" if m >= 0 else "top",
                    fontsize=6, color="gray")

        ax.axhline(0, color="black", linewidth=0.5, zorder=2)
        ax.set_xlabel("Max depth reached")
        ax.set_ylabel("Mean $\\Delta$F1 ($K{=}0$ $-$ Full)")
        ax.set_title(dataset, fontweight="bold")
        ax.set_xticks(depths)
        ax.set_axisbelow(True)
        ax.yaxis.grid(True, linewidth=0.3, alpha=0.5)

    fig.tight_layout(w_pad=2.0)
    out = FIG / "qwen35_depth_analysis.pdf"
    fig.savefig(out)
    print(f"\nSaved depth figure: {out}")
    plt.close(fig)

except ImportError:
    print("\nmatplotlib not available, skipping figure.", file=sys.stderr)

# ═══════════════════════════════════════════════════════════
# 7. LaTeX-ready summary table data
# ═══════════════════════════════════════════════════════════
print("\n" + "=" * 80)
print("7. LATEX-READY WIN/LOSS/TIE TABLE")
print("=" * 80)
print("Model & Dataset & Comparison & Wins & Losses & Ties & Mean $\\Delta$F1 \\\\")
for (model, dataset), kdata in data.items():
    if "K0" not in kdata or "Kfull" not in kdata:
        continue
    ids = sorted(set(kdata["K0"].keys()) & set(kdata["Kfull"].keys()))
    w, l, t, avg = wlt(kdata["K0"], kdata["Kfull"], ids)
    print(f"{model} & {dataset} & $K{{=}}0$ vs Full & {w} & {l} & {t} & ${avg:+.004f}$ \\\\")
    if "K1" in kdata:
        ids1 = sorted(set(kdata["K1"].keys()) & set(kdata["Kfull"].keys()))
        w1, l1, t1, avg1 = wlt(kdata["K1"], kdata["Kfull"], ids1)
        print(f"{model} & {dataset} & $K{{=}}1$ vs Full & {w1} & {l1} & {t1} & ${avg1:+.004f}$ \\\\")

print("\nDone.")
