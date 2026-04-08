import re
import csv
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

ROOT = Path(".")
RESULTS = ROOT / "04_results"
OUTDIR = ROOT / "05_fig_tables"
OUTDIR.mkdir(parents=True, exist_ok=True)

SUMMARY_GLOB = "stage2_hom40_*_summary.txt"
PERLABEL_GLOB = "stage2_hom40_*_perlabel_metrics.csv"

# -----------------------------
# Helpers
# -----------------------------
def parse_method_from_name(stem: str) -> str:
    # stem is filename without suffix
    # Examples:
    # stage2_hom40_baseline_lr_kmer13_summary
    # stage2_hom40_esm2_t6_8M_UR50D_mean_ovr_lr_summary
    # stage2_hom40_fusion_kmer13_esm2_t30_150M_UR50D_mean_ovr_lr_summary
    # stage2_hom40_fusion_kmer13_esm2_t6_8M_UR50D_cwbalanced_mean_ovr_lr_summary
    s = stem
    s = s.replace("_summary", "")
    s = s.replace("stage2_hom40_", "")

    tag = []
    if "cwbalanced" in s:
        tag.append("cwbalanced")
        s = s.replace("_cwbalanced", "")
        s = s.replace("cwbalanced_", "")

    if s.startswith("baseline_lr_kmer13"):
        base = "kmer13_LR"
    elif s.startswith("esm2_"):
        # embedding-only
        m = re.search(r"esm2_(t\d+_.+?)_mean_ovr_lr", s)
        base = f"ESM2_{m.group(1)}_mean_LR" if m else f"ESM2_mean_LR"
    elif s.startswith("fusion_kmer13_esm2_"):
        m = re.search(r"fusion_kmer13_esm2_(t\d+_.+?)_mean_ovr_lr", s)
        base = f"Fusion(kmer13+ESM2_{m.group(1)}_mean)_OVR_LR" if m else "Fusion(kmer13+ESM2)_OVR_LR"
    else:
        base = s

    if tag:
        return base + "__" + "__".join(tag)
    return base


def parse_summary_file(path: Path) -> dict:
    txt = path.read_text(errors="ignore").splitlines()
    out = {
        "file": str(path),
        "method": parse_method_from_name(path.stem),
    }
    # capture:
    # [valid] mean AUPRC = 0.7426
    # [test_hom40] mean AUROC = 0.8941
    pat = re.compile(r"^\[(?P<split>[^\]]+)\]\s+mean\s+(?P<metric>AUPRC|AUROC|F1@0\.5)\s*=\s*(?P<val>[-+0-9\.eE]+)")
    for line in txt:
        m = pat.search(line.strip())
        if not m:
            continue
        split = m.group("split").strip()
        metric = m.group("metric").strip()
        val = float(m.group("val"))
        key = f"{split}__{metric}"
        out[key] = val
    return out


def read_perlabel(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    # expected cols: split,label,pos,neg,AUROC,AUPRC,F1@0.5
    need = {"split","label","AUPRC","AUROC"}
    if not need.issubset(set(df.columns)):
        raise SystemExit(f"[ERROR] perlabel missing required cols in {path}: {df.columns.tolist()}")
    df["file"] = str(path)
    df["method"] = parse_method_from_name(path.stem.replace("_perlabel_metrics",""))
    return df


# -----------------------------
# Collect
# -----------------------------
summary_rows = []
for p in sorted(RESULTS.glob(SUMMARY_GLOB)):
    summary_rows.append(parse_summary_file(p))

if not summary_rows:
    raise SystemExit(f"[ERROR] No summary files matched: {RESULTS}/{SUMMARY_GLOB}")

summary_df = pd.DataFrame(summary_rows).sort_values("method")

# standardize columns we care
def pickcol(split, metric):
    col = f"{split}__{metric}"
    return col if col in summary_df.columns else None

cols_out = ["method", "file"]
for split in ["valid","test_hom40"]:
    for metric in ["AUPRC","AUROC","F1@0.5"]:
        c = pickcol(split, metric)
        if c:
            cols_out.append(c)

summary_out = summary_df[cols_out].copy()
summary_csv = OUTDIR / "stage2_model_comparison_summary.csv"
summary_out.to_csv(summary_csv, index=False)
print("[WROTE]", summary_csv)

# -----------------------------
# per-label (test_hom40)
# -----------------------------
perlabel_files = sorted(RESULTS.glob(PERLABEL_GLOB))
per_df = pd.concat([read_perlabel(p) for p in perlabel_files], ignore_index=True) if perlabel_files else pd.DataFrame()

if not per_df.empty:
    per_test = per_df[per_df["split"].astype(str) == "test_hom40"].copy()
    # keep key cols
    keep = ["method","label","pos","neg","AUPRC","AUROC"]
    if "F1@0.5" in per_test.columns:
        keep.append("F1@0.5")
    per_test = per_test[keep].copy()

    per_csv = OUTDIR / "stage2_model_comparison_perlabel_testhom40.csv"
    per_test.to_csv(per_csv, index=False)
    print("[WROTE]", per_csv)

    # heatmap for AUPRC
    piv = per_test.pivot_table(index="label", columns="method", values="AUPRC", aggfunc="mean")
    piv = piv.sort_index()
    # order methods by overall mean test_hom40 AUPRC if available
    mean_col = "test_hom40__AUPRC"
    if mean_col in summary_df.columns:
        order = summary_df.sort_values(mean_col, ascending=False)["method"].tolist()
        piv = piv.reindex(columns=[m for m in order if m in piv.columns])

    fig = plt.figure(figsize=(max(8, 0.55 * len(piv.columns) + 2), max(3.5, 0.45 * len(piv.index) + 1.5)))
    ax = fig.add_subplot(111)
    im = ax.imshow(piv.values, aspect="auto")
    ax.set_xticks(range(len(piv.columns)))
    ax.set_xticklabels(piv.columns, rotation=45, ha="right")
    ax.set_yticks(range(len(piv.index)))
    ax.set_yticklabels(piv.index)
    ax.set_title("Stage2 test_hom40 per-label AUPRC (higher is better)")
    fig.colorbar(im, ax=ax, shrink=0.8)
    fig.tight_layout()
    heat_png = OUTDIR / "stage2_model_comparison_heatmap_testhom40_AUPRC.png"
    fig.savefig(heat_png, dpi=300)
    print("[WROTE]", heat_png)

# -----------------------------
# bar plot: mean AUPRC on test_hom40
# -----------------------------
if "test_hom40__AUPRC" in summary_df.columns:
    plot_df = summary_df.sort_values("test_hom40__AUPRC", ascending=False).copy()

    fig = plt.figure(figsize=(max(8, 0.55 * len(plot_df) + 2), 4.8))
    ax = fig.add_subplot(111)
    ax.bar(plot_df["method"].astype(str), plot_df["test_hom40__AUPRC"].astype(float))
    ax.set_ylabel("mean AUPRC (test_hom40)")
    ax.set_title("Stage2 comparison: mean AUPRC on homology-aware test (test_hom40)")
    ax.set_xticklabels(plot_df["method"].astype(str), rotation=45, ha="right")
    fig.tight_layout()
    bar_png = OUTDIR / "stage2_model_comparison_meanAUPRC_bar.png"
    fig.savefig(bar_png, dpi=300)
    print("[WROTE]", bar_png)

print("[DONE] Stage2 model comparison collected.")
