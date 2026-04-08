import argparse
import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scores_npz", required=True)
    ap.add_argument("--Ks", default="50,100,200", help="comma-separated, e.g. 50,100,200")
    ap.add_argument("--out_csv", required=True)
    ap.add_argument("--out_csv_k100", required=True)
    ap.add_argument("--out_png", required=True)
    ap.add_argument("--plot_metric", default="enrichment", choices=["precision","recall","enrichment"])
    ap.add_argument("--title", default="Stage2 Top-K (test_hom40)")
    ap.add_argument("--method", default="")
    return ap.parse_args()

def safe_div(a, b):
    return float(a)/float(b) if b else np.nan

def main():
    args = parse_args()
    Ks = [int(x.strip()) for x in args.Ks.split(",") if x.strip()]
    z = np.load(args.scores_npz, allow_pickle=True)

    labels = [str(x) for x in z["labels"]]
    Yt = z["Y_test"].astype(int)      # (N, K)
    Pt = z["P_test"].astype(float)    # (N, K)

    N = Yt.shape[0]
    assert Pt.shape[0] == N and Pt.shape[1] == len(labels)

    method = args.method.strip() or Path(args.scores_npz).stem.replace("_scores_valid_test","")

    rows = []
    for j, lab in enumerate(labels):
        y = Yt[:, j]
        p = Pt[:, j]
        total_pos = int(y.sum())
        base_rate = safe_div(total_pos, N)

        order = np.argsort(-p)  # descending
        for K in Ks:
            kk = min(K, N)
            top_idx = order[:kk]
            tp_at_k = int(y[top_idx].sum())
            prec_k = safe_div(tp_at_k, kk)
            rec_k  = safe_div(tp_at_k, total_pos) if total_pos else np.nan
            enr_k  = safe_div(prec_k, base_rate) if base_rate else np.nan

            rows.append({
                "method": method,
                "label": lab,
                "K": kk,
                "N_test": N,
                "test_pos": total_pos,
                "base_rate": base_rate,
                "TP_at_K": tp_at_k,
                "Precision@K": prec_k,
                "Recall@K": rec_k,
                "Enrichment@K": enr_k,
            })

    df = pd.DataFrame(rows)
    Path(args.out_csv).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out_csv, index=False)

    # K=100 compact table for paper
    k_main = 100
    sub = df[df["K"] == min(k_main, N)].copy()
    sub = sub.sort_values("Enrichment@K", ascending=False)
    keep = ["label","test_pos","base_rate","K","TP_at_K","Precision@K","Recall@K","Enrichment@K"]
    sub[keep].to_csv(args.out_csv_k100, index=False)

    # Plot (K=100) per-label bar
    metric_col = {"precision":"Precision@K", "recall":"Recall@K", "enrichment":"Enrichment@K"}[args.plot_metric]
    plot = sub[["label", metric_col]].copy()
    plot = plot.sort_values(metric_col, ascending=True)

    plt.figure(figsize=(8.8, 4.8))
    plt.barh(plot["label"], plot[metric_col])
    plt.xlabel(metric_col.replace("@K", f"@{min(k_main, N)}"))
    plt.title(args.title + f" — {metric_col.replace('@K', f'@{min(k_main, N)}')}")
    plt.tight_layout()
    Path(args.out_png).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(args.out_png, dpi=300)

    print("[WROTE]", args.out_csv)
    print("[WROTE]", args.out_csv_k100)
    print("[WROTE]", args.out_png)

if __name__ == "__main__":
    main()
