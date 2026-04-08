import argparse
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tsv", required=True, help="Stage2 hom40 CLEAN tsv")
    ap.add_argument("--out_prefix", required=True, help="output prefix path without extension")
    ap.add_argument("--max_ngram", type=int, default=3, help="k-mer max n (1..max_ngram)")
    ap.add_argument("--C", type=float, default=2.0, help="LR inverse regularization strength")
    return ap.parse_args()

def main():
    args = parse_args()
    df = pd.read_csv(args.tsv, sep="\t")

    label_cols = ["VIP","VEIP","VINIP","MAP","PIP","RTIP","SFIP"]
    for c in ["Sequence","split40"] + label_cols:
        if c not in df.columns:
            raise ValueError(f"missing column: {c}")

    # Simple k-mer bag-of-words features (NO GPU)
    seqs = df["Sequence"].astype(str).tolist()
    # analyzer='char' makes character n-grams
    vec = CountVectorizer(analyzer="char", ngram_range=(1, args.max_ngram), lowercase=False)
    X = vec.fit_transform(seqs)
    Y = df[label_cols].astype(int).values

    # Split
    split = df["split40"].astype(str).values
    idx_train = np.where(split=="train")[0]
    idx_valid = np.where(split=="valid")[0]
    idx_test  = np.where(split=="test_hom40")[0]

    clf = OneVsRestClassifier(
        LogisticRegression(
            C=args.C, max_iter=5000, solver="liblinear"
        )
    )
    clf.fit(X[idx_train], Y[idx_train])

    def eval_split(name, idx):
        prob = clf.predict_proba(X[idx])
        pred = (prob >= 0.5).astype(int)
        y = Y[idx]

        # per-label AUPRC/AUROC (skip labels with all-0 or all-1 in that split)
        rows = []
        for j, lab in enumerate(label_cols):
            yj = y[:, j]
            pj = prob[:, j]
            if len(np.unique(yj)) < 2:
                auroc = np.nan
                auprc = np.nan
            else:
                auroc = roc_auc_score(yj, pj)
                auprc = average_precision_score(yj, pj)
            f1 = f1_score(yj, pred[:, j], zero_division=0)
            rows.append([name, lab, int(yj.sum()), int((1-yj).sum()), auroc, auprc, f1])

        out = pd.DataFrame(rows, columns=["split","label","pos","neg","AUROC","AUPRC","F1@0.5"])
        return out

    out_valid = eval_split("valid", idx_valid)
    out_test  = eval_split("test_hom40", idx_test)

    out_all = pd.concat([out_valid, out_test], ignore_index=True)
    out_all.to_csv(args.out_prefix + "_perlabel_metrics.csv", index=False)

    # Also write a short text summary
    with open(args.out_prefix + "_summary.txt", "w", encoding="utf-8") as f:
        f.write("Stage2 hom40 baseline (k-mer + OneVsRest LogisticRegression)\n")
        f.write(f"tsv={args.tsv}\n")
        f.write(f"ngram=1..{args.max_ngram}\n")
        f.write(f"C={args.C}\n\n")
        for split_name in ["valid","test_hom40"]:
            sub = out_all[out_all["split"]==split_name].copy()
            f.write(f"[{split_name}] mean AUPRC (nanmean) = {np.nanmean(sub['AUPRC']):.4f}\n")
            f.write(f"[{split_name}] mean AUROC (nanmean) = {np.nanmean(sub['AUROC']):.4f}\n")
            f.write(f"[{split_name}] mean F1@0.5 (mean) = {np.mean(sub['F1@0.5']):.4f}\n\n")

    print("WROTE:")
    print(args.out_prefix + "_perlabel_metrics.csv")
    print(args.out_prefix + "_summary.txt")

if __name__ == "__main__":
    main()
