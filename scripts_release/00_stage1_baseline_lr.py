import argparse
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    confusion_matrix,
)


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", required=True)
    ap.add_argument("--valid", required=True)
    ap.add_argument("--test_easy", required=True)
    ap.add_argument("--test_hard", required=True)
    ap.add_argument("--out_prefix", required=True)
    ap.add_argument("--max_ngram", type=int, default=3)
    ap.add_argument("--C", type=float, default=2.0)
    return ap.parse_args()


def metrics(y_true, prob, thr=0.5):
    pred = (prob >= thr).astype(int)
    auroc = roc_auc_score(y_true, prob) if len(np.unique(y_true)) == 2 else np.nan
    auprc = average_precision_score(y_true, prob) if len(np.unique(y_true)) == 2 else np.nan
    f1 = f1_score(y_true, pred, zero_division=0)
    mcc = matthews_corrcoef(y_true, pred)
    prec = precision_score(y_true, pred, zero_division=0)
    rec = recall_score(y_true, pred, zero_division=0)
    tn, fp, fn, tp = confusion_matrix(y_true, pred, labels=[0, 1]).ravel()
    spec = tn / (tn + fp) if (tn + fp) else np.nan

    return dict(
        AUROC=auroc,
        AUPRC=auprc,
        F1=f1,
        MCC=mcc,
        Precision=prec,
        Recall=rec,
        Specificity=spec,
        TN=tn,
        FP=fp,
        FN=fn,
        TP=tp,
    )


def find_best_thr_on_valid(yv, pv):
    # Select the validation threshold that maximizes MCC.
    best = (-1.0, 0.5)
    for thr in np.linspace(0.05, 0.95, 19):
        m = matthews_corrcoef(yv, (pv >= thr).astype(int))
        if m > best[0]:
            best = (m, float(thr))
    return best[1], best[0]


def main():
    args = parse_args()
    files = dict(
        train=args.train,
        valid=args.valid,
        test_easy=args.test_easy,
        test_hard=args.test_hard,
    )
    dfs = {k: pd.read_csv(v, sep="\t") for k, v in files.items()}

    for k, df in dfs.items():
        for col in ["Sequence", "y"]:
            if col not in df.columns:
                raise ValueError(f"{k} missing {col}. cols={list(df.columns)[:20]}")

    # Character k-mer features (1..max_ngram).
    vec = CountVectorizer(
        analyzer="char",
        ngram_range=(1, args.max_ngram),
        lowercase=False,
    )
    X_train = vec.fit_transform(dfs["train"]["Sequence"].astype(str))
    y_train = dfs["train"]["y"].astype(int).values

    clf = LogisticRegression(C=args.C, max_iter=5000, solver="liblinear")
    clf.fit(X_train, y_train)

    # Transform validation sequences with the training-fitted vectorizer.
    X_valid = vec.transform(dfs["valid"]["Sequence"].astype(str))
    y_valid = dfs["valid"]["y"].astype(int).values
    p_valid = clf.predict_proba(X_valid)[:, 1]

    best_thr, best_mcc = find_best_thr_on_valid(y_valid, p_valid)

    rows = []
    # Evaluate each split at both 0.5 and the validation-selected threshold.
    for split in ["valid", "test_easy", "test_hard"]:
        X = vec.transform(dfs[split]["Sequence"].astype(str))
        y = dfs[split]["y"].astype(int).values
        p = clf.predict_proba(X)[:, 1]

        m05 = metrics(y, p, thr=0.5)
        mbt = metrics(y, p, thr=best_thr)

        rows.append([
            split, "thr=0.5", len(y), int(y.sum()), best_thr, best_mcc, *m05.values()
        ])
        rows.append([
            split, f"thr={best_thr:.2f}", len(y), int(y.sum()), best_thr, best_mcc, *mbt.values()
        ])

    cols = [
        "split", "setting", "n", "pos", "best_thr_from_valid", "best_valid_MCC",
        "AUROC", "AUPRC", "F1", "MCC", "Precision", "Recall", "Specificity",
        "TN", "FP", "FN", "TP",
    ]
    out = pd.DataFrame(rows, columns=cols)
    out.to_csv(args.out_prefix + "_metrics.csv", index=False)

    with open(args.out_prefix + "_summary.txt", "w", encoding="utf-8") as f:
        f.write("Stage1 baseline (k-mer + LogisticRegression)\n")
        f.write(f"ngram=1..{args.max_ngram}, C={args.C}\n")
        f.write(f"best_thr_from_valid={best_thr:.2f}, best_valid_MCC={best_mcc:.4f}\n\n")
        f.write(out.to_string(index=False))
        f.write("\n")

    print("WROTE:")
    print(args.out_prefix + "_metrics.csv")
    print(args.out_prefix + "_summary.txt")
    print(f"best_thr_from_valid={best_thr:.2f} best_valid_MCC={best_mcc:.4f}")


if __name__ == "__main__":
    main()
