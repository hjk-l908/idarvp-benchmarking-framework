import argparse
from pathlib import Path
import numpy as np
import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.multiclass import OneVsRestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score

def load_emb(npz_path: str, map_path: str):
    z = np.load(npz_path, allow_pickle=True)
    X = z["X"]
    df_map = pd.read_csv(map_path, sep="\t")
    id2row = dict(zip(df_map["seq_id"].astype(str), df_map["row"].astype(int)))
    return X, id2row

def attach_X(df: pd.DataFrame, X_all: np.ndarray, id2row: dict, id_col: str):
    ids = df[id_col].astype(str).tolist()
    missing = [i for i in ids if i not in id2row]
    if missing:
        raise SystemExit(f"[FATAL] missing embeddings for {len(missing)} ids, examples={missing[:5]}")
    rows = [id2row[i] for i in ids]
    return X_all[rows]

def safe_auc(func, y_true, y_score):
    try:
        return float(func(y_true, y_score))
    except Exception:
        return np.nan

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tsv", required=True)
    ap.add_argument("--emb_npz", required=True)
    ap.add_argument("--emb_map", required=True)

    ap.add_argument("--id_col", default="seq_id")
    ap.add_argument("--split_col", default="split40")
    ap.add_argument("--train_value", default="train")
    ap.add_argument("--valid_value", default="valid")
    ap.add_argument("--test_value", default="test_hom40")

    ap.add_argument("--labels", required=True, help="comma-separated, e.g. VIP,VEIP,VINIP,MAP,PIP,RTIP,SFIP")
    ap.add_argument("--C", type=float, default=2.0)
    ap.add_argument("--out_prefix", required=True)
    args = ap.parse_args()

    labels = [x.strip() for x in args.labels.split(",") if x.strip()]
    out_prefix = Path(args.out_prefix)
    out_prefix.parent.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.tsv, sep="\t")
    for c in [args.id_col, args.split_col] + labels:
        if c not in df.columns:
            raise SystemExit(f"[FATAL] missing column: {c}")

    X_all, id2row = load_emb(args.emb_npz, args.emb_map)

    df_tr = df[df[args.split_col].astype(str) == args.train_value].copy()
    df_va = df[df[args.split_col].astype(str) == args.valid_value].copy()
    df_te = df[df[args.split_col].astype(str) == args.test_value].copy()

    Xtr = attach_X(df_tr, X_all, id2row, args.id_col)
    Xva = attach_X(df_va, X_all, id2row, args.id_col)
    Xte = attach_X(df_te, X_all, id2row, args.id_col)

    Ytr = df_tr[labels].astype(int).to_numpy()
    Yva = df_va[labels].astype(int).to_numpy()
    Yte = df_te[labels].astype(int).to_numpy()

    clf = Pipeline([
        ("scaler", StandardScaler()),
        ("ovr", OneVsRestClassifier(LogisticRegression(C=args.C, max_iter=5000, solver="liblinear")))
    ])
    clf.fit(Xtr, Ytr)

    Pva = clf.predict_proba(Xva)  # shape [n, L]
    Pte = clf.predict_proba(Xte)

    rows = []
    def eval_split(split_name, Y, P):
        for j, lab in enumerate(labels):
            y = Y[:, j]
            p = P[:, j]
            pos = int(y.sum())
            neg = int(len(y) - pos)
            auroc = safe_auc(roc_auc_score, y, p)
            auprc = safe_auc(average_precision_score, y, p)
            f1 = float(f1_score(y, (p >= 0.5).astype(int), zero_division=0))
            rows.append(dict(
                split=split_name, label=lab,
                pos=pos, neg=neg,
                AUROC=auroc, AUPRC=auprc, F1_0_5=f1
            ))

    eval_split(args.valid_value, Yva, Pva)
    eval_split(args.test_value, Yte, Pte)

    df_out = pd.DataFrame(rows)
    out_csv = str(out_prefix) + "_perlabel_metrics.csv"
    df_out.to_csv(out_csv, index=False)

    def mean_metrics(split_name):
        sub = df_out[df_out["split"] == split_name]
        return {
            "mean_AUROC": float(np.nanmean(sub["AUROC"].to_numpy())),
            "mean_AUPRC": float(np.nanmean(sub["AUPRC"].to_numpy())),
            "mean_F1_0_5": float(np.nanmean(sub["F1_0_5"].to_numpy())),
        }

    out_txt = str(out_prefix) + "_summary.txt"
    with open(out_txt, "w", encoding="utf-8") as f:
        f.write("Stage2 multilabel embedding baseline (StandardScaler + OVR LogisticRegression)\n")
        f.write(f"labels={labels}\n")
        f.write(f"C={args.C}\n")
        f.write(f"\n[{args.valid_value}] {mean_metrics(args.valid_value)}\n")
        f.write(f"[{args.test_value}] {mean_metrics(args.test_value)}\n\n")
        f.write(df_out.to_string(index=False))
        f.write("\n")

    print("[WROTE]", out_csv)
    print("[WROTE]", out_txt)

if __name__ == "__main__":
    main()
