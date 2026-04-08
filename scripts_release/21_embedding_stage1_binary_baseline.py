import argparse
from pathlib import Path
import numpy as np
import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    roc_auc_score, average_precision_score,
    f1_score, matthews_corrcoef, confusion_matrix,
    precision_score, recall_score
)

def load_npz_and_map(npz_path: str, map_path: str):
    z = np.load(npz_path, allow_pickle=True)
    X = z["X"]
    df_map = pd.read_csv(map_path, sep="\t")
    id2row = dict(zip(df_map["seq_id"].astype(str), df_map["row"].astype(int)))
    return X, id2row

def build_Xy(tsv_path: str, X_all: np.ndarray, id2row: dict, id_col: str, y_col: str):
    df = pd.read_csv(tsv_path, sep="\t")
    ids = df[id_col].astype(str).tolist()
    missing = [i for i in ids if i not in id2row]
    if missing:
        raise SystemExit(f"[FATAL] {tsv_path}: missing embeddings for {len(missing)} ids, examples={missing[:5]}")
    rows = [id2row[i] for i in ids]
    X = X_all[rows]
    y = df[y_col].astype(int).to_numpy()
    return df, X, y

def compute_metrics(y_true, y_prob, thr):
    y_pred = (y_prob >= thr).astype(int)
    # confusion matrix: [[TN, FP],[FN, TP]]
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0,1]).ravel()

    # AUROC/AUPRC might fail if only one class exists
    auroc = np.nan
    auprc = np.nan
    try:
        auroc = roc_auc_score(y_true, y_prob)
    except Exception:
        pass
    try:
        auprc = average_precision_score(y_true, y_prob)
    except Exception:
        pass

    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    spec = (tn / (tn + fp)) if (tn + fp) > 0 else np.nan
    f1 = f1_score(y_true, y_pred, zero_division=0)
    mcc = matthews_corrcoef(y_true, y_pred) if len(np.unique(y_pred)) > 1 or len(np.unique(y_true)) > 1 else 0.0

    return dict(
        AUROC=float(auroc) if auroc==auroc else np.nan,
        AUPRC=float(auprc) if auprc==auprc else np.nan,
        F1=float(f1),
        MCC=float(mcc),
        Precision=float(prec),
        Recall=float(rec),
        Specificity=float(spec) if spec==spec else np.nan,
        TN=int(tn), FP=int(fp), FN=int(fn), TP=int(tp),
    )

def best_thr_by_valid_mcc(y_true, y_prob):
    best_thr = 0.5
    best_mcc = -1e9
    # scan thresholds
    for thr in np.linspace(0, 1, 501):
        y_pred = (y_prob >= thr).astype(int)
        try:
            mcc = matthews_corrcoef(y_true, y_pred)
        except Exception:
            mcc = -1e9
        if mcc > best_mcc:
            best_mcc = mcc
            best_thr = float(thr)
    return best_thr, float(best_mcc)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_tsv", required=True)
    ap.add_argument("--valid_tsv", required=True)
    ap.add_argument("--test_easy_tsv", required=True)
    ap.add_argument("--test_hard_tsv", required=True)

    ap.add_argument("--train_npz", required=True)
    ap.add_argument("--train_map", required=True)
    ap.add_argument("--valid_npz", required=True)
    ap.add_argument("--valid_map", required=True)
    ap.add_argument("--easy_npz", required=True)
    ap.add_argument("--easy_map", required=True)
    ap.add_argument("--hard_npz", required=True)
    ap.add_argument("--hard_map", required=True)

    ap.add_argument("--id_col", default="seq_id")
    ap.add_argument("--label_col", default="y")
    ap.add_argument("--C", type=float, default=2.0)
    ap.add_argument("--out_prefix", required=True)
    args = ap.parse_args()

    out_prefix = Path(args.out_prefix)
    out_prefix.parent.mkdir(parents=True, exist_ok=True)

    Xtr_all, id2tr = load_npz_and_map(args.train_npz, args.train_map)
    Xva_all, id2va = load_npz_and_map(args.valid_npz, args.valid_map)
    Xe_all, id2e = load_npz_and_map(args.easy_npz, args.easy_map)
    Xh_all, id2h = load_npz_and_map(args.hard_npz, args.hard_map)

    _, Xtr, ytr = build_Xy(args.train_tsv, Xtr_all, id2tr, args.id_col, args.label_col)
    _, Xva, yva = build_Xy(args.valid_tsv, Xva_all, id2va, args.id_col, args.label_col)
    _, Xe, ye = build_Xy(args.test_easy_tsv, Xe_all, id2e, args.id_col, args.label_col)
    _, Xh, yh = build_Xy(args.test_hard_tsv, Xh_all, id2h, args.id_col, args.label_col)

    clf = Pipeline([
        ("scaler", StandardScaler()),
        ("lr", LogisticRegression(C=args.C, max_iter=5000, solver="liblinear"))
    ])
    clf.fit(Xtr, ytr)

    p_tr = clf.predict_proba(Xtr)[:,1]
    p_va = clf.predict_proba(Xva)[:,1]
    p_e  = clf.predict_proba(Xe)[:,1]
    p_h  = clf.predict_proba(Xh)[:,1]

    best_thr, best_valid_mcc = best_thr_by_valid_mcc(yva, p_va)

    rows = []
    def add(split, setting, y, p, thr):
        m = compute_metrics(y, p, thr)
        rows.append(dict(
            split=split,
            setting=setting,
            n=int(len(y)),
            pos=int(y.sum()),
            best_thr_from_valid=best_thr,
            best_valid_MCC=best_valid_mcc,
            threshold=float(thr),
            **m
        ))

    # setting: thr=0.5 and thr=best_thr_from_valid
    for split, y, p in [
        ("valid", yva, p_va),
        ("test_easy", ye, p_e),
        ("test_hard", yh, p_h),
    ]:
        add(split, "thr=0.5", y, p, 0.5)
        add(split, f"thr={best_thr:.3f}", y, p, best_thr)

    df_out = pd.DataFrame(rows)
    out_csv = str(out_prefix) + "_metrics.csv"
    df_out.to_csv(out_csv, index=False)

    out_txt = str(out_prefix) + "_summary.txt"
    with open(out_txt, "w", encoding="utf-8") as f:
        f.write("Stage1 embedding baseline (StandardScaler + LogisticRegression)\n")
        f.write(f"C={args.C}\n")
        f.write(f"best_thr_from_valid={best_thr:.4f}, best_valid_MCC={best_valid_mcc:.4f}\n\n")
        f.write(df_out.to_string(index=False))
        f.write("\n")

    print("[WROTE]", out_csv)
    print("[WROTE]", out_txt)

if __name__ == "__main__":
    main()
