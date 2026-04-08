import argparse
import numpy as np
import pandas as pd
from pathlib import Path

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import average_precision_score, roc_auc_score, f1_score

from scipy.sparse import csr_matrix, hstack
import matplotlib.pyplot as plt

LABELS_DEFAULT = ["VIP","VEIP","VINIP","MAP","PIP","RTIP","SFIP"]

def safe_auroc(y, p):
    pos = int(y.sum())
    neg = int(len(y) - pos)
    if pos == 0 or neg == 0:
        return np.nan
    return roc_auc_score(y, p)

def safe_auprc(y, p):
    pos = int(y.sum())
    if pos == 0:
        return np.nan
    return average_precision_score(y, p)

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tsv", required=True)
    ap.add_argument("--split_col", default="split40")
    ap.add_argument("--train_value", default="train")
    ap.add_argument("--valid_value", default="valid")
    ap.add_argument("--test_value", default="test_hom40")

    ap.add_argument("--labels", default=",".join(LABELS_DEFAULT))
    ap.add_argument("--max_ngram", type=int, default=3)
    ap.add_argument("--C", type=float, default=2.0)
    ap.add_argument("--class_weight", default="", help="e.g. balanced")

    ap.add_argument("--model", default="esm2_t6_8M_UR50D")
    ap.add_argument("--emb_npz", default=None)
    ap.add_argument("--emb_map", default=None)
    ap.add_argument("--id_col", default="seq_id")

    ap.add_argument("--out_prefix", required=True)
    return ap.parse_args()

def load_npz(npz_path: str):
    npz = np.load(npz_path)
    if "X" in npz:
        return npz["X"]
    return npz[list(npz.files)[0]]

def load_map(map_path: str):
    mp = pd.read_csv(map_path, sep="\t")
    if "row" not in mp.columns and "row_idx" in mp.columns:
        mp = mp.rename(columns={"row_idx": "row"})
    if "seq_id" not in mp.columns and "id" in mp.columns:
        mp = mp.rename(columns={"id": "seq_id"})
    if "row" not in mp.columns or "seq_id" not in mp.columns:
        raise SystemExit(f"[ERROR] map missing required cols. cols={list(mp.columns)}")
    mp = mp.drop_duplicates(subset=["seq_id"], keep="first")
    return mp

def align_embeddings(df: pd.DataFrame, X: np.ndarray, mp: pd.DataFrame, id_col: str):
    if id_col not in df.columns:
        raise SystemExit(f"[ERROR] TSV missing id_col={id_col}. cols={list(df.columns)[:20]}")
    id2row = dict(zip(mp["seq_id"].astype(str), mp["row"].astype(int)))
    ids = df[id_col].astype(str).tolist()

    rows = []
    missing = []
    for sid in ids:
        r = id2row.get(sid, None)
        if r is None:
            missing.append(sid)
        else:
            rows.append(r)
    if missing:
        raise SystemExit(f"[ERROR] missing embeddings for {len(missing)} ids, examples={missing[:5]}")
    return X[np.array(rows, dtype=int)]

def main():
    args = parse_args()
    labels = [x.strip() for x in args.labels.split(",") if x.strip()]

    df = pd.read_csv(args.tsv, sep="\t")
    need_cols = ["Sequence", args.split_col, args.id_col] + labels
    for c in need_cols:
        if c not in df.columns:
            raise SystemExit(f"[ERROR] missing column: {c}")

    split = df[args.split_col].astype(str).values
    idx_tr = np.where(split == args.train_value)[0]
    idx_va = np.where(split == args.valid_value)[0]
    idx_te = np.where(split == args.test_value)[0]
    if len(idx_tr)==0 or len(idx_va)==0 or len(idx_te)==0:
        raise SystemExit(f"[ERROR] split sizes: train={len(idx_tr)} valid={len(idx_va)} test={len(idx_te)}")

    # k-mer features
    vec = CountVectorizer(analyzer="char", ngram_range=(1, args.max_ngram), lowercase=False)
    vec.fit(df.loc[idx_tr, "Sequence"].astype(str).tolist())
    Xk = vec.transform(df["Sequence"].astype(str).tolist())

    # embeddings
    if args.emb_npz is None:
        args.emb_npz = f"03_embeddings/{args.model}/stage2_all_mean.npz"
    if args.emb_map is None:
        args.emb_map = f"03_embeddings/{args.model}/stage2_all_mean.map.tsv"

    Xemb_raw = load_npz(args.emb_npz)
    mp = load_map(args.emb_map)
    Xemb = align_embeddings(df, Xemb_raw, mp, args.id_col)

    scaler = StandardScaler()
    scaler.fit(Xemb[idx_tr])
    Xemb_s = scaler.transform(Xemb)
    Xemb_sp = csr_matrix(Xemb_s)

    # fusion
    Xfusion = hstack([Xk, Xemb_sp], format="csr")
    Y = df[labels].astype(int).values

    cw = None if str(args.class_weight).strip()=="" else str(args.class_weight).strip()
    base = LogisticRegression(C=args.C, max_iter=5000, solver="liblinear", class_weight=cw)
    clf = OneVsRestClassifier(base)
    clf.fit(Xfusion[idx_tr], Y[idx_tr])

    P_va = clf.predict_proba(Xfusion[idx_va])
    P_te = clf.predict_proba(Xfusion[idx_te])

    rows = []
    def eval_split(name, idx, P):
        Ytrue = Y[idx]
        for j, lab in enumerate(labels):
            y = Ytrue[:, j]
            p = P[:, j]
            pos = int(y.sum())
            neg = int(len(y) - pos)
            auroc = safe_auroc(y, p)
            auprc = safe_auprc(y, p)
            f1 = np.nan
            if pos > 0 and neg > 0:
                yhat = (p >= 0.5).astype(int)
                f1 = f1_score(y, yhat, zero_division=0)
            rows.append({"split": name, "label": lab, "pos": pos, "neg": neg,
                         "AUROC": auroc, "AUPRC": auprc, "F1@0.5": f1})

    eval_split(args.valid_value, idx_va, P_va)
    eval_split(args.test_value, idx_te, P_te)

    out_df = pd.DataFrame(rows)
    out_prefix = Path(args.out_prefix)
    out_prefix.parent.mkdir(parents=True, exist_ok=True)

    perlabel_csv = str(out_prefix) + "_perlabel_metrics.csv"
    out_df.to_csv(perlabel_csv, index=False)
    print("[WROTE]", perlabel_csv)

    def mean_metrics(split_name):
        sub = out_df[out_df["split"] == split_name]
        return {
            "mean_AUPRC": float(np.nanmean(sub["AUPRC"].values)),
            "mean_AUROC": float(np.nanmean(sub["AUROC"].values)),
            "mean_F1@0.5": float(np.nanmean(sub["F1@0.5"].values)),
        }

    m_va = mean_metrics(args.valid_value)
    m_te = mean_metrics(args.test_value)

    summary_txt = str(out_prefix) + "_summary.txt"
    with open(summary_txt, "w") as f:
        f.write(f"Stage2 hom40 fusion baseline (k-mer 1..{args.max_ngram} + {args.model} mean + OVR LR)\n")
        f.write(f"tsv={args.tsv}\nC={args.C}\n")
        f.write(f"split_col={args.split_col}, train={args.train_value}, valid={args.valid_value}, test={args.test_value}\n")
        f.write(f"labels={','.join(labels)}\n\n")
        f.write(f"[valid] mean AUPRC = {m_va['mean_AUPRC']:.4f}\n")
        f.write(f"[valid] mean AUROC = {m_va['mean_AUROC']:.4f}\n")
        f.write(f"[valid] mean F1@0.5 = {m_va['mean_F1@0.5']:.4f}\n\n")
        f.write(f"[test_hom40] mean AUPRC = {m_te['mean_AUPRC']:.4f}\n")
        f.write(f"[test_hom40] mean AUROC = {m_te['mean_AUROC']:.4f}\n")
        f.write(f"[test_hom40] mean F1@0.5 = {m_te['mean_F1@0.5']:.4f}\n")
    print("[WROTE]", summary_txt)

    # figure: per-label AUPRC on test_hom40
    sub = out_df[out_df["split"] == args.test_value].copy().sort_values("AUPRC", ascending=True)
    plt.figure(figsize=(8.8, 4.8))
    plt.barh(sub["label"], sub["AUPRC"])
    plt.xlabel("AUPRC")
    plt.title("Stage2 hom40 fusion (k-mer + ESM2) — test_hom40 per-label AUPRC")
    plt.tight_layout()

    fig_path = Path("05_fig_tables") / (out_prefix.name + "_perlabel_AUPRC_testhom40.png")
    fig_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(fig_path, dpi=300)
    print("[WROTE]", str(fig_path))

if __name__ == "__main__":
    main()
