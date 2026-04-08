import argparse
import numpy as np
import pandas as pd
from pathlib import Path

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import StandardScaler

from scipy.sparse import csr_matrix, hstack

LABELS_DEFAULT = ["VIP","VEIP","VINIP","MAP","PIP","RTIP","SFIP"]

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
    ap.add_argument("--emb_npz", required=True)
    ap.add_argument("--emb_map", required=True)
    ap.add_argument("--id_col", default="seq_id")

    ap.add_argument("--out_npz", required=True)
    ap.add_argument("--out_meta_tsv", required=True)
    return ap.parse_args()

def load_npz(npz_path: str):
    npz = np.load(npz_path)
    if "X" in npz:
        return npz["X"]
    return npz[list(npz.files)[0]]

def load_map(map_path: str):
    mp = pd.read_csv(map_path, sep="\t")
    if "row" not in mp.columns and "row_idx" in mp.columns:
        mp = mp.rename(columns={"row_idx":"row"})
    if "seq_id" not in mp.columns and "id" in mp.columns:
        mp = mp.rename(columns={"id":"seq_id"})
    need={"seq_id","row"}
    miss=need - set(mp.columns)
    if miss:
        raise SystemExit(f"[ERROR] map missing cols: {miss}, cols={list(mp.columns)}")
    mp["seq_id"]=mp["seq_id"].astype(str)
    mp["row"]=mp["row"].astype(int)
    mp = mp.drop_duplicates(subset=["seq_id"], keep="first")
    return mp

def align_embeddings(df: pd.DataFrame, X: np.ndarray, mp: pd.DataFrame, id_col: str):
    id2row = dict(zip(mp["seq_id"], mp["row"]))
    ids = df[id_col].astype(str).tolist()
    rows=[]
    missing=[]
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
    args=parse_args()
    labels=[x.strip() for x in args.labels.split(",") if x.strip()]

    df=pd.read_csv(args.tsv, sep="\t")
    for c in ["Sequence", args.split_col, args.id_col] + labels:
        if c not in df.columns:
            raise SystemExit(f"[ERROR] missing column: {c}")

    split=df[args.split_col].astype(str).values
    idx_tr=np.where(split==args.train_value)[0]
    idx_va=np.where(split==args.valid_value)[0]
    idx_te=np.where(split==args.test_value)[0]
    if len(idx_tr)==0 or len(idx_va)==0 or len(idx_te)==0:
        raise SystemExit(f"[ERROR] split sizes: train={len(idx_tr)} valid={len(idx_va)} test={len(idx_te)}")

    # k-mer
    vec=CountVectorizer(analyzer="char", ngram_range=(1,args.max_ngram), lowercase=False)
    vec.fit(df.loc[idx_tr,"Sequence"].astype(str).tolist())
    Xk=vec.transform(df["Sequence"].astype(str).tolist())

    # emb
    Xemb_raw=load_npz(args.emb_npz)
    mp=load_map(args.emb_map)
    Xemb=align_embeddings(df, Xemb_raw, mp, args.id_col)

    scaler=StandardScaler()
    scaler.fit(Xemb[idx_tr])
    Xemb_s=scaler.transform(Xemb)
    Xemb_sp=csr_matrix(Xemb_s)

    Xfusion=hstack([Xk, Xemb_sp], format="csr")
    Y=df[labels].astype(int).values

    cw = None if str(args.class_weight).strip()=="" else str(args.class_weight).strip()
    base=LogisticRegression(C=args.C, max_iter=5000, solver="liblinear", class_weight=cw)
    clf=OneVsRestClassifier(base)
    clf.fit(Xfusion[idx_tr], Y[idx_tr])

    P_va=clf.predict_proba(Xfusion[idx_va])
    P_te=clf.predict_proba(Xfusion[idx_te])

    out_npz=Path(args.out_npz)
    out_npz.parent.mkdir(parents=True, exist_ok=True)

    np.savez_compressed(
        out_npz,
        labels=np.array(labels, dtype=object),
        idx_valid=idx_va.astype(int),
        idx_test=idx_te.astype(int),
        Y_valid=Y[idx_va].astype(int),
        Y_test=Y[idx_te].astype(int),
        P_valid=P_va.astype(float),
        P_test=P_te.astype(float),
    )
    print("[WROTE]", str(out_npz))

    meta_cols=[args.id_col, "Sequence", args.split_col]
    meta=df[meta_cols].copy()
    meta.to_csv(args.out_meta_tsv, sep="\t", index=False)
    print("[WROTE]", args.out_meta_tsv)

if __name__=="__main__":
    main()
