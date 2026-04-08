import argparse, os, glob
import numpy as np
import pandas as pd
import torch

def load_vec(pt_path):
    obj = torch.load(pt_path, map_location="cpu")
    # 支援：tensor / numpy / dict(常見 fair-esm 輸出)
    if isinstance(obj, torch.Tensor):
        v = obj
    elif isinstance(obj, np.ndarray):
        v = torch.from_numpy(obj)
    elif isinstance(obj, dict):
        # 盡量猜常見 key（你如果知道自己存的是哪個 key，也可以在這裡固定）
        for k in ["mean", "embedding", "repr", "vector"]:
            if k in obj:
                v = obj[k]
                if isinstance(v, np.ndarray):
                    v = torch.from_numpy(v)
                break
        else:
            # fair-esm extract 常見：obj["representations"][layer] 或 obj["mean_representations"][layer]
            if "mean_representations" in obj:
                layer = sorted(obj["mean_representations"].keys())[-1]
                v = obj["mean_representations"][layer]
            elif "representations" in obj:
                layer = sorted(obj["representations"].keys())[-1]
                v = obj["representations"][layer]
                # 若是 (L, D) 取 mean
                if v.ndim == 2:
                    v = v.mean(dim=0)
            else:
                raise ValueError(f"Unrecognized dict keys in {pt_path}: {list(obj.keys())[:20]}")
    else:
        raise ValueError(f"Unrecognized pt object type: {type(obj)} from {pt_path}")

    v = v.detach().float().view(-1).numpy()
    return v

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pt_dir", required=True, help="directory containing *.pt and (optional) manifest.tsv")
    ap.add_argument("--out_npz", required=True)
    ap.add_argument("--out_map", required=True)
    ap.add_argument("--manifest", default=None, help="default: <pt_dir>/manifest.tsv if exists")
    args = ap.parse_args()

    pt_dir = args.pt_dir.rstrip("/")
    manifest = args.manifest or os.path.join(pt_dir, "manifest.tsv")

    rows = []
    if os.path.exists(manifest):
        mf = pd.read_csv(manifest, sep="\t")
        # 期待至少有 id / pt_path
        id_col = "id" if "id" in mf.columns else mf.columns[0]
        pt_col = "pt_path" if "pt_path" in mf.columns else ("path" if "path" in mf.columns else None)
        if pt_col is None:
            raise SystemExit(f"[ERROR] manifest has no pt_path/path col: {manifest} cols={list(mf.columns)}")
        for _, r in mf.iterrows():
            sid = str(r[id_col])
            p = str(r[pt_col])
            if not os.path.isabs(p):
                p = os.path.join(os.path.dirname(manifest), p)
            rows.append((sid, p))
    else:
        pts = sorted(glob.glob(os.path.join(pt_dir, "*.pt")))
        if not pts:
            raise SystemExit(f"[ERROR] no .pt found in {pt_dir}")
        for p in pts:
            sid = os.path.splitext(os.path.basename(p))[0]
            rows.append((sid, p))

    X_list = []
    for i, (sid, p) in enumerate(rows):
        if not os.path.exists(p):
            raise SystemExit(f"[ERROR] missing pt: {p}")
        X_list.append(load_vec(p))

    X = np.stack(X_list, axis=0).astype(np.float32)
    os.makedirs(os.path.dirname(args.out_npz), exist_ok=True)
    np.savez_compressed(args.out_npz, X=X)

    mp = pd.DataFrame({
        "seq_id": [sid for sid, _ in rows],
        "row": np.arange(len(rows), dtype=int),
        "row":     np.arange(len(rows), dtype=int),
        "pt_path": [p for _, p in rows],
    })
    mp.to_csv(args.out_map, sep="\t", index=False)

    print("[WROTE]", args.out_npz, "X.shape=", X.shape)
    print("[WROTE]", args.out_map, "rows=", len(mp))

if __name__ == "__main__":
    main()
