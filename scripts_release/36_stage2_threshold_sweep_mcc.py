import argparse
import numpy as np
import pandas as pd
from pathlib import Path

def mcc_from_counts(tp, tn, fp, fn):
    num = tp*tn - fp*fn
    den = (tp+fp)*(tp+fn)*(tn+fp)*(tn+fn)
    if den <= 0:
        return 0.0
    return float(num / np.sqrt(den))

def counts(y, p, thr):
    yhat = (p >= thr).astype(int)
    tp = int(((y==1) & (yhat==1)).sum())
    tn = int(((y==0) & (yhat==0)).sum())
    fp = int(((y==0) & (yhat==1)).sum())
    fn = int(((y==1) & (yhat==0)).sum())
    return tp, tn, fp, fn

def metrics_from_counts(tp, tn, fp, fn):
    prec = tp / (tp+fp) if (tp+fp)>0 else 0.0
    rec  = tp / (tp+fn) if (tp+fn)>0 else 0.0
    spec = tn / (tn+fp) if (tn+fp)>0 else 0.0
    f1   = 2*prec*rec/(prec+rec) if (prec+rec)>0 else 0.0
    mcc  = mcc_from_counts(tp, tn, fp, fn)
    return prec, rec, spec, f1, mcc

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scores_npz", required=True)
    ap.add_argument("--out_thresholds_csv", required=True)
    ap.add_argument("--out_test_csv", required=True)
    ap.add_argument("--k_grid", type=int, default=200)  # threshold grid size
    return ap.parse_args()

def main():
    args = parse_args()
    z = np.load(args.scores_npz, allow_pickle=True)
    labels = list(z["labels"])
    Yv = z["Y_valid"]
    Yt = z["Y_test"]
    Pv = z["P_valid"]
    Pt = z["P_test"]

    # threshold grid (0..1)
    grid = np.linspace(0.0, 1.0, args.k_grid+1)

    thr_rows = []
    test_rows = []

    for j, lab in enumerate(labels):
        yv = Yv[:, j].astype(int)
        pv = Pv[:, j].astype(float)
        yt = Yt[:, j].astype(int)
        pt = Pt[:, j].astype(float)

        # find best MCC on valid
        best = {"mcc": -1e9, "thr": 0.5, "tp":0,"tn":0,"fp":0,"fn":0}
        for thr in grid:
            tp, tn, fp, fn = counts(yv, pv, thr)
            prec, rec, spec, f1, mcc = metrics_from_counts(tp, tn, fp, fn)
            if mcc > best["mcc"]:
                best = {"mcc": mcc, "thr": float(thr), "tp":tp, "tn":tn, "fp":fp, "fn":fn,
                        "prec":prec, "rec":rec, "spec":spec, "f1":f1}

        thr_rows.append({
            "label": lab,
            "best_thr_valid_MCC": round(best["thr"], 6),
            "valid_MCC": round(best["mcc"], 6),
            "valid_F1": round(best["f1"], 6),
            "valid_precision": round(best["prec"], 6),
            "valid_recall": round(best["rec"], 6),
            "valid_specificity": round(best["spec"], 6),
            "valid_TP": best["tp"], "valid_TN": best["tn"], "valid_FP": best["fp"], "valid_FN": best["fn"],
            "valid_pos": int(yv.sum()), "valid_neg": int(len(yv)-yv.sum()),
        })

        # apply best thr to test
        tp, tn, fp, fn = counts(yt, pt, best["thr"])
        prec, rec, spec, f1, mcc = metrics_from_counts(tp, tn, fp, fn)
        test_rows.append({
            "label": lab,
            "thr_from_valid": round(best["thr"], 6),
            "test_MCC": round(mcc, 6),
            "test_F1": round(f1, 6),
            "test_precision": round(prec, 6),
            "test_recall": round(rec, 6),
            "test_specificity": round(spec, 6),
            "test_TP": tp, "test_TN": tn, "test_FP": fp, "test_FN": fn,
            "test_pos": int(yt.sum()), "test_neg": int(len(yt)-yt.sum()),
        })

    out_thr = pd.DataFrame(thr_rows).sort_values("valid_MCC", ascending=False)
    out_test = pd.DataFrame(test_rows).merge(out_thr[["label","best_thr_valid_MCC"]], on="label", how="left")
    out_test = out_test.sort_values("test_MCC", ascending=False)

    Path(args.out_thresholds_csv).parent.mkdir(parents=True, exist_ok=True)
    out_thr.to_csv(args.out_thresholds_csv, index=False)
    out_test.to_csv(args.out_test_csv, index=False)
    print("[WROTE]", args.out_thresholds_csv)
    print("[WROTE]", args.out_test_csv)

if __name__=="__main__":
    main()
