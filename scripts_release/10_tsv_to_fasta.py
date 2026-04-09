import argparse
import pandas as pd


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tsv", required=True)
    ap.add_argument("--out_fasta", required=True)
    ap.add_argument("--id_col", default="seq_id")
    ap.add_argument("--seq_col", default="Sequence")
    ap.add_argument("--prefix", default="")
    return ap.parse_args()


def main():
    args = parse_args()
    df = pd.read_csv(args.tsv, sep="\t")

    if args.seq_col not in df.columns:
        raise ValueError(
            f"missing seq_col={args.seq_col}, cols={list(df.columns)[:30]}"
        )

    if args.id_col in df.columns:
        ids = df[args.id_col].astype(str).tolist()
    else:
        # If id_col is absent, fall back to row-based identifiers.
        ids = [f"row{i}" for i in range(len(df))]

    seqs = df[args.seq_col].astype(str).tolist()

    with open(args.out_fasta, "w", encoding="utf-8") as f:
        for sid, seq in zip(ids, seqs):
            sid = sid.replace(" ", "_")
            f.write(f">{args.prefix}{sid}\n{seq}\n")

    print("WROTE:", args.out_fasta, "n=", len(df))


if __name__ == "__main__":
    main()
