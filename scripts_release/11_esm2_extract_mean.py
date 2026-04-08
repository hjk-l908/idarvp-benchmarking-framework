import argparse
from pathlib import Path

import torch
import esm


def read_fasta(fp: str):
    """Yield (id, seq) from FASTA. id = text after '>' until first whitespace."""
    seq_id = None
    seq_chunks = []
    with open(fp, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.startswith(">"):
                if seq_id is not None:
                    yield seq_id, "".join(seq_chunks)
                seq_id = line[1:].split()[0]
                seq_chunks = []
            else:
                seq_chunks.append(line)
        if seq_id is not None:
            yield seq_id, "".join(seq_chunks)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="e.g., esm2_t6_8M_UR50D")
    ap.add_argument("--layer", type=int, required=True, help="repr layer index, e.g., 6")
    ap.add_argument("--fasta", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--truncate", type=int, default=0, help="0=do not truncate; otherwise cut seq to this length")
    ap.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    ap.add_argument("--dtype", default="float16", choices=["float16", "float32"])
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = out_dir / "_manifest.tsv"

    # 1) load model
    if not hasattr(esm.pretrained, args.model):
        raise SystemExit(f"[ERROR] esm.pretrained has no model named: {args.model}")

    print(f"[INFO] loading model: {args.model} (this may download weights on first run)")
    model, alphabet = getattr(esm.pretrained, args.model)()
    model.eval()

    device = torch.device(args.device if (args.device == "cpu" or torch.cuda.is_available()) else "cpu")
    model = model.to(device)

    batch_converter = alphabet.get_batch_converter()
    pad_idx = alphabet.padding_idx

    save_dtype = torch.float16 if args.dtype == "float16" else torch.float32

    # 2) read sequences
    seqs = []
    n_total = 0
    for sid, s in read_fasta(args.fasta):
        if args.truncate and len(s) > args.truncate:
            s = s[: args.truncate]
        seqs.append((sid, s))
        n_total += 1

    print(f"[INFO] FASTA sequences: {n_total}")
    if n_total == 0:
        raise SystemExit("[ERROR] FASTA has 0 sequences")

    # 3) extract + save
    written = 0
    skipped = 0

    # write manifest header if not exists
    if not manifest_path.exists() or args.overwrite:
        manifest_path.write_text("id\tseq_len\tpt_path\n", encoding="utf-8")

    with open(manifest_path, "a", encoding="utf-8") as mf:
        for b_start in range(0, n_total, args.batch_size):
            batch = seqs[b_start : b_start + args.batch_size]

            # skip already-done if not overwrite
            if not args.overwrite:
                batch2 = []
                for sid, s in batch:
                    pt_path = out_dir / f"{sid}.pt"
                    if pt_path.exists():
                        skipped += 1
                    else:
                        batch2.append((sid, s))
                batch = batch2
                if not batch:
                    continue

            labels, strs, toks = batch_converter(batch)
            toks = toks.to(device)
            lens = (toks != pad_idx).sum(1).tolist()  # includes special tokens

            with torch.no_grad():
                out = model(toks, repr_layers=[args.layer], return_contacts=False)
                reps = out["representations"][args.layer]  # (B, T, D)

            for i, (sid, s) in enumerate(batch):
                L = int(lens[i])
                # exclude BOS (0) and EOS (L-1) if possible
                if L > 2:
                    mean_vec = reps[i, 1 : L - 1].mean(0)
                else:
                    mean_vec = reps[i, :L].mean(0)

                mean_vec = mean_vec.detach().to("cpu").to(save_dtype)

                pt_path = out_dir / f"{sid}.pt"
                torch.save(mean_vec, pt_path)
                mf.write(f"{sid}\t{len(s)}\t{pt_path}\n")
                written += 1

            if (b_start // args.batch_size) % 10 == 0:
                print(f"[PROGRESS] {min(b_start+args.batch_size, n_total)}/{n_total} ... written={written}, skipped={skipped}")

    print(f"[DONE] out_dir={out_dir}")
    print(f"[DONE] written={written}, skipped={skipped}")
    print(f"[DONE] manifest={manifest_path}")


if __name__ == "__main__":
    main()
