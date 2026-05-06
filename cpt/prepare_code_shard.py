"""Pre-tokenize Python code from bigcode/the-stack-smol into uint16 .bin shards
matching the FineWeb shard format used elsewhere in this repo.

Output shard format:
  - 256 int32 header: [magic=20240520, version=1, num_tokens, 0, 0, ...]
  - num_tokens uint16 GPT-2 BPE tokens

We emit two shards:
  cpt/data/code_train.bin  (~50M tokens, used for CPT)
  cpt/data/code_val.bin    (~5M tokens, used for new-domain eval)

Both are tokenized with the standard GPT-2 BPE so the loaded HF gpt2 checkpoint
can be continued without retraining the embedding matrix.
"""

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import tiktoken
from datasets import load_dataset
from tqdm import tqdm

MAGIC = 20240520
VERSION = 1
EOT = 50256  # <|endoftext|> in GPT-2 BPE


def write_shard(path: Path, tokens: np.ndarray) -> None:
    assert tokens.dtype == np.uint16
    header = np.zeros(256, dtype=np.int32)
    header[0] = MAGIC
    header[1] = VERSION
    header[2] = tokens.size
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as f:
        f.write(header.tobytes())
        f.write(tokens.tobytes())
    print(f"wrote {path}  ({tokens.size:,} tokens, {path.stat().st_size / 1e6:.1f} MB)")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir", default="cpt/data")
    ap.add_argument("--train_tokens", type=int, default=50_000_000)
    ap.add_argument("--val_tokens", type=int, default=5_000_000)
    ap.add_argument("--dataset", default="codeparrot/codeparrot-clean")
    ap.add_argument("--data_dir", default=None)
    args = ap.parse_args()

    enc = tiktoken.get_encoding("gpt2")

    target = args.train_tokens + args.val_tokens
    out_dir = Path(args.out_dir)

    print(f"streaming {args.dataset} (data_dir={args.data_dir}) -> {target:,} tokens")
    kwargs = {"split": "train", "streaming": True}
    if args.data_dir:
        kwargs["data_dir"] = args.data_dir
    ds = load_dataset(args.dataset, **kwargs)

    buf: list[int] = []
    pbar = tqdm(total=target, unit="tok", unit_scale=True, dynamic_ncols=True)
    last = 0
    for ex in ds:
        text = ex.get("content") or ex.get("text") or ""
        if not text:
            continue
        toks = enc.encode_ordinary(text)
        toks.append(EOT)
        buf.extend(toks)
        pbar.update(len(buf) - last)
        last = len(buf)
        if len(buf) >= target:
            break
    pbar.close()

    arr = np.array(buf[:target], dtype=np.uint16)
    train, val = arr[: args.train_tokens], arr[args.train_tokens : args.train_tokens + args.val_tokens]
    write_shard(out_dir / "code_train.bin", train)
    write_shard(out_dir / "code_val.bin", val)


if __name__ == "__main__":
    main()
