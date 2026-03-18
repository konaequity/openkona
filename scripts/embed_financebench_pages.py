#!/usr/bin/env python3
"""Embed all FinanceBench page-level chunks using Qwen3-Embedding-0.6B.

Produces a prebuilt_index.npz compatible with konash's Corpus loader.

IMPORTANT: Qwen3-Embedding uses LAST-TOKEN pooling, not mean pooling.
This must match how queries are embedded at search time (via
sentence-transformers which auto-detects last-token pooling from the
model config).

Usage (on GPU machine):
    pip install torch transformers numpy
    python embed_financebench_pages.py --pages-dir ./pages --output prebuilt_index.npz

The pages directory should contain .txt files named like:
    3M_2018_10K_p0.txt, 3M_2018_10K_p1.txt, ...
"""

from __future__ import annotations

import argparse
import glob
import os
import time

import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer


EMBED_MODEL = "Qwen/Qwen3-Embedding-0.6B"
BATCH_SIZE = 64
MAX_LENGTH = 512  # Match KARL paper token budget per chunk


def load_pages(pages_dir: str) -> tuple[list[str], list[str]]:
    """Load all .txt page files, return (doc_ids, texts)."""
    files = sorted(glob.glob(os.path.join(pages_dir, "*.txt")))
    doc_ids = []
    texts = []
    for f in files:
        doc_id = os.path.splitext(os.path.basename(f))[0]
        with open(f, "r", encoding="utf-8") as fh:
            text = fh.read().strip()
        if len(text) >= 5:  # Skip empty/near-empty pages (bare page numbers, etc.)
            doc_ids.append(doc_id)
            texts.append(text)
    return doc_ids, texts


def embed_batch(
    texts: list[str],
    model: AutoModel,
    tokenizer: AutoTokenizer,
    max_length: int = MAX_LENGTH,
) -> np.ndarray:
    """Embed a batch of texts using last-token pooling.

    Qwen3-Embedding models use last-token pooling (the embedding of the
    final non-padding token), NOT mean pooling or CLS pooling. This is
    confirmed by the sentence-transformers config:
        pooling_mode_lasttoken: True
    """
    encoded = tokenizer(
        texts,
        padding=True,
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
    ).to(model.device)

    with torch.no_grad():
        outputs = model(**encoded)
        attention_mask = encoded["attention_mask"]
        hidden = outputs.last_hidden_state

        # Last-token pooling: get the embedding at the last non-padding position
        # For each sequence, find the index of the last real token
        seq_lens = attention_mask.sum(dim=1) - 1  # (batch_size,)
        batch_indices = torch.arange(hidden.size(0), device=hidden.device)
        embeddings = hidden[batch_indices, seq_lens.long()]

    # L2 normalize
    embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)
    return embeddings.cpu().float().numpy().astype(np.float16)


def main():
    parser = argparse.ArgumentParser(description="Embed FinanceBench pages with Qwen3-0.6B")
    parser.add_argument("--pages-dir", required=True, help="Directory with page .txt files")
    parser.add_argument("--output", default="prebuilt_index.npz", help="Output .npz path")
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE, help="Batch size")
    parser.add_argument("--max-length", type=int, default=MAX_LENGTH, help="Max token length")
    args = parser.parse_args()

    print(f"Loading pages from {args.pages_dir}...")
    doc_ids, texts = load_pages(args.pages_dir)
    print(f"  {len(doc_ids)} non-empty pages loaded")

    print(f"Loading model {EMBED_MODEL}...")
    tokenizer = AutoTokenizer.from_pretrained(EMBED_MODEL, trust_remote_code=True)
    model = AutoModel.from_pretrained(EMBED_MODEL, trust_remote_code=True)
    model.eval()

    if torch.cuda.is_available():
        model = model.cuda()
        print(f"  Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        print("  WARNING: No GPU detected, running on CPU (will be slow)")

    print(f"Embedding {len(texts)} pages in batches of {args.batch_size} (last-token pooling)...")
    all_embeddings = []
    t0 = time.monotonic()

    for i in range(0, len(texts), args.batch_size):
        batch = texts[i : i + args.batch_size]
        emb = embed_batch(batch, model, tokenizer, max_length=args.max_length)
        all_embeddings.append(emb)

        done = min(i + args.batch_size, len(texts))
        elapsed = time.monotonic() - t0
        rate = done / elapsed
        eta = (len(texts) - done) / rate if rate > 0 else 0
        print(f"  {done}/{len(texts)}  ({rate:.0f} pages/s, ETA {eta:.0f}s)")

    vectors = np.concatenate(all_embeddings, axis=0)
    total_time = time.monotonic() - t0
    print(f"\nDone! {vectors.shape[0]} vectors, dim={vectors.shape[1]}, dtype={vectors.dtype}")
    print(f"  Total time: {total_time:.1f}s ({vectors.shape[0]/total_time:.0f} pages/s)")

    # Save in konash-compatible format
    np.savez(
        args.output,
        vectors=vectors,
        doc_ids=np.array(doc_ids),
        embed_model=np.array(EMBED_MODEL),
    )
    size_mb = os.path.getsize(args.output) / (1024 * 1024)
    print(f"  Saved to {args.output} ({size_mb:.1f} MB)")


if __name__ == "__main__":
    main()
