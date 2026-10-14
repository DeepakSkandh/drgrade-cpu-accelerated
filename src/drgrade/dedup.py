"""
Near-duplicate grouping from 256-bit structure hashes (see preprocess.structure_hash).

Two images are linked if their hashes differ in <= max_bits bits; linked images
are merged into groups (union-find). Groups are kept together when splitting,
so the same retina never appears in both training and test data.
"""

import numpy as np
import pandas as pd


def _popcount64(x):
    """Bit count of each uint64 (SWAR algorithm, vectorized)."""
    x = x - ((x >> np.uint64(1)) & np.uint64(0x5555555555555555))
    x = (x & np.uint64(0x3333333333333333)) + ((x >> np.uint64(2)) & np.uint64(0x3333333333333333))
    x = (x + (x >> np.uint64(4))) & np.uint64(0x0F0F0F0F0F0F0F0F)
    return (x * np.uint64(0x0101010101010101)) >> np.uint64(56)


def hashes_to_words(hashes_hex):
    """Hex hash strings (any multiple of 16 hex chars) -> (n, n_words) uint64 array."""
    raw = np.frombuffer(b"".join(bytes.fromhex(s) for s in hashes_hex), dtype=">u8")
    return raw.astype(np.uint64).reshape(len(hashes_hex), -1)


def hamming_matrix(words, rows=slice(None)):
    """Bit distances between words[rows] and all hashes."""
    a = words[rows]
    dist = np.zeros((a.shape[0], words.shape[0]), dtype=np.uint64)
    for k in range(words.shape[1]):
        dist += _popcount64(a[:, k, None] ^ words[None, :, k])
    return dist.astype(np.int64)


def duplicate_groups(hashes_hex, max_bits=24, chunk=512):
    words = hashes_to_words(list(hashes_hex))
    n = len(words)
    parent = np.arange(n)

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    pairs = []
    for s in range(0, n, chunk):
        dist = hamming_matrix(words, slice(s, s + chunk))
        ii, jj = np.nonzero(dist <= max_bits)
        ii += s
        keep = ii < jj
        pairs.append(np.stack([ii[keep], jj[keep]], axis=1))
    pairs = np.concatenate(pairs) if pairs else np.empty((0, 2), int)

    for i, j in pairs:
        ri, rj = find(i), find(j)
        if ri != rj:
            parent[max(ri, rj)] = min(ri, rj)
    groups = np.array([find(i) for i in range(n)])
    return groups, pairs


def duplicate_report(df, groups):
    """Groups with more than one image, and whether their grades disagree."""
    d = df.assign(dup_group=groups)
    sizes = d.groupby("dup_group")["image_id"].transform("size")
    multi = d[sizes > 1]
    summary = multi.groupby("dup_group").agg(
        n_images=("image_id", "size"),
        grades=("grade", lambda g: sorted(set(g))),
        image_ids=("image_id", list),
    )
    summary["label_conflict"] = summary["grades"].apply(len) > 1
    return summary.sort_values("n_images", ascending=False)
