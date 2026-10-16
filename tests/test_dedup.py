"""
Tests for the duplicate-hash machinery. Read-only: no dataset files are touched.

    python -m pytest tests/test_dedup.py -q        (or)    python tests/test_dedup.py

Mechanics tests use random data only. The behaviour tests build small synthetic
fundus images in a temporary folder.
"""

import random
import sys
import tempfile
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from src.drgrade.dedup import (_popcount64, hashes_to_words, hamming_matrix,  # noqa: E402
                               duplicate_groups)
from src.drgrade.preprocess import preprocess, structure_hash                  # noqa: E402
from make_splits import nearest_distances                                     # noqa: E402

RNG = random.Random(0)


def _rand_hex(nbytes=32):
    return bytes(RNG.getrandbits(8) for _ in range(nbytes)).hex()


def _brute_hamming(a_hex, b_hex):
    return bin(int(a_hex, 16) ^ int(b_hex, 16)).count("1")


# ---------------------------------------------------------------- mechanics
def test_popcount_matches_python():
    vals = [0, 1, 2 ** 64 - 1, 0x8000000000000000, 0x5555555555555555] + \
           [RNG.getrandbits(64) for _ in range(2000)]
    got = _popcount64(np.array(vals, dtype=np.uint64))
    assert [int(g) for g in got] == [bin(v).count("1") for v in vals]


def test_hex_roundtrip_is_lossless_256_bits():
    hs = [_rand_hex() for _ in range(50)] + ["0" * 64, "f" * 64, "00" * 31 + "01"]
    w = hashes_to_words(hs)
    assert w.shape == (len(hs), 4)                       # 4 x 64 = 256 bits
    back = [b"".join(int(x).to_bytes(8, "big") for x in row).hex() for row in w]
    assert back == hs                                    # leading zeros preserved


def test_hamming_matches_bruteforce():
    hs = [_rand_hex() for _ in range(60)]
    hs[10] = hs[3]                                       # an exact duplicate
    d = hamming_matrix(hashes_to_words(hs))
    for i in range(len(hs)):
        for j in range(len(hs)):
            assert d[i, j] == _brute_hamming(hs[i], hs[j])
    assert d[3, 10] == 0


def test_nearest_excludes_self_across_chunk_boundaries():
    hs = [_rand_hex() for _ in range(1100)]              # > 2 chunks of 512
    hs[700] = hs[5]                                      # duplicate across chunks
    nn = nearest_distances(hs)
    assert nn[5] == 0 and nn[700] == 0
    others = np.delete(nn, [5, 700])
    assert others.min() > 60                             # random 256-bit hashes: ~128 apart
    # brute force on a few rows
    for i in (0, 511, 512, 1099):
        assert nn[i] == min(_brute_hamming(hs[i], hs[j]) for j in range(len(hs)) if j != i)


def test_groups_are_transitive_and_pairs_counted_once():
    base = _rand_hex()
    flip = lambda h, k: f"{int(h, 16) ^ ((1 << k) - 1):064x}"        # flip k low bits
    hs = [base, flip(base, 20), flip(base, 40), _rand_hex()]
    groups, pairs = duplicate_groups(hs, max_bits=24)
    # 0-1 are 20 apart, 1-2 are 20 apart, 0-2 are 40 apart -> chained into one group
    assert groups[0] == groups[1] == groups[2] != groups[3]
    assert sorted(map(tuple, pairs)) == [(0, 1), (1, 2)]


# ---------------------------------------------------------------- behaviour
def _synthetic_fundus(seed, size=1200, n_spots=25):
    rng = np.random.default_rng(seed)
    img = np.zeros((size, size, 3), np.uint8)
    c = size / 2
    yy, xx = np.mgrid[0:size, 0:size]
    r = np.hypot(xx - c, yy - c)
    fov = r < 0.47 * size
    shade = 1 - 0.4 * (r / (0.47 * size)) ** 2
    img[fov] = np.clip(np.stack([200 * shade, 105 * shade, 45 * shade], -1)[fov]
                       + rng.normal(0, 2.5, (fov.sum(), 3)), 0, 255)
    for k in range(9):
        a = rng.uniform(0, 2 * np.pi)
        t = np.arange(0, 0.8 * size, 6)
        ang = a + 0.4 * np.sin(t / 150 + rng.uniform(0, 6))
        pts = np.stack([c - 0.2 * size + t * np.cos(ang), c + t * np.sin(ang)], 1).astype(np.int32)
        cv2.polylines(img, [pts], False, (130, 40, 20), 10)
    for _ in range(n_spots):
        p = rng.integers(int(0.2 * size), int(0.8 * size), 2)
        cv2.circle(img, tuple(int(v) for v in p), int(rng.integers(3, 14)), (110, 30, 18), -1)
    img[~fov] = 0
    return img[..., ::-1].copy()          # BGR, ready for cv2.imwrite


def _hash_file(path):
    pre = preprocess(path)
    return pre["quality"]["q_dhash"], pre


def test_structure_hash_is_256_bits_and_deterministic():
    z = np.random.default_rng(1).normal(size=(768, 768)).astype(np.float32)
    inner = np.ones_like(z, bool)
    h1, h2 = structure_hash(z, inner), structure_hash(z, inner)
    assert h1 == h2 and len(h1) == 64 and int(h1, 16) >= 0


def test_hash_distances_on_copies_and_different_eyes():
    with tempfile.TemporaryDirectory() as d:
        d = Path(d)
        a, b = _synthetic_fundus(1), _synthetic_fundus(2)
        cv2.imwrite(str(d / "a.png"), a)
        cv2.imwrite(str(d / "a_copy.png"), a)                                   # byte-identical
        cv2.imwrite(str(d / "a_q90.jpg"), a, [cv2.IMWRITE_JPEG_QUALITY, 90])     # re-encoded
        cv2.imwrite(str(d / "a_small.png"), cv2.resize(a, (700, 700), interpolation=cv2.INTER_AREA))
        cv2.imwrite(str(d / "b.png"), b)
        h = {k: _hash_file(d / f"{k}.{'jpg' if k == 'a_q90' else 'png'}")[0]
             for k in ["a", "a_copy", "a_q90", "a_small", "b"]}
    dist = lambda x, y: _brute_hamming(h[x], h[y])
    print("distances:", {k: dist("a", k) for k in h})
    # Measured behaviour (see the audit notes): pixel-identical copies hash to exactly 0,
    # but re-encoded / resized copies land at roughly 10-45 bits, so the 24-bit threshold
    # in make_splits is NOT a reliable cut-off. What must hold is a clear gap to other eyes:
    assert dist("a", "a_copy") == 0
    assert dist("a", "a_q90") < dist("a", "b") / 2
    assert dist("a", "a_small") < dist("a", "b") / 2
    assert dist("a", "b") > 60


def test_degenerate_images_do_not_share_a_constant_hash():
    """A flat retina (no structure) gives many tied cells; check it does not collapse
    to the same hash as another flat retina with different noise."""
    with tempfile.TemporaryDirectory() as d:
        d = Path(d)
        hashes = []
        for seed in (10, 11):
            img = _synthetic_fundus(seed, n_spots=0)
            img[...] = np.where(img.any(axis=2, keepdims=True),
                                np.uint8([200, 100, 40]), 0).astype(np.uint8)   # flat colour disc
            img = np.clip(img + np.random.default_rng(seed).normal(0, 1.0, img.shape) * (img > 0),
                          0, 255).astype(np.uint8)
            cv2.imwrite(str(d / f"flat{seed}.png"), img)
            hashes.append(_hash_file(d / f"flat{seed}.png")[0])
    pops = [bin(int(h, 16)).count("1") for h in hashes]
    print("flat-image popcounts:", pops, "distance:", _brute_hamming(*hashes))
    assert _brute_hamming(*hashes) > 0


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print("PASS", name)
