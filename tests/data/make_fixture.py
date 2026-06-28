"""Generate a tiny synthetic laterally-translating sequence for fast, deterministic tests."""
import os
import numpy as np
from imageio import imwrite


def make_sequence(out_dir, prefix="syn", n=8, w=120, h=80, dx=12, dy=0, seed=0):
    os.makedirs(out_dir, exist_ok=True)
    rng = np.random.default_rng(seed)
    base = (rng.random((h + dy * n, w + dx * n)) * 255).astype(np.uint8)
    base = np.dstack([base, base, base])
    for i in range(n):
        x0, y0 = i * dx, i * dy
        frame = base[y0:y0 + h, x0:x0 + w, :]
        imwrite(os.path.join(out_dir, "%s%03d.jpg" % (prefix, i + 1)), frame)
    return out_dir, prefix, n


if __name__ == "__main__":
    make_sequence(os.path.join(os.path.dirname(__file__), "syn"))
