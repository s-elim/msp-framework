"""Figure: every object in the corpus, one panel each.

    python scripts/paper/make_object_gallery.py

Reads the cached test corpus only. For each of the 13 HOPE grocery meshes it shows the
first test scene containing that object, view 0, with the per-object base success rate
underneath, because the corpus is strongly heterogeneous -- base rates run from 0.04 to
1.00 -- and that heterogeneity is why the paper refereess ablations on within-scene AUC
rather than a pooled number.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

DCOL = 7.16
plt.rcParams.update({"font.family": "STIXGeneral", "font.size": 6.5,
                     "savefig.dpi": 400, "savefig.bbox": "tight",
                     "savefig.pad_inches": 0.02})


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--corpus", default="results/libero/corpus/libero_197aaee1a663467e.pt")
    ap.add_argument("--out", default="ICRA'27_MSP-FRAMEWORK/figures/fig_objects")
    ap.add_argument("--cols", type=int, default=7)
    args = ap.parse_args()

    d = torch.load(args.corpus, map_location="cpu", weights_only=False, mmap=True)
    names = d["object_names"]
    oi, succ, ex = d["object_index"], d["succ"].squeeze(-1), d["executable"].bool()

    rows = int(np.ceil(len(names) / args.cols))
    fig, axes = plt.subplots(rows, args.cols, figsize=(DCOL, DCOL / args.cols * rows * 1.22))
    for k, ax in enumerate(np.atleast_1d(axes).ravel()):
        if k >= len(names):
            ax.axis("off")
            continue
        i = int(torch.nonzero(oi == k)[0])
        ax.imshow(np.clip(d["observation"][i, 0, :3].permute(1, 2, 0).numpy(), 0, 1))
        m = (oi == k).unsqueeze(-1) & ex
        base = float(succ[m].mean())
        ax.set_title(names[k].replace("_", " "), fontsize=6, pad=2)
        ax.set_xlabel(f"base {base:.2f}", fontsize=5.8, labelpad=1)
        ax.set_xticks([]); ax.set_yticks([])
        for s in ax.spines.values():
            s.set_color("#C8D0DC"); s.set_linewidth(0.6)
    fig.subplots_adjust(wspace=0.05, hspace=0.35)
    out = Path(args.out)
    fig.savefig(out.with_suffix(".pdf"))
    print("wrote", out.with_suffix(".pdf"))


if __name__ == "__main__":
    main()
