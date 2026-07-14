"""Datasets.

Scenes carry the state x (J(x) is a derivative with respect to it), the analytic margin computed on
a RECONSTRUCTION, and the simulator's verdict on the TRUE geometry. Those three together are what
make the paper's decisive experiment possible.
"""

from msp.data.libero import LiberoCorpusSpec, LiberoGraspDataset, generate_libero_corpus
from msp.data.rgbd import CorpusSpec, RGBDGraspDataset, generate_corpus
from msp.data.synthetic import SyntheticGraspDataset, collate

__all__ = [
    "CorpusSpec",
    "LiberoCorpusSpec",
    "LiberoGraspDataset",
    "RGBDGraspDataset",
    "SyntheticGraspDataset",
    "collate",
    "generate_corpus",
    "generate_libero_corpus",
]
