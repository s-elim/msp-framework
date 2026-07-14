"""Datasets. Scenes carry the state x (J(x) needs it) and importance weights (V4 needs them)."""

from msp.data.synthetic import SyntheticGraspDataset, collate

__all__ = ["SyntheticGraspDataset", "collate"]
