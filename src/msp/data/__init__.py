"""Datasets: LM-Grasp (real LINEMOD RGB-D crops with simulated grasp outcomes, `linemod`) and the
collate function shared by every scorer (`synthetic.collate`)."""

from msp.data.synthetic import SyntheticGraspDataset, collate

__all__ = ["SyntheticGraspDataset", "collate"]
