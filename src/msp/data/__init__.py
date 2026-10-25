"""Datasets: LM-Grasp (real LINEMOD RGB-D crops with simulated grasp outcomes, `linemod`) and
the batch collation shared by every scorer (`collate`)."""

from msp.data.collate import collate

__all__ = ["collate"]
