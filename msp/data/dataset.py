import logging
import torch
from torch.utils.data import Dataset
from typing import Dict, Any, List, Optional
import h5py
import numpy as np

from msp.data import DATASET_REGISTRY

logger = logging.getLogger(__name__)

class BaseDataset(Dataset):
    """
    Abstract interface for all MSP Datasets.
    Enforces that __getitem__ returns a standardized dictionary dictionary of tensors.
    """
    def __len__(self) -> int:
        raise NotImplementedError

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        raise NotImplementedError


@DATASET_REGISTRY.register()
class MSPOfflineDataset(BaseDataset):
    """
    Loads offline simulation or real-world trajectory data from HDF5 files.
    
    Expected HDF5 structure:
    /observations/rgb      (N, H, W, 3)
    /observations/depth    (N, H, W, 1)
    /actions               (N, N_a, 7)  [x, y, z, qx, qy, qz, qw]
    /outcomes/success      (N, N_a, 1)
    /outcomes/margin       (N, N_a, 1)
    /outcomes/slip         (N, N_a, 1)
    """

    def __init__(self, data_path: str, transform: Optional[Any] = None) -> None:
        """
        Args:
            data_path (str): Path to the HDF5 dataset.
            transform (Optional[Any]): Transformations to apply to the observations.
        """
        super().__init__()
        self.data_path = data_path
        self.transform = transform
        
        # We do not keep the file open to prevent multiprocessing issues in Dataloader workers.
        # Instead, we open it dynamically in __getitem__ or map it into memory if small enough.
        # For this production-ready blueprint, we assume it fits in RAM or is chunked.
        
        logger.info(f"Initializing MSPOfflineDataset from {self.data_path}...")
        try:
            with h5py.File(self.data_path, 'r') as f:
                self.num_samples = f['observations']['rgb'].shape[0]
                logger.info(f"Loaded dataset with {self.num_samples} samples.")
        except Exception as e:
            logger.error(f"Failed to load dataset at {self.data_path}. Error: {e}")
            raise e

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Retrieves a single frame observation and its evaluated candidate actions/outcomes.
        """
        with h5py.File(self.data_path, 'r') as f:
            rgb = f['observations']['rgb'][idx]      # (H, W, 3), uint8
            depth = f['observations']['depth'][idx]  # (H, W, 1), float32
            
            # (N_a, 7) - The candidate actions evaluated for this scene
            actions = f['actions'][idx]              
            
            # Ground truth physics outcomes M(. | x, a)
            succ = f['outcomes']['success'][idx]     # (N_a, 1)
            margin = f['outcomes']['margin'][idx]    # (N_a, 1)
            slip = f['outcomes']['slip'][idx]        # (N_a, 1)

        # Combine RGB and Depth into a single [4, H, W] tensor
        rgb_t = torch.from_numpy(rgb).permute(2, 0, 1).float() / 255.0
        depth_t = torch.from_numpy(depth).permute(2, 0, 1).float()
        obs_t = torch.cat([rgb_t, depth_t], dim=0)

        if self.transform:
            obs_t = self.transform(obs_t)

        return {
            "observation": obs_t,
            "actions": torch.from_numpy(actions).float(),
            "outcomes": {
                "success": torch.from_numpy(succ).float(),
                "margin": torch.from_numpy(margin).float(),
                "slip": torch.from_numpy(slip).float(),
            }
        }


def msp_collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Custom collate function to handle batching of nested outcome dictionaries.
    """
    observations = torch.stack([item["observation"] for item in batch])
    actions = torch.stack([item["actions"] for item in batch])
    
    outcomes = {
        "success": torch.stack([item["outcomes"]["success"] for item in batch]),
        "margin": torch.stack([item["outcomes"]["margin"] for item in batch]),
        "slip": torch.stack([item["outcomes"]["slip"] for item in batch]),
    }
    
    return {
        "observation": observations,
        "actions": actions,
        "outcomes": outcomes
    }
