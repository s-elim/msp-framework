import os
import random
import numpy as np
import torch
import logging

logger = logging.getLogger(__name__)

def seed_everything(seed: int = 42) -> None:
    """
    Sets the seed across all stochastic modules to ensure exact reproducibility.
    
    This is critical for the Variational Information Bottleneck (VIB) objective,
    which relies on reparameterized sampling (Z ~ N(mu, sigma)).

    Args:
        seed (int): The global integer seed. Default is 42.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # For multi-GPU setups

    # Enforce deterministic convolutions (may impact performance slightly)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    logger.info(f"Global seed set to {seed} (CuDNN deterministic=True).")
