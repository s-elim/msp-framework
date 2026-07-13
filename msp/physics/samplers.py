import torch
from msp.physics import SAMPLER_REGISTRY

class BaseActionSampler:
    """Abstract interface for proposing candidate actions (rho)."""
    def sample(self, num_samples: int) -> torch.Tensor:
        raise NotImplementedError

@SAMPLER_REGISTRY.register()
class HeuristicGraspSampler(BaseActionSampler):
    """Samples grasps heuristically, e.g., pointing towards the object centroid."""
    def sample(self, num_samples: int) -> torch.Tensor:
        """
        Returns:
            torch.Tensor: Random grasp actions [N_a, 7]
        """
        # 3 for translation, 4 for quaternion
        return torch.randn(num_samples, 7)
