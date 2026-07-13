import torch
import torch.nn as nn

class ActivePerceptionManager:
    """
    Handles the Value of Information (VoI) and triggers Next-Best-View sensing.
    """
    def __init__(self, acquisition_net: nn.Module, tau_U: float = 0.1):
        """
        Args:
            acquisition_net: The amortized estimator for IG.
            tau_U: Threshold for ambiguity above which sensing is triggered.
        """
        self.acq_net = acquisition_net
        self.tau_U = tau_U

    def should_sense(self, ambiguity: torch.Tensor) -> bool:
        """
        Checks if the epistemic ambiguity U(o) exceeds the threshold.
        """
        return bool(ambiguity.mean().item() > self.tau_U)

    @torch.no_grad()
    def get_next_best_view(self, obs_features: torch.Tensor, candidate_views: torch.Tensor) -> torch.Tensor:
        """
        Evaluates candidate camera movements and returns the one that maximizes Expected IG.
        
        Args:
            obs_features: Fused features from the backbone.
            candidate_views: SE(3) transforms of potential next camera poses.
            
        Returns:
            The index of the optimal view.
        """
        ig_scores = self.acq_net(obs_features, candidate_views)
        best_view_idx = torch.argmax(ig_scores, dim=-1)
        return best_view_idx
