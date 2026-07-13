import torch
from typing import Dict, Any, Optional

class InferenceEngine:
    """
    Handles robust action selection via epistemic uncertainty marginalization.
    Implements Algorithm 2.
    """
    def __init__(self, encoder: torch.nn.Module, head: torch.nn.Module, K_samples: int = 32):
        self.encoder = encoder
        self.head = head
        self.K = K_samples

    @torch.no_grad()
    def evaluate_actions(self, obs: torch.Tensor, actions: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Evaluates candidate actions against K posterior samples of the belief.
        
        Args:
            obs: RGB-D observation (B, C, H, W).
            actions: Candidate actions (B, N_a, act_dim).
            
        Returns:
            Dict containing mean success probability s(o, a) and epistemic variance v(o, a).
        """
        # Encode observation to belief distribution parameters
        mu, logvar = self.encoder(obs)
        std = torch.exp(0.5 * logvar)
        
        # Sample K latents Z ~ q(z|o)
        # mu is (B, d). We want z to be (B, K, d)
        eps = torch.randn(mu.size(0), self.K, mu.size(1), device=mu.device)
        z_samples = mu.unsqueeze(1) + std.unsqueeze(1) * eps
        
        # Predict outcomes for all K samples and all actions
        # z_samples: (B, K, d), actions: (B, N_a, act_dim)
        # We need the head to process (B, K, N_a, d) + (B, K, N_a, act_dim)
        
        # Expand actions to match K dimension
        B, N_a, act_dim = actions.shape
        actions_exp = actions.unsqueeze(1).expand(-1, self.K, -1, -1) # (B, K, N_a, act_dim)
        z_exp = z_samples.unsqueeze(2).expand(-1, -1, N_a, -1)       # (B, K, N_a, d)
        
        # Flatten B and K for the head
        z_flat = z_exp.reshape(B * self.K, N_a, -1)
        a_flat = actions_exp.reshape(B * self.K, N_a, -1)
        
        preds = self.head(z_flat, a_flat)
        
        # Extract success probabilities (Sigmoid over logits)
        succ_probs_flat = torch.sigmoid(preds["succ_logit"]) # (B*K, N_a, 1)
        succ_probs = succ_probs_flat.view(B, self.K, N_a)
        
        # Marginalize over K to get s(o, a) and epistemic variance v(o, a)
        s_oa = succ_probs.mean(dim=1) # (B, N_a)
        v_oa = succ_probs.var(dim=1)  # (B, N_a)
        
        return {
            "s_oa": s_oa,
            "v_oa": v_oa,
            "ambiguity": v_oa.mean(dim=-1) # U(o)
        }
        
    def select_best_action(self, s_oa: torch.Tensor, v_oa: torch.Tensor, 
                           lambda_risk: float = 1.0, 
                           certified_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Risk-averse selection: a* = argmax (s - lambda * v)
        """
        scores = s_oa - lambda_risk * v_oa
        
        if certified_mask is not None:
            # Mask out uncertified actions by setting their score to -inf
            scores = scores.masked_fill(~certified_mask, float('-inf'))
            
        best_action_idxs = torch.argmax(scores, dim=-1)
        return best_action_idxs
