import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
from typing import Dict, Any

from msp.core.inference import InferenceEngine
from msp.core.calibration import ConformalCalibrator
from msp.utils.logger import LoggerManager

class Evaluator:
    """
    Offline evaluation orchestrator. Validates prediction coverage and average success.
    """
    def __init__(self, encoder: nn.Module, head: nn.Module, 
                 calibrator: ConformalCalibrator,
                 logger_mgr: LoggerManager,
                 device: torch.device):
        self.engine = InferenceEngine(encoder.to(device), head.to(device), K_samples=32)
        self.calibrator = calibrator
        self.logger_mgr = logger_mgr
        self.device = device

    @torch.no_grad()
    def calibrate(self, calib_loader: DataLoader) -> float:
        """Runs the dataset to compute q_hat for conformal prediction."""
        self.engine.encoder.eval()
        self.engine.head.eval()
        
        all_s = []
        all_y = []
        
        for batch in calib_loader:
            obs = batch["observation"].to(self.device)
            actions = batch["actions"].to(self.device)
            labels = batch["outcomes"]["success"].to(self.device)
            
            eval_dict = self.engine.evaluate_actions(obs, actions)
            
            all_s.append(eval_dict["s_oa"].cpu().numpy())
            all_y.append(labels.cpu().numpy())
            
        s_arr = np.concatenate(all_s).flatten()
        y_arr = np.concatenate(all_y).flatten()
        
        q_hat = self.calibrator.fit(s_arr, y_arr)
        self.logger_mgr.info(f"Calibration complete. q_hat = {q_hat:.4f}")
        return q_hat

    @torch.no_grad()
    def test_coverage(self, test_loader: DataLoader) -> Dict[str, float]:
        """Tests if the conformal set actually bounds the target alpha."""
        total_samples = 0
        covered_samples = 0
        
        for batch in test_loader:
            obs = batch["observation"].to(self.device)
            actions = batch["actions"].to(self.device)
            labels = batch["outcomes"]["success"].to(self.device)
            
            eval_dict = self.engine.evaluate_actions(obs, actions)
            s_oa = eval_dict["s_oa"]
            
            certified_mask = self.calibrator.get_certified_set(s_oa)
            
            # Coverage: If the action is in the certified set, is the true label 1?
            # Or strictly: is the true label contained in the predicted set?
            # For marginal coverage of success=1:
            for i in range(certified_mask.size(0)):
                for j in range(certified_mask.size(1)):
                    if certified_mask[i, j]:
                        total_samples += 1
                        if labels[i, j, 0] == 1.0:
                            covered_samples += 1
                            
        coverage_rate = covered_samples / max(1, total_samples)
        self.logger_mgr.info(f"Empirical Coverage Rate: {coverage_rate:.4f} (Target: {1-self.calibrator.alpha:.4f})")
        return {"coverage_rate": coverage_rate}
