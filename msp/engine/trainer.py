import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Dict, Any
import logging

from msp.core.losses import VIBLoss
from msp.utils.logger import LoggerManager

logger = logging.getLogger(__name__)

class Trainer:
    """
    Orchestrates the training loop for the MSP framework.
    """
    def __init__(self, encoder: nn.Module, head: nn.Module, 
                 train_loader: DataLoader, val_loader: DataLoader,
                 optimizer: torch.optim.Optimizer, 
                 loss_fn: VIBLoss,
                 logger_mgr: LoggerManager,
                 device: torch.device):
        self.encoder = encoder.to(device)
        self.head = head.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.logger_mgr = logger_mgr
        self.device = device

    def train_epoch(self, epoch: int) -> Dict[str, float]:
        self.encoder.train()
        self.head.train()
        
        epoch_metrics = {}
        num_batches = len(self.train_loader)
        
        for batch_idx, batch in enumerate(self.train_loader):
            # Move to device
            obs = batch["observation"].to(self.device)
            actions = batch["actions"].to(self.device)
            outcomes = {k: v.to(self.device) for k, v in batch["outcomes"].items()}
            
            # Forward Encoder
            mu, logvar = self.encoder(obs)
            std = torch.exp(0.5 * logvar)
            
            # Reparameterization Trick (Z ~ N(mu, std^2))
            eps = torch.randn_like(std)
            z = mu + std * eps
            
            # Forward Head
            y_pred = self.head(z, actions)
            
            # Compute VIB Loss
            loss, metrics = self.loss_fn(y_pred, outcomes, mu, logvar)
            
            # Backward
            self.optimizer.zero_grad()
            loss.backward()
            
            # Gradient clipping to prevent instability from NLL
            torch.nn.utils.clip_grad_norm_(self.encoder.parameters(), max_norm=1.0)
            torch.nn.utils.clip_grad_norm_(self.head.parameters(), max_norm=1.0)
            
            self.optimizer.step()
            
            # Accumulate metrics
            for k, v in metrics.items():
                epoch_metrics[k] = epoch_metrics.get(k, 0.0) + v
                
        # Average metrics
        epoch_metrics = {k: v / num_batches for k, v in epoch_metrics.items()}
        return epoch_metrics

    def fit(self, num_epochs: int):
        for epoch in range(num_epochs):
            self.logger_mgr.info(f"Starting Epoch {epoch}")
            
            metrics = self.train_epoch(epoch)
            
            # Log metrics
            self.logger_mgr.log_metrics(metrics, step=epoch)
            
            # Save checkpoint (simplified for blueprint)
            if epoch % 5 == 0:
                torch.save({
                    'encoder_state_dict': self.encoder.state_dict(),
                    'head_state_dict': self.head.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                }, f"checkpoint_epoch_{epoch}.pth")
                self.logger_mgr.info(f"Saved checkpoint for epoch {epoch}")
