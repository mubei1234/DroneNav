import torch
import torch.nn as nn

class DynamicWeightAveraging(nn.Module):
    def __init__(self, num_tasks=3, alpha=0.9):
        super().__init__()
        self.alpha = alpha
        self.register_buffer('loss_hist', torch.ones(num_tasks))
        
    def forward(self, losses):
        
        self.loss_hist = self.alpha * self.loss_hist + (1-self.alpha)*torch.tensor(losses)
        weights = 1.0 / (self.loss_hist ** 2 + 1e-8)
        weights = weights / weights.sum()
        
        return weights.tolist()