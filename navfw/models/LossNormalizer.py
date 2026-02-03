import torch
import torch.nn as nn

class LossNormalizer(nn.Module):
    def __init__(self, num_tasks=3, momentum=0.9, init_scale=1.0):
        super().__init__()
        self.num_tasks = num_tasks
        self.momentum = momentum
        self.init_scale = init_scale
        self.register_buffer('running_mean', None)
        self.step = 0
        
    def forward(self, losses):
        if self.running_mean is None:
            self.running_mean = torch.tensor([self.init_scale] * self.num_tasks, device=losses[0].device)
        
        current = torch.stack(losses)
        normalized = current / self.running_mean
        
        if self.training:
            self.step += 1
            if self.step <= 100:
                new_mean = (self.running_mean * self.step + current.detach()) / (self.step + 1)
            else:
                new_mean = self.momentum * self.running_mean + (1 - self.momentum) * current.detach()
            self.running_mean = new_mean.detach().clone()
        
        return [normalized[i] for i in range(self.num_tasks)]