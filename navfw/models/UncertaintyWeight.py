import torch
import torch.nn as nn

class UncertaintyWeight(nn.Module):
    def __init__(self, num_tasks=3, init_logvar=0.0, min_logvar=-4.0):
        super().__init__()
        self.min_logvar = min_logvar
        self.log_vars = nn.Parameter(torch.full((num_tasks,), init_logvar))
        
    def forward(self, losses):
        log_vars = torch.clamp(self.log_vars, min=self.min_logvar, max=4.0)

        task_terms = [
            losses[i]/(2*torch.exp(log_vars[i])) + 0.5*log_vars[i]
            for i in range(len(losses))
        ]

        boundary_penalty = 0.1 * torch.sum(torch.exp(-log_vars))
        total_loss = sum(task_terms) + boundary_penalty
        return torch.clamp_min(total_loss, 0.0), log_vars.detach()