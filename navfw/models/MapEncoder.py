import torch
import torch.nn as nn

class MapEncoder(nn.Module):
    
    def __init__(self, map_size):
        super().__init__()
        self.conv_high = nn.Sequential(
            nn.MaxPool2d(2),
            nn.Conv2d(5, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            
            nn.MaxPool2d(2),
            nn.Conv2d(128, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
        )

        self.conv_low = nn.Sequential(
            nn.Conv2d(5, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            
            nn.AdaptiveAvgPool2d((15, 15)),
            nn.Conv2d(128, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
        )

        self.attn = nn.Sequential(
            nn.Conv2d(32+64, 1, 1),
            nn.Sigmoid()
        )

        self.channel_adjust = nn.Conv2d(32+64, 32, 1)
        
        self.out_features = (map_size // 2**4)**2 * 32

    def forward(self, x):
        feat_high = self.conv_high(x)
        feat_low = self.conv_low(x)
        
        combined = torch.cat([feat_high, feat_low], dim=1)
        attn_map = self.attn(combined)
        
        weighted_feat = combined * attn_map
        out = self.channel_adjust(weighted_feat)
        
        return torch.flatten(out, 1)
