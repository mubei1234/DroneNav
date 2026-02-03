from collections.abc import Callable

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
import math

from .resenet_encoders import TorchVisionResNet50, ResnetDepthEncoder
from .MapEncoder import MapEncoder
from transformers import BertTokenizer, BertModel


class InstructionEncoder(nn.Module):
    def __init__(self,):

        super().__init__()

        self.model = BertModel.from_pretrained("bert-base-uncased")
        
    def forward(self, instructions) -> Tensor:

        input_ids = instructions["input_ids"]
        attention_mask = instructions["attention_mask"]
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)

        return outputs.pooler_output   

class Map_guided_by_Instruction(nn.Module):
    def __init__(self, out_features = 7200):
        
        super().__init__()
        
        self.fc1 = nn.Linear(768, 7200)
        self.activate1 = nn.ReLU()

        self.fc2 = nn.Linear(7200, 7200)
        self.activate2 = nn.ReLU()

        self.fc3 = nn.Linear(7200, 7200)
        self.activate3 = nn.Sigmoid()

        self.out_features = out_features

    def forward(self, map_features, instruction_features):

        new_instruction_features = self.fc1(instruction_features)
        new_instruction_features = self.activate1(new_instruction_features)

        new_map_features = self.fc2(map_features)
        new_map_features = self.activate2(new_map_features)


        doted = torch.mul(new_instruction_features, new_map_features)

        fused = self.fc3(doted)
        fused = self.activate3(fused)

        resault = torch.mul(fused, map_features)

        return resault
    
class LiteMemory(nn.Module):
    def __init__(self, 
                 input_dim=7584,
                 bottleneck_dim=128,
                 #mem_dim=256):
                 mem_dim=128):
        super().__init__()
        
        self.compress = nn.Sequential(
            nn.Linear(input_dim, bottleneck_dim),
            nn.GELU(),
            nn.Dropout(0.1)
        )
        self.gate = nn.Sequential(
            nn.Linear(bottleneck_dim, mem_dim),
            nn.Sigmoid()
        )
        self.proj = nn.Sequential(
            nn.Linear(mem_dim, 256*256),
            nn.ReLU()
        )

    def forward(self, x):
        compressed = self.compress(x)
        mem = self.gate(compressed)
        enhanced = compressed + mem[:, :128]
        neural_map = self.proj(enhanced).view(-1, 1, 256, 256)
        return neural_map
    
class NavigationPolicy(nn.Module):
    def __init__(self):
        super().__init__()
        self.global_planner = nn.Sequential(
            nn.AdaptiveAvgPool2d((16, 32)),
            nn.Flatten(start_dim=1)
        )
        
        self.local_controller = nn.Sequential(
            nn.Linear(512+7200, 256),
            nn.Tanh(),
            nn.Linear(256, 4)
        )
        
    def forward(self, map_feat, attn_feat):
        global_feat = self.global_planner(map_feat)
        policy = self.local_controller(torch.cat([global_feat, attn_feat], dim=1))
        return policy 

class GoalPredictionHead(nn.Module):

    def __init__(self, n_map_features: int):
        super(GoalPredictionHead, self).__init__()
        
        self.prediction_head = nn.Sequential(
            nn.Linear(n_map_features, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 2),
            nn.Sigmoid(),
        )

    def forward(self, map_features):
        return self.prediction_head(map_features)


class ProgressPredictionHead(nn.Module):

    def __init__(self, n_map_features: int, n_rgb_features: int, n_depth_featuers):
        super(ProgressPredictionHead, self).__init__()

        self.prediction_head = nn.Sequential(
            nn.Linear(n_map_features + n_rgb_features + n_depth_featuers, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 1),
            nn.Sigmoid(),
        )

    def forward(self, map_features, rgb_features, depth_features):
        return self.prediction_head(torch.cat((map_features, rgb_features, depth_features), dim=1))
    

class Navigator(nn.Module):

    def __init__(self, map_size: int):
        super(Navigator, self).__init__()

        self.instruction_encoder = InstructionEncoder().eval()
        self.map_encoder = MapEncoder(map_size)
        self.map_guided_by_instruction = Map_guided_by_Instruction()
        self.memory = LiteMemory()
        self.rgb_encoder = TorchVisionResNet50().eval()
        self.policy = NavigationPolicy()
        self.depth_encoder = ResnetDepthEncoder().eval()
        self.goal_prediction_head = GoalPredictionHead(self.map_guided_by_instruction.out_features)
        self.progress_prediction_head = ProgressPredictionHead(
            self.map_guided_by_instruction.out_features, self.rgb_encoder.out_features, self.depth_encoder.out_features
        )
    
    def forward(self, instructions: Tensor, maps: Tensor, rgbs: Tensor, depths: Tensor, flip_depth=True):
        """rgb & depth (B, C, H, W)"""

        if flip_depth:
            depths = depths.flip(-2)

        instruction_feature = self.instruction_encoder(instructions)
        map_features = self.map_encoder(maps)
        map_guided_by_instruction_feature = self.map_guided_by_instruction(map_features, instruction_feature)

        rgb_features = self.rgb_encoder(rgbs)
        depth_features = self.depth_encoder(depths)
        
        neural_map = self.memory(torch.cat([map_guided_by_instruction_feature, rgb_features, depth_features], dim=1))
        action_probs = self.policy(neural_map, map_guided_by_instruction_feature)

        pred_normalized_goal_xys = self.goal_prediction_head(map_guided_by_instruction_feature)
        pred_progress = self.progress_prediction_head(map_guided_by_instruction_feature, rgb_features, depth_features)

        return pred_normalized_goal_xys, pred_progress, action_probs
