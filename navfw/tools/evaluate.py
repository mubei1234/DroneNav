from collections import defaultdict
from dataclasses import dataclass, asdict
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F
import math
from torch.utils.data.dataloader import DataLoader
from tqdm import tqdm, trange

from navfw.configs.dronenav.parser import ExperimentArgs
from navfw.dataset.episode import Episode, EpisodeID
from navfw.observation import cropclient
from navfw.models.navigator import Navigator
from navfw.maps.mapdata import MAP_BOUNDS
from navfw.maps.landmark_nav_map import LandmarkNavMap
from navfw.scr.space import Point2D, Point3D, Pose4D
from navfw.Trajectory_Planning.lookahead import lookahead_discrete_action
from navfw.Trajectory_Planning.trajectory import _moved_pose

from navfw.scr.actions import Action, DiscreteAction
from transformers import BertTokenizerFast


@dataclass
class EvalMetrics:
    mean_final_pos_to_goal_dist: float = np.inf
    mean_final_pred_to_goal_dist: float = np.inf
    success_rate_final_pos_to_goal: float = 0.
    success_rate_final_pred_to_goal: float = 0.
    mean_oracle_pos_to_goal_dist: float = np.inf
    mean_oracle_pred_to_goal_dist: float = np.inf
    success_rate_oracle_pos_to_goal: float = 0.
    success_rate_oracle_pred_to_goal: float = 0.
    mean_progress_mse: float = np.inf
    mean_final_progress_mse: float = np.inf
    mean_spl: float = 0.
    
    @classmethod
    def names(cls):
        return list(asdict(cls()))
    
    def to_dict(self):
        return asdict(self)


def eval(
    args: ExperimentArgs,
    episodes: list[Episode],
    trajectory_logs: dict[EpisodeID, list[Pose4D]],
    pred_goal_logs: dict[EpisodeID, list[Point2D]],
    pred_progress_logs: dict[EpisodeID, list[float]],
):
    final_pos_to_goal_dists = np.array([trajectory_logs[eps.id][-1].xy.dist_to(eps.target_position.xy) for eps in episodes])
    final_pred_to_goal_dists = np.array([pred_goal_logs[eps.id][-1].dist_to(eps.target_position.xy) for eps in episodes])

  
    def oracle_distance(goal: Point2D, trajectory: list[Point2D]) -> float:
        goal = np.array(goal)
        trajectory = np.array(trajectory)
        distances = np.linalg.norm(goal - trajectory, axis=-1)
        return distances.min()

    oracle_pos_to_goal_dists = np.array([oracle_distance(eps.target_position.xy, [pose.xy for pose in trajectory_logs[eps.id]]) for eps in episodes])
    oracle_pred_to_goal_dists = np.array([oracle_distance(eps.target_position.xy, pred_goal_logs[eps.id]) for eps in episodes])

    mean_progress_mse = np.mean([
        np.mean([
            ((1 - min(eps.target_position.xy.dist_to(pose.xy) / eps.target_position.xy.dist_to(eps.start_pose.xy), 1)) - pred_progress) ** 2
            for pose, pred_progress in zip(trajectory_logs[eps.id], pred_progress_logs[eps.id])
        ]) for eps in episodes
    ])
    
    mean_final_progress_mse = np.mean([
        ((1 - min(eps.target_position.xy.dist_to(trajectory_logs[eps.id][-1].xy) / eps.target_position.xy.dist_to(eps.start_pose.xy), 1)) - pred_progress_logs[eps.id][-1]) ** 2
        for eps in episodes
    ])
    path_lengths = []
    for eps in episodes:
        trajectory = trajectory_logs[eps.id]
        path_length = 0.0
        for i in range(1, len(trajectory)):
            prev_xy = trajectory[i-1].xy
            curr_xy = trajectory[i].xy
            path_length += prev_xy.dist_to(curr_xy)
        path_lengths.append(path_length)
    
    shortest_lengths = [eps.start_pose.xy.dist_to(eps.target_position.xy) for eps in episodes]
    success = final_pos_to_goal_dists <= args.success_dist
    
    spl_values = []
    for pl, sl, s in zip(path_lengths, shortest_lengths, success):
        if sl == 0:
            spl = float(s)
        else:
            spl = s * (sl / max(pl, sl))
        spl_values.append(spl)
    mean_spl = np.mean(spl_values)


    metrics = EvalMetrics(
        final_pos_to_goal_dists.mean(),
        final_pred_to_goal_dists.mean(),
        (final_pos_to_goal_dists <= args.success_dist).mean(),
        (final_pred_to_goal_dists <= args.success_dist).mean(),
        oracle_pos_to_goal_dists.mean(),
        oracle_pred_to_goal_dists.mean(),
        (oracle_pos_to_goal_dists <= args.success_dist).mean(),
        (oracle_pred_to_goal_dists <= args.success_dist).mean(),
        mean_progress_mse, mean_final_progress_mse, 
        mean_spl=mean_spl
    )

    return metrics


@torch.no_grad()
def run_episodes_batch(
    args: ExperimentArgs,
    predictor: Navigator,
    episodes: list[Episode],
    device: str,
):
    ACTION_WEIGHT = 0.1
    ACTION_DECAY = 0.95

    cropclient.load_image_cache(alt_env=args.alt_env)
    
    dataloader = DataLoader(episodes, args.eval_batch_size, shuffle=False, collate_fn=lambda x: x, num_workers=0)
    tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased')

    pose_logs: dict[EpisodeID, list[Pose4D]] = defaultdict(list)
    pred_goal_logs: dict[EpisodeID, list[Point2D]] = defaultdict(list)
    pred_progress_logs: dict[EpisodeID, list[float]] = defaultdict(list)
    pred_action_logs: dict[EpisodeID, list[DiscreteAction]] = defaultdict(list)

    action_influence = defaultdict(lambda: 1.0)

    episodes_batch: list[Episode]
    for episodes_batch in tqdm(dataloader, desc='eval episodes', unit='batch', colour='#88dd88', position=1):

        batch_size = len(episodes_batch)
        poses = [eps.start_pose for eps in episodes_batch]
        dones = np.zeros(batch_size, dtype=bool)
        nav_maps = [
            LandmarkNavMap(
                eps.map_name, args.map_shape, args.map_pixels_per_meter,
                eps.description_landmarks, eps.description_target, eps.description_surroundings, args.gsam_params
            ) for eps in episodes_batch
        ]

        encoded_instructions : torch.Tensor = tokenizer(
        [episode.target_description for episode in episodes_batch],
        padding=True,
        truncation=True,
        return_tensors='pt'
        )

        for t in trange(args.eval_max_timestep, desc='eval timestep', unit='step', colour='#66aa66', position=2, leave=False):

            gps_noise_batch = np.random.normal(scale=args.gps_noise_scale, size=(batch_size, 2))
            noisy_poses = [Pose4D(x + n_x, y + n_y, z, yaw) for (x, y, z, yaw), (n_x, n_y) in zip(poses, gps_noise_batch)]
            for eps, pose, noisy_pose, nav_map, done in tqdm(zip(episodes_batch, poses, noisy_poses, nav_maps, dones), desc='updating maps', unit='map', colour='#448844', position=3, leave=False):
                if not done:
                    gsam_rgb = cropclient.crop_image(eps.map_name, pose, args.gsam_rgb_shape, 'rgb')
                    nav_map.update_observations(noisy_pose, gsam_rgb, None, args.gsam_use_map_cache)
                    pose_logs[eps.id].append(pose)

            maps = np.stack([nav_map.to_array() for nav_map in nav_maps])
            rgbs = np.stack([cropclient.crop_image(eps.map_name, pose, (224, 224), 'rgb') for eps, pose in zip(episodes_batch, poses)]).transpose(0, 3, 1, 2)
            normalized_depths = np.stack([cropclient.crop_image(eps.map_name, pose, (256, 256), 'depth') for eps, pose in zip(episodes_batch, poses)]).transpose(0, 3, 1, 2) / args.max_depth

            maps = torch.tensor(maps, device=device)
            rgbs = torch.tensor(rgbs, device=device)
            normalized_depths = torch.tensor(normalized_depths, device=device, dtype=torch.float32)
            input_ids = encoded_instructions["input_ids"].to(device)
            attention_mask = encoded_instructions["attention_mask"].to(device)
            pred_normalized_goal_xys, pred_progresses, action_logits  = predictor({"input_ids": input_ids, "attention_mask": attention_mask}, maps, rgbs, normalized_depths, flip_depth=True)

            action_probs = F.softmax(action_logits, dim=1).cpu().numpy()
            action_indices = np.argmax(action_probs, axis=1)
            predicted_actions = [DiscreteAction.from_index(idx) for idx in action_indices]

            pred_goal_xys = [unnormalize_position(xy.tolist(), eps.map_name, args.map_meters) for eps, xy in zip(episodes_batch, pred_normalized_goal_xys)]

            for eps, done, xy, progress, action in zip(episodes_batch, dones, pred_goal_xys, pred_progresses.flatten().tolist(), predicted_actions):
                if not done:
                    pred_goal_logs[eps.id].append(xy)
                    pred_progress_logs[eps.id].append(progress)
                    pred_action_logs[eps.id].append(action)

            dones = dones | (pred_progresses.cpu().numpy().flatten() >= args.progress_stop_val)
            
            if dones.all():
                break

            new_poses = []

            for i, (pose, noisy_pose, xy, done, predicted_action) in enumerate(zip(poses, noisy_poses, pred_goal_xys, dones, predicted_actions)):
                eps = episodes_batch[i]
                eps_id = eps.id
    
                if action_influence[eps_id] > 0.01:
                    influence_factor = ACTION_WEIGHT * action_influence[eps_id]
                    new_pose = move1(pose, xy, args.move_iteration, noisy_pose, predicted_action, influence_factor) if not done else pose
                    new_poses.append(new_pose)
                    action_influence[eps_id] *= ACTION_DECAY
                else:
                    new_pose = move2(pose, xy, args.move_iteration, noisy_pose) if not done else pose
                    new_poses.append(new_pose)
            poses = new_poses

    return dict(pose_logs), dict(pred_goal_logs), dict(pred_progress_logs), dict(pred_action_logs)


def move1(pose: Pose4D, dst: Point2D, iterations: int, noisy_pose: Pose4D, predicted_actions: DiscreteAction, influence_factor: float):
    
    dst = Point3D(dst.x, dst.y, pose.z)
    
    for i in range(iterations):
        action = lookahead_discrete_action(noisy_pose, [dst])
        
        if i == 0:
            blended_yaw = action.value[1] + influence_factor * predicted_actions.value[1]
            pose = _moved_pose(pose, action.value[0], blended_yaw, action.value[2])
        else:
            pose = _moved_pose(pose, *action.value)
        
        noisy_pose = pose
    
    return pose    

def move2(pose: Pose4D, dst: Point2D, iterations: int, noisy_pose: Pose4D):

    dst = Point3D(dst.x, dst.y, pose.z)

    for _ in range(iterations):
        action = lookahead_discrete_action(noisy_pose, [dst])
        pose = _moved_pose(pose, *action.value)
        noisy_pose = pose
    
    return pose

def unnormalize_position(normalized_xy: tuple[float, float], map_name: str, map_meters: float):
    nx, ny = normalized_xy
    return Point2D(nx * map_meters + MAP_BOUNDS[map_name].x_min, MAP_BOUNDS[map_name].y_max - ny * map_meters)