import argparse
from typing import Literal, Optional
from dataclasses import dataclass, asdict

from navfw.maps.gsam_map import GSamParams


@dataclass
class ExperimentArgs:

    seed: int
    mode: Literal['train', 'eval']
    model: Literal['dronenav']

    log: bool
    silent: bool
    resume_log_id: str

    map_size: int
    map_meters: float
    map_update_interval: int
    max_depth: float
    altitude: float
    alt_env: Literal['']

    gsam_rgb_shape: tuple[int, int]
    gsam_use_segmentation_mask: bool
    gsam_use_bbox_confidence: bool
    gsam_use_map_cache: bool
    gsam_box_threshold: float
    gsam_text_threshold: float
    gsam_max_box_size: float
    gsam_max_box_area: float

    learning_rate: float
    train_batch_size: int
    epochs: int
    checkpoint: Optional[str]
    save_every: int
    train_episode_sample_size: int
    
    eval_every: int
    eval_batch_size: int
    eval_at_start: bool
    eval_max_timestep: int
    success_dist: float
    success_iou: float
    move_iteration: int
    progress_stop_val: float
    gps_noise_scale: float

    def to_dict(self):
        return asdict(self)
    
    @property
    def map_shape(self):
        return self.map_size, self.map_size
    
    @property
    def map_pixels_per_meter(self):
        return self.map_size / self.map_meters
    
    @property
    def gsam_params(self):
        return GSamParams(
            self.gsam_use_segmentation_mask,
            self.gsam_use_bbox_confidence,
            self.gsam_box_threshold, self.gsam_text_threshold,
            self.gsam_max_box_size, self.gsam_max_box_area
        )


def parse_args():

    parser = argparse.ArgumentParser()

    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--mode', type=str, choices=['train', 'eval'], default='train')
    parser.add_argument('--model', type=str, default='dronenav')

    parser.add_argument('--log', action='store_true', default=False, help="log results to wandb")
    parser.add_argument('--silent', action='store_true', default=False, help="disable printing log info to stdout")
    parser.add_argument('--resume_log_id', type=str, default='')

    parser.add_argument('--map_size', type=int, default=240)
    parser.add_argument('--map_meters', type=float, default=410.)
    parser.add_argument('--map_update_interval', type=int, default=5)
    parser.add_argument('--max_depth', type=float, default=200.)
    parser.add_argument('--altitude', type=float, default=50)
    parser.add_argument('--alt_env', type=str, default='')

    parser.add_argument('--gsam_rgb_shape', type=int, default=500)
    parser.add_argument('--gsam_use_segmentation_mask', action='store_true', default=False)
    parser.add_argument('--gsam_use_bbox_confidence', action='store_true', default=False)
    parser.add_argument('--gsam_use_map_cache', action='store_true', default=False)
    parser.add_argument('--gsam_box_threshold', type=float, default=0.35)
    parser.add_argument('--gsam_text_threshold', type=float, default=0.25)
    parser.add_argument('--gsam_max_box_size', type=float, default=50.)
    parser.add_argument('--gsam_max_box_area', type=float, default=3000.)
    
    parser.add_argument('--learning_rate', type=float, default=1.0e-03)
    parser.add_argument('--train_batch_size', type=int, default=8)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--checkpoint', type=str, default='')
    parser.add_argument('--save_every', type=int, default=1)
    parser.add_argument('--train_episode_sample_size', type=int, default=-1)
    
    parser.add_argument('--eval_every', type=int, default=1)
    parser.add_argument('--eval_batch_size', type=int, default=100)
    parser.add_argument('--eval_at_start', action='store_true', default=False)
    parser.add_argument('--eval_max_timestep', type=int, default=20)
    parser.add_argument('--success_dist', type=float, default=20.)
    parser.add_argument('--success_iou', type=float, default=0.4)
    parser.add_argument('--move_iteration', type=int, default=5)
    parser.add_argument('--progress_stop_val', type=float, default=0.75)
    parser.add_argument('--gps_noise_scale', type=float, default=0.)


    args = parser.parse_args()
    args.gsam_rgb_shape = args.gsam_rgb_shape, args.gsam_rgb_shape

    return ExperimentArgs(**vars(args))