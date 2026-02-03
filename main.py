import json
import torch
import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

from navfw.configs.dronenav.defaultpaths import PROJECT_ROOT
from navfw.configs.dronenav.parser import parse_args
from navfw.tools.evaluate import eval
from navfw.scr.cityreferobject import get_city_refer_objects
from navfw.dataset.generate import generate_episodes_from_mturk_trajectories
from navfw.dataset.mturk_trajectory import load_mturk_trajectories
from navfw.models.navigator import Navigator
from navfw.models.visual_grounding import llava_visual_grounding
from navfw.tools.train import train
from navfw.tools.evaluate import run_episodes_batch

DEVICE = 'cuda'
args = parse_args()

Model = {'dronenav': Navigator,}[args.model]

if args.mode == 'train':
    train(args, DEVICE)

if args.mode == 'eval':
    model_trajectory = args.checkpoint.split('/')[-2]
    epoch = args.checkpoint.split('/')[-1].split('.')[0]
    objects = get_city_refer_objects()
    model : Navigator = Model(args.map_size).to(DEVICE)
    if args.checkpoint:
        model.load_state_dict(torch.load(args.checkpoint)['predictor_state_dict'])
    for split in ('val_seen', 'val_unseen', 'test_unseen'):
        test_episodes = generate_episodes_from_mturk_trajectories(objects, load_mturk_trajectories(split, 'easy', args.altitude))
        teacher_trajectory_logs = {
            f"{episode.target_object.map_name}_{episode.target_object.id}_{episode.description_id}": [
                tuple(pose) for pose in episode.teacher_trajectory
            ]
            for episode in test_episodes
        }
        trajectory_logs, pred_goal_logs, pred_progress_logs, pred_action_logs = run_episodes_batch(args, model, test_episodes, DEVICE)  
        predicted_positions = llava_visual_grounding(args, pred_goal_logs)
        for eps_id, pose in predicted_positions.items():
            trajectory_logs[eps_id].append(pose)
        metrics = eval(args, test_episodes, trajectory_logs, pred_goal_logs, pred_progress_logs)
        print(f"{split} -- {metrics.mean_final_pos_to_goal_dist: .1f}, {metrics.success_rate_final_pos_to_goal*100: .2f}, {metrics.success_rate_oracle_pos_to_goal*100: .2f}, {metrics.mean_spl*100: .2f}")

        save_subdir = PROJECT_ROOT / "resaults" / f'{args.model}'
        save_subdir.mkdir(parents=True, exist_ok=True)
        file_path = save_subdir / f"{split}.json"

        with open(file_path, 'w') as f:
            json.dump({
                'metrics': metrics.to_dict(),
                'trajectory_logs': {str(eps_id): [tuple(pose) for pose in trajectory] for eps_id, trajectory in trajectory_logs.items()},
                'pred_goal_logs': {str(eps_id): [tuple(pos) for pos in pred_goals] for eps_id, pred_goals in pred_goal_logs.items()},
                'pred_progress_logs': {str(eps_id): pred_progresses for eps_id, pred_progresses in pred_progress_logs.items()},
                'teacher_trajectory_logs': {
                    str(eps_id): [tuple(pose) for pose in trajectory] 
                    for eps_id, trajectory in teacher_trajectory_logs.items()
                }
            }, f)