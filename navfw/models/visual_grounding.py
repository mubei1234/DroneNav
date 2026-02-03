import os
import re
import numpy as np
from datetime import datetime
from tqdm import tqdm
from PIL import Image

from navfw.scr.cityreferobject import get_city_refer_objects
from navfw.observation import cropclient
from navfw.dataset.episode import EpisodeID
from navfw.maps.mapdata import GROUND_LEVEL
from navfw.configs.dronenav.parser import ExperimentArgs
from navfw.scr.space import Point2D, Point3D, Pose4D, bbox_corners_to_position, crwh_to_global_bbox, view_area_corners
from navfw.models import lora_merged
from navfw.configs import som

def llava_visual_grounding(
    args: ExperimentArgs,
    pred_goal_logs: dict[EpisodeID, list[Point2D]]
) -> dict[EpisodeID, Pose4D]:
    
    NOT_IN_IMAGE = -1
    INVALID_RESPONSE = -2

    lora_merged.load_model(load_8bit=False, device_map="auto", device="cuda", use_flash_attn=False)
    
    som.load_model('semantic-sam')
    cropclient.load_image_cache(alt_env=args.alt_env)
    objects = get_city_refer_objects()

    predicted_positions = {}
    for (map_name, obj_id, desc_id), pred_goals in tqdm(pred_goal_logs.items(), desc='selecting target bbox', unit='trajectory'):

        (x, y), yaw = pred_goals[-1], 0.
        ground_level = GROUND_LEVEL[map_name]
        target_object = objects[map_name][obj_id]
        camera_pose = Pose4D(x, y, args.altitude + ground_level, yaw)
        
        target_size = (int(args.altitude * 10), int(args.altitude * 10)) 
        rgb = cropclient.crop_image(map_name, camera_pose, target_size, 'rgb')
        annotated_rgb, masks = som.annotate(rgb, 'semantic-sam', [4])
        """
        prompt = f"Here is a description of the target object. To help you better complete the task, I have broken down the description in advance and identified the specific target object, some key landmarks, and nearby background information."

        prompt += f":\n{target_object.descriptions[desc_id]}"
        prompt += f":\n{target_object.processed_descriptions[desc_id].target}"
        prompt += f":\n{target_object.processed_descriptions[desc_id].landmarks}"
        prompt += f":\n{target_object.processed_descriptions[desc_id].surroundings}"

        prompt += f"\nYour task is to find the TARGET object and  MUST return its label number which marked beside the target(e.g. 1 2 3). If the target is not present in the image, answer {NOT_IN_IMAGE} instead."
        """        
        prompt = (
            f"### Visual Navigation Task\n"
            f"You are an aerial image analysis expert. Analyze the current drone view using these steps:\n\n"
    
            f"### Step 1: Scene Comprehension (模仿微调数据描述风格)\n"
            f"Describe the current scene in the style of our training examples, focusing on:\n"
            f"- Roof colors and building types\n"
            f"- Vehicle distribution and colors\n"
            f"- Natural elements (trees/lawns)\n"
            f"- Road structures and landmarks\n"
            f"Example format: 'An aerial view of... featuring... with...'\n\n"
    
            f"### Step 2: Target Specification\n"
            f"Locate this target object:\n"
            f"**TARGET**: {target_object.processed_descriptions[desc_id].target}\n"
            f"**Near landmarks**: {target_object.processed_descriptions[desc_id].landmarks}\n"
            f"**Surroundings**: {target_object.processed_descriptions[desc_id].surroundings}\n\n"
    
            f"### Step 3: Verification Check\n"
            f"Compare your scene description with the target specification. Check for:\n"
            f"- Matching object characteristics (size/color/type)\n"
            f"- Spatial relationships to landmarks\n"
            f"- Consistency with described surroundings\n\n"
    
            f"### Step 4: Decision Protocol\n"
            f"IF target is present:\n"
            f"  ➤ Output ONLY its numeric label (e.g. '1')\n"
            f"ELSE:\n"
            f"  ➤ Output EXACTLY: '{NOT_IN_IMAGE}'\n\n"
    
            f"### Critical Constraints\n"
            f"- Your scene description MUST use aerial perspective terminology\n"
            f"- Final output MUST be a single number or {NOT_IN_IMAGE}\n"
            f"- DO NOT explain your reasoning in the final output"
        )
        response = lora_merged.query(annotated_rgb, prompt)

        try:
            label = int(response)
        except ValueError:
            label = INVALID_RESPONSE
        
        bbox_corners = crwh_to_global_bbox(masks[label - 1]['bbox'], rgb.shape[:2], camera_pose, ground_level) if 0 < label <= len(masks) else view_area_corners(camera_pose, ground_level)
        pred_pos = bbox_corners_to_position(bbox_corners, ground_level) if 0 < label <= len(masks) else camera_pose.xyz

        camera_z = GROUND_LEVEL[map_name] + args.altitude
        camera_pose = Pose4D(pred_pos.x, pred_pos.y, camera_z, 0)
        depth = cropclient.crop_image(map_name, camera_pose, (100, 100), 'depth')
        z_around_center = camera_pose.z - depth[45:55, 45:55].mean()
        final_pose = Pose4D(pred_pos.x, pred_pos.y, z_around_center + 5, 0)

        predicted_positions[(map_name, obj_id, desc_id)] = final_pose
    
    return predicted_positions

