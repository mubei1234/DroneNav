import os
import json
import numpy as np
from PIL import Image

from navfw.observation import cropclient
from navfw.dataset.episode import EpisodeID
from navfw.scr.space import Point2D, Point3D, Pose4D
from navfw.configs.dronenav.parser import ExperimentArgs
from navfw.maps.mapdata import GROUND_LEVEL


altitude = 50.0  
alt_env = ""  

cropclient.load_image_cache(alt_env = alt_env)

json_path = ''
output_dir = ''

os.makedirs(output_dir, exist_ok=True)

with open(json_path, 'r') as f:
    data = json.load(f)
crop_size = (int(altitude * 10), int(altitude * 10)) 
for idx, item in enumerate(data):

    map_name = f"{item['area']}_block_{item['block']}"
    target_position = item['target_positions'][0]
    x, y, z = target_position
    ground_level = GROUND_LEVEL.get(map_name, 0.0)
    final_pose = Pose4D(x, y, altitude + ground_level, 0)
    rgb_array = cropclient.crop_image(map_name, final_pose, crop_size, 'rgb')
    rgb_img = Image.fromarray(rgb_array.astype(np.uint8))
    filename = f"{item['area']}_block_{item['block']}_{item['object_ids'][0]}_{item['ann_ids'][0]}.png"
    output_path = os.path.join(output_dir, filename)
    rgb_img.save(output_path)
    
