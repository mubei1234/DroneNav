import gc
import re
from typing import Literal

import numpy as np
import torch
from PIL import Image
from peft import PeftModel
from transformers import AutoTokenizer

from llava.constants import DEFAULT_IM_END_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IMAGE_TOKEN, IMAGE_PLACEHOLDER, IMAGE_TOKEN_INDEX
from llava.conversation import conv_templates
from llava.mm_utils import get_model_name_from_path, process_images, tokenizer_image_token
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from llava.model import LlavaMistralForCausalLM

IGNORE_INDEX = -100
DEFAULT_IMAGE_PATCH_TOKEN = "<im_patch>"

tokenizer = None
model = None
image_processor = None

def load_model(load_8bit=False, device_map="auto", device="cuda", use_flash_attn=False):
    global tokenizer, model, image_processor
    
    kwargs = {"device_map": device_map}
    
    if device != "cuda":
        kwargs['device_map'] = {"": device}
    
    if load_8bit:
        kwargs['load_in_8bit'] = True
    else:
        kwargs['torch_dtype'] = torch.float16
    
    if use_flash_attn:
        kwargs['attn_implementation'] = 'flash_attention_2'
    
    base_model_path = ""
    finetuned_model_path = ""
    
    tokenizer = AutoTokenizer.from_pretrained(base_model_path)
    
    model = LlavaMistralForCausalLM.from_pretrained(
        base_model_path,
        low_cpu_mem_usage=True,
        **kwargs
    )
    
    model = PeftModel.from_pretrained(
            model,
            finetuned_model_path,
            adapter_name="lora_adapter",
            #config=peft_config,
            is_trainable=False
        )
        
    model = model.merge_and_unload()

    image_processor = None
    mm_use_im_start_end = getattr(model.config, "mm_use_im_start_end", False)
    mm_use_im_patch_token = getattr(model.config, "mm_use_im_patch_token", True)
    if mm_use_im_patch_token:
        tokenizer.add_tokens([DEFAULT_IMAGE_PATCH_TOKEN], special_tokens=True)
    if mm_use_im_start_end:
        tokenizer.add_tokens([DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN], special_tokens=True)
    model.resize_token_embeddings(len(tokenizer))

    vision_tower = model.get_vision_tower()
    if not vision_tower.is_loaded:
        vision_tower.load_model(device_map=device_map)
    if device_map != 'auto':
        vision_tower.to(device=device_map, dtype=torch.float16)
    image_processor = vision_tower.image_processor
    
    return model, tokenizer, image_processor

def query(
    image: np.ndarray,
    query: str,
    temperature=0,
    top_p=None,
    num_beams=1,
    max_new_tokens=512,
):
    global tokenizer, model, image_processor

    disable_torch_init()

    images = [Image.fromarray(image)]

    image_token = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN if model.config.mm_use_im_start_end else DEFAULT_IMAGE_TOKEN
    query = re.sub(IMAGE_PLACEHOLDER, image_token, query) if IMAGE_PLACEHOLDER in query else image_token + "\n" + query

    conv = conv_templates["chatml_direct"].copy()
    conv.append_message(conv.roles[0], query)
    conv.append_message(conv.roles[1], None)
    prompt = conv.get_prompt()
    
    images_tensor = process_images(images, image_processor, model.config).to(model.device, dtype=torch.float16)
    input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt").unsqueeze(0).cuda()
    
    attention_mask = input_ids.ne(tokenizer.pad_token_id).int().cuda()

    with torch.inference_mode():
        output_ids = model.generate(
            
            input_ids,
            attention_mask=attention_mask,
            images=images_tensor,
            image_sizes=[img.size for img in images],
            do_sample=temperature > 0,
            temperature=temperature,
            top_p=top_p,
            num_beams=num_beams,
            max_new_tokens=max_new_tokens,
            use_cache=None,
            pad_token_id=tokenizer.pad_token_id
        )

    outputs = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()
    return outputs


def unload_model():

    if _model is not None:
        del _tokenizer
        del _model
        del _image_processor
        gc.collect()
        torch.cuda.empty_cache()
    
