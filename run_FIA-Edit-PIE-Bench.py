import os
import json
import torch
import random
import argparse
import numpy as np
from PIL import Image
from utils import FIAEdit
from diffusers import StableDiffusion3Pipeline


if __name__ == "__main__":
    
    data_path = "data"
    output_path = "outputs"
    mapping_json_path = f"{data_path}/mapping_file.json"
    rerun_exist_images = False

    # set device
    device = torch.device(f"cuda:0" if torch.cuda.is_available() else "cpu")

    exp_configs = [
        {
            "exp_name": "FIA-Edit",
            "model_type": "SD35",
            "T_steps": 50,
            "n_avg": 1,
            "src_guidance_scale": 3.5,
            "tar_guidance_scale": 13.5,
            "n_min": 0,
            "n_max": 33,
            "seed": 42,
        }
    ]

    model_type = exp_configs[0]["model_type"] # currently only one model type per run

    if model_type == 'SD3': # diffusers == 0.30.1
        pipe = StableDiffusion3Pipeline.from_pretrained("stabilityai/stable-diffusion-3-medium-diffusers", torch_dtype=torch.float16)
        # pipe.enable_xformers_memory_efficient_attention()
    elif model_type == 'SD35': # diffusers == 0.33.1
        pipe = StableDiffusion3Pipeline.from_pretrained("stabilityai/stable-diffusion-3.5-medium", torch_dtype=torch.float16).to("cuda")
        # pipe.enable_xformers_memory_efficient_attention()
    else:
        raise NotImplementedError(f"Model type {model_type} not implemented")
    
    scheduler = pipe.scheduler
    pipe = pipe.to(device)

    for exp_dict in exp_configs:

        exp_name = exp_dict["exp_name"]
        # model_type = exp_dict["model_type"]
        T_steps = exp_dict["T_steps"]
        n_avg = exp_dict["n_avg"]
        src_guidance_scale = exp_dict["src_guidance_scale"]
        tar_guidance_scale = exp_dict["tar_guidance_scale"]
        n_min = exp_dict["n_min"]
        n_max = exp_dict["n_max"]
        seed = exp_dict["seed"]
        
        with open(mapping_json_path, "r") as f:
            editing_instruction = json.load(f)

        for key, item in editing_instruction.items():
            
            src_prompt = item["original_prompt"].replace("[", "").replace("]", "")
            tar_prompt = item["editing_prompt"].replace("[", "").replace("]", "")
            image_src_path = os.path.join(f"{data_path}/annotation_images", item["image_path"])

            present_image_save_path=image_src_path.replace(data_path, os.path.join(output_path,f'{exp_name}_{model_type}'))
            if ((not os.path.exists(present_image_save_path)) or rerun_exist_images):
                print(f"editing image [{image_src_path}] with FIA-Edit-{model_type}")
                
                # set seed
                random.seed(seed)
                np.random.seed(seed)
                torch.manual_seed(seed)
                torch.cuda.manual_seed_all(seed)
        
                negative_prompt =  "" # optionally add support for negative prompts (SD3)
                
                # load image
                image = Image.open(image_src_path)
                
                # crop image to have both dimensions divisibe by 16 - avoids issues with resizing
                image = image.crop((0, 0, image.width - image.width % 16, image.height - image.height % 16))
                image_src = pipe.image_processor.preprocess(image)
                
                # cast image to half precision
                image_src = image_src.to(device).half()
                with torch.autocast("cuda"), torch.inference_mode():
                    x0_src_denorm = pipe.vae.encode(image_src).latent_dist.mode()
                x0_src = (x0_src_denorm - pipe.vae.config.shift_factor) * pipe.vae.config.scaling_factor
                x0_src = x0_src.to(device)
                
                if model_type == 'SD3' or model_type == 'SD35':
                    x0_tar = FIAEdit(pipe,
                                     scheduler,
                                     x0_src,
                                     src_prompt,
                                     tar_prompt,
                                     negative_prompt,
                                     T_steps,
                                     n_avg,
                                     src_guidance_scale,
                                     tar_guidance_scale,
                                     n_min,
                                     n_max,)
                else:
                    raise NotImplementedError(f"Sampler type {model_type} not implemented")


                x0_tar_denorm = (x0_tar / pipe.vae.config.scaling_factor) + pipe.vae.config.shift_factor
                with torch.autocast("cuda"), torch.inference_mode():
                    image_tar = pipe.vae.decode(x0_tar_denorm, return_dict=False)[0]
                image_tar = pipe.image_processor.postprocess(image_tar)

                if not os.path.exists(os.path.dirname(present_image_save_path)):
                    os.makedirs(os.path.dirname(present_image_save_path))
                image_tar[0].save(present_image_save_path)

            else:
                print(f"skip image [{image_src_path}] with {exp_name}-{model_type}")

    print("Done")