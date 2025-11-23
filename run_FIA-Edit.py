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
    
    src_prompt = "a cat sitting on a wooden chair"
    tar_prompt = "a dog sitting on a wooden chair"
    negative_prompt =  "" # optionally add support for negative prompts (SD3)
    image_src_path = "example/cat.jpg"

    # set device
    device = torch.device(f"cuda:0" if torch.cuda.is_available() else "cpu")
    
    model_type = 'SD35'
    T_steps = 50
    n_avg = 1
    src_guidance_scale = 3.5
    tar_guidance_scale = 13.5
    n_min = 0
    n_max = 33
    seed = 42


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


    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
        
    # load image
    image = Image.open(image_src_path)
    image = image.resize((512, 512), Image.BILINEAR)
    # crop image to have both dimensions divisibe by 16 - avoids issues with resizing
    image = image.crop((0, 0, image.width - image.width % 16, image.height - image.height % 16))
    image_src = pipe.image_processor.preprocess(image)
    # cast image to half precision
    image_src = image_src.to(device).half()
    with torch.autocast("cuda"), torch.inference_mode():
        x0_src_denorm = pipe.vae.encode(image_src).latent_dist.mode()
    x0_src = (x0_src_denorm - pipe.vae.config.shift_factor) * pipe.vae.config.scaling_factor
    x0_src = x0_src.to(device)
    
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
        

    x0_tar_denorm = (x0_tar / pipe.vae.config.scaling_factor) + pipe.vae.config.shift_factor
    with torch.autocast("cuda"), torch.inference_mode():
        image_tar = pipe.vae.decode(x0_tar_denorm, return_dict=False)[0]
    image_tar = pipe.image_processor.postprocess(image_tar)

    src_prompt_txt = os.path.basename(image_src_path).split('.')[0]

    # make sure to create the directories before saving
    save_dir = f"outputs/{model_type}/src_{src_prompt_txt}/"
    os.makedirs(save_dir, exist_ok=True)
    
    image_tar[0].save(f"{save_dir}/output_T_steps_{T_steps}_n_avg_{n_avg}_cfg_enc_{src_guidance_scale}_cfg_dec{tar_guidance_scale}_n_min_{n_min}_n_max_{n_max}_seed{seed}.jpg")
    # also save source and target prompt in txt file
    with open(f"{save_dir}/prompts.txt", "w") as f:
        f.write(f"Source prompt: {src_prompt}\n")
        f.write(f"Target prompt: {tar_prompt}\n")
        f.write(f"Seed: {seed}\n")
        f.write(f"Sampler type: {model_type}\n")

    print("Done")