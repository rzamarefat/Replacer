import argparse
from functools import partial
import cv2
import requests
import os
from io import BytesIO
from PIL import Image
import numpy as np
from pathlib import Path
import gradio as gr
from src.groundingdino.models import build_model
from src.groundingdino.util.slconfig import SLConfig
from src.groundingdino.util.utils import clean_state_dict
from src.groundingdino.util.inference import annotate, load_image, predict
import src.groundingdino.datasets.transforms as T
from huggingface_hub import hf_hub_download
import warnings
from src.segment_anything import sam_model_registry, SamPredictor
import matplotlib.pyplot as plt
from .config import *
from src.fastsam import FastSAM, FastSAMPrompt
import torch
import os

from src.controlnet.share import *


import cv2
import einops
import gradio as gr
import numpy as np

import random

from pytorch_lightning import seed_everything
from src.controlnet.annotator.util import resize_image, HWC3
from src.controlnet.cldm.model import create_model, load_state_dict
from src.controlnet.cldm.ddim_hacked import DDIMSampler






class Replacer:
    def __init__(self):
        self._config_file_groundingdino = groundingdino_config["config_file"]
        self._ckpt_repo_id_groundingdino = groundingdino_config["ckpt_repo_id"]
        self._ckpt_filename_groundingdino = groundingdino_config["ckpt_filename"]
        self._groundingdino_ckpt_path = groundingdino_config["groundingdino_ckpt_path"]
        self.device = "cuda"


        self._box_threshold = 0.85 
        self._text_threshold = 0.85


        self._model_groundingdino = self._load_model_groundingdino(self._config_file_groundingdino, self._ckpt_repo_id_groundingdino, self._ckpt_filename_groundingdino)


        self._fastsam_checkpoint = fastsam_config["fastsam_checkpoint"]
        self._fastsam_model = FastSAM(fastsam_config["fastsam_checkpoint"])



        self._controlnet_model_name = controlnet_config['model_name']
        self._controlnet_model = create_model(os.path.join(controlnet_config["path_to_config_yaml_file"], f"{self._controlnet_model_name}.yaml")).cpu()
        self._controlnet_model.load_state_dict(load_state_dict(controlnet_config["path_to_sd_model"], location=self.device), strict=False)
        self._controlnet_model.load_state_dict(load_state_dict(controlnet_config["path_to_cldm"], location=self.device), strict=False)
        self._controlnet_model = self._controlnet_model.cuda()
        self._ddim_sampler = DDIMSampler(self._controlnet_model)

    def _load_model_groundingdino(self, model_config_path, repo_id, filename, device='cpu'):
        args = SLConfig.fromfile(model_config_path) 
        model = build_model(args)
        args.device = device

        checkpoint = torch.load(self._groundingdino_ckpt_path, map_location='cpu')
        log = model.load_state_dict(clean_state_dict(checkpoint['model']), strict=False)

        _ = model.eval()
        
        return model


    def _image_transform_grounding(self, init_image):
        transform = T.Compose([
            T.RandomResize([800], max_size=1333),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        image, _ = transform(init_image, None) # 3, h, w
        return init_image, image

    def _image_transform_grounding_for_vis(self, init_image):
        transform = T.Compose([
            T.RandomResize([800], max_size=1333),
        ])
        image, _ = transform(init_image, None) # 3, h, w
        return image


    def _predict(self, model_groundingdino, image_tensor, grounding_caption, box_threshold, text_threshold, device):
        return predict(model_groundingdino, image_tensor, grounding_caption, box_threshold, text_threshold, device)


    def _run_grounding(self, input_image, grounding_caption):
        init_image = input_image.convert("RGB")
        
        original_size = init_image.size

        _, image_tensor = self._image_transform_grounding(init_image)
        image_pil: Image = self._image_transform_grounding_for_vis(init_image)

        # run grounidng
        boxes, logits, phrases = predict(self._model_groundingdino, image_tensor, grounding_caption, self._box_threshold, self._text_threshold, device='cpu')

        # annotated_frame = annotate(image_source=np.asarray(image_pil), boxes=boxes, logits=logits, phrases=phrases)
        # image_with_box = Image.fromarray(cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB))

        boxes = boxes.detach().to('cpu').numpy()

        return boxes

    def _convert_xywh_to_xyxy(self, boxes, image_width, image_height, return_np=True):
        converted_holder = []
        for i in range(boxes.shape[0]):
            x = int(boxes[i][0] * image_width)
            y = int(boxes[i][1] * image_height)
            w = int(boxes[i][2] * image_width)
            h = int(boxes[i][3] * image_height)

            converted_holder.append([int(x - w / 2), int(y - h / 2), int(x + w / 2), int(y + h / 2)])
            

        if return_np:
            print("====================")
            return np.array(converted_holder)
        
        return converted_holder

    
    def show_mask(self, mask, ax, random_color=False):
        if random_color:
            color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
        else:
            color = np.array([30/255, 144/255, 255/255, 0.6])
        h, w = mask.shape[-2:]
        mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
        ax.imshow(mask_image)
    
    def show_points(self, coords, labels, ax, marker_size=375):
        pos_points = coords[labels==1]
        neg_points = coords[labels==0]
        ax.scatter(pos_points[:, 0], pos_points[:, 1], color='green', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)
        ax.scatter(neg_points[:, 0], neg_points[:, 1], color='red', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)   
        
    def show_box(self, box, ax):
        x0, y0 = box[0], box[1]
        w, h = box[2] - box[0], box[3] - box[1]
        ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', facecolor=(0,0,0,0), lw=2))  


    def _do_controlnet_process(self, 
                               input_image, 
                               input_mask,
                               prompt, 
                               a_prompt, 
                               n_prompt, 
                               num_samples, 
                               image_resolution, 
                               ddim_steps, 
                               guess_mode, 
                               strength, 
                               scale, 
                               seed, 
                               eta, 
                               mask_blur):
        with torch.no_grad():
            input_image = HWC3(input_image)
            

            img_raw = resize_image(input_image, image_resolution).astype(np.float32)
            H, W, C = img_raw.shape
            
            mask_pixel = cv2.resize(input_mask, (W, H), interpolation=cv2.INTER_LINEAR).astype(np.float32) / 255.0
            mask_pixel = cv2.GaussianBlur(mask_pixel, (0, 0), mask_blur)

            mask_latent = cv2.resize(mask_pixel, (W // 8, H // 8), interpolation=cv2.INTER_AREA)

            detected_map = img_raw.copy()
            detected_map[mask_pixel > 0.5] = - 255.0

            control = torch.from_numpy(detected_map.copy()).float().cuda() / 255.0
            control = torch.stack([control for _ in range(num_samples)], dim=0)
            control = einops.rearrange(control, 'b h w c -> b c h w').clone()

            mask = 1.0 - torch.from_numpy(mask_latent.copy()).float().cuda()
            mask = torch.stack([mask for _ in range(num_samples)], dim=0)
            mask = einops.rearrange(mask, 'b h w -> b 1 h w').clone()

            x0 = torch.from_numpy(img_raw.copy()).float().cuda() / 127.0 - 1.0
            x0 = torch.stack([x0 for _ in range(num_samples)], dim=0)
            x0 = einops.rearrange(x0, 'b h w c -> b c h w').clone()

            mask_pixel_batched = mask_pixel[None, :, :, None]
            img_pixel_batched = img_raw.copy()[None]

            if seed == -1:
                seed = random.randint(0, 65535)
            seed_everything(seed)

            if controlnet_config["save_memory"]:
                self._controlnet_model.low_vram_shift(is_diffusing=False)

            cond = {"c_concat": [control], "c_crossattn": [self._controlnet_model.get_learned_conditioning([prompt + ', ' + a_prompt] * num_samples)]}
            un_cond = {"c_concat": None if guess_mode else [control], "c_crossattn": [self._controlnet_model.get_learned_conditioning([n_prompt] * num_samples)]}
            shape = (4, H // 8, W // 8)

            if controlnet_config["save_memory"]:
                self._controlnet_model.low_vram_shift(is_diffusing=False)

            self._ddim_sampler.make_schedule(ddim_steps, ddim_eta=eta, verbose=True)
            x0 = self._controlnet_model.get_first_stage_encoding(self._controlnet_model.encode_first_stage(x0))

            if controlnet_config["save_memory"]:
                self._controlnet_model.low_vram_shift(is_diffusing=True)

            self._controlnet_model.control_scales = [strength * (0.825 ** float(12 - i)) for i in range(13)] if guess_mode else ([strength] * 13)
            # Magic number. IDK why. Perhaps because 0.825**12<0.01 but 0.826**12>0.01

            samples, intermediates = self._ddim_sampler.sample(ddim_steps, num_samples,
                                                        shape, cond, verbose=False, eta=eta,
                                                        unconditional_guidance_scale=scale,
                                                        unconditional_conditioning=un_cond, x0=x0, mask=mask)

            if controlnet_config["save_memory"]:
                self._controlnet_model.low_vram_shift(is_diffusing=False)

            x_samples = self._controlnet_model.decode_first_stage(samples)
            x_samples = (einops.rearrange(x_samples, 'b c h w -> b h w c') * 127.5 + 127.5).cpu().numpy().astype(np.float32)
            x_samples = x_samples * mask_pixel_batched + img_pixel_batched * (1.0 - mask_pixel_batched)

            results = [x_samples[i].clip(0, 255).astype(np.uint8) for i in range(num_samples)]
        return [detected_map.clip(0, 255).astype(np.uint8)] + results

    
    def replace(self, 
                input_image, 
                caption,
                a_prompt, 
                n_prompt, 
                num_samples, 
                image_resolution, 
                ddim_steps, 
                guess_mode, 
                strength, 
                scale, 
                seed, 
                eta, 
                mask_blur
                ):

        image_width, image_height = input_image.size

        xywh_boxes = self._run_grounding(input_image, caption)

        input_image = np.array(input_image)
        input_image = cv2.cvtColor(input_image, cv2.COLOR_BGR2RGB)


        xyxy_boxes = self._convert_xywh_to_xyxy(xywh_boxes, image_width, image_height, return_np=False)

        everything_results = self._fastsam_model(input_image, device=self.device, retina_masks=True, imgsz=1024, conf=0.4, iou=0.9,)
        prompt_process = FastSAMPrompt(input_image, everything_results, device=self.device)
        ann = prompt_process.box_prompt(bboxes=xyxy_boxes)

        for mask in ann:
            mask *= 255
            input_mask = mask.astype(np.uint8)

            result = self._do_controlnet_process(input_image, 
                                        input_mask, 
                                        caption, 
                                        a_prompt, 
                                        n_prompt, 
                                        num_samples, 
                                        image_resolution, 
                                        ddim_steps, 
                                        guess_mode, 
                                        strength, 
                                        scale, 
                                        seed, 
                                        eta, 
                                        mask_blur)

            return result




if __name__ == "__main__":
    from PIL import Image
    replacer = Replacer()
    # input_image = cv2.imread()
    input_image = Image.open("/home/evobits/vyrodrive/rzamarefat/Replacer/test_images/01.jpg")
    print("type(input_image)", type(input_image))

    grounding_caption = "replace the dog with a monkey"
    replacer.replace(input_image=input_image, grounding_caption=grounding_caption)
