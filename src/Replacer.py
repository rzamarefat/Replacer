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
import torch

from .config import groundingdino_config

class Replacer:
    def __init__(self):
        self._config_file_groundingdino = groundingdino_config["config_file"]
        self._ckpt_repo_id_groundingdino = groundingdino_config["ckpt_repo_id"]
        self._ckpt_filename_groundingdino = groundingdino_config["ckpt_filename"]
        self._groundingdino_ckpt_path = groundingdino_config["groundingdino_ckpt_path"]


        self._box_threshold = 0.85 
        self._text_threshold = 0.85


        self._model_groundingdino = self._load_model_groundingdino(self._config_file_groundingdino, self._ckpt_repo_id_groundingdino, self._ckpt_filename_groundingdino)

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

        annotated_frame = annotate(image_source=np.asarray(image_pil), boxes=boxes, logits=logits, phrases=phrases)
        image_with_box = Image.fromarray(cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB))

        return image_with_box

    
    def replace(self, input_image, grounding_caption):
        image_with_box = self._run_grounding(input_image, grounding_caption)


if __name__ == "__main__":
    from PIL import Image
    replacer = Replacer()
    # input_image = cv2.imread()
    input_image = Image.open("/home/evobits/vyrodrive/rzamarefat/Replacer/test_images/01.jpg")
    print("type(input_image)", type(input_image))

    grounding_caption = "replace the dog with a monkey"
    replacer.replace(input_image=input_image, grounding_caption=grounding_caption)
