import os

root_path = os.getcwd()

groundingdino_config = {
    "config_file": os.path.join(root_path, "src", "groundingdino","config", "GroundingDINO_SwinT_OGC.py"),
    "ckpt_repo_id": "ShilongLiu/GroundingDINO",
    "ckpt_filename": os.path.join(root_path, "groundingdino_swint_ogc.pth"),
    "groundingdino_ckpt_path": os.path.join(root_path, "pretrained_weights", "groundingdino_swint_ogc.pth"),
    "box_threshold": 0.85,
    "text_threshold": 0.85,

}

fastsam_config = {
    "fastsam_checkpoint": os.path.join(root_path, "pretrained_weights", "FastSAM-x.pt"),
}


controlnet_config = {
    "save_memory": False,
    "model_name": "control_v11p_sd15_inpaint",
    "path_to_config_yaml_file": "/home/evobits/vyrodrive/rzamarefat/Replacer/src/controlnet/models",
    "path_to_sd_model":"/home/evobits/vyrodrive/rzamarefat/Replacer/pretrained_weights/v1-5-pruned.ckpt",
    "path_to_cldm": "/home/evobits/vyrodrive/rzamarefat/Replacer/pretrained_weights/control_v11p_sd15_inpaint.pth",
}