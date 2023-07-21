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



