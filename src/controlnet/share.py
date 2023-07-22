from ..config import controlnet_config
from src.controlnet.cldm.hack import disable_verbosity, enable_sliced_attention


disable_verbosity()

if controlnet_config["save_memory"]:
    enable_sliced_attention()
