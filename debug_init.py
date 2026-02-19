
import os
import torch
from TTS.tts.configs.xtts_config import XttsConfig
from TTS.tts.models.xtts import Xtts, XttsAudioConfig, XttsArgs
from TTS.config.shared_configs import BaseDatasetConfig

# Add safe globals
torch.serialization.add_safe_globals([XttsConfig, XttsAudioConfig, BaseDatasetConfig, XttsArgs])

# Path to the config file
checkpoint_dir = "/home/wiwi/.local/share/tts/tts_models--multilingual--multi-dataset--xtts_v2/"
config_path = os.path.join(checkpoint_dir, "config.json")

print(f"Loading config from: {config_path}")
config = XttsConfig()
config.load_json(config_path)

print("Initializing model from config...")
model = Xtts.init_from_config(config)

print(f"Result of Xtts.init_from_config: {model}")
