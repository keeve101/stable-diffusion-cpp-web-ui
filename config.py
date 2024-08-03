from pathlib import Path

"""
Configuration file, containing all relevant paths.
"""

SD_EXECUTABLE_FILE_PATH = Path(
    "D:\\Repositories\\stable-diffusion.cpp\\build\\bin\\Release\\sd.exe"
)

MODELS_FOLDER_PATH = Path("D:\\Repositories\\stable-diffusion\\models")

CHECKPOINTS_FOLDER_PATH = MODELS_FOLDER_PATH / "checkpoints"
LORAS_FOLDER_PATH = MODELS_FOLDER_PATH / "loras"
TAESD_FOLDER_PATH = MODELS_FOLDER_PATH / "taesd"
VAE_FOLDER_PATH = MODELS_FOLDER_PATH / "vae"
CONTROLNET_FOLDER_PATH = MODELS_FOLDER_PATH / "controlnet"
UPSCALE_MODELS_FOLDER_PATH = MODELS_FOLDER_PATH / "upscale_models"

IMAGES_FOLDER_PATH = Path.cwd() / "images"
