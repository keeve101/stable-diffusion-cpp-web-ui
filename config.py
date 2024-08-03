from pathlib import Path

"""
Configuration file, containing all relevant paths.
"""

SD_EXECUTABLE_FILE_PATH = Path(
    "D:\\Repositories\\stable-diffusion.cpp\\build\\bin\\Release\\sd.exe"
)
TAESD_FILE_PATH = Path(
    "D:\\Repositories\\stable-diffusion.cpp\\models\\taesd\\diffusion_pytorch_model.safetensors"
)
MODELS_FOLDER_PATH = Path("D:\\Repositories\\stable-diffusion.cpp\\models")
IMAGES_FOLDER_PATH = Path.cwd() / "images"
