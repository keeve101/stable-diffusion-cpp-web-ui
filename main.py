from stable_diffusion_cpp import run_sd
from config import (
    SD_EXECUTABLE_FILE_PATH,
    MODELS_FOLDER_PATH,
    IMAGES_FOLDER_PATH,
    TAESD_FILE_PATH,
)
from subprocess import Popen
from PIL import Image

import datetime
import asyncio
import streamlit as st

st.set_page_config(layout="wide")

IMAGES_FOLDER_PATH.mkdir(exist_ok=True)
TASKS = ["txt2img", "img2img"]


def run_stable_diffusion():
    """
    Run the Stable Diffusion model with the specified prompt.
    """
    prompt = st.session_state["prompt"]
    model = st.session_state["model"]
    task = st.session_state["task"]

    output_file_path = (
        IMAGES_FOLDER_PATH
        / f"{datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.png"
    )

    kwargs = {
        "executable_file_path": SD_EXECUTABLE_FILE_PATH,
        "mode": task,
        "prompt": prompt,
        "output": output_file_path,
        "model": st.session_state["model_paths"].get(model),
        "height": 512,
        "width": 512,
        "strength": 0.75,
        "cfg_scale": 7.0,
        "steps": 15,
        "seed": 4141843455,
        "batch_count": 1,
        "sampling_method": "euler",
    }

    if task == "img2img":
        image = st.session_state["image"]
        image_path = st.session_state["image_paths"].get(image)
        kwargs["init_img"] = image_path

    process = run_sd(**kwargs)
    st.session_state["process"] = process
    st.session_state["output_file_path"] = output_file_path.name


async def get_process_output(process: Popen, logs_container: st.empty):
    async def lines(process: Popen):
        for line in process.stdout:
            yield line

    async for line in lines(process):
        with logs_container:
            st.markdown(f"*{line.strip()}*")

    st.session_state["reload_image"] = str(st.session_state["output_file_path"])
    del st.session_state["output_file_path"]
    del st.session_state["process"]
    st.rerun()


def stop_process():
    """
    Stop the Stable Diffusion process.
    """
    if "process" in st.session_state:
        st.session_state["process"].terminate()
        del st.session_state["process"]


def get_model_paths():
    paths = MODELS_FOLDER_PATH.glob("**/*")
    options = [file for file in paths if file.is_file()]
    st.session_state["model_paths"] = {file.name: file.as_posix() for file in options}


def get_image_paths():
    paths = IMAGES_FOLDER_PATH.glob("**/*")
    options = [file for file in paths if file.suffix in [".png", ".jpg", ".jpeg"]]
    st.session_state["image_paths"] = {file.name: file.as_posix() for file in options}


if "reload_image" in st.session_state:
    st.session_state["image"] = str(st.session_state["reload_image"])
    del st.session_state["reload_image"]


with st.sidebar:
    st.title("Stable Diffusion Models")
    model_paths = get_model_paths()
    image_paths = get_image_paths()

    st.selectbox("*Model Select*", st.session_state["model_paths"], key="model")
    st.selectbox("*Images*", st.session_state["image_paths"], key="image")
    st.selectbox("*Tasks*", TASKS, key="task")

    if "process" in st.session_state:
        st.button("Stop", on_click=stop_process)

st.chat_input(
    placeholder="Enter your prompt here...",
    key="prompt",
    on_submit=run_stable_diffusion,
)

image_display, logs = st.columns(2)
with image_display:
    st.markdown(
        """
        ## Image Display
        ---
        """
    )
    if st.session_state.get("image"):
        image_path = st.session_state["image_paths"].get(st.session_state["image"])
        st.image(Image.open(image_path), output_format="PNG", width=512)

with logs:
    st.markdown("## Logs")
    with st.container(border=True):
        if "process" in st.session_state:
            logs_container = st.empty()
            asyncio.run(get_process_output(st.session_state["process"], logs_container))
        else:
            st.markdown(
                """
                *Logs will be displayed here when the Stable Diffusion process is running.*
                """
            )
