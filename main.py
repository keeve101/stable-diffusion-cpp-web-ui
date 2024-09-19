from stable_diffusion_cpp import run_sd
from pathlib import Path
from config import (
    SD_EXECUTABLE_FILE_PATH,
    CHECKPOINTS_FOLDER_PATH,
    LORAS_FOLDER_PATH,
    TAESD_FOLDER_PATH,
    VAE_FOLDER_PATH,
    UPSCALE_MODELS_FOLDER_PATH,
    IMAGES_FOLDER_PATH,
)
from subprocess import Popen

import base64
import datetime
import asyncio
import streamlit as st

st.set_page_config(layout="wide")

IMAGES_FOLDER_PATH.mkdir(exist_ok=True)
TASKS = ["txt2img", "img2img"]
SAMPLING_METHODS = [
    "euler",
    "euler_a",
    "dpm2",
    "dpm++2s_a",
    "dpm++2m",
    "dpm++2mv2",
    "lcm",
]


def run_stable_diffusion():
    """
    Run the Stable Diffusion model with the specified prompt.
    """
    prompt = st.session_state["prompt"]
    negative_prompt = st.session_state["negative_prompt"]
    sampling_method = st.session_state["sampling_method"]
    seed = st.session_state["seed"]
    steps = st.session_state["steps"]
    height = st.session_state["height"]
    width = st.session_state["width"]
    cfg_scale = st.session_state["cfg_scale"]
    strength = st.session_state["strength"]
    style_ratio = st.session_state["style_ratio"]
    batch_count = st.session_state["batch_count"]
    model = st.session_state["model"]
    task = st.session_state["task"]
    lora = st.session_state[LORAS_FOLDER_PATH.name].get(st.session_state["lora_select"])
    taesd = st.session_state[TAESD_FOLDER_PATH.name].get(
        st.session_state["taesd_select"]
    )
    vae = st.session_state[VAE_FOLDER_PATH.name].get(st.session_state["vae_select"])
    upscale_model = st.session_state[UPSCALE_MODELS_FOLDER_PATH.name].get(
        st.session_state["upscale_model_select"]
    )

    output_file_path = (
        IMAGES_FOLDER_PATH
        / f"{datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.png"
    )

    kwargs = {
        "executable_file_path": SD_EXECUTABLE_FILE_PATH,
        "mode": task,
        "prompt": prompt,
        "negative_prompt": negative_prompt,
        "output": output_file_path,
        "model": st.session_state[CHECKPOINTS_FOLDER_PATH.name].get(model),
        "height": height,
        "width": width,
        "cfg_scale": cfg_scale,
        "strength": strength,
        "style_ratio": style_ratio,
        "steps": steps,
        "seed": seed,
        "batch_count": batch_count,
        "sampling_method": sampling_method,
    }

    if task == "img2img":
        image = st.session_state["image"]
        image_path = st.session_state["image_paths"].get(image)
        kwargs["init_img"] = image_path
    if lora is not None:
        kwargs["lora_model_dir"] = lora
    if taesd is not None:
        kwargs["taesd"] = taesd
    elif vae is not None:
        kwargs["vae"] = vae
    if upscale_model is not None:
        kwargs["upscale_model"] = upscale_model
        kwargs["upscale_repeats"] = st.session_state.get("upscale_repeats")

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


def get_model_paths(folder_path: Path):
    paths = folder_path.glob("**/*")
    options = [file for file in paths if file.is_file()]
    st.session_state[folder_path.name] = {
        file.name: file.as_posix() for file in options
    }


def get_image_paths():
    paths = IMAGES_FOLDER_PATH.glob("**/*")
    options = [file for file in paths if file.suffix in [".png", ".jpg", ".jpeg"]]
    st.session_state["image_paths"] = {file.name: file.as_posix() for file in options}

    if (
        "image" in st.session_state
        and st.session_state["image"] not in st.session_state["image_paths"]
    ):
        del st.session_state["image"]


if "reload_image" in st.session_state:
    st.session_state["image"] = str(st.session_state["reload_image"])
    del st.session_state["reload_image"]


with st.sidebar:
    st.title("Stable Diffusion Models")
    get_model_paths(CHECKPOINTS_FOLDER_PATH)
    get_model_paths(LORAS_FOLDER_PATH)
    get_model_paths(TAESD_FOLDER_PATH)
    get_model_paths(VAE_FOLDER_PATH)
    get_model_paths(UPSCALE_MODELS_FOLDER_PATH)
    get_image_paths()

    st.selectbox(
        "*Model Select*", st.session_state[CHECKPOINTS_FOLDER_PATH.name], key="model"
    )
    st.selectbox(
        "*Loras*",
        st.session_state[LORAS_FOLDER_PATH.name],
        key="lora_select",
        index=None,
    )
    st.selectbox(
        "*TAESD*",
        st.session_state[TAESD_FOLDER_PATH.name],
        key="taesd_select",
        index=None,
    )
    st.selectbox(
        "*VAE*", st.session_state[VAE_FOLDER_PATH.name], key="vae_select", index=None
    )
    st.selectbox(
        "*Upscale Model*",
        st.session_state[UPSCALE_MODELS_FOLDER_PATH.name],
        key="upscale_model_select",
        index=None,
    )
    if st.session_state.get("upscale_model_select") is not None:
        st.number_input("*Upscale Repeats*", key="upscale_repeats", value=1)
    st.selectbox("*Images*", st.session_state["image_paths"], key="image")
    st.selectbox("*Tasks*", TASKS, key="task")
    st.button("Generate", on_click=run_stable_diffusion, use_container_width=True)

    if "process" in st.session_state:
        st.button("Stop", on_click=stop_process, use_container_width=True)


image_display, parameters = st.columns(2)
with image_display:
    if st.session_state.get("image"):
        image_path = st.session_state["image_paths"].get(st.session_state["image"])

        with open(image_path, "rb") as image_file:
            image_bytes = base64.b64encode(image_file.read()).decode("utf-8")

        img_html = f'<img src="data:image/{image_path};base64,{image_bytes}" width="512" height="512" />'

        st.markdown(img_html, unsafe_allow_html=True)

with parameters:
    column1, column2 = st.columns(2)
    with column1:
        st.selectbox("*Sampling Method*", SAMPLING_METHODS, key="sampling_method")
        st.number_input("*Seed*", key="seed", value=42)
        st.number_input("*Height*", key="height", value=512, min_value=64)
        st.number_input("*Width*", key="width", value=512, min_value=64)
        st.number_input("*Batch Count*", key="batch_count", value=1)
    with column2:
        st.number_input("*Steps*", key="steps", value=15, min_value=1)
        st.number_input(
            "*CFG Scale*", key="cfg_scale", value=7.0, max_value=30.0, min_value=0.0
        )
        st.number_input(
            "*Strength*", key="strength", value=0.75, max_value=1.0, min_value=0.0
        )
        st.number_input(
            "*Style Ratio*",
            key="style_ratio",
            value=20,
            max_value=100,
            min_value=0,
        )

st.text_area(
    label="Prompt",
    placeholder="Enter your prompt here...",
    key="prompt",
    label_visibility="hidden",
)
st.text_area(
    label="Negative Prompt",
    placeholder="Enter your negative prompt here...",
    key="negative_prompt",
    label_visibility="hidden",
)

with st.sidebar:
    with st.container(border=True):
        st.markdown(
            """
            *Logs will be displayed here when the Stable Diffusion process is running.*
            """
        )
        if "process" in st.session_state:
            logs_container = st.empty()
            asyncio.run(get_process_output(st.session_state["process"], logs_container))
