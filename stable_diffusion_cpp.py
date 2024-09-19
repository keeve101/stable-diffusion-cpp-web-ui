import subprocess


def run_sd(
    executable_file_path=None,
    mode="txt2img",
    threads=-1,
    model=None,
    vae=None,
    taesd=None,
    control_net=None,
    embd_dir=None,
    stacked_id_embd_dir=None,
    input_id_images_dir=None,
    normalize_input=False,
    upscale_model=None,
    upscale_repeats=1,
    weight_type=None,
    lora_model_dir=None,
    init_img=None,
    control_image=None,
    output="./output.png",
    prompt=None,
    negative_prompt="",
    cfg_scale=7.0,
    strength=0.75,
    style_ratio=20.0,
    control_strength=0.9,
    height=512,
    width=512,
    sampling_method="euler_a",
    steps=20,
    rng="cuda",
    seed=42,
    batch_count=1,
    schedule="discrete",
    clip_skip=-1,
    vae_tiling=False,
    control_net_cpu=False,
    canny=False,
    color=False,
    verbose=False,
) -> subprocess.Popen:
    """
    Run sd.exe from stable-diffusion.cpp with the specified arguments.

    Example usage:
    run_sd(prompt="A beautiful landscape", output="landscape.png")
    """
    cmd = [executable_file_path]

    # Add the arguments
    cmd += ["-M", mode]
    cmd += ["-t", str(threads)]
    if model:
        cmd += ["-m", model]
    if vae:
        cmd += ["--vae", vae]
    if taesd:
        cmd += ["--taesd", taesd]
    if control_net:
        cmd += ["--control-net", control_net]
    if embd_dir:
        cmd += ["--embd-dir", embd_dir]
    if stacked_id_embd_dir:
        cmd += ["--stacked-id-embd-dir", stacked_id_embd_dir]
    if input_id_images_dir:
        cmd += ["--input-id-images-dir", input_id_images_dir]
    if normalize_input:
        cmd += ["--normalize-input"]
    if upscale_model:
        cmd += ["--upscale-model", upscale_model]
    if upscale_repeats:
        cmd += ["--upscale-repeats", str(upscale_repeats)]
    if weight_type:
        cmd += ["--type", weight_type]
    if lora_model_dir:
        cmd += ["--lora-model-dir", lora_model_dir]
    if init_img:
        cmd += ["-i", init_img]
    if control_image:
        cmd += ["--control-image", control_image]
        cmd += ["--control-strength", str(control_strength)]
    cmd += ["-o", output]
    if prompt:
        cmd += ["-p", prompt]
    if negative_prompt:
        cmd += ["-n", negative_prompt]
    cmd += ["--cfg-scale", str(cfg_scale)]
    cmd += ["--strength", str(strength)]
    cmd += ["--style-ratio", f"{str(style_ratio)}%"]
    cmd += ["-H", str(height)]
    cmd += ["-W", str(width)]
    cmd += ["--sampling-method", sampling_method]
    cmd += ["--steps", str(steps)]
    cmd += ["--rng", rng]
    cmd += ["-s", str(seed)]
    cmd += ["-b", str(batch_count)]
    cmd += ["--schedule", schedule]
    cmd += ["--clip-skip", str(clip_skip)]
    if vae_tiling:
        cmd += ["--vae-tiling"]
    if control_net_cpu:
        cmd += ["--control-net-cpu"]
    if canny:
        cmd += ["--canny"]
    if color:
        cmd += ["--color"]
    if verbose:
        cmd += ["--verbose"]

    print(r" ".join([str(t) for t in cmd]))

    # Execute the command, return the process
    process = subprocess.Popen(
        cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
    )

    return process
