import torch
import argparse
import json
import base64
import numpy as np
from io import BytesIO
from PIL import Image

import sys, os
import io
import contextlib

from diffusers import StableDiffusion3Pipeline
from FlowEdit_utils import FlowEditSD3   

os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
os.environ["TRANSFORMERS_NO_ADVISORY_WARNINGS"] = "1"
os.environ["DIFFUSERS_VERBOSITY"] = "error"
os.environ["TRANSFORMERS_VERBOSITY"] = "error"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

ERROR_LOG = "flowedit_error.log"

@contextlib.contextmanager
def suppress_stdout():
    saved_stdout = sys.stdout
    saved_stderr = sys.stderr

    temp_out = io.StringIO()
    temp_err = io.StringIO()
    sys.stdout = temp_out
    sys.stderr = temp_err
    try:
        yield
    finally:
        sys.stdout = saved_stdout
        sys.stderr = saved_stderr

        with open(ERROR_LOG, "a") as f:
            out = temp_out.getvalue().strip()
            err = temp_err.getvalue().strip()
            if out:
                f.write("\n--- STDOUT ---\n" + out + "\n")
            if err:
                f.write("\n--- STDERR ---\n" + err + "\n")

@contextlib.contextmanager
def suppress_output():
    saved_stdout = sys.stdout
    saved_stderr = sys.stderr
    sys.stdout = io.StringIO()
    sys.stderr = io.StringIO()
    try:
        yield
    finally:
        sys.stdout = saved_stdout
        sys.stderr = saved_stderr


def pil_to_b64(img):
    buf = BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def b64_to_pil(b64_string):
    img_bytes = base64.b64decode(b64_string)
    return Image.open(BytesIO(img_bytes)).convert("RGB")


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("--request", type=str, help="JSON request as string")
    parser.add_argument("--request_file", type=str, help="Path to JSON request file")
    args = parser.parse_args()

    if args.request_file:
        with open(args.request_file) as f:
            req = json.load(f)
    else:
        req = json.loads(args.request)
    try: 

        img_b64 = req["image"]
        src_prompt = req.get("src_prompt", "")
        tar_prompt = req.get("tar_prompt", "")
        negative_prompt = ""
        
        with suppress_output():
            pipe = StableDiffusion3Pipeline.from_pretrained(
                "stabilityai/stable-diffusion-3-medium-diffusers",
                torch_dtype=torch.float16
            ).to("cuda")
        
        scheduler = pipe.scheduler

        pil_img = b64_to_pil(img_b64)
        W, H = pil_img.width, pil_img.height
        pil_img = pil_img.crop((0, 0, W - (W % 16), H - (H % 16)))

        with suppress_output():
            image_tensor = pipe.image_processor.preprocess(pil_img)
            image_tensor = image_tensor.half().cuda()

        with torch.autocast("cuda"), torch.inference_mode():
            x0_src_denorm = pipe.vae.encode(image_tensor).latent_dist.mode()

        with suppress_output():
            x0_src = (x0_src_denorm - pipe.vae.config.shift_factor) * pipe.vae.config.scaling_factor
            x0_src = x0_src.to("cuda")

        with suppress_output():
            x0_tar = FlowEditSD3(
                pipe=pipe,
                scheduler=scheduler,
                x_src=x0_src,
                src_prompt=src_prompt,
                tar_prompt=tar_prompt,
                negative_prompt=negative_prompt,
                T_steps=50,
                n_avg=1,
                src_guidance_scale=3.5,
                tar_guidance_scale=14.5,
                n_min=0,
                n_max=28
            )

        x0_tar_denorm = (x0_tar / pipe.vae.config.scaling_factor) + pipe.vae.config.shift_factor

        with suppress_output():
            with torch.autocast("cuda"), torch.inference_mode():
                image_tar = pipe.vae.decode(x0_tar_denorm, return_dict=False)[0]

        pil_out = pipe.image_processor.postprocess(image_tar)[0]
        json_payload = json.dumps({"edited": pil_to_b64(pil_out)})

    except Exception as e:
        with open(ERROR_LOG, "a") as f:
            f.write("\n===== FLOWEDIT EXCEPTION ======\n")
            f.write(traceback.format_exc())
            f.write("\n===============================\n")

    sys.stdout.write(json_payload + "\n")
    sys.stdout.flush()


if __name__ == "__main__":
    main()
