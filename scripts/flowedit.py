"""
FlowEdit latent editing module for Instruct-GS2GS
Equivalent in structure to instructpix2pix.py
"""

import torch
from torch import nn
from dataclasses import dataclass
from typing import Union
from rich.console import Console
from jaxtyping import Float
import numpy as np
import random

from diffusers import StableDiffusion3Pipeline

CONSOLE = Console(width=120)


@dataclass
class FlowEditOutput:
    sample: torch.FloatTensor   


class FlowEditSD3(nn.Module):
    """
    FlowEdit implementation for Stable Diffusion 3.
    This mirrors InstructPix2Pix structure so IG2GS can directly plug it in.

    Args:
        device: torch.device
        model_id: HF ID for SD3 (default: stabilityai/stable-diffusion-3-medium-diffusers)
        use_fp16: use half precision for memory efficiency
    """

    def __init__(
        self,
        device: Union[torch.device, str],
        model_id: str = "stabilityai/stable-diffusion-3-medium-diffusers",
        use_fp16: bool = True,
    ) -> None:
        super().__init__()

        self.device = device

        dtype = torch.float16 if use_fp16 else torch.float32

        CONSOLE.print(f"[green]Loading SD3 for FlowEdit from {model_id}…[/green]")

        pipe = StableDiffusion3Pipeline.from_pretrained(
            model_id,
            torch_dtype=dtype,
        ).to(device)

        pipe.vae.eval()
        pipe.unet.eval()

        self.pipe = pipe
        self.scheduler = pipe.scheduler

        CONSOLE.print("[green]FlowEdit SD3 loaded successfully![/green]")

    def edit_image(
        self,
        latent_src: Float[torch.Tensor, "1 4 H W"],
        src_prompt: str,
        tar_prompt: str,
        T_steps: int,
        n_avg: int,
        src_guidance_scale: float,
        tar_guidance_scale: float,
        n_min: int,
        n_max: int,
        negative_prompt: str = "",
        seed: int = 0,
    ) -> torch.Tensor:
        """
        Perform FlowEdit latent editing on the GS-rendered latent.

        Matches the API of InstructPix2Pix.edit_image() so IG2GS code
        can call it directly.
        """

        pipe = self.pipe
        unet = pipe.unet
        scheduler = self.scheduler

        generator = torch.Generator(device=self.device).manual_seed(seed)
        x0_src = latent_src.clone().to(self.device)

        prompt_embeds = pipe._encode_prompt(
            tar_prompt,
            device=self.device,
            num_images_per_prompt=1,
            do_classifier_free_guidance=True,
        )
        negative_embeds = pipe._encode_prompt(
            negative_prompt,
            device=self.device,
            num_images_per_prompt=1,
            do_classifier_free_guidance=True,
        )

        x_tar_accum = torch.zeros_like(x0_src)

        for _ in range(n_avg):
            T = random.randint(n_min, n_max)

            x = x0_src.clone()

            for _ in range(T_steps):

                noise = torch.randn_like(x, generator=generator)
                t = scheduler.timesteps[_] if _ < len(scheduler.timesteps) else scheduler.timesteps[-1]

                latent_model_input = x

                noise_pred = unet(
                    latent_model_input,
                    t,
                    encoder_hidden_states=prompt_embeds,
                ).sample

                noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                noise_pred = noise_pred_uncond + tar_guidance_scale * ( noise_pred_text - noise_pred_uncond )
                x = scheduler.step(noise_pred, t, x).prev_sample

            x_tar_accum += x

        x0_tar = x_tar_accum / n_avg

        return x0_tar
        
    def edit(self, image, prompt):
        """
        Convenience wrapper to mimic InstructPix2Pix API.
        image: 1×3×H×W float tensor in [0,1]
        """
        device = self.device
        pipe = self.pipe

        # → Convert RGB to latent
        with torch.no_grad(), torch.autocast("cuda"):
            latents = pipe.vae.encode(image * 2 - 1).latent_dist.sample()
            latents = latents * pipe.vae.config.scaling_factor

        # → Run FlowEdit latent editing
        edited_latent = self.edit_image(
            latent_src=latents,
            src_prompt="",           # FlowEdit usually uses neutral prompt
            tar_prompt=prompt,       # user editing prompt
            T_steps=50,
            n_avg=4,
            src_guidance_scale=1.0,
            tar_guidance_scale=7.5,
            n_min=5,
            n_max=25,
            negative_prompt="",
            seed=0,
        )

        # → Decode latent to image
        with torch.no_grad():
            image_out = self.latent_to_image(edited_latent)

        # Convert PIL → torch
        image_out = torch.from_numpy(np.array(image_out)).float() / 255.0
        image_out = image_out.permute(2,0,1)  # C,H,W

        return image_out


    def latent_to_image(self, latent):
        """
        Decode a latent using SD3 VAE into a PIL image.
        """
        pipe = self.pipe

        x0_denorm = (latent / pipe.vae.config.scaling_factor) + pipe.vae.config.shift_factor

        with torch.no_grad(), torch.autocast("cuda"):
            image = pipe.vae.decode(x0_denorm, return_dict=False)[0]

        image = pipe.image_processor.postprocess(image)
        return image[0]
