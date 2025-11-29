import torch
from torch import nn
from PIL import Image
import numpy as np
from torchvision import transforms
from diffusers import StableDiffusionPipeline, DDIMScheduler
from tqdm.auto import tqdm


class DDIMInversionEditor:
    def __init__(
        self,
        model_name="runwayml/stable-diffusion-v1-5",
        device=None,
        vae_scale=0.18215,
        dtype=None,
    ):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.vae_scale = vae_scale

        if dtype is None:
            self.dtype = torch.float16 if self.device == "cuda" else torch.float32
        else:
            self.dtype = dtype

        self.pipe = StableDiffusionPipeline.from_pretrained(
            model_name,
            torch_dtype=self.dtype,
            revision="fp16" if self.dtype == torch.float16 else None,
            safety_checker=None,
        )

        self.pipe.scheduler = DDIMScheduler.from_config(self.pipe.scheduler.config)

        self.pipe = self.pipe.to(self.device)
        if self.device == "cuda":
            try:
                self.pipe.enable_xformers_memory_efficient_attention()
            except Exception:
                pass

    def _get_text_embeddings(self, prompt, negative_prompt=""):
        tokenizer = self.pipe.tokenizer

        text_in = tokenizer(
            prompt,
            padding="max_length",
            max_length=tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        )
        with torch.no_grad():
            cond = self.pipe.text_encoder(text_in.input_ids.to(self.device))[0]

        uncond_in = tokenizer(
            negative_prompt or "",
            padding="max_length",
            max_length=tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        )
        with torch.no_grad():
            uncond = self.pipe.text_encoder(uncond_in.input_ids.to(self.device))[0]

        return torch.cat([uncond, cond])

    def encode_image(self, image: Image.Image):
        w, h = image.size
        w, h = w - w % 8, h - h % 8
        image = image.resize((w, h), Image.LANCZOS)

        transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5]),
            ]
        )

        img = transform(image).unsqueeze(0).to(self.device, dtype=self.pipe.vae.dtype)

        with torch.no_grad():
            enc = self.pipe.vae.encode(img)
            if hasattr(enc, "latent_dist"):
                lat = enc.latent_dist.sample()
            else:
                lat = enc.sample()
            lat = lat * self.vae_scale

        return lat  # [1,4,H/8,W/8]

    def invert(
        self,
        latent,
        prompt,
        num_steps=50,
        guidance_scale=7.5,
    ):
        scheduler = self.pipe.scheduler
        scheduler.set_timesteps(num_steps, device=self.device)
        timesteps = scheduler.timesteps

        text_emb = self._get_text_embeddings(prompt)

        z = latent.clone().to(self.device).to(self.pipe.unet.dtype)
        trajectory = []

        for t in tqdm(timesteps[::-1], desc="DDIM inversion"):
            t_tensor = torch.tensor([t], device=self.device, dtype=torch.long)

            z_in = torch.cat([z] * 2)
            with torch.no_grad():
                out = self.pipe.unet(
                    z_in, t_tensor, encoder_hidden_states=text_emb
                ).sample

            eps_uncond, eps_cond = out.chunk(2)
            eps = eps_uncond + guidance_scale * (eps_cond - eps_uncond)

            alpha_t = scheduler.alphas_cumprod[t]

            idx = (scheduler.timesteps == t).nonzero().item()
            if idx == 0:
                alpha_next = torch.tensor(0.0, device=self.device)
            else:
                t_next = scheduler.timesteps[idx - 1]
                alpha_next = scheduler.alphas_cumprod[t_next]

            pred_x0 = (z - torch.sqrt(1 - alpha_t) * eps) / torch.sqrt(alpha_t)

            z_next = torch.sqrt(alpha_next) * pred_x0 + torch.sqrt(1 - alpha_next) * eps

            trajectory.append(z.clone())
            z = z_next

        return z.detach(), trajectory

    def sample_from_zT(
        self,
        z_T,
        prompt,
        num_steps=50,
        guidance_scale=7.5,
    ):
        scheduler = self.pipe.scheduler
        scheduler.set_timesteps(num_steps, device=self.device)
        timesteps = scheduler.timesteps

        text_emb = self._get_text_embeddings(prompt)
        z = z_T.clone().to(self.device).to(self.pipe.unet.dtype)

        for t in tqdm(timesteps, desc="DDIM sampling"):
            t_tensor = torch.tensor([t], device=self.device, dtype=torch.long)

            z_in = torch.cat([z] * 2)
            with torch.no_grad():
                out = self.pipe.unet(
                    z_in, t_tensor, encoder_hidden_states=text_emb
                ).sample

            eps_uncond, eps_cond = out.chunk(2)
            eps = eps_uncond + guidance_scale * (eps_cond - eps_uncond)

            alpha_t = scheduler.alphas_cumprod[t]

            idx = (scheduler.timesteps == t).nonzero().item()
            if idx == len(scheduler.timesteps) - 1:
                alpha_next = torch.tensor(0.0, device=self.device)
            else:
                t_next = scheduler.timesteps[idx + 1]
                alpha_next = scheduler.alphas_cumprod[t_next]

            pred_x0 = (z - torch.sqrt(1 - alpha_t) * eps) / torch.sqrt(alpha_t)
            z_prev = torch.sqrt(alpha_next) * pred_x0 + torch.sqrt(1 - alpha_next) * eps

            z = z_prev

        # decode
        with torch.no_grad():
            x = self.pipe.vae.decode(z / self.vae_scale).sample

        x = (x / 2 + 0.5).clamp(0, 1)
        x = x.cpu().permute(0, 2, 3, 1).numpy()[0]
        img = Image.fromarray((x * 255).round().astype("uint8"))
        return img

    def edit(
        self,
        image: Image.Image,
        inversion_prompt: str,
        edit_prompt: str,
        steps=50,
        guidance=7.5,
        return_trajectory=False,
    ):
        latent = self.encode_image(image)
        z_T, traj = self.invert(latent, inversion_prompt, steps, guidance)
        edited_img = self.sample_from_zT(z_T, edit_prompt, steps, guidance)

        if return_trajectory:
            return edited_img, traj, z_T
        return edited_img
