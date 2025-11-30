from diffusers import StableDiffusionPipeline, DDIMScheduler
from diffusers import (
    StableDiffusionControlNetPipeline,
    ControlNetModel,
    UNet2DConditionModel,
)
from diffusers.schedulers import DDIMScheduler, DDIMInverseScheduler
import torchvision
import tqdm
import cv2
import torch


class DDIMInversionEditor:
    def __init__(self, device="cuda"):
        model_id = "runwayml/stable-diffusion-v1-5"
        self.pipe = StableDiffusionPipeline.from_pretrained(model_id).to(
            torch.device(device)
        )
        self.pipe.scheduler = DDIMScheduler.from_config(self.pipe.scheduler.config)
        self.device = torch.device(device)
        self.guidance_scale = 3.5

    @torch.no_grad()
    def sample(
        self,
        prompt,
        start_step=0,
        start_latents=None,
        guidance_scale=3.5,
        num_inference_steps=30,
        num_images_per_prompt=1,
        do_classifier_free_guidance=True,
        negative_prompt="",
    ):
        # Encode prompt
        text_embeddings = self.pipe._encode_prompt(
            prompt,
            self.device,
            num_images_per_prompt,
            do_classifier_free_guidance,
            negative_prompt,
        )

        # Set num inference steps
        self.pipe.scheduler.set_timesteps(num_inference_steps, device=self.device)

        # Create a random starting point if we don't have one already
        if start_latents is None:
            start_latents = torch.randn(1, 4, 64, 64, device=self.device)
            start_latents *= self.pipe.scheduler.init_noise_sigma

        latents = start_latents.clone()

        for i in tqdm.tqdm(range(start_step, num_inference_steps)):
            t = self.pipe.scheduler.timesteps[i]

            # Expand the latents if we are doing classifier free guidance
            latent_model_input = (
                torch.cat([latents] * 2) if do_classifier_free_guidance else latents
            )
            latent_model_input = self.pipe.scheduler.scale_model_input(
                latent_model_input, t
            )

            # Predict the noise residual
            noise_pred = self.pipe.unet(
                latent_model_input, t, encoder_hidden_states=text_embeddings
            ).sample

            # Perform guidance
            if do_classifier_free_guidance:
                noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                noise_pred = noise_pred_uncond + guidance_scale * (
                    noise_pred_text - noise_pred_uncond
                )

            # Instead, let's do it ourselves:
            prev_t = max(1, t.item() - (1000 // num_inference_steps))  # t-1
            alpha_t = self.pipe.scheduler.alphas_cumprod[t.item()]
            alpha_t_prev = self.pipe.scheduler.alphas_cumprod[prev_t]
            predicted_x0 = (
                latents - (1 - alpha_t).sqrt() * noise_pred
            ) / alpha_t.sqrt()
            direction_pointing_to_xt = (1 - alpha_t_prev).sqrt() * noise_pred
            latents = alpha_t_prev.sqrt() * predicted_x0 + direction_pointing_to_xt

        # Post-processing
        images = self.pipe.decode_latents(latents)

        return images

    @torch.no_grad()
    def invert(
        self,
        start_latents,
        prompt,
        guidance_scale=3.5,
        num_inference_steps=80,
        num_images_per_prompt=1,
        do_classifier_free_guidance=True,
        negative_prompt="",
    ):
        # Encode prompt
        text_embeddings = self.pipe._encode_prompt(
            prompt,
            self.device,
            num_images_per_prompt,
            do_classifier_free_guidance,
            negative_prompt,
        )

        # Latents are now the specified start latents
        latents = start_latents.clone()

        # We'll keep a list of the inverted latents as the process goes on
        intermediate_latents = []

        # Set num inference steps
        self.pipe.scheduler.set_timesteps(num_inference_steps, device=self.device)

        # Reversed timesteps <<<<<<<<<<<<<<<<<<<<
        timesteps = reversed(self.pipe.scheduler.timesteps)

        for i in tqdm.tqdm(
            range(1, num_inference_steps), total=num_inference_steps - 1
        ):
            # We'll skip the final iteration
            if i >= num_inference_steps - 1:
                continue

            t = timesteps[i]

            # Expand the latents if we are doing classifier free guidance
            latent_model_input = (
                torch.cat([latents] * 2) if do_classifier_free_guidance else latents
            )
            latent_model_input = self.pipe.scheduler.scale_model_input(
                latent_model_input, t
            )

            # Predict the noise residual
            noise_pred = self.pipe.unet(
                latent_model_input, t, encoder_hidden_states=text_embeddings
            ).sample

            # Perform guidance
            if do_classifier_free_guidance:
                noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                noise_pred = noise_pred_uncond + guidance_scale * (
                    noise_pred_text - noise_pred_uncond
                )

            current_t = max(0, t.item() - (1000 // num_inference_steps))  # t
            next_t = t  # min(999, t.item() + (1000//num_inference_steps)) # t+1
            alpha_t = self.pipe.scheduler.alphas_cumprod[current_t]
            alpha_t_next = self.pipe.scheduler.alphas_cumprod[next_t]

            # Inverted update step (re-arranging the update step to get x(t) (new latents) as a function of x(t-1) (current latents)
            latents = (latents - (1 - alpha_t).sqrt() * noise_pred) * (
                alpha_t_next.sqrt() / alpha_t.sqrt()
            ) + (1 - alpha_t_next).sqrt() * noise_pred

            # Store
            intermediate_latents.append(latents)

        return torch.cat(intermediate_latents)

    def edit(
        self,
        input_image,
        input_image_prompt,
        edit_prompt,
        num_steps=100,
        start_step=30,
        guidance_scale=3.5,
    ):
        # Resize image to 512x512 for SD v1.5
        original_size = (input_image.shape[1], input_image.shape[2])
        input_image = torchvision.transforms.Resize((512, 512))(input_image)
        input_image = input_image.unsqueeze(0).to(self.device)

        with torch.no_grad():
            latent = self.pipe.vae.encode((input_image * 2.0) - 1.0)

        l = 0.18215 * latent.latent_dist.sample()

        inverted_latents = self.invert(
            l, input_image_prompt, num_inference_steps=num_steps
        )

        final_im = self.sample(
            edit_prompt,
            start_latents=inverted_latents[-(start_step + 1)][None],
            start_step=start_step,
            num_inference_steps=num_steps,
            guidance_scale=guidance_scale,
        )
        final_im = torch.from_numpy(final_im).permute(0, 3, 1, 2)
        return torchvision.transforms.Resize(original_size)(final_im)[0]


class DDIMInversionEditorDepth:
    def __init__(self, device="cuda"):
        diffusion_ckpt = "runwayml/stable-diffusion-v1-5"
        self.device = device

        self.ddim_scheduler = DDIMScheduler.from_pretrained(
            diffusion_ckpt, subfolder="scheduler"
        )
        self.ddim_inverser = DDIMInverseScheduler.from_pretrained(
            diffusion_ckpt, subfolder="scheduler"
        )

        controlnet = ControlNetModel.from_pretrained("lllyasviel/sd-controlnet-depth")

        self.pipe = StableDiffusionControlNetPipeline.from_pretrained(
            diffusion_ckpt, controlnet=controlnet, scheduler=self.ddim_scheduler
        ).to(self.device)

    @torch.no_grad()
    def image2latent(self, image):
        image = image * 2 - 1
        latents = self.pipe.vae.encode(image)["latent_dist"].mean
        latents = latents * 0.18215
        return latents

    def depth2disparity(self, depth):
        disparity = 1 / (depth + 1e-5)
        disparity_map = disparity / torch.max(disparity)
        disparity_map = torch.concatenate(
            [disparity_map, disparity_map, disparity_map], dim=0
        )
        return disparity_map[None]

    def edit(
        self,
        input_image,
        depth,
        edit_prompt,
        source_prompt="",
        negative_prompt="",
        controlnet_conditioning_scale=1.0,
        guidance_scale=3.5,
        num_inference_steps=100,
    ):
        # print(input_image.shape, depth.shape)
        # exit()
        original_size = (input_image.shape[1], input_image.shape[2])

        input_image = torchvision.transforms.Resize((512, 512))(input_image)
        depth = torchvision.transforms.Resize((512, 512))(depth)

        input_image = input_image.unsqueeze(0).to(self.device)

        # Encode image to latent
        rgb_latents = self.image2latent(input_image)

        # Convert depth to disparity
        disparity = self.depth2disparity(depth[0, :, :][None])

        # Step 1: DDIM Inversion (RGB -> Noise)
        self.pipe.scheduler = self.ddim_inverser

        inverted_latent, _ = self.pipe(
            prompt=source_prompt,  # Empty or source description
            num_inference_steps=num_inference_steps,
            latents=rgb_latents,
            image=disparity,
            return_dict=False,
            guidance_scale=0,  # No CFG for inversion
            output_type="latent",
        )

        # Step 2: Forward Sampling (Noise -> Edited RGB)
        self.pipe.scheduler = self.ddim_scheduler

        edited_image = self.pipe(
            prompt=edit_prompt,
            negative_prompt=negative_prompt,
            latents=inverted_latent,
            image=disparity,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            controlnet_conditioning_scale=controlnet_conditioning_scale,
            eta=0.0,  # Deterministic DDIM
            output_type="pt",  # Return as torch tensor
        ).images.squeeze()

        edited_image = cv2.resize(edited_image, original_size[::-1])
        return torch.Tensor(edited_image)


# # for debugging
# device = "cuda" if torch.cuda.is_available() else "cpu"
# ddim_editor = DDIMInversionEditorDepth(device=device)

# sample_img = torch.load("render_colors.pt")
# depth = torch.load("depth.pt")

# print(f"Input image shape: {sample_img.shape}, Depth shape: {depth.shape}")

# edited = ddim_editor.edit(
#     input_image=sample_img.squeeze().permute(2, 0, 1),  # [3, H, W]
#     depth=depth,  # [1, H, W]
#     edit_prompt="there is a white marble table with a vase on it in the yard",
#     source_prompt="there is a wooden table with a vase on it in the yard",
#     controlnet_conditioning_scale=0.8,
#     guidance_scale=3.5,
#     num_inference_steps=50,
# )

# print(
#     f"Edited image shape: {edited.shape}, range: [{edited.min():.3f}, {edited.max():.3f}]"
# )

# # Save comparison with mixup
# import matplotlib.pyplot as plt

# # Create mixup versions
# alpha_values = [0.0, 0.3, 0.5, 0.7, 1.0]  # 0=original, 1=fully edited
# mixup_images = []

# original_np = sample_img.squeeze().cpu().numpy()
# edited_np = edited  # Already in numpy format from cv2.resize

# for alpha in alpha_values:
#     # Blend: (1-alpha)*original + alpha*edited
#     blended = (1 - alpha) * original_np + alpha * edited_np
#     mixup_images.append(blended)

# # Create visualization
# fig, axes = plt.subplots(1, len(alpha_values), figsize=(20, 4))
# titles = ["Original", "30% Edit", "50% Mix", "70% Edit", "Full Edit"]

# for idx, (img, title) in enumerate(zip(mixup_images, titles)):
#     axes[idx].imshow(img)
#     axes[idx].set_title(title)
#     axes[idx].axis("off")

# plt.tight_layout()
# plt.savefig("edited_mixup_comparison.png", dpi=150, bbox_inches="tight")
# print("Saved mixup comparison to edited_mixup_comparison.png")

# # Also save simple side-by-side
# fig2, axes2 = plt.subplots(1, 2, figsize=(12, 6))
# axes2[0].imshow(original_np)
# axes2[0].set_title("Original (Rendered)")
# axes2[0].axis("off")

# axes2[1].imshow(edited_np)
# axes2[1].set_title("Edited (ControlNet + Depth)")
# axes2[1].axis("off")

# plt.tight_layout()
# plt.savefig("edited_comparison.png")
# print("Saved comparison to edited_comparison.png")
