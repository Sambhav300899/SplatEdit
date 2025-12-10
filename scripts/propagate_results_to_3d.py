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
import torch.nn.functional as F
import numpy as np
import random


class ImageWarper:
    """Handles 3D warping of images between viewpoints using depth information."""

    def __init__(self, device="cuda"):
        self.device = device

    def get_reprojection_error(
        self, forward_grid, backward_grid, curr_grid, thres=0.01
    ):
        """
        Calculate reprojection error for forward-backward consistency check.

        Args:
            forward_grid: Projected coordinates from current to reference view [H, W, 2]
            backward_grid: Projected coordinates from reference to current view [H, W, 2]
            curr_grid: Current view pixel coordinates [H, W, 2]
            thres: Reprojection error threshold

        Returns:
            valid_mask: Boolean mask of valid pixels
        """
        forward_grid = forward_grid.squeeze()  # H x W x 2
        backward_grid = backward_grid.squeeze()
        H, W = forward_grid.shape[:2]

        # Normalize to [-1, 1] for grid_sample
        forward_grid_norm = forward_grid.clone()
        forward_grid_norm[..., 0] = forward_grid_norm[..., 0] / (W) * 2 - 1
        forward_grid_norm[..., 1] = forward_grid_norm[..., 1] / (H) * 2 - 1

        W_bound = 1 - 1 / W
        H_bound = 1 - 1 / H
        forward_mask = (
            (forward_grid_norm[..., 0] > -W_bound)
            & (forward_grid_norm[..., 0] < W_bound)
            & (forward_grid_norm[..., 1] > -H_bound)
            & (forward_grid_norm[..., 1] < H_bound)
        )

        backward_grid_norm = backward_grid.clone()
        backward_grid_norm[..., 0] = backward_grid_norm[..., 0] / (W) * 2 - 1
        backward_grid_norm[..., 1] = backward_grid_norm[..., 1] / (H) * 2 - 1

        backward_mask = (
            (backward_grid_norm[..., 0] > -W_bound)
            & (backward_grid_norm[..., 0] < W_bound)
            & (backward_grid_norm[..., 1] > -H_bound)
            & (backward_grid_norm[..., 1] < H_bound)
        )

        backward_grid_norm[~backward_mask, :] = 10

        curr_grid_norm = curr_grid.clone()
        curr_grid_norm[..., 0] = curr_grid_norm[..., 0] / (W) * 2 - 1
        curr_grid_norm[..., 1] = curr_grid_norm[..., 1] / (H) * 2 - 1

        # Sample backward grid at forward locations
        re_proj_grid = (
            F.grid_sample(
                backward_grid_norm[None, ...].permute(0, 3, 1, 2),
                forward_grid_norm[None, ...],
                mode="nearest",
                padding_mode="zeros",
                align_corners=False,
            )
            .squeeze()
            .permute(1, 2, 0)
        )

        error = torch.norm(re_proj_grid - curr_grid_norm.to(self.device), dim=-1)
        valid_mask = (error < thres) & forward_mask

        return valid_mask

    def warp_image(self, image, src_grid, mask, mode="bilinear"):
        """
        Warp an image using a grid of source coordinates.

        Args:
            image: Image to warp [H, W, 3]
            src_grid: Source pixel coordinates [H, W, 2]
            mask: Valid pixel mask [H, W]
            mode: Interpolation mode

        Returns:
            Warped image [H, W, 3]
        """
        image = image.permute(2, 0, 1).unsqueeze(0)  # 1, 3, H, W

        H, W = image.shape[2:]
        src_grid = src_grid.squeeze().unsqueeze(0)  # 1, H, W, 2
        src_grid_norm = src_grid.clone()
        src_grid_norm[..., 0] = src_grid_norm[..., 0] / ((W) / 2) - 1
        src_grid_norm[..., 1] = src_grid_norm[..., 1] / ((H) / 2) - 1
        src_grid_norm = src_grid_norm.float()

        warped_rgb = (
            F.grid_sample(
                image,
                src_grid_norm,
                mode=mode,
                padding_mode="zeros",
                align_corners=False,
            )
            .squeeze()
            .permute(1, 2, 0)
        )

        new_image = torch.zeros(H, W, 3, device=image.device)
        new_image[mask[..., None].expand(*mask.shape, 3)] = warped_rgb[
            mask[..., None].expand(*mask.shape, 3)
        ]

        return new_image

    def project_points_to_camera(self, points_3d, camera_to_world, K, image_size):
        """
        Project 3D points to camera image coordinates.

        Args:
            points_3d: 3D points in world coordinates [H, W, 3]
            camera_to_world: Camera to world transformation [4, 4]
            K: Camera intrinsics [3, 3]
            image_size: (height, width)

        Returns:
            pixel_coords: Pixel coordinates [H, W, 2] in (x, y) format
        """
        H, W = image_size

        # Transform to camera coordinates
        world_to_camera = torch.inverse(camera_to_world)
        points_3d_flat = points_3d.reshape(-1, 3)
        points_3d_homo = torch.cat(
            [
                points_3d_flat,
                torch.ones(points_3d_flat.shape[0], 1, device=points_3d.device),
            ],
            dim=1,
        )
        points_cam_homo = (world_to_camera @ points_3d_homo.T).T
        points_cam = points_cam_homo[:, :3]

        # Project to image plane
        points_2d_homo = (K @ points_cam.T).T
        pixel_coords = points_2d_homo[:, :2] / (points_2d_homo[:, 2:3] + 1e-8)
        pixel_coords = pixel_coords.reshape(H, W, 2)

        return pixel_coords

    def warp_between_views(
        self,
        ref_image,
        ref_depth,
        ref_camera_to_world,
        ref_K,
        cur_depth,
        cur_camera_to_world,
        cur_K,
        cur_image=None,
        thres=0.01,
        mode="bilinear",
    ):
        """
        Warp reference image to current view using depth and camera parameters.

        Args:
            ref_image: Reference image [H, W, 3]
            ref_depth: Reference depth map [H, W]
            ref_camera_to_world: Reference camera pose [4, 4]
            ref_K: Reference camera intrinsics [3, 3]
            cur_depth: Current depth map [H, W]
            cur_camera_to_world: Current camera pose [4, 4]
            cur_K: Current camera intrinsics [3, 3]
            cur_image: Current image for blending [H, W, 3] (optional)
            thres: Reprojection error threshold
            mode: Interpolation mode

        Returns:
            warped_image: Warped reference image [H, W, 3]
            valid_mask: Valid pixel mask [H, W]
        """
        H, W = ref_depth.shape

        # Create pixel grid for current view
        y_coords, x_coords = torch.meshgrid(
            torch.arange(H, device=self.device),
            torch.arange(W, device=self.device),
            indexing="ij",
        )
        curr_grid = torch.stack([x_coords, y_coords], dim=-1).float()

        # Unproject current view pixels to 3D
        cur_pixels_homo = torch.stack(
            [x_coords, y_coords, torch.ones_like(x_coords)], dim=-1
        ).float()
        cur_K_inv = torch.inverse(cur_K)
        cur_rays = (cur_K_inv @ cur_pixels_homo.reshape(-1, 3).T).T.reshape(H, W, 3)
        cur_points_3d_cam = cur_rays * cur_depth[..., None]

        # Transform to world coordinates
        cur_points_3d_homo = torch.cat(
            [
                cur_points_3d_cam.reshape(-1, 3),
                torch.ones(H * W, 1, device=self.device),
            ],
            dim=1,
        )
        cur_points_3d_world = (
            (cur_camera_to_world @ cur_points_3d_homo.T).T[:, :3].reshape(H, W, 3)
        )

        # Project to reference view
        coords_cur_to_ref = self.project_points_to_camera(
            cur_points_3d_world, ref_camera_to_world, ref_K, (H, W)
        )

        # Do the same for reference to current (for consistency check)
        ref_pixels_homo = torch.stack(
            [x_coords, y_coords, torch.ones_like(x_coords)], dim=-1
        ).float()
        ref_K_inv = torch.inverse(ref_K)
        ref_rays = (ref_K_inv @ ref_pixels_homo.reshape(-1, 3).T).T.reshape(H, W, 3)
        ref_points_3d_cam = ref_rays * ref_depth[..., None]

        ref_points_3d_homo = torch.cat(
            [
                ref_points_3d_cam.reshape(-1, 3),
                torch.ones(H * W, 1, device=self.device),
            ],
            dim=1,
        )
        ref_points_3d_world = (
            (ref_camera_to_world @ ref_points_3d_homo.T).T[:, :3].reshape(H, W, 3)
        )

        coords_ref_to_cur = self.project_points_to_camera(
            ref_points_3d_world, cur_camera_to_world, cur_K, (H, W)
        )

        # Check reprojection consistency
        valid_mask = self.get_reprojection_error(
            coords_cur_to_ref, coords_ref_to_cur, curr_grid, thres=thres
        )

        # Warp the reference image
        warped_image = self.warp_image(
            ref_image, coords_cur_to_ref, valid_mask, mode=mode
        )

        # Blend with current image if provided
        if cur_image is not None:
            blended_image = (
                warped_image * valid_mask[..., None]
                + cur_image * (~valid_mask)[..., None]
            )
            return blended_image, valid_mask

        return warped_image, valid_mask


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

        controlnet = ControlNetModel.from_pretrained(
            "lllyasviel/sd-controlnet-depth", torch_dtype=torch.float16
        )

        self.pipe = StableDiffusionControlNetPipeline.from_pretrained(
            diffusion_ckpt,
            controlnet=controlnet,
            scheduler=self.ddim_scheduler,
            safety_checker=None,
            requires_safety_checker=False,
            torch_dtype=torch.float16,
        ).to(self.device)

    @torch.no_grad()
    def image2latent(self, image):
        image = image.to(torch.float16)  # Convert to FP16 to match VAE dtype
        image = image * 2 - 1
        latents = self.pipe.vae.encode(image)["latent_dist"].mean
        latents = latents * 0.18215
        return latents

    def depth2disparity(self, depth):
        depth = depth.to(torch.float16)  # Convert to FP16 to match pipeline dtype
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


class DDIMInversionEditorDepthMultiview:
    def __init__(self, device="cuda"):
        diffusion_ckpt = "runwayml/stable-diffusion-v1-5"
        self.device = device

        self.ddim_scheduler = DDIMScheduler.from_pretrained(
            diffusion_ckpt, subfolder="scheduler"
        )
        self.ddim_inverser = DDIMInverseScheduler.from_pretrained(
            diffusion_ckpt, subfolder="scheduler"
        )

        controlnet = ControlNetModel.from_pretrained(
            "lllyasviel/sd-controlnet-depth", torch_dtype=torch.float16
        )

        self.pipe = StableDiffusionControlNetPipeline.from_pretrained(
            diffusion_ckpt,
            controlnet=controlnet,
            scheduler=self.ddim_scheduler,
            safety_checker=None,
            requires_safety_checker=False,
            torch_dtype=torch.float16,
        ).to(self.device)

        # Initialize warper
        self.warper = ImageWarper(device=device)

    @torch.no_grad()
    def image2latent(self, image):
        image = image.to(torch.float16)  # Convert to FP16 to match VAE dtype
        image = image * 2 - 1
        latents = self.pipe.vae.encode(image)["latent_dist"].mean
        latents = latents * 0.18215
        return latents

    def depth2disparity(self, depth):
        depth = depth.to(torch.float16)  # Convert to FP16 to match VAE dtype
        disparity = 1 / (depth + 1e-5)
        disparity_map = disparity / torch.max(disparity)
        disparity_map = torch.concatenate(
            [disparity_map, disparity_map, disparity_map], dim=0
        )
        return disparity_map[None]

    def run_edits(
        self,
        idx2image: dict,
        idx2depth: dict,
        idx2camera: dict,  # New: camera parameters {idx: {'c2w': [4,4], 'K': [3,3]}}
        idxs,
        edit_prompt,
        source_prompt,
        negative_prompt="",
        controlnet_conditioning_scale=1.0,
        guidance_scale=3.5,
        num_inference_steps=100,
        num_ref_views=4,
        bs=1,
        use_warping=True,  # New: enable/disable 3D warping
        warp_threshold=0.01,  # New: reprojection error threshold
    ):
        idxs = sorted(idxs)
        partition_size = len(idxs) // num_ref_views

        ref_indices = []
        for i in range(num_ref_views):
            ref_indices.append(
                random.choice(idxs[i * partition_size : (i + 1) * partition_size])
            )

        print(f"Using {ref_indices} as reference views")

        print("Performing DDIM inversion on reference views...")
        ref_inverted_latents = []
        ref_disparities = []

        resize_transform = torchvision.transforms.Resize((512, 512))
        resize_to_orig = torchvision.transforms.Resize(
            (idx2depth[idxs[0]].shape[1], idx2depth[idxs[0]].shape[2])
        )

        idx2edited = {}

        ref_images = torch.cat(
            [
                idx2image[ref_index].unsqueeze(0).to(self.device)
                for ref_index in ref_indices
            ],
            dim=0,
        )
        ref_disparities = torch.cat(
            [
                self.depth2disparity(idx2depth[ref_index]).to(self.device)
                for ref_index in ref_indices
            ],
            dim=0,
        )

        ref_images = resize_transform(ref_images)
        ref_disparities = resize_transform(ref_disparities)
        ref_rgb_latents = self.image2latent(ref_images)

        self.pipe.scheduler = self.ddim_inverser
        ref_inverted_latents, _ = self.pipe(
            prompt=[source_prompt] * num_ref_views,
            num_inference_steps=num_inference_steps,
            latents=ref_rgb_latents,
            image=ref_disparities,
            return_dict=False,
            guidance_scale=0,  # No CFG for inversion
            output_type="latent",
        )

        # Edit reference views first
        self.pipe.scheduler = self.ddim_scheduler
        ref_edited = self.pipe(
            prompt=[edit_prompt] * num_ref_views,
            negative_prompt=[negative_prompt] * num_ref_views,
            latents=ref_inverted_latents,
            image=ref_disparities,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            controlnet_conditioning_scale=controlnet_conditioning_scale,
            eta=0.0,
            output_type="pt",
        ).images

        ref_edited = resize_to_orig(torch.Tensor(ref_edited).permute(0, 3, 1, 2))

        # Store edited reference views
        for i, ref_idx in enumerate(ref_indices):
            idx2edited[ref_idx] = ref_edited[i].permute(1, 2, 0)

        # Process chunks
        for idx in tqdm.tqdm(range(0, len(idxs), bs), total=len(idxs) // bs):
            batch_idxs = idxs[idx : idx + bs]
            print(f"Editing chunk: {batch_idxs}")

            # Perform DDIM inversion on chunk images
            batch_imgs = torch.cat(
                [
                    idx2image[img_idx].unsqueeze(0).to(self.device)
                    for img_idx in batch_idxs
                ],
                dim=0,
            )
            batch_disparities = torch.cat(
                [
                    self.depth2disparity(idx2depth[img_idx]).to(self.device)
                    for img_idx in batch_idxs
                ],
                dim=0,
            )

            batch_imgs = resize_transform(batch_imgs)
            batch_disparities = resize_transform(batch_disparities)

            batch_latents = self.image2latent(batch_imgs)

            self.pipe.scheduler = self.ddim_inverser

            actual_batch_size = len(batch_idxs)
            source_prompts = [source_prompt] * actual_batch_size

            batch_inverted_latents, _ = self.pipe(
                prompt=source_prompts,
                num_inference_steps=num_inference_steps,
                latents=batch_latents,
                image=batch_disparities,
                return_dict=False,
                guidance_scale=0,
                output_type="latent",
            )

            # Concatenate reference views with chunk data
            disp_ctrl_batch = torch.cat((ref_disparities, batch_disparities), dim=0)
            latents_batch = torch.cat(
                (ref_inverted_latents, batch_inverted_latents), dim=0
            )

            # Run forward sampling with DDIM scheduler
            edit_prompts = [edit_prompt] * (num_ref_views + actual_batch_size)
            negative_prompts = [negative_prompt] * (num_ref_views + actual_batch_size)

            self.pipe.scheduler = self.ddim_scheduler
            edited_batch = self.pipe(
                prompt=edit_prompts,
                negative_prompt=negative_prompts,
                latents=latents_batch,
                image=disp_ctrl_batch,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                controlnet_conditioning_scale=controlnet_conditioning_scale,
                eta=0.0,
                output_type="pt",
            ).images[num_ref_views:]  # Skip reference views in output

            edited_batch = resize_to_orig(
                torch.Tensor(edited_batch).permute(0, 3, 1, 2)
            )

            for j, img_idx in enumerate(batch_idxs):
                edited_img = edited_batch[j].permute(1, 2, 0)

                # Apply 3D warping if enabled
                if use_warping and idx2camera is not None:
                    # Find closest reference view for warping
                    best_ref_idx = ref_indices[0]  # Default to first ref

                    # Warp edited reference view to current view
                    warped_img, warp_mask = self.warper.warp_between_views(
                        ref_image=idx2edited[best_ref_idx],
                        ref_depth=idx2depth[best_ref_idx].squeeze(),
                        ref_camera_to_world=idx2camera[best_ref_idx]["c2w"],
                        ref_K=idx2camera[best_ref_idx]["K"],
                        cur_depth=idx2depth[img_idx].squeeze(),
                        cur_camera_to_world=idx2camera[img_idx]["c2w"],
                        cur_K=idx2camera[img_idx]["K"],
                        cur_image=edited_img,
                        thres=warp_threshold,
                        mode="bilinear",
                    )
                    print(warped_img, warp_mask)

                    # Use warped result where valid, otherwise use direct edit
                    edited_img = warped_img

                idx2edited[img_idx] = edited_img

            # Debug visualization
            import matplotlib.pyplot as plt
            import os

            os.makedirs("view_renders", exist_ok=True)
            for j, img_idx in enumerate(batch_idxs):
                fig, axes = plt.subplots(1, 2, figsize=(10, 5))

                # Original image
                original_img = (
                    resize_to_orig(batch_imgs)[j].permute(1, 2, 0).cpu().numpy()
                )
                axes[0].imshow(original_img)
                axes[0].set_title(f"Original Image {img_idx}")
                axes[0].axis("off")

                # Edited image
                edited_img = idx2edited[img_idx].cpu().numpy()
                axes[1].imshow(edited_img)
                axes[1].set_title(f"Edited Image {img_idx}")
                axes[1].axis("off")

                plt.tight_layout()
                plt.savefig(f"view_renders/{img_idx}.png")
                plt.close()

        return idx2edited
