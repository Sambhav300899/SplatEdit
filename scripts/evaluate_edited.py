import torch
from datasets.colmap import Parser, Dataset
from gsplat.rendering import rasterization
import tqdm
import torch.nn.functional as F
import math
import metrics
from dataclasses import dataclass


@dataclass
class Config:
    original_splat_ckpt: str = "base_splats/garden/ckpts/ckpt_19999_rank0.pt"
    # edited_splat_ckpt: str = "/home/sambhav/ml/SplatEdit/results/garden_edited_igs2gs/ckpts/ckpt_4999_rank0.pt"
    edited_splat_ckpt: str = "/home/sambhav/ml/SplatEdit/results/garden_edited_igs2gs_naive/ckpts/ckpt_2499_rank0.pt"

    data_dir: str = "data/360_v2/garden/"

    data_factor: int = 8
    original_prompt: str = "there is a wooden table with a vase on it in the yard"
    edited_prompt: str = "make it like van goghs starry night"


def get_spiral_trajectory(dataset, num_steps=60, num_rotations=2, camera_down_tilt=0.3):
    # COORDS USED BY GSPLAT
    # -Z is UP
    camtoworlds_all = torch.stack([data["camtoworld"] for data in dataset], dim=0)
    centers = camtoworlds_all[:, :3, 3]
    centers_mean = centers.mean(dim=0)

    avg_scene_radii = torch.norm(
        centers[:, [0, 1]] - centers_mean[[0, 1]], dim=1
    ).mean()

    avg_height = centers_mean[2]

    trajectory_camtoworlds = []
    world_up = torch.tensor([0.0, 0.0, -1.0])

    for t in torch.linspace(0, 1, num_steps):
        theta = 2 * math.pi * num_rotations * t

        cam_pos = torch.tensor(
            [
                centers_mean[0] + avg_scene_radii * math.cos(theta),
                centers_mean[1] + avg_scene_radii * math.sin(theta),
                avg_height,
            ]
        )

        tilt_offset = avg_scene_radii * camera_down_tilt
        look_at_target = centers_mean.clone()
        look_at_target[2] -= tilt_offset

        forward = look_at_target - cam_pos
        forward /= torch.norm(forward)

        right = torch.cross(forward, world_up)
        right /= torch.norm(right)

        up_true = torch.cross(right, forward)
        up_true /= torch.norm(up_true)

        camtoworld = torch.eye(4)
        camtoworld[:3, 0] = right
        camtoworld[:3, 1] = up_true
        camtoworld[:3, 2] = forward
        camtoworld[:3, 3] = cam_pos

        trajectory_camtoworlds.append(camtoworld)

    return torch.stack(trajectory_camtoworlds, dim=0)


@torch.no_grad()
def render_loader(dataset, device, means, quats, scales, opacities, colors, sh_degree):
    images = []
    depths = []
    # trajectory = get_spiral_trajectory(dataset, num_steps=len(dataset), num_rotations=1)

    print(f"rendering {len(dataset)} images...")
    for i in tqdm.tqdm(range(len(dataset))):
        data = dataset[i]

        camtoworld = data["camtoworld"].to(device)
        # print(camtoworld)
        K = data["K"].to(device)

        gt_image = data["image"].to(device)  # [H, W, 3]
        height, width = gt_image.shape[:2]

        viewmat = torch.linalg.inv(camtoworld)

        render_colors, render_alphas, info = rasterization(
            means=means,
            quats=quats,
            scales=scales,
            opacities=opacities,
            colors=colors,
            viewmats=viewmat.unsqueeze(0),
            Ks=K.unsqueeze(0),
            width=width,
            height=height,
            sh_degree=sh_degree,
            packed=False,
            render_mode="RGB+ED",
        )

        depth = render_colors[:, :, :, 3]
        render_colors = render_colors[:, :, :, :3]

        # torch.save(depth, "depth.pt")
        # torch.save(render_colors, "render_colors.pt")

        # print(depth.shape, render_colors.shape, depth.max(), depth.min())
        # exit()
        # # uncomment to debug
        # import cv2

        # display_image = (
        #     render_colors.squeeze().cpu().numpy()[:, :, ::-1] * 255
        # ).astype("uint8")
        # cv2.imshow("Rendered Image", display_image)
        # cv2.waitKey(1)

        images.append(render_colors.squeeze().permute(2, 1, 0))
        depths.append(depth.squeeze().permute(2, 1, 0))

    # cv2.destroyAllWindows()
    images = torch.stack(images, dim=0)
    depths = torch.stack(depths, dim=0)

    return images, depths


@torch.no_grad()
def get_clip_based_metrics(
    orig_rendered, edited_rendered, original_prompt, edited_prompt, device
):
    clip_metrics = metrics.ClipSimilarity(device=device)

    try:
        # Attempt to run everything at once
        sim_0, sim_1, sim_direction, sim_image = clip_metrics(
            orig_rendered, edited_rendered, original_prompt, edited_prompt
        )
        print("Calculated CLIP metrics without batching.")
    except RuntimeError as e:
        # If it fails (e.g., CUDA out of memory), fall back to batched calculation
        if "CUDA out of memory" in str(e) or "out of memory" in str(e):
            print("CUDA out of memory, falling back to batched calculation.")
            batch_size = 16
            num_images = orig_rendered.shape[0]

            all_sim_0, all_sim_1, all_sim_direction, all_sim_image = [], [], [], []

            print(
                f"Calculating CLIP metrics for {num_images} images in batches of {batch_size}..."
            )
            for i in tqdm.tqdm(range(0, num_images, batch_size)):
                batch_orig_rendered = orig_rendered[i : i + batch_size]
                batch_edited_rendered = edited_rendered[i : i + batch_size]

                sim_0_batch, sim_1_batch, sim_direction_batch, sim_image_batch = (
                    clip_metrics(
                        batch_orig_rendered,
                        batch_edited_rendered,
                        original_prompt,
                        edited_prompt,
                    )
                )
                all_sim_0.append(sim_0_batch)
                all_sim_1.append(sim_1_batch)
                all_sim_direction.append(sim_direction_batch)
                all_sim_image.append(sim_image_batch)

            sim_0 = torch.cat(all_sim_0)
            sim_1 = torch.cat(all_sim_1)
            sim_direction = torch.cat(all_sim_direction)
            sim_image = torch.cat(all_sim_image)
        else:
            # Re-raise other RuntimeErrors
            raise e

    return sim_0.cpu(), sim_1.cpu(), sim_direction.cpu(), sim_image.cpu()


if __name__ == "__main__":
    config = Config()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    parser = Parser(
        data_dir=config.data_dir,
        factor=config.data_factor,
        normalize=True,
        test_every=8,
    )
    trainset = Dataset(parser, split="train", patch_size=None, load_depths=False)

    orig_rendered = []
    edited_rendered = []
    orig_depths = []
    edited_depths = []

    for i, splat_ckpt in enumerate(
        [config.original_splat_ckpt, config.edited_splat_ckpt]
    ):
        gs = torch.load(splat_ckpt, map_location=device, weights_only=True)

        splats = gs["splats"]
        means = splats["means"].to(device)
        quats = splats["quats"].to(device)
        scales = splats["scales"].to(device)
        opacities = splats["opacities"].to(device)

        # Handle SH
        sh0 = splats["sh0"].to(device)
        if "shN" in splats:
            shN = splats["shN"].to(device)
            colors = torch.cat([sh0, shN], dim=-2)
        else:
            colors = sh0
        sh_degree = int(math.sqrt(colors.shape[-2]) - 1)

        # Process parameters for rendering
        quats = F.normalize(quats, p=2, dim=-1)
        scales = torch.exp(scales)
        opacities = torch.sigmoid(opacities)
        print(f"Loaded {len(means)} Gaussians with SH degree {sh_degree}.")

        images, depths = render_loader(
            dataset=trainset,
            device=device,
            means=means,
            quats=quats,
            scales=scales,
            opacities=opacities,
            colors=colors,
            sh_degree=sh_degree,
        )

        if i == 0:
            orig_rendered = images
            orig_depths = depths
        else:
            edited_rendered = images
            edited_depths = depths

    sim_0, sim_1, sim_direction, sim_image = get_clip_based_metrics(
        orig_rendered,
        edited_rendered,
        config.original_prompt,
        config.edited_prompt,
        device,
    )

    sim_0_mean = sim_0.mean()
    sim_1_mean = sim_1.mean()
    sim_direction_mean = sim_direction.mean()
    sim_image_mean = sim_image.mean()

    print(f"CLIP Similarity (Original to Prompt 0): {sim_0_mean:.4f}")
    print(f"CLIP Similarity (Edited to Prompt 1): {sim_1_mean:.4f}")
    print(f"CLIP Directional Similarity: {sim_direction_mean:.4f}")
    print(f"CLIP Image Similarity: {sim_image_mean:.4f}")

    depth_diff = torch.abs(orig_depths - edited_depths).mean()
    print(f"Mean Absolute Depth Difference: {depth_diff:.4f}")
