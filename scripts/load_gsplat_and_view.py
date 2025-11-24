import torch
import torch.nn.functional as F
import math
import numpy as np
import imageio
import os
import tqdm
from gsplat.rendering import rasterization
from datasets.colmap import Parser, Dataset


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt_path = "results/counter/ckpts/ckpt_6999_rank0.pt"
    data_dir = "../data/360_v2/counter/"
    data_factor = 4

    if not torch.cuda.is_available():
        print(
            "Warning: CUDA not available, running on CPU might be slow or not supported for rasterization."
        )

    if not os.path.exists(ckpt_path):
        print(f"Checkpoint not found at {ckpt_path}. Please check the path.")
        return

    print(f"Loading checkpoint from {ckpt_path}...")
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=True)

    splats = ckpt["splats"]

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

    # Process parameters for rendering
    quats = F.normalize(quats, p=2, dim=-1)
    scales = torch.exp(scales)
    opacities = torch.sigmoid(opacities)

    sh_degree = int(math.sqrt(colors.shape[-2]) - 1)
    print(f"Loaded {len(means)} Gaussians with SH degree {sh_degree}.")

    # Load Dataset
    print(f"Loading data from {data_dir}...")
    # Note: The paths in cfg.yml might be relative to where simple_trainer was run.
    # We are running from examples/, so ../data/360_v2/counter/ should be correct if data is in /home/ubuntu/gsplat/data

    if not os.path.exists(data_dir):
        print(f"Data directory {data_dir} does not exist. Checking absolute path...")
        # Fallback or check
        pass

    parser = Parser(
        data_dir=data_dir,
        factor=data_factor,
        normalize=True,  # Matches simple_trainer default
        test_every=8,
    )
    trainset = Dataset(parser, split="train", patch_size=None, load_depths=False)

    print(f"Found {len(trainset)} training images.")

    output_dir = "renders_train_views"
    os.makedirs(output_dir, exist_ok=True)

    print("Rendering training views...")
    for i in tqdm.tqdm(range(len(trainset))):
        data = trainset[i]

        camtoworld = data["camtoworld"].to(device)
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
        )

        # Clamp and convert to numpy
        img = render_colors[0, ..., :3].clamp(0, 1).detach().cpu().numpy()
        img = (img * 255).astype(np.uint8)

        # Save individual frame
        imageio.imwrite(f"{output_dir}/train_{i:04d}.png", img)

    print(f"Saved renders to {output_dir}")


if __name__ == "__main__":
    main()
