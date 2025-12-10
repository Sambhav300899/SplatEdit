import torch
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration
import glob
import os
from collections import Counter
from tqdm import tqdm
from dataclasses import dataclass
import tyro


@dataclass
class Config:
    """Configuration for the caption generation script."""

    # Help text can go in the docstring
    image_folder: str  # e.g., 'data/360_v2/garden/images_8_png/*.png'
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    img_extension: str = "png"
    output_path: str = "view_captions.txt"


def generate_multiview_captions(cfg: Config):
    device = cfg.device
    # 1. Load Model
    print(f"Loading BLIP model on {device}...")
    processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-large")
    model = BlipForConditionalGeneration.from_pretrained(
        "Salesforce/blip-image-captioning-large"
    ).to(device)

    image_paths = sorted(
        glob.glob(os.path.join(cfg.image_folder, f"*.{cfg.img_extension}"))
    )
    captions = []

    print(f"Generating captions for {len(image_paths)} images...")

    # 2. Batch Generation Loop
    # (Doing one by one for simplicity, but batching is faster for massive datasets)
    for img_path in tqdm(image_paths):
        raw_image = Image.open(img_path).convert("RGB")

        # Unconditional image captioning
        inputs = processor(raw_image, return_tensors="pt").to(device)

        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=20)

        caption = processor.decode(out[0], skip_special_tokens=True)
        captions.append(caption)

    return image_paths, captions


if __name__ == "__main__":
    cfg = tyro.cli(Config)

    # --- RUN IT ---
    paths, generated_captions = generate_multiview_captions(cfg)

    # --- ANALYSIS ---

    # 2. Find the "Consensus" (Most Frequent) Caption
    # This filters out weird angles where the model got confused.
    counts = Counter(generated_captions)
    most_common = counts.most_common(1)[0]

    with open(cfg.output_path, "w") as f:
        f.write(f"--- CONSENSUS CAPTION ---\n")
        f.write(f"Most frequent caption: '{most_common[0]}'\n")
        f.write(f"(Appeared in {most_common[1]} out of {len(paths)} frames)\n\n")

        f.write(f"--- ALL UNIQUE CAPTIONS AND THEIR COUNTS ---\n")
        for caption, count in counts.most_common():
            f.write(f"'{caption}': {count} times\n")
        f.write("\n")

        f.write(f"--- INDIVIDUAL IMAGE CAPTIONS ---\n")
        for p, c in zip(paths, generated_captions):
            f.write(f"{os.path.basename(p)}: {c}\n")
