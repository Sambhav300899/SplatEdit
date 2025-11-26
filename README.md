# SplatEdit

Our pipeline is based on the gsplat pipeline, we will use gsplat to train the base splat and then the added methods to modify the splat

## AMI and config used

**AMI Name**: Deep Learning OSS Nvidia Driver AMI GPU PyTorch 2.4.1 (Ubuntu 22.04) 20250223
**AMI ID**: ami-00100bf567866e453
**Instance type**: g5.2xlarge

I put 500 gb storage on the instance to not have to bother with it. It expensive though, the overall setup, better to use a smaller one for dev.

## Env setup 
**Install prereqs and nerfstudio** - We aren't using nerfstudio but it installs a lot of dependencies
```
# Taken from nerfstudio
conda create --name nerfstudio -y python=3.10
conda activate nerfstudio
python -m pip install --upgrade pip

pip install torch==2.1.2+cu118 torchvision==0.16.2+cu118 --extra-index-url https://download.pytorch.org/whl/cu118
conda install -c "nvidia/label/cuda-11.8.0" cuda-toolkit

# this will take a while to finish running
pip install ninja git+https://github.com/NVlabs/tiny-cuda-nn/#subdirectory=bindings/torch --no-build-isolation

pip install nerfstudio
```

**Setup gsplat and the dependencies we need**
```
pip install git+https://github.com/nerfstudio-project/gsplat --no-build-isolation

pip install git+https://github.com/rahul-goel/fused-ssim --no-build-isolation
pip install git+https://github.com/harry7557558/fused-bilagrid --no-build-isolation

pip install -r requirements.txt
```

**fix issues dependencies**
```
pip install hugging-face==0.25.2
pip install "transformers==4.45.2"
pip install nerfstudio[gen]
pip install gsplat==1.5.3

```

## To download the dataset run
```
python scripts/datasets/download_dataset.py

gdown 1x4pf17vjt9IslKorC4BMZNZogyFsFiXS
```

## Train a gsplat
```
CUDA_VISIBLE_DEVICES=0 \
python scripts/simple_trainer.py default \
  --data_dir data/360_v2/garden/ \
  --data_factor 8 \
  --result_dir results/garden \
  --max_steps 20000

## Available datasets and their image counts:
*   `bicycle`: 194 images
*   `bonsai`: 292 images
*   `counter`: 240 images
*   `garden`: 185 images
*   `kitchen`: 279 images
*   `room`: 311 images
*   `stump`: 125 images
## data_factor defines how much our dataset should be downsample. Select one of [2, 4, 8]
## result_dir is where the results will be saved. results/ckpts will have the checkpoints and results/ply will have the ply files
```

## Run editing on gsplat
Assuming previous method produced checkpoint `results/garden/ckpts/ckpt_19999_rank0.pt`
```
CUDA_VISIBLE_DEVICES=0 python scripts/igs2gs.py default \
    --data_dir data/360_v2/garden/ \
    --data_factor 8 \
    --result_dir results/garden_edited_igs2gs \
    --start_ckpt data/base_gsplats/results/garden/ckpts/ckpt_19999_rank0.pt \
    --prompt "make it autumn" \
    --max_steps 5000 \
    --ip2p_method iterative \
    --guidance_scale 7.5 \
    --image_guidance_scale 1.5 \
    --pix2pix_iterations 10 \
    --update_iters 2500
```

update_iters defines how many iterations we run after editing. So we will basically end up running `max_steps/update_iters` editings, after which we will traing for `updater_iters`.

E.g. - `max_steps = 5000`, `update_iters = 2500`
Update images -> 2500 steps
Update images -> 2500 steps

## Getting captions from original dataset
For evaluating clip directional similarity, we need the originial and edited captions. Since we lack the original captions, we can generate them using BLIP. For doing this run - 
```
python helpers/get_original_captions_from_set_of_imgs.py --image_folder data/360_v2/garden/images_8_png --output_path data/garden_captions.txt
```

## Evaluating an edited splat 
```
python scripts/evaluate_edited.py \
  --original_splat_ckpt "../base_splats/results/garden/ckpts/ckpt_19999_rank0.pt" \
  --edited_splat_ckpt "/home/sambhav/ml/SplatEdit/results/results/garden_edited_igs2gs/ckpts/ckpt_4999_rank0.pt" \
  --data_dir "../data/360_v2/garden/" \
  --data_factor 8 \
  --original_prompt "there is a wooden table with a vase on it in the yard" \
  --edited_prompt "Make it autumn"
```
