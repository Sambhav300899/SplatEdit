CUDA_VISIBLE_DEVICES=0 python scripts/igs2gs.py default \
    --data_dir data/360_v2/garden/ \
    --data_factor 8 \
    --result_dir results/garden_edited_igs2gs \
    --start_ckpt results/garden/ckpts/ckpt_19999_rank0.pt \
    --prompt "make it autumn" \
    --ip2p_method iterative \
    --max_steps 5000 \
    --guidance_scale 7.5 \
    --image_guidance_scale 1.5 \
    --pix2pix_iterations 10 \
    --update_iters 2500


CUDA_VISIBLE_DEVICES=0 python scripts/igs2gs_cli.py default \
    --data_dir data/360_v2/garden/ \
    --data_factor 8 \
    --result_dir results/garden_flowedit_marble_table\
    --start_ckpt results/garden/ckpts/ckpt_19999_rank0.pt \
    --prompt "turn the table into a white marble table" \
    --ip2p_method naive \
    --update_iters 10000 \
    --max_steps 5000


CUDA_VISIBLE_DEVICES=0 python scripts/igs2gs_cli.py default \
    --data_dir data/360_v2/garden/ \
    --data_factor 8 \
    --result_dir results/garden_flowedit_iterative \
    --start_ckpt results/garden/ckpts/ckpt_19999_rank0.pt \
    --prompt "Turn the existing table into pink color table. Do not modify any other object." \
    --ip2p_method iterative \
    --pix2pix_iterations 10 \
    --update_iters 1000 \
    --max_steps 5000

