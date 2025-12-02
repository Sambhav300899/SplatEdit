python -m pip install --upgrade pip

pip install torch==2.1.2+cu118 torchvision==0.16.2+cu118 --extra-index-url https://download.pytorch.org/whl/cu118
conda install -c "nvidia/label/cuda-11.8.0" -y cuda-toolkit

# this will take a while to finish running
pip install ninja git+https://github.com/NVlabs/tiny-cuda-nn/#subdirectory=bindings/torch --no-build-isolation

pip install nerfstudio

pip install git+https://github.com/nerfstudio-project/gsplat --no-build-isolation

pip install git+https://github.com/rahul-goel/fused-ssim --no-build-isolation
pip install git+https://github.com/harry7557558/fused-bilagrid --no-build-isolation

pip install -r requirements.txt

pip install huggingface-hub==0.25.2
pip install "transformers==4.45.2"
pip install nerfstudio[gen]
pip install gsplat==1.5.3



pip install spacy
pip install -U "spacy==3.8.2" "numpy<2"

python -m spacy download en_core_web_sm

export CUDA_HOME=/opt/conda/envs/splatedit

git clone https://github.com/IDEA-Research/GroundingDINO.git

cd GroundingDINO

python setup.py install

pip install yapf

pip install supervision
