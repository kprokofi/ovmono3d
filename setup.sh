uv pip install torch torchvision
uv pip install git+https://github.com/facebookresearch/pytorch3d.git@055ab3a --no-build-isolation
uv pip install git+https://github.com/yaojin17/detectron2.git  --no-build-isolation # slightly modified detectron2 for OVMono3D
uv pip install cython opencv-python scipy pandas einops open_clip_torch open3d --no-build-isolation

uv pip install git+https://github.com/apple/ml-depth-pro.git@b2cd0d5 --no-build-isolation
uv pip install git+https://github.com/facebookresearch/segment-anything.git@dca509f --no-build-isolation
uv pip install git+https://github.com/IDEA-Research/GroundingDINO.git --no-build-isolation
