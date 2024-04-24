work_dir="/data/project/DiffPASTIS"
export PYTHONPATH=$work_dir

# diffusion free

# utae
CONFIG_FILE="configs/PASTIS/utae/utae_semantic.yaml"
# exchanger unet
CONFIG_FILE="configs/PASTIS/exchanger/exchanger_unet_semantic.yml"
# exchanger mask2former
CONFIG_FILE="configs/PASTIS/exchanger/exchanger_mask2former_semantic.yml"
# tsvit
CONFIG_FILE="configs/PASTIS/tsvit/TSViT_semantic.yaml"


# diffusion based

CONFIG_FILE="configs/PASTIS/utae/utae_semantic_diffusion.yaml"
# exchanger unet
CONFIG_FILE="configs/PASTIS/exchanger/exchanger_unet_diffusion_semantic.yml"
# exchanger mask2former
CONFIG_FILE="configs/PASTIS/exchanger/exchanger_mask2former_diffusion_semantic.yml"
# tsvit
CONFIG_FILE="configs/PASTIS/tsvit/TSViT_semantic_diffusion.yaml"




CUDA_VISIBLE_DEVICES=0 deepspeed --master_port 32996 scripts/get_flops.py $CONFIG_FILE


