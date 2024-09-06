work_dir="/data/mamba/Ablation"
export PYTHONPATH=$work_dir
MASTER_PORT=32990

# CONFIG_FILE="config/uv/unet_beijing.yaml"

CONFIG_FILE="config/uv/uvmamba_cityscapes.yaml"

CUDA_VISIBLE_DEVICES=0,1 deepspeed --master_port $MASTER_PORT tools/train.py --config_file $CONFIG_FILE

