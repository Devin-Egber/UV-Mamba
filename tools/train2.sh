work_dir="/data/mamba/Ablation"
export PYTHONPATH=$work_dir
MASTER_PORT=32991

# CONFIG_FILE="config/uv/unet_xian.yaml"

CONFIG_FILE="config/uv/uvmamba_cityscapes2.yaml"

CUDA_VISIBLE_DEVICES=0,1 deepspeed --master_port $MASTER_PORT tools/train.py --config_file $CONFIG_FILE
