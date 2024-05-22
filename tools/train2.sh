work_dir="/data/project/UV-Mamba"
export PYTHONPATH=$work_dir
MASTER_PORT=32991

CONFIG_FILE="config/uv/uv_mamba/uvmamba_beijing.yaml"


CUDA_VISIBLE_DEVICES=1 deepspeed --master_port $MASTER_PORT tools/train.py --config_file $CONFIG_FILE


#CUDA_VISIBLE_DEVICES=0 deepspeed --master_port $MASTER_PORT scripts/test_semantic.py $CONFIG_FILE
