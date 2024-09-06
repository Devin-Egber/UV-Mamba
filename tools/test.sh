work_dir="/data/mamba/Ablation"
export PYTHONPATH=$work_dir
MASTER_PORT=32999


# CONFIG_FILE="config/uv/uvmamba_xian.yaml"

CONFIG_FILE="config/uv/uvmamba_beijing.yaml"

# CONFIG_FILE="config/uv/unet_beijing.yaml"

weight_folder="weights/global_step8721"
CUDA_VISIBLE_DEVICES=0 deepspeed --master_port $MASTER_PORT tools/test.py --config_file $CONFIG_FILE --weight_folder $weight_folder

