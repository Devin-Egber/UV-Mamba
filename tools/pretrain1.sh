work_dir="/data/project/UV-Mamba"
export PYTHONPATH=$work_dir
MASTER_PORT=32991

# CONFIG_FILE="config/uv/uv_mamba/deform_uvmamba_beijing_crop.yaml"


CONFIG_FILE="config/uv/uv_mamba/deform_uvmamba_beijing_pretrain.yaml"



CUDA_VISIBLE_DEVICES=1 deepspeed --master_port $MASTER_PORT tools/train.py --config_file $CONFIG_FILE --fine_tune


#CUDA_VISIBLE_DEVICES=0 deepspeed --master_port $MASTER_PORT scripts/test_semantic.py $CONFIG_FILE
