work_dir="/data/mamba/Segment"
export PYTHONPATH=$work_dir
MASTER_PORT=32990

# CONFIG_FILE="config/uv/uv_mamba/deform_uvmamba_beijing_crop.yaml"


# CONFIG_FILE="config/uv/uv_mamba/deform_unet_uvmamba_xian.yaml"


CONFIG_FILE="config/uv/uv_mamba/deform_uvmamba_inria.yaml"


CUDA_VISIBLE_DEVICES=0 deepspeed --master_port $MASTER_PORT tools/train.py --config_file $CONFIG_FILE


#CUDA_VISIBLE_DEVICES=0 deepspeed --master_port $MASTER_PORT scripts/test_semantic.py $CONFIG_FILE
