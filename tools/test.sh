work_dir="/data/mamba/Segment"
export PYTHONPATH=$work_dir

MASTER_PORT=32999


#CONFIG_FILE="config/uv/segformer/deform_uvmamba_beijing.yaml"

# CONFIG_FILE="config/uv/segformer/segformer_shenzhen.yaml"

#CONFIG_FILE="config/uv/segformer/segmamba_xian.yaml"
# CONFIG_FILE="config/uv/segmamba/segmamba_beijing.yaml"
# CONFIG_FILE="config/uv/segmamba/segmamba_cityspace.yaml"

# CONFIG_FILE="config/uv/uv_mamba/deform_uvmamba_beijing.yaml"
# CONFIG_FILE="config/uv/uv_mamba/deform_unet_uvmamba_beijing.yaml"

# CONFIG_FILE="config/uv/uv_mamba/deform_uvmamba_beijing_crop.yaml"

CONFIG_FILE="config/uv/uv_mamba/deform_unet_uvmamba_xian.yaml"
# CONFIG_FILE="config/uv/uv_mamba/deform_unet_uvmamba_beijing.yaml"


# CONFIG_FILE="config/uv/uv_mamba/deform_unet_uvmamba_whu.yaml"

#CONFIG_FILE="config/uv/uv_mamba/deform_unet_uvmamba_road.yaml"


# CONFIG_FILE="config/uv/uv_mamba/deform_uvmamba_beijing_pretrain.yaml"
# test one of five fold script
weight_folder="weights/global_step3168"
CUDA_VISIBLE_DEVICES=0 deepspeed --master_port $MASTER_PORT tools/test.py --config_file $CONFIG_FILE --weight_folder $weight_folder

