import warnings
import argparse
import deepspeed
import yaml

from addict import Dict
from utils.file_utils import *
from utils.dataset_utils import *
from utils.performance_utils import *
from utils.model_utils import *
from utils.model_runner import run_iterate
from utils.distributed_utils import logger
from utils.dataset_utils import get_dataset, build_uv_dataloader

# from src.losses import get_loss


def main(config):
    if config.torch_dtype == "bf16":
        dtype = torch.bfloat16
    elif config.torch_dtype == "fp16":
        dtype = torch.half
    else:
        dtype = torch.float32

    prepare_output(config.PATH)
    dataset_config = config.DATASET
    dataset_config.batch_size = config.batch_size

    if config.local_rank == -1:
        device = torch.device("cuda")
    else:
        device = torch.device("cuda", config.local_rank)

    _, _, dt_test = get_dataset(config)

    test_loader = build_uv_dataloader(dt_test, 1, dataset_config, False, False, sampler=None)

    model = get_model(config)
    model = model.to(device)
    logger.info(model)

    # Load weights
    checkpoint_path = os.path.join(config.weight_path, os.listdir(config.weight_path)[0])
    logger.info(f"Loading checkpoint from {checkpoint_path}...")
    pretrained_dict = torch.load(checkpoint_path, map_location=device)["module"]
    model.load_state_dict(pretrained_dict)
    model = model.to(device)

    model = deepspeed.init_inference(model, mp_size=1, dtype=dtype, replace_with_kernel_inject=False)

    # Loss
    criterion = nn.CrossEntropyLoss()

    # Inference
    logger.info(f"***** Running testing *****")
    logger.info(f"Num examples = {len(dt_test)}")
    logger.info(f"Batch size = {config.batch_size}")
    model.eval()
    test_metrics, conf_mat = run_iterate(
        model,
        data_loader=test_loader,
        criterion=criterion,
        config=config,
        mode="test",
        device=device,
        dtype=dtype
    )

    log_metrics(test_metrics, 'test')
    if config.save_metrics:
        save_results(test_metrics, conf_mat.cpu().numpy(), config.PATH)

if __name__ == "__main__":
    warnings.filterwarnings("ignore", category=UserWarning)

    parser = argparse.ArgumentParser()
    parser.add_argument('--config_file',
                        default="config/uv/segformer/segformer.yaml",
                        type=str,
                        help='Configuration (.json) file to use')
    parser.add_argument('--weight_folder',
                        default="weights/global_step7568",
                        type=str,
                        help='Path to the main folder containing the pre-trained weights')
    parser.add_argument('--local_rank', type=int, default=-1, help='Specifying the default GPU')
    parser.add_argument('--batch_size', default=4, type=int, help='Batch size')
    parser.add_argument('--show_dir', help='Directory where painted images will be saved')
    parser.add_argument('--torch_dtype', default="fp32", type=str, choices=['fp32', 'fp16', 'bf16'],
                        help='Override the default `torch.dtype` and load the model under this dtype')
    parser.add_argument('--save_metrics', action='store_true', default=True,
                        help='Whether to save metrics or not')

    parser = deepspeed.add_config_arguments(parser)
    test_config = parser.parse_args()


    with open(test_config.config_file) as file:
        model_config = yaml.safe_load(file)

    config = Dict({**model_config, **vars(test_config)})
    config = argparse.Namespace(**config)
    config.weight_path = os.path.join(config.PATH.res_dir, config.weight_folder)
    logger.info(config)
    main(config)