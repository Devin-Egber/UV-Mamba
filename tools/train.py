import argparse
import deepspeed
import warnings
import yaml
from addict import Dict
from utils.file_utils import *
from utils.dataset_utils import *
from utils.performance_utils import *
from utils.model_utils import *
from utils.model_runner import run_iterate, init_random_seed
from utils.distributed_utils import get_dist_info, all_metrics_reduce, dist_barrier, logger

from utils.dataset_utils import get_dataset, build_uv_dataloader
# from src.losses import get_loss

# from src.utils.lr_scheduler import build_schduler

def main(config):
    start_epoch = 1
    checkpoint_path = os.path.join(config.PATH.res_dir, "weights")
    dataset_config = config.DATASET
    pipeline_config = config.PIPELINE
    config.ds_config = read_ds_config(config.deepspeed_config)
    dataset_config.batch_size = config.ds_config.get("train_batch_size", 1)

    logger.info(config)

    if config.local_rank == -1:
        device = torch.device("cuda")
    else:
        device = torch.device("cuda", config.local_rank)

    prepare_output(config.PATH)

    # set random seed
    seed = init_random_seed(pipeline_config.train.rdm_seed, device)
    logger.info(f"Set random seed to {seed}")
    deepspeed.runtime.utils.set_random_seed(seed)
    torch.cuda.manual_seed_all(seed)

    _, world_size = get_dist_info()

    dt_train, dt_val, dt_test = get_dataset(config)
    train_sampler, val_sampler, test_sampler = None, None, None

    if world_size > 1:
        train_sampler = data.distributed.DistributedSampler(dt_train, shuffle=True, drop_last=False)
        val_sampler = data.distributed.DistributedSampler(dt_val, shuffle=False, drop_last=False)
        test_sampler = data.distributed.DistributedSampler(dt_test, shuffle=False, drop_last=False)

    train_loader = build_uv_dataloader(dt_train, world_size, dataset_config, (train_sampler is None), False,
                                         sampler=train_sampler)
    val_loader = build_uv_dataloader(dt_val, world_size, dataset_config, False, False,
                                         sampler=val_sampler)
    test_loader = build_uv_dataloader(dt_test, world_size, dataset_config, False, False,
                                          sampler=test_sampler)
    # Model definition
    model = get_model(config)
    # model.apply(init_weights)

    logger.info(model)


    # Load weights from pre-trained models
    # if config.fine_tune:
    #     assert os.path.isfile(config.PRETRAINED), f'the path of pretrained model {config.PRETRAINED}" is not valid!!'
    #     model_state_file = config.PRETRAINED
    #     logger.info(f'=> Loading model from {model_state_file}')
    #     pretrain_dict = torch.load(model_state_file)['state_dict']
    #     with open('exchanger_weight/exchanger_dict.pkl', 'rb') as file:
    #         loaded_exchanger_dict = pickle.load(file)
    #
    #     pretrained_dict = {'base_model.' + k: v for k, v in pretrain_dict.items()
    #                        if k in loaded_exchanger_dict.keys()}
    #     model_dict = model.state_dict()
    #     pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict.keys()}
    #     for k in pretrained_dict.keys():
    #         logger.info(f'=> Loading {k} from pretrained model')
    #     model_dict.update(pretrained_dict)
    #     model.load_state_dict(model_dict)

    model = model.to(device)

    with open(os.path.join(config.PATH.res_dir, "conf.json"), "w") as file:
        file.write(json.dumps(vars(config), indent=4))

    # optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=0.0)
    # scheduler = build_schduler(config, optimizer, len(train_loader))
    #
    # model, optimizer, _, lr_scheduler = deepspeed.initialize(
    #     args=config, model=model, model_parameters=model.parameters(), optimizer=optimizer, lr_scheduler=scheduler)

    # Initialize the model
    model, optimizer, _, scheduler = deepspeed.initialize(
        args=config, model=model, model_parameters=model.parameters())
    model.lr_scheduler.total_num_steps = pipeline_config.train.epochs * len(train_loader)

    if model.bfloat16_enabled():
        dtype = torch.bfloat16
    elif model.fp16_enabled():
        dtype = torch.half
    else:
        dtype = torch.float32

    if config.auto_resume:
        logger.info(f"Resuming from the latest checkpoint from {checkpoint_path}...")
        resume_steps = model.resume_state_dict(model, checkpoint_path)
        start_epoch = int(resume_steps / len(train_loader) + 1)

    all_params, trainable_params = get_ntrainparams(model)
    logger.info(f"Train set: {len(dt_train)}, Val set: {len(dt_val)}, Test set: {len(dt_test)}")
    logger.info(f"trainable params: {trainable_params} || all params: {all_params} "
                f"|| trainable (%): {trainable_params / all_params * 100}")

    # criterion = get_loss(config, device=device)
    criterion = nn.CrossEntropyLoss()

    # Training loop
    trainlog = {}
    best_mIoU = 0
    logger.info(f"***** Running training *****")
    logger.info(f"Num examples = {len(train_loader)}")
    logger.info(f"Batch size = {dataset_config.batch_size}")

    for epoch in range(start_epoch, pipeline_config.train.epochs + 1):
        logger.info(f"EPOCH {epoch}/{config.PIPELINE.train.epochs}")

        if world_size > 1:
            train_sampler.set_epoch(epoch)

        train_metrics = run_iterate(
            model,
            data_loader=train_loader,
            criterion=criterion,
            config=config,
            mode="train",
            device=device,
            dtype=dtype
        )
        if epoch % pipeline_config.train.val_every == 0 and epoch > pipeline_config.train.val_after:
            logger.info(f"***** Running Validation *****")
            logger.info(f"Num examples = {len(val_loader)}")
            logger.info(f"Batch size = {dataset_config.batch_size}")

            val_metrics = run_iterate(
                model,
                data_loader=val_loader,
                criterion=criterion,
                config=config,
                mode="val",
                device=device,
                dtype=dtype
            )

            # Synchronize processes in distributed model to ensure all val_metrics calculations are complete.
            dist_barrier(world_size)
            for key, value in val_metrics.items():
                val_metrics[key] = all_metrics_reduce(value, device, world_size)

            summary_events = [(f"Eval/Samples/val_mIoU", val_metrics['val_mIoU'], model.global_samples)]
            model.monitor.write_events(summary_events)
            log_metrics(val_metrics, 'val')

            trainlog[epoch] = {**train_metrics, **val_metrics}
            checkpoint(trainlog, config.PATH)

            if val_metrics["val_mIoU"] >= best_mIoU:
                best_mIoU = val_metrics["val_mIoU"]
                save_state_dict(model, checkpoint_path, config.PIPELINE.train.save_total_limit)
        else:
            trainlog[epoch] = {**train_metrics}
            checkpoint(trainlog, config.PATH)

    # 保存最后训练的模型
    model.save_checkpoint(checkpoint_path)

    logger.info(f"***** Running testing *****")
    logger.info(f"Num examples = {len(test_loader)}")
    logger.info(f"Batch size = {dataset_config.batch_size}")

    # Ensure each process's model is saved before loading the checkpoint.
    dist_barrier(world_size)
    model.load_checkpoint(checkpoint_path)

    test_metrics, conf_mat = run_iterate(
        model,
        data_loader=test_loader,
        criterion=criterion,
        config=config,
        mode="test",
        device=device,
        dtype=dtype
    )

    # Synchronize processes in distributed model to ensure all test_metrics calculations are complete.
    dist_barrier(world_size)
    for key, value in test_metrics.items():
        test_metrics[key] = all_metrics_reduce(value, device, world_size)

    log_metrics(test_metrics, 'test')
    save_results(test_metrics, conf_mat.cpu().numpy(), config.PATH)


if __name__ == "__main__":
    warnings.filterwarnings("ignore", category=UserWarning)

    parser = argparse.ArgumentParser()
    parser.add_argument('--config_file',
                        type=str,
                        default="config/uv/segformer/segformer_shenzhen.yaml",
                        help='Configuration (.json) file to use')
    parser.add_argument('--rdm_seed', type=int, default=None, help='Random seed')
    parser.add_argument('--local_rank', type=int, default=-1,
                        help='Specifying the default GPU')
    parser.add_argument('--auto_resume', action='store_true',
                        help='Resume from the latest checkpoint automatically.')
    parser.add_argument('--fine_tune', action='store_true', default=False,
                        help='fine tune exchanger from a pretrained model')

    parser = deepspeed.add_config_arguments(parser)
    config = parser.parse_args()

    with open(config.config_file, 'r') as config_file:
        model_config = yaml.safe_load(config_file)

    config = Dict({**model_config, **vars(config)})
    config = argparse.Namespace(**config)
    config.deepspeed_config = config.PATH.ds_config

    main(config)
