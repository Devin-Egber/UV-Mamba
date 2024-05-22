from torch.utils import data
import numpy as np
import cv2
import os
from dataset.uv.uv_seg import UVSegDataset
from dataset.city.cityscapes import CityscapesSegmentation
import json
import albumentations as A
from albumentations.pytorch import ToTensorV2


# import warnings
# import deepspeed
# import  argparse
# from addict import Dict
# import yaml
# from file_utils import read_ds_config


def normalize(imgs_path):
    """
    This function is used to normalize the dataset

    The dataset organization should have the following format:

    ├── dataset
    │   ├── img_dir
    │   │   ├── train
    │   │   ├── val
    │   │   ├── test
    │   ├── ann_dir
    │   │   ├── train
    │   │   ├── val
    │   │   ├── test

    :param imgs_path: the path of the ima_dir
    :return: norm_mean, norm_std
    """

    img_h, img_w = 1024, 1024
    means, stdevs = [], []
    img_list, all_img_list = [], []

    for path in os.listdir(imgs_path):
        img_path = os.path.join(imgs_path, path)
        img_path_list = os.listdir(img_path)
        for img in img_path_list:
            all_img_list.append(os.path.join(img_path, img))

    len_ = len(all_img_list)
    i = 0
    for item in all_img_list:
        img = cv2.imread(item)
        img = cv2.resize(img, (img_w, img_h))
        img = img[:, :, :, np.newaxis]
        img_list.append(img)
        i += 1
        print(i, '/', len_)
    imgs = np.concatenate(img_list, axis=3)
    imgs = imgs.astype(np.float32) / 255.
    imgs = imgs.astype(np.float32)
    for i in range(3):
        pixels = imgs[:, :, i, :].ravel()
        means.append(np.mean(pixels))
        stdevs.append(np.std(pixels))

    # BGR --> RGB
    means.reverse()
    stdevs.reverse()
    print("normMean = {}".format(means))
    print("normStd = {}".format(stdevs))


def get_dataset(config):
    dataset_config = config.DATASET
    img_path = dataset_config.dataset_folder
    IMAGE_HEIGHT, IMAGE_WIDTH = dataset_config.crop_size

    with open(os.path.join(img_path, "NORM.json"), "r") as file:
        normvals = json.loads(file.read())

    train_transform = A.Compose([
        # A.Resize(IMAGE_HEIGHT, IMAGE_WIDTH),
        A.RandomCrop(IMAGE_HEIGHT, IMAGE_WIDTH),
        # A.Rotate(limit=35, p=1.0),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.Normalize(
            mean=normvals["mean"],
            std=normvals['std'],
            # mean=[0.485, 0.456, 0.406],
            # std=[0.229, 0.224, 0.225],
            max_pixel_value=255.0
        ),
        ToTensorV2()
    ])

    val_transform = A.Compose([
        # A.Resize(IMAGE_HEIGHT, IMAGE_WIDTH),
        A.RandomCrop(IMAGE_HEIGHT, IMAGE_WIDTH),
        A.Normalize(
            mean=normvals["mean"],
            std=normvals['std'],
            # mean=[0.485, 0.456, 0.406],
            # std=[0.229, 0.224, 0.225],
            max_pixel_value=255.0,
        ),
        ToTensorV2()
    ])

    test_transform = A.Compose([
        A.Resize(IMAGE_HEIGHT, IMAGE_WIDTH),
        # A.RandomCrop(IMAGE_HEIGHT, IMAGE_WIDTH),
        A.Normalize(
            mean=normvals["mean"],
            std=normvals['std'],
            # mean=[0.485, 0.456, 0.406],
            # std=[0.229, 0.224, 0.225],
            max_pixel_value=255.0,
        ),
        ToTensorV2()
    ])

    if dataset_config.dataset == "uvseg":
        train_dataset = UVSegDataset(img_path=img_path, mode="train", transform=train_transform)
        val_dataset = UVSegDataset(img_path=img_path, mode="val", transform=val_transform)
        test_dataset = UVSegDataset(img_path=img_path, mode="test", transform=test_transform)
        return train_dataset, val_dataset, test_dataset

    elif dataset_config.dataset == "cityscapes":
        train_dataset = CityscapesSegmentation(root=img_path, mode="train", transform=train_transform)
        val_dataset = CityscapesSegmentation(root=img_path, mode="val", transform=val_transform)
        test_dataset = CityscapesSegmentation(root=img_path, mode="test", transform=test_transform)
        return train_dataset, val_dataset, test_dataset

    else:
        raise NotImplementedError


def build_uv_dataloader(
        dataset,
        world_size,
        dataset_config,
        shuffle=True,
        drop_last=False,
        collate_fn=None,
        sampler=None):

    dataloader = data.DataLoader(
        dataset,
        batch_size=dataset_config.batch_size // world_size,
        shuffle=shuffle,
        drop_last=drop_last,
        num_workers=dataset_config.num_workers,
        collate_fn=collate_fn,
        sampler=sampler
    )
    return dataloader


# if __name__ == '__main__':
#     warnings.filterwarnings("ignore", category=UserWarning)
#
#     parser = argparse.ArgumentParser()
#     parser.add_argument('--config_file',
#                         type=str,
#                         default="config/uv/segformer/segformer.yaml",
#                         help='Configuration (.json) file to use')
#     parser.add_argument('--rdm_seed', type=int, default=None, help='Random seed')
#     parser.add_argument('--local_rank', type=int, default=-1,
#                         help='Specifying the default GPU')
#     parser.add_argument('--auto_resume', action='store_true',
#                         help='Resume from the latest checkpoint automatically.')
#     parser.add_argument('--fine_tune', action='store_true', default=False,
#                         help='fine tune exchanger from a pretrained model')
#
#     parser = deepspeed.add_config_arguments(parser)
#     config = parser.parse_args()
#
#     with open(config.config_file, 'r') as config_file:
#         model_config = yaml.safe_load(config_file)
#
#     config = Dict({**model_config, **vars(config)})
#     config = argparse.Namespace(**config)
#     config.deepspeed_config = config.PATH.ds_config
#     config.ds_config = read_ds_config(config.deepspeed_config)
#
#     dataset_config = config.DATASET
#     dataset_config.batch_size = config.ds_config.get("train_batch_size", 1)
#
#     dt_train, dt_val, dt_test = get_dataset(config)
#
#     train_loader = build_uv_dataloader(dt_train, 1, dataset_config, False)
#
#     for i, batch in enumerate(train_loader):
#         x, y = batch
