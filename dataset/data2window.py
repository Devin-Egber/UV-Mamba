import os
import argparse
from PIL import Image


def crop_image(image_path, block_size, save_path):
    """

    :param image_path: the image file path
    :param block_size: the patch size you want to crop
    :param save_path: the path you want to save the cropped image
    :return:
    """

    img_name = image_path.split('\\')[-1][:-4]
    image = Image.open(image_path)

    width, height = image.size
    file_number = 0

    # Crop image
    for i in range(0, width + 1, block_size):
        for j in range(0, height + 1, block_size):

            # calculate the lower right corner coordinates of the current block
            right = i + block_size
            bottom = j + block_size

            if right <= width and bottom <= height:
                block = image.crop((i, j, right, bottom))
            if right > width and bottom <= height:
                block = image.crop((width - block_size, j, width, bottom))
            if right <= width and bottom > height:
                block = image.crop((i, height - block_size, right, height))
            if right > width and bottom > height:
                block = image.crop((width - block_size, height - block_size, width, height))

            block.save(os.path.join(save_path, img_name + "-" + str(file_number) + ".png"))
            file_number += 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Image cropping')

    parser.add_argument('--img_dir',
                        type=str,
                        default="",
                        help='the img_dir path of the UIS-shenzhen Dataset')
    parser.add_argument('--ann_dir',
                        type=str,
                        default="",
                        help='the ann_dir path of the UIS-shenzhen Dataset')
    parser.add_argument('--target_img_path',
                        type=str,
                        default="",
                        help='the path to save cropped image')
    parser.add_argument('--target_mask_path',
                        type=str,
                        default="",
                        help='the path to save cropped image mask')
    parser.add_argument('--crop_size',
                        type=int,
                        default=96,
                        help='size of extracted windows')

    args = parser.parse_args()

    origin_img_path = args.img_dir
    origin_mask_path = args.ann_dir
    target_img_path = args.target_img_path
    target_mask_path =args.target_mask_path

    for mode in ['train', 'val', 'test']:
        img_path = os.path.join(origin_img_path, mode)
        mask_path = os.path.join(origin_mask_path, mode)

        target_img_path = os.path.join(target_img_path, mode)
        target_mask_path = os.path.join(target_mask_path, mode)


        for train_img in os.listdir(img_path):
            file_path = os.path.join(img_path, train_img)
            crop_image(file_path, 96, target_img_path)

        for train_mask in os.listdir(mask_path):
            file_path = os.path.join(mask_path, train_mask)
            crop_image(file_path, 96, target_mask_path)
