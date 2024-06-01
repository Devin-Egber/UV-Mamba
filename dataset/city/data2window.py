import os
from glob import glob
from PIL import Image


def genarate_dataset(data_dir, target_size, save_dir=None, flags=['train', 'val', 'test']):
    for flag in flags:

        # 获取待裁剪影像和label的路径
        images_paths = glob(data_dir + "leftImg8bit/" + flag + "/*/*_leftImg8bit.png")
        images_paths = sorted(images_paths)
        gts_paths = glob(data_dir + "gtFine/" + flag + "/*/*gtFine_labelIds.png")
        gts_paths = sorted(gts_paths)
        print(len(gts_paths))

        # 遍历每一张图片
        for image_path, gt_path in zip(images_paths, gts_paths):
            # 确保图片和标签对应
            image_name = os.path.split(image_path)[-1].split('_')[0:3]
            # e.g. ['zurich', '000121', '000019']
            gt_name = os.path.split(gt_path)[-1].split('_')[0:3]
            assert image_name == gt_name

            img_save_path = os.path.join(save_dir,"leftImg8bit", flag, image_name[0])
            if not os.path.exists(img_save_path):
                os.makedirs(img_save_path)

            gt_save_path = os.path.join(save_dir,"gtFine", flag, gt_name[0])
            if not os.path.exists(gt_save_path):
                os.makedirs(gt_save_path)

            # 读取图片和标签
            # image = cv2.imread(image_path)
            # gt = cv2.imread(gt_path, 0)
            image = Image.open(image_path)
            gt = Image.open(gt_path)
            img_width, img_height = image.size

            # 尺寸
            crop_height, crop_width = target_size

            num_cols = img_width // crop_width
            num_rows = img_height // crop_height

            save_num = 0
            for row in range(num_rows):
                for col in range(num_cols):
                    # 计算每个裁剪区域的左、上、右、下坐标
                    left = col * crop_width
                    upper = row * crop_height
                    right = left + crop_width
                    lower = upper + crop_height

                    # 裁剪图片
                    cropped_image = image.crop((left, upper, right, lower))
                    croped_gt = gt.crop((left, upper, right, lower))

                    # save_path
                    image_path = os.path.join(img_save_path, '_'.join(image_name) + str(save_num) + "_leftImg8bit.png")
                    gt_path = os.path.join(gt_save_path, '_'.join(gt_name) + str(save_num) + "_gtFine_labelIds.png")

                    cropped_image.save(image_path)
                    croped_gt.save(gt_path)
                    # 每保存一次图片和标签，计数加一
                    save_num += 1


if __name__ == '__main__':
    genarate_dataset(data_dir='D:\\迅雷下载\\city\\',
                    target_size=(1024, 1024),
                    save_dir='D:\\迅雷下载\\CityScapes\\')
