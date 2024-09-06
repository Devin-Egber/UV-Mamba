import os
import numpy as np
from PIL import Image
from torch.utils.data import Dataset


class CityscapesSegmentation(Dataset):
    """
    The dataset organization should have the following attributes:

    ├── dataset
    │   ├── leftImg8bit
    │   │   ├── train
    │   │   ├── val
    │   │   ├── test
    │   ├── gtFine
    │   │   ├── train
    │   │   ├── val
    │   │   ├── test

    """

    def __init__(self, root, mode="train", transform=None):

        self.root = root
        self.mode = mode
        self.transform = transform
        self.files = {}
        self.NUM_CLASSES = 19

        self.images_base = os.path.join(self.root, 'leftImg8bit', self.mode)
        self.annotations_base = os.path.join(self.root, 'gtFine', self.mode)

        self.files[mode] = self.recursive_glob(rootdir=self.images_base, suffix='.png')

        self.void_classes = [0, 1, 2, 3, 4, 5, 6, 9, 10, 14, 15, 16, 18, 29, 30, -1]
        self.valid_classes = [7, 8, 11, 12, 13, 17, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 31, 32, 33]
        self.class_names = ['unlabelled', 'road', 'sidewalk', 'building', 'wall', 'fence', \
                            'pole', 'traffic_light', 'traffic_sign', 'vegetation', 'terrain', \
                            'sky', 'person', 'rider', 'car', 'truck', 'bus', 'train', \
                            'motorcycle', 'bicycle']

        self.ignore_index = 255
        self.class_map = dict(zip(self.valid_classes, range(self.NUM_CLASSES)))

        if not self.files[mode]:
            raise Exception("No files for split=[%s] found in %s" % (mode, self.images_base))

    def __len__(self):
        return len(self.files[self.mode])

    def __getitem__(self, index):

        img_path = self.files[self.mode][index].rstrip()
        lbl_path = os.path.join(self.annotations_base,
                                img_path.split(os.sep)[-2],
                                os.path.basename(img_path)[:-15] + 'gtFine_labelIds.png')

        image = np.array(Image.open(img_path).convert('RGB'))
        _tmp = np.array(Image.open(lbl_path), dtype=np.uint8)
        mask = self.encode_segmap(_tmp)

        if self.transform:
            augmentations = self.transform(image=image, mask=mask)
            image = augmentations['image']
            mask = augmentations['mask']

        return image, mask

    def encode_segmap(self, mask):
        # Put all void classes to zero
        for _voidc in self.void_classes:
            mask[mask == _voidc] = self.ignore_index
        for _validc in self.valid_classes:
            mask[mask == _validc] = self.class_map[_validc]
        return mask

    def recursive_glob(self, rootdir='.', suffix=''):
        """Performs recursive glob with given suffix and rootdir
            :param rootdir is the root directory
            :param suffix is the suffix to be searched
        """
        return [os.path.join(looproot, filename)
                for looproot, _, filenames in os.walk(rootdir)
                for filename in filenames if filename.endswith(suffix)]
