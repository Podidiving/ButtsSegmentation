import os
import torch
from collections import OrderedDict
from typing import Callable, Dict, List, Union

import albumentations as A
from albumentations.pytorch import ToTensorV2 as ToTensor

from .helpers import imread


class SegmentationDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        root_path: str,
        images: List[str],
        images_folder: str = "images",
        masks_folder: str = "masks",
        image_mask_transforms: Callable = None,
        image_transforms: Callable = None,
        to_tensor: bool = True,
    ):
        """
        each image has .jpg extension, so looks like <image_id>.jpg
        each mask has .png extension, so corresponding to <image_id>.jpg looks like <image_id>.png
        structure is following:
        - <root_path>
            - <images_folder>
                - images...
            - <masks_folder>
                - masks....
        :param root_path:
        :param images:
        :param images_folder:
        :param masks_folder:
        :param image_mask_transforms: apply both on images and masks
        :param image_transforms: apply only to images, after image_masks_transforms
        :param to_tensor: apply ToTensor transform or not
        """
        self.root_path = root_path
        self.images = images
        self.images_folder = images_folder
        self.masks_folder = masks_folder
        self.image_mask_transforms = image_mask_transforms
        self.image_transforms = image_transforms
        if self.image_transforms is not None:
            if isinstance(self.image_transforms, A.core.composition.Compose):
                self._image_transforms = self.image_transforms
                self.image_transforms = lambda image: self._image_transforms(
                    image=image
                )["image"]
        self.to_tensor = None
        if to_tensor:
            self.to_tensor = ToTensor()

    def __len__(self):
        return len(self.images)

    def __getitem__(self, item):
        image_name = self.images[item]
        mask_name = f"{image_name.split('.', 1)[0]}.png"
        image_path = os.path.join(
            self.root_path, self.images_folder, image_name
        )
        mask_path = os.path.join(self.root_path, self.masks_folder, mask_name)
        image = imread(image_path)
        mask = imread(mask_path)

        if self.image_mask_transforms is not None:
            transformed = self.image_mask_transforms(image=image, mask=mask)
            image = transformed["image"]
            mask = transformed["mask"]

        if self.image_transforms is not None:
            image = self.image_transforms(image)

        if self.to_tensor is not None:
            transformed = self.to_tensor(image=image, mask=mask)
            image = transformed["image"]
            mask = transformed["mask"].unsqueeze(0)

        result = OrderedDict({"image": image.float(), "mask": mask.float()})
        return result


def create_transforms_from_configs(config: Union[Dict, None]) -> Callable:
    transforms = []
    if config is not None:
        apply_rescale = False
        image_size = None
        if config.get("RescaleTransforms", None) is not None:
            apply_rescale = config.pop("RescaleTransforms")
            assert (
                config.get("ImageSize", None) is not None
            ), "ImageSize must be specified"
            image_size = config.pop("ImageSize")
        for k, v in config.items():
            params = v or dict()
            transforms.append(A.__dict__[k](**params))
        if apply_rescale:
            pre_size = int(image_size * 1.5)

            random_crop = A.Compose(
                [
                    A.SmallestMaxSize(pre_size, p=1),
                    A.RandomCrop(image_size, image_size, p=1),
                ]
            )

            rescale = A.Compose([A.Resize(image_size, image_size, p=1)])

            random_crop_big = A.Compose(
                [
                    A.LongestMaxSize(pre_size, p=1),
                    A.RandomCrop(image_size, image_size, p=1),
                ]
            )

            # Converts the image to a square of size image_size x image_size
            result = [A.OneOf([random_crop, rescale, random_crop_big], p=1)]
            transforms = transforms + result
    return A.Compose(transforms)
