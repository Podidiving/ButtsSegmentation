from argparse import Namespace
from collections import OrderedDict
import os
import torch
from torch import nn
import pytorch_lightning as pl
import segmentation_models_pytorch as smp

import numpy as np

from .model import model_creator
from .dataset import SegmentationDataset, create_transforms_from_configs


class LightningModule(pl.LightningModule):
    def __init__(self, params_: Namespace):
        super().__init__()
        if isinstance(params_, dict):
            params_ = Namespace(**params_)
        self.params = params_
        self.model = model_creator(
            model_name=self.params.model["name"],
            model_params=self.params.model["params"],
        )

        self.loss_functions = {
            "dice": smp.utils.losses.DiceLoss(activation="sigmoid"),
            "iou": smp.utils.losses.JaccardLoss(activation="sigmoid"),
            "bce": nn.BCEWithLogitsLoss(),
        }
        self.loss_coefs = OrderedDict(
            {
                "dice": self.params.loss["dice_coef"],
                "iou": self.params.loss["iou_coef"],
                "bce": self.params.loss["bce_coef"],
            }
        )

        self.iou = smp.utils.metrics.IoU(activation="sigmoid")
        self.dice = smp.utils.metrics.Fscore(activation="sigmoid")

    def forward(self, x):
        return self.model(x)

    def loss_fn(self, output, target):
        return (
            self.loss_coefs["dice"]
            * self.loss_functions["dice"](output, target)
            + self.loss_coefs["iou"]
            * self.loss_functions["iou"](output, target)
            + self.loss_coefs["bce"]
            * self.loss_functions["bce"](output, target)
        )

    def training_step(self, batch, batch_idx):
        images = batch["image"]
        masks = batch["mask"]
        output = self(images)

        loss = self.loss_fn(output, masks)
        tensorboard_logs = {"train_loss": loss}
        return {"loss": loss, "log": tensorboard_logs}

    def validation_step(self, batch, batch_idx):
        images = batch["image"]
        masks = batch["mask"]
        output = self(images)

        loss = self.loss_fn(output, masks).detach().cpu().item()
        dice = self.dice(output, masks).detach().cpu().item()
        iou = self.iou(output, masks).detach().cpu().item()
        tensorboard_logs = {"train_loss": loss}
        return {
            "dice": dice,
            "iou": iou,
            "val_loss": loss,
            "log": tensorboard_logs,
        }

    def validation_epoch_end(self, outputs):
        mean_dice = np.mean([output["dice"] for output in outputs])
        mean_iou = np.mean([output["iou"] for output in outputs])
        val_loss = np.mean([output["val_loss"] for output in outputs])
        tensorboard_logs = {"mean_dice": mean_dice, "mean_iou": mean_iou}
        return {"val_loss": val_loss, "log": tensorboard_logs}

    def configure_optimizers(self):
        optim = torch.optim.__dict__[self.params.optimizer["name"]]
        if self.params.optimizer.get("encoder_params", None) is not None:
            return optim(
                [
                    {
                        "params": self.model.encoder.parameters(),
                        **self.params.optimizer["encoder_params"],
                    },
                    {"params": self.model.decoder.parameters()},
                    {"params": self.model.segmentation_head.parameters()},
                ],
                **self.params.optimizer["params"]
            )
        else:
            return optim(self.parameters(), **self.params.optimizer["params"])

    def _prepare_dataset(self, type_: str):
        image_transforms = smp.encoders.get_preprocessing_fn(
            self.params.model["params"]["encoder_name"], pretrained="imagenet",
        )

        image_mask_transforms = create_transforms_from_configs(
            self.params.transforms[type_]
        )

        root_path = self.params.dataset[type_]["root_path"]
        images_folder = self.params.dataset[type_]["images_folder"]
        masks_folder = self.params.dataset[type_]["masks_folder"]
        to_tensor = self.params.dataset[type_]["to_tensor"]
        images = os.listdir(os.path.join(root_path, images_folder))
        dataset = SegmentationDataset(
            root_path=root_path,
            images=images,
            images_folder=images_folder,
            masks_folder=masks_folder,
            image_mask_transforms=image_mask_transforms,
            image_transforms=image_transforms,
            to_tensor=to_tensor,
        )
        return dataset

    def _prepare_dataloader(
        self, dataset: torch.utils.data.Dataset, type_: str
    ):
        return torch.utils.data.DataLoader(
            dataset, **self.params.dataloader[type_]["params"]
        )

    def val_dataloader(self):
        dataset = self._prepare_dataset("val")
        return self._prepare_dataloader(dataset, "val")

    def train_dataloader(self):
        dataset = self._prepare_dataset("train")
        return self._prepare_dataloader(dataset, "train")
