from PIL import Image
import numpy as np
import torch
from collections import OrderedDict
from typing import Callable, Dict, Union
import safitty
import segmentation_models_pytorch as smp
from albumentations.pytorch import ToTensorV2 as ToTensor

from .model import model_creator


def _build_model(params: Dict) -> torch.nn.Module:
    model_name = params["model"]["name"]
    model_params = params["model"]["params"]
    return model_creator(model_name=model_name, model_params=model_params)


def _load_from_checkpoint(
    params: Dict, checkpoint: str, map_location: Union[str, Dict] = "cpu",
) -> torch.nn.Module:
    # Since, I'm using pytorch_lightning==0.9.0
    # It has issues with LightningModule.load_from_checkpoint method
    # When __init__ has parameters
    loaded = torch.load(checkpoint, map_location=map_location)
    state_dict = OrderedDict()
    # is there anything better?
    for k, v in loaded["state_dict"].items():
        state_dict[k.split("model.")[1]] = v

    model = _build_model(params)
    model.load_state_dict(state_dict)
    return model


def _get_preproc_func(params: Dict) -> Callable:
    preprocessing_fn = smp.encoders.get_preprocessing_fn(
        params["model"]["params"]["encoder_name"], pretrained="imagenet",
    )
    to_tensor = ToTensor()
    return lambda image: to_tensor(image=preprocessing_fn(image))[
        "image"
    ].float()


def prepare_for_inference(
    param_path: str, checkpoint: str, map_location: Union[str, Dict] = "cpu",
):
    params = OrderedDict(safitty.load(param_path))

    # prepare model
    model = _load_from_checkpoint(
        params=params, checkpoint=checkpoint, map_location=map_location
    )
    model.eval()

    preproc_func = _get_preproc_func(params)
    return OrderedDict({"model": model, "preprocessing_function": preproc_func})


def imread(path: str) -> np.ndarray:
    image = np.array(Image.open(path))
    if len(image.shape) == 3 and image.shape[2] == 4:
        image = image[:, :, :3]
    return image
