import segmentation_models_pytorch as smp
from torch.nn import Module


def model_creator(model_name: str, model_params: dict,) -> Module:
    return smp.__dict__[model_name](**model_params)
