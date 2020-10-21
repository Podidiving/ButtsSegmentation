from argparse import ArgumentParser
import numpy as np
import os
import json
from tqdm import tqdm
import cv2


def parse_args():
    parser = ArgumentParser()
    parser.add_argument("-p", "--path", type=str, required=True)
    parser.add_argument("-o", "--out-path", type=str, required=True)
    parser.add_argument("-v", "--verbose", type=bool, default=True)
    return parser.parse_args()


def get_mask(img_id, annotations):
    img_info = annotations["images"][img_id]
    assert img_info["id"] == img_id
    w, h = img_info["width"], img_info["height"]
    mask = np.zeros((h, w)).astype(np.uint8)
    gt = annotations["annotations"][img_id]
    assert gt["id"] == img_id
    polygon = np.array(gt["segmentation"][0]).reshape((-1, 2))
    cv2.fillPoly(mask, [polygon.astype(np.int8)], color=1)

    return mask.astype(np.int8)


def create_masks_from_annotations(
    in_path: str, out_path: str, annotations: dict, verbose: bool = True
):
    def _get_mask(img_id):
        img_info = annotations["images"][img_id]
        assert img_info["id"] == img_id
        w, h = img_info["width"], img_info["height"]
        mask_ = np.zeros((h, w)).astype(np.uint8)
        gt = annotations["annotations"][img_id]
        assert gt["id"] == img_id
        polygon = np.array(gt["segmentation"][0]).reshape((-1, 2))
        cv2.fillPoly(mask_, [polygon.astype(np.int32)], color=1)

        return mask_.astype(np.int8)

    images_path = os.path.join(in_path, "images")
    assert os.path.isdir(images_path), f"{images_path} doesn't exist"
    images_list = os.listdir(images_path)
    if verbose:
        print(f"Total num of images: {len(images_list)}")
        images_list = tqdm(images_list)

    for image_id in images_list:
        image_id = int(image_id.split(".", 1)[0])
        filename = os.path.join(out_path, f"{image_id:08}.png")
        mask = _get_mask(image_id)
        cv2.imwrite(filename, mask)


def prepare_masks(in_path: str, out_path: str, verbose: bool = True):
    annotations_path = os.path.join(in_path, "coco_annotations.json")
    assert os.path.isfile(annotations_path), f"{annotations_path} doesn't exist"
    annotations = json.load(open(annotations_path, "r"))
    create_masks_from_annotations(in_path, out_path, annotations, verbose)


if __name__ == "__main__":
    parser_ = parse_args()
    path_ = parser_.path
    out_path_ = parser_.out_path
    verbose_ = parser_.verbose
    os.makedirs(out_path_, exist_ok=True)
    prepare_masks(path_, out_path_, verbose_)
