import torch
import os
import numpy as np
import sys
import argparse

from PIL import Image
from torch.nn import functional as F

from detectron2.checkpoint import DetectionCheckpointer
from detectron2.config import get_cfg
from detectron2.data import detection_utils as utils
from detectron2.projects.deeplab import add_deeplab_config
from detectron2.modeling import build_model
from detectron2.utils.logger import setup_logger

from tapps import add_maskformer2_config
from tapps import (
    register_all_pascal_panoptic_parts,
    register_all_cityscapes_panoptic_parts,
    register_all_pascal_panoptic_parts_107,
)

sys.path.append("utils/panoptic_parts")
from panoptic_parts.utils.format import encode_ids
from panoptic_parts.utils.visualization import experimental_colorize_label
from panoptic_parts.specs.eval_spec import PartPQEvalSpec

def setup(config_file):
    cfg = get_cfg()
    add_maskformer2_config(cfg)
    add_deeplab_config(cfg)
    cfg.merge_from_file(config_file)
    cfg.freeze()
    setup_logger(name="fvcore")
    setup_logger()
    return cfg

def visualize_pps(pps_pred, eval_spec, dataset):
    sid2color = eval_spec.dataset_spec.sid2scene_color
    sids = pps_pred[0]
    iids = pps_pred[1]
    pids = pps_pred[2]

    sids_no_parts = eval_spec.eval_sid_no_parts
    sids_stuff = eval_spec.eval_sid_stuff

    sids_wo_parts = np.isin(sids, sids_no_parts)
    sids_w_stuff = np.isin(sids, sids_stuff)
    iids[sids_w_stuff] = -1
    pids[sids_wo_parts] = -1

    sids[sids == 255] = 0
    pids[pids == 255] = -1
    iids[iids == 255] = -1
    uids = encode_ids(sids, iids, pids)

    if dataset == 'Cityscapes':
        is_cpp = True
    else:
        is_cpp = False

    pps_colors = experimental_colorize_label(uids, sid2color=sid2color, is_cpp=is_cpp, emphasize_instance_boundaries=False, experimental_alpha=0)
    return pps_colors

def resize_image(img, config):
    h, w = img.shape[1:3]

    if config.DATASETS.NAME == "Cityscapes":
        h_new = 1024
        w_new = 2048
    else:
        if h == w:
            (h_new, w_new) = (800, 800)
        elif h > w:
            h_new = int(min(1333, 800/w * h))
            w_new = 800
        elif w > h:
            h_new = 800
            w_new = int(min(1333, 800/h * w))

    return F.interpolate(img.unsqueeze(0), size=(h_new, w_new), mode='bilinear').squeeze(0)

def inference_single_image(config, image_fn, model_weights, save_dir):
    # Parse the config file
    cfg = setup(config)

    # Load the information about the dataset that the model is trained on, e.g., class definitions
    if cfg.DATASETS.NAME == "Pascal":
        register_all_pascal_panoptic_parts(cfg)
        eval_spec_path = "utils/panoptic_parts/panoptic_parts/specs/eval_specs/ppq_ppp_59_57_cvpr21_default_evalspec.yaml"
    if cfg.DATASETS.NAME == "Pascal107":
        register_all_pascal_panoptic_parts_107(cfg)
        eval_spec_path = "utils/panoptic_parts/panoptic_parts/specs/eval_specs/ppq_ppp_59_107_cvpr21_default_evalspec.yaml"
    if cfg.DATASETS.NAME == "Cityscapes":
        register_all_cityscapes_panoptic_parts(cfg)
        eval_spec_path = "utils/panoptic_parts/panoptic_parts/specs/eval_specs/ppq_cpp_19_23_cvpr21_grouped_evalspec.yaml"
    eval_spec = PartPQEvalSpec(eval_spec_path)

    # Build the model and load the model weights
    model = build_model(cfg)
    model.cuda()
    DetectionCheckpointer(model).load(model_weights)
    model.eval()

    # Load the image and resize it
    image = utils.read_image(image_fn, format=cfg.INPUT.FORMAT)
    image = torch.as_tensor(np.ascontiguousarray(image.transpose(2, 0, 1))).cuda().float()
    h, w = image.shape[1:3]
    image = resize_image(image, config=cfg)
    inputs = [{"image": image}]

    # Feed the image to the model and output the predictions.
    # The 'output' list contains the raw predictions for part segmentation, semantic segmentation, panoptic segmentation and PPS.
    output = model(inputs)

    # Visualize the PPS predictions
    pps = output[0]['pps']
    pps = torch.stack(pps, dim=0)
    pps_pred = pps.detach().cpu().numpy().astype(np.int32)
    pps_colors = visualize_pps(pps_pred, eval_spec, cfg.DATASETS.NAME)

    # Create save_dir if it doesn't already exist
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)

    # Store the PPS predictions
    pps_colors_img = Image.fromarray(pps_colors.astype(np.uint8))
    pps_colors_img = pps_colors_img.resize((w, h), Image.BILINEAR)
    pps_colors_img.save(os.path.join(save_dir, f"{os.path.splitext(os.path.basename(image_fn))[0]}_pps.png"))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='configs/pascal/pps/tapps_pascal_swinb_cocoinit.yaml')
    parser.add_argument('--image', type=str)
    parser.add_argument('--model_weights', type=str, default='checkpoints/tapps_pascal_swinb_cocoinit.bin')
    parser.add_argument('--save_dir', type=str)
    args = parser.parse_args()

    config = args.config
    image = args.image
    model_weights = args.model_weights
    save_dir = args.save_dir

    inference_single_image(config=config,
                           image_fn=image,
                           model_weights=model_weights,
                           save_dir=save_dir)
