import argparse
import logging
from pathlib import Path
import torch

import numpy as np
import cv2
import os, json
from PIL import Image
from tqdm import tqdm

from src import models
from src.methods import ddim_inversion, vton
from src.utils import IImage, resize, poisson_blend, resize_and_insert, find_mask_boundary, warping_cloth

logging.disable(logging.INFO)

root_path = Path(__file__).resolve().parent.parent
negative_prompt = "text, bad anatomy, bad proportions, blurry, cropped, deformed, disfigured, duplicate, error, extra limbs, gross proportions, jpeg artifacts, long neck, low quality, lowres, malformed, morbid, mutated, mutilated, out of frame, ugly, worst quality"
positive_prompt = "Full HD, 4K, high quality, high resolution"


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--image-path', type=Path, help='Image path.', required=True)
    parser.add_argument('--cloth-path', type=Path, help='Image path.', required=True)
    parser.add_argument('--in-mask-path', type=Path, help='Mask path.', required=True)
    parser.add_argument('--out-mask-path', type=Path, help='Mask path.', required=True)
    parser.add_argument('--output-path', type=Path, help='Output path.', required=True)
    parser.add_argument('--condition-path', type=Path, help='Condition path.', required=True)
    parser.add_argument('--stage', type=str, default='vton', help='One of [outpainting, vton]')
    parser.add_argument('--num-samples', type=int, help='Num of samples', default=1)
    parser.add_argument('--H', type=int, default=512)
    parser.add_argument('--W', type=int, default=384)
    parser.add_argument('--sub_type', type=int, default=0, help='trouser=0, skirt=1')
    parser.add_argument('--c_type', type=str, default='0', help='upper=0, lower=1, dresses=2')

    parser.add_argument('--model-id', type=str, default='sd2_inp',
        help='One of [sd2_inp, sd15_inp]', required=False)
    parser.add_argument('--blending', type=str, default=None,
        help='Direct blending or Poisson blending', required=False)
    parser.add_argument('--guidance-scale', type=float, default=7.5,
        help='Classifier-free guidance scale.', required=False)
    parser.add_argument('--num-steps', type=int, default=50,
        help='Num of DDIM steps.', required=False)
    parser.add_argument('--seed', type=int, default=1,
        help='Seed to use for generation.', required=False)
    return parser.parse_args()


def get_inpainting_function(
    model,
    negative_prompt: str = '',
    positive_prompt: str = '',
    num_steps: int = 50,
    guidance_scale: float = 7.5,
    invert=False
):
    if invert:
        def run(image: Image, cloth: Image, in_mask: Image, out_mask: Image, cprompt: str,
                pprompt: str, seed: int = 1) -> Image:
            invert = ddim_inversion.run(
                ddim=model,
                cprompt=cprompt,
                pprompt=pprompt,
                image=IImage(image),
                cloth=IImage(cloth),
                in_mask=IImage(in_mask),
                out_mask=IImage(out_mask),
                seed=seed,
                negative_prompt=negative_prompt,
                positive_prompt=positive_prompt,
                num_steps=num_steps,
                guidance_scale=guidance_scale
            )
            return invert
    else:
        def run(image: Image, cloth: Image, warped_cloth, in_mask: Image, out_mask: Image,
                cprompt: str, pprompt: str, stage: str, seed: int = 1, xT=None) -> Image:
            painted_image = vton.run(
                ddim=model,
                cprompt=cprompt,
                pprompt=pprompt,
                image=IImage(image),
                cloth=IImage(cloth),
                warped_cloth=warped_cloth,
                in_mask=IImage(in_mask),
                out_mask=IImage(out_mask),
                init_noise=xT,
                seed=seed,
                negative_prompt=negative_prompt,
                positive_prompt=positive_prompt,
                num_steps=num_steps,
                guidance_scale=guidance_scale,
                stage=stage
            ).pil()
            w, h = image.size
            inpainted_image = Image.fromarray(np.array(painted_image[1])[:h, :w])
            outpainted_image = Image.fromarray(np.array(painted_image[0])[:h, :w])
            return outpainted_image, inpainted_image
    return run


def main():
    args = get_args()
    if args.stage == 'vton':
        (args.output_path / 'vton').mkdir(exist_ok=True, parents=True)
        (args.output_path / 'warped_cloth').mkdir(exist_ok=True, parents=True)
    else:
        (args.output_path / 'outpainting').mkdir(exist_ok=True, parents=True)

    inp_model = models.load_inpainting_model(args.model_id, device='cuda:0', cache=True)
    run_invert = get_inpainting_function(
        model=inp_model,
        guidance_scale=args.guidance_scale,
        num_steps=args.num_steps,
        negative_prompt=negative_prompt,
        positive_prompt=positive_prompt,
        invert=True
    )
    run_inpainting = get_inpainting_function(
        model=inp_model,
        guidance_scale=args.guidance_scale,
        num_steps=args.num_steps,
        negative_prompt=negative_prompt,
        positive_prompt=positive_prompt
    )

    image = Image.open(args.image_path).convert('RGB')
    cloth = Image.open(args.cloth_path).convert('RGB')
    in_mask = Image.open(args.in_mask_path).convert('RGB')
    out_mask = Image.open(args.out_mask_path).convert('RGB')

    image_name = str(Path(args.image_path)).split('/')[-1]
    cloth_name = str(Path(args.cloth_path)).split('/')[-1]
    cloth_ci_name = cloth_name.replace('.jpg', '_ci.json')
    with open(os.path.join(args.condition_path, 'cloth_clip_interrogate', cloth_ci_name), 'r') as f:
        ci_caption = json.load(f)
        cprompt = ci_caption['cloth_clip_interrogate']

    person_ci_name = image_name.replace('.jpg', '_ci.json')
    with open(os.path.join(args.condition_path, 'image_clip_interrogate', person_ci_name), 'r') as f:
        ci_caption = json.load(f)
        pprompt = ci_caption['cloth_clip_interrogate']

    resized_image = resize(image, (args.W, args.H))
    resized_cloth = resize(cloth, (args.W, args.H))
    resized_in_mask = resize(in_mask, (args.W, args.H), resample=Image.NEAREST)
    resized_out_mask = resize(out_mask, (args.W, args.H), resample=Image.NEAREST)

    in_boundary = find_mask_boundary(resized_in_mask)
    out_boundary = find_mask_boundary(resized_out_mask)
    ri_cloth = resize_and_insert(resized_cloth, in_boundary, out_boundary)
    ri_out_mask = resize_and_insert(resized_out_mask, in_boundary, out_boundary, mask=True)

    warped_cloth = None
    if args.stage == 'vton':
        ori_parsing_path = os.path.join(args.condition_path, 'image_tapps_parse', image_name.replace(".jpg", "_pps.png"))
        out_parsing_path = os.path.join(args.condition_path, 'cloth_tapps_parse', cloth_name.replace(".jpg", "_pps.png"))
        # Load human pose
        human_pose_name = os.path.join(args.condition_path, 'image_openpose_json', image_name.replace('.jpg', '_keypoints.json'))
        with open(human_pose_name, "r") as f:
            pose_label = json.load(f)
            if 'people' in pose_label and len(pose_label['people']) == 1:
                human_pose_data = pose_label['people'][0]['pose_keypoints_2d']
                human_pose_data = np.array(human_pose_data)
                human_pose_data = human_pose_data.reshape((-1, 3))
            else:
                print(f"No pose_keypoints_2d data found in {human_pose_name}, skipping.")
                human_pose_data = None
        # Load cloth pose
        cloth_pose_name = os.path.join(args.condition_path, 'cloth_openpose_json', cloth_name.replace('.jpg', '_keypoints.json'))
        with open(cloth_pose_name, "r") as f:
            cloth_pose_label = json.load(f)
            if 'people' in cloth_pose_label and len(cloth_pose_label['people']) == 1:
                cloth_pose_data = cloth_pose_label['people'][0]['pose_keypoints_2d']
                cloth_pose_data = np.array(cloth_pose_data)
                cloth_pose_data = cloth_pose_data.reshape((-1, 3))
            else:
                print(f"No pose_keypoints_2d data found in {cloth_pose_name}, skipping.")
                cloth_pose_data = None

        ori_parsing = Image.open(ori_parsing_path).convert('RGB')
        out_parsing = Image.open(out_parsing_path).convert('RGB')
        resized_ori_parsing = resize(ori_parsing, (args.W, args.H))
        resized_out_parsing = resize(out_parsing, (args.W, args.H))

        out_mask_array = np.array(ri_out_mask)
        in_mask_array = np.array(resized_in_mask)
        ori_parsing_array = np.array(resized_ori_parsing)
        out_parsing_array = np.array(resized_out_parsing)
        cloth_array = np.array(ri_cloth)

        cloth_array = cloth_array / 127.5 - 1
        ori_parsing_array = ori_parsing_array / 127.5 - 1
        out_parsing_array = out_parsing_array / 127.5 - 1
        cloth_array[out_mask_array == 0] = 0
        out_parsing_array[out_mask_array == 0] = 0
        w, h = resized_cloth.size
        if cloth_pose_data is not None and human_pose_data is not None:
            if args.c_type == '2':
                upper_dress, lower_dress = cloth_array.copy(), cloth_array.copy()
                warped_upper = warping_cloth(upper_dress, out_parsing_array, ori_parsing_array, cloth_pose_data,
                                             human_pose_data, w, h, c_type='0', sub_type=args.sub_type)
                warped_lower = warping_cloth(lower_dress, out_parsing_array, ori_parsing_array, cloth_pose_data,
                                             human_pose_data, w, h, c_type='1', sub_type=1)
                warped_lower[warped_upper != 0] = 0
                warped_cloth = warped_upper + warped_lower
            else:
                warped_cloth = warping_cloth(cloth_array, out_parsing_array, ori_parsing_array, cloth_pose_data,
                                             human_pose_data, w, h, c_type=args.c_type, sub_type=args.sub_type, adjust=True)
            warped_cloth[in_mask_array == 0] = 0
            warped_cloth = ((warped_cloth + 1) * 127.5).clip(0, 255).astype(np.uint8)
        else:
            warped_cloth = None

    out_mask_array = np.array(resized_out_mask)
    out_mask_array = 255 - out_mask_array
    resized_out_mask = Image.fromarray(out_mask_array)

    for idx in tqdm(range(1, args.num_samples+1)):
        seed = args.seed + (idx-1) * 1000
        xt = None
        if args.stage == 'vton':
            xt = \
                run_invert(resized_image, resized_cloth, resized_in_mask, resized_out_mask,
                           cprompt, pprompt, seed=seed)
        outpainted_image, inpainted_image = \
            run_inpainting(resized_image, resized_cloth, warped_cloth, resized_in_mask, resized_out_mask,
                           cprompt, pprompt, args.stage, seed=seed, xT=xt)

        if args.stage == 'vton':
            output_path = args.output_path / f'vton/{image_name}'
            output_warp_path = args.output_path / f'warped_cloth/{image_name}'
            if args.blending == 'poisson':
                inpainted_image = poisson_blend(
                    orig_img=IImage(resized_image).data[0],
                    fake_img=IImage(inpainted_image).data[0],
                    mask=IImage(resized_in_mask).alpha().data[0]
                )
                inpainted_image = IImage(inpainted_image).data[0]
                Image.fromarray(inpainted_image).save(output_path)
            else:
                inpainted_image = IImage(inpainted_image).data[0]
                mask_array = IImage(resized_in_mask).data[0] / 255.0
                gt_array = IImage(resized_image).data[0]
                inpainted_image = inpainted_image * mask_array + gt_array * (1 - mask_array)
                inpainted_image = Image.fromarray(inpainted_image.astype(np.uint8))
                inpainted_image.save(output_path)
            if warped_cloth is not None:
                warped_cloth = Image.fromarray(warped_cloth.astype(np.uint8))
                warped_cloth.save(output_warp_path)
        else:
            output_path = args.output_path / f'outpainting/{cloth_name}'
            outpainted_image = poisson_blend(
                orig_img=IImage(resized_cloth).data[0],
                fake_img=IImage(outpainted_image).data[0],
                mask=IImage(resized_out_mask).alpha().data[0]
            )
            outpainted_image = Image.fromarray(IImage(outpainted_image).data[0])
            outpainted_image.save(output_path)


if __name__ == '__main__':
    main()