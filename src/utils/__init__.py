import base64
from typing import Tuple, Union

import cv2
import numpy as np
import open_clip
from PIL import Image
from tqdm import tqdm

from .iimage import IImage


def tokenize(prompt):
    tokens = open_clip.tokenize(prompt)[0]
    return [open_clip.tokenizer._tokenizer.decoder[x.item()] for x in tokens]


def poisson_blend(
    orig_img: np.ndarray,
    fake_img: np.ndarray,
    mask: np.ndarray,
    pad_width: int = 32,
    dilation: int = 48
) -> np.ndarray:
    """Does poisson blending with some tricks.

    Args:
        orig_img (np.ndarray): Original image.
        fake_img (np.ndarray): Generated fake image to blend.
        mask (np.ndarray): Binary 0-1 mask to use for blending.
        pad_width (np.ndarray): Amount of padding to add before blending (useful to avoid some issues).
        dilation (np.ndarray): Amount of dilation to add to the mask before blending (useful to avoid some issues).

    Returns:
        np.ndarray: Blended image.
    """
    mask = mask[:, :, 0]
    padding_config = ((pad_width, pad_width), (pad_width, pad_width), (0, 0))
    padded_fake_img = np.pad(fake_img, pad_width=padding_config, mode="reflect")
    padded_orig_img = np.pad(orig_img, pad_width=padding_config, mode="reflect")
    padded_orig_img[:pad_width, :, :] = padded_fake_img[:pad_width, :, :]
    padded_orig_img[:, :pad_width, :] = padded_fake_img[:, :pad_width, :]
    padded_orig_img[-pad_width:, :, :] = padded_fake_img[-pad_width:, :, :]
    padded_orig_img[:, -pad_width:, :] = padded_fake_img[:, -pad_width:, :]
    padded_mask = np.pad(mask, pad_width=padding_config[:2], mode="constant")
    padded_dmask = cv2.dilate(padded_mask, np.ones((dilation, dilation), np.uint8), iterations=1)
    x_min, y_min, rect_w, rect_h = cv2.boundingRect(padded_dmask)
    center = (x_min + rect_w // 2, y_min + rect_h // 2)
    output = cv2.seamlessClone(padded_fake_img, padded_orig_img, padded_dmask, center, cv2.NORMAL_CLONE)
    output = output[pad_width:-pad_width, pad_width:-pad_width]
    return output


def image_from_url_text(filedata):
    if filedata is None:
        return None

    if type(filedata) == list and filedata and type(filedata[0]) == dict and filedata[0].get("is_file", False):
        filedata = filedata[0]

    if type(filedata) == dict and filedata.get("is_file", False):
        filename = filedata["name"]
        filename = filename.rsplit('?', 1)[0]
        return Image.open(filename)

    if type(filedata) == list:
        if len(filedata) == 0:
            return None

        filedata = filedata[0]

    if filedata.startswith("data:image/png;base64,"):
        filedata = filedata[len("data:image/png;base64,"):]

    filedata = base64.decodebytes(filedata.encode('utf-8'))
    image = Image.open(io.BytesIO(filedata))
    return image


def resize(image: Image, size: Union[int, Tuple[int, int]], resample=Image.BICUBIC):
    if isinstance(size, int):
        w, h = image.size
        aspect_ratio = w / h
        size = (min(size, int(size * aspect_ratio)),
                min(size, int(size / aspect_ratio)))
    return image.resize(size, resample=resample)


def find_mask_boundary(mask: Image):
    mask_array = np.array(mask)
    condition = np.all(mask_array == [255, 255, 255], axis=-1)

    y_indices, x_indices = np.where(condition)

    if len(x_indices) == 0 or len(y_indices) == 0:
        return None

    x_min, x_max = x_indices.min(), x_indices.max()
    y_min, y_max = y_indices.min(), y_indices.max()
    return (x_min, x_max, y_min, y_max)


def resize_and_insert(img: Image, boundary_a: tuple, boundary_b: tuple, mask=False):
    W, H = img.size
    x_min_a, x_max_a, y_min_a, y_max_a = boundary_a
    x_min_b, x_max_b, y_min_b, y_max_b = boundary_b

    b_cropped = img.crop((x_min_b, y_min_b, x_max_b, y_max_b))

    a_width = x_max_a - x_min_a
    a_height = y_max_a - y_min_a
    b_width = x_max_b - x_min_b
    b_height = y_max_b - y_min_b

    if a_width / b_width < a_height / b_height:
        scale = a_width / b_width
    else:
        scale = a_height / b_height

    if mask:
        b_resized = b_cropped.resize((int(b_width * scale), int(b_height * scale)), Image.NEAREST)
    else:
        b_resized = b_cropped.resize((int(b_width * scale), int(b_height * scale)), Image.BICUBIC)

    new_b_width, new_b_height = b_resized.size

    x_offset = (a_width - new_b_width) // 2
    y_offset = (a_height - new_b_height) // 2

    if mask:
        b_new = Image.new("RGB", (W, H), (0, 0, 0))
    else:
        b_new = Image.new("RGB", (W, H), (255, 255, 255))

    b_new.paste(b_resized, (x_min_a + x_offset, y_min_a + y_offset))

    return b_new


def static_resize(img: Image, boundary, mask=False, type=0):
    W, H = img.size
    x_min, x_max, y_min, y_max = boundary
    cloth_width = x_max - x_min
    cloth_height = y_max - y_min

    target_width = W // 3
    scale = target_width / cloth_width
    target_height = int(cloth_height * scale)

    x_offset = (W - target_width) // 2
    y_offset = H // 4

    resized_img = img.crop((x_min, y_min, x_max, y_max)).resize((target_width, target_height),
                                                                Image.NEAREST if mask else Image.BICUBIC)

    bg_color = (0, 0, 0) if mask else (255, 255, 255)
    new_img = Image.new("RGB", (W, H), bg_color)

    if y_offset + target_height > H:
        y_offset = H - target_height

    new_img.paste(resized_img, (x_offset, y_offset))

    return new_img


def multi_static_resize(img1: Image, img2: Image, boundary1: tuple, boundary2: tuple, size: Tuple[int, int], mask=False):
    W, H = size
    x_min1, x_max1, y_min1, y_max1 = boundary1
    x_min2, x_max2, y_min2, y_max2 = boundary2

    # 提取 b 图像的边界区域
    img1_cropped = img1.crop((x_min1, y_min1, x_max1, y_max1))
    img2_cropped = img2.crop((x_min2, y_min2, x_max2, y_max2))
    width1 = x_max1 - x_min1
    height1 = y_max1 - y_min1
    width2 = x_max2 - x_min2
    height2 = y_max2 - y_min2

    half_W = W // 2
    y_min = int(H * 2 / 7)

    target_w1 = int(half_W * 4 / 5)
    target_w2 = int(half_W * 4 / 5)

    scale1 = target_w1 / width1
    scale2 = target_w2 / width2

    new_h1 = int(height1 * scale1)
    new_h2 = int(height2 * scale2)

    if mask:
        img1_resized = img1_cropped.resize((target_w1, new_h1), Image.NEAREST)
        img2_resized = img2_cropped.resize((target_w2, new_h2), Image.NEAREST)
    else:
        img1_resized = img1_cropped.resize((target_w1, new_h1), Image.BICUBIC)
        img2_resized = img2_cropped.resize((target_w2, new_h2), Image.BICUBIC)
    bg_color = (0, 0, 0) if mask else (255, 255, 255)
    img_combine = Image.new("RGB", size, bg_color)
    only_img1 = Image.new("RGB", size, bg_color)
    only_img2 = Image.new("RGB", size, bg_color)
    y_offset1 = y_min
    y_offset2 = y_min
    x_offset1 = (half_W - target_w1) // 2
    x_offset2 = half_W + (half_W - target_w2) // 2
    img_combine.paste(img1_resized, (x_offset1, y_offset1))
    img_combine.paste(img2_resized, (x_offset2, y_offset2))
    only_img1.paste(img1_resized, (x_offset1, y_offset1))
    only_img2.paste(img2_resized, (x_offset2, y_offset2))

    return img_combine, only_img1, only_img2


def perpendicular(a):
    b = np.empty_like(a)
    b[0] = -a[1]
    b[1] = a[0]
    return b


def normalize(a):
    a = np.array(a)
    return a/np.linalg.norm(a)


def get_box(a, b):
    width = 0.2 * np.linalg.norm(a-b)
    u = normalize(perpendicular(a-b)) * width
    a = a + 0.2 * (a-b)
    b = b + 0.2 * (b-a)
    return np.array([a+u, a-u, b-u, b+u])


def get_dist(a, b):
    return np.linalg.norm(a-b)


def get_uintVec(a, b):
    # from a point toward b
    return (b-a) / get_dist(a,b)


def get_Vec(a, b):
    # from a point toward b
    return (b-a)


def warping_cloth(cloth_rgb, parsing_array, ori_parsing_array, cloth_pose_val, human_pose_val, w, h,
                  c_type='0', sub_type=0, adjust=False):
    torso_colors = [(66, 48, 197), (67, 51, 205), (68, 54, 211), (69, 57, 217)]
    uarm_colors = [(71, 75, 238), (71, 78, 241), (71, 82, 244), (71, 85, 246),]
    larm_colors = [(70, 60, 223), (70, 64, 227), (71, 67, 231), (71, 71, 235),]
    uleg_colors = [(66, 104, 253), (64, 108, 254), (61, 112, 254), (57, 116, 254)]
    lleg_colors = [(71, 89, 248), (70, 93, 250), (69, 96, 251), (68, 100, 252)]
    other_colors = [(0, 0, 0)]

    cloth_pose, human_pose = cloth_pose_val[:, :2], human_pose_val[:, :2]
    cloth_val, human_val = cloth_pose_val[:, 2], human_pose_val[:, 2]
    final_result = np.zeros_like(cloth_rgb)

    if cloth_pose[9][0] > cloth_pose[12][0]:
        temp_9, temp_10, temp_11 = cloth_pose[9].copy(), cloth_pose[10].copy(), cloth_pose[11].copy()
        cloth_pose[9], cloth_pose[10], cloth_pose[11] = cloth_pose[12], cloth_pose[13], cloth_pose[14]
        cloth_pose[12], cloth_pose[13], cloth_pose[14] = temp_9, temp_10, temp_11
    if cloth_pose[2][0] > cloth_pose[5][0]:
        temp_2, temp_3, temp_4 = cloth_pose[2].copy(), cloth_pose[3].copy(), cloth_pose[4].copy()
        cloth_pose[2], cloth_pose[3], cloth_pose[4] = cloth_pose[5], cloth_pose[6], cloth_pose[7]
        cloth_pose[5], cloth_pose[6], cloth_pose[7] = temp_2, temp_3, temp_4

    if c_type == '0':
        human_limb_length = 0
        limbs = [(2, 3), (3, 4), (5, 6), (6, 7)]
        for limb in limbs:
            if human_pose[limb[0]].sum() == 0 or human_pose[limb[1]].sum() == 0:
                continue
            human_limb_length = max(human_limb_length, get_dist(human_pose[limb[0]], human_pose[limb[1]]))

        if adjust:
            human_shoulder_length = get_dist(human_pose[2], human_pose[5])
            cloth_shoulder_length = get_dist(cloth_pose[2], cloth_pose[5])

            human_torso_length = get_dist(human_pose[1], human_pose[8])
            human_waist_length = get_dist(human_pose[9], human_pose[12])

            # Get body ratio
            limb_ratio = human_limb_length / human_shoulder_length
            torso_ratio = human_torso_length / human_shoulder_length
            waist_ratio = human_waist_length / human_shoulder_length

            # Start adjusting
            cloth_pose_adjusted = cloth_pose.copy()
            cloth_pose_adjusted[3] = cloth_pose_adjusted[2] + get_uintVec(cloth_pose[2], cloth_pose[
                3]) * cloth_shoulder_length * limb_ratio  # right_elbow
            cloth_pose_adjusted[4] = cloth_pose_adjusted[3] + get_uintVec(cloth_pose[3], cloth_pose[
                4]) * cloth_shoulder_length * limb_ratio  # right_wrist

            cloth_pose_adjusted[6] = cloth_pose_adjusted[5] + get_uintVec(cloth_pose[5], cloth_pose[
                6]) * cloth_shoulder_length * limb_ratio  # left_elbow
            cloth_pose_adjusted[7] = cloth_pose_adjusted[6] + get_uintVec(cloth_pose[6], cloth_pose[
                7]) * cloth_shoulder_length * limb_ratio  # left_wrist

            cloth_pose_adjusted[8] = cloth_pose_adjusted[1] + np.array([0, cloth_shoulder_length * torso_ratio])
            cloth_pose_adjusted[9] = cloth_pose_adjusted[8] + np.array([-cloth_shoulder_length * waist_ratio / 2, 0])
            cloth_pose_adjusted[12] = cloth_pose_adjusted[8] + np.array([cloth_shoulder_length * waist_ratio / 2, 0])

            cloth_pose = cloth_pose_adjusted

        arms = np.zeros_like(cloth_rgb)
        cloth_boxes = []
        human_boxes = []
        for limb in limbs:
            if cloth_pose[limb[0]].sum() == 0 or cloth_pose[limb[1]].sum() == 0 or human_pose[limb[0]].sum() == 0 or \
                    human_pose[limb[1]].sum() == 0:
                continue
            cloth_boxes.append(get_box(cloth_pose[limb[0]], cloth_pose[limb[1]]).astype(np.int32))
            human_boxes.append(get_box(human_pose[limb[0]], human_pose[limb[1]]).astype(np.int32))


        out_torso_mask_array = np.zeros(parsing_array.shape[:2], dtype=np.uint8)
        ori_torso_mask_array = np.zeros(parsing_array.shape[:2], dtype=np.uint8)
        out_uarm_mask_array = np.zeros(parsing_array.shape[:2], dtype=np.uint8)
        out_larm_mask_array = np.zeros(parsing_array.shape[:2], dtype=np.uint8)
        ori_uarm_mask_array = np.zeros(parsing_array.shape[:2], dtype=np.uint8)
        ori_larm_mask_array = np.zeros(parsing_array.shape[:2], dtype=np.uint8)
        for color in torso_colors:
            out_torso_mask_array[np.all(parsing_array == (np.array(color) / 127.5 - 1), axis=-1)] = 1
            ori_torso_mask_array[np.all(ori_parsing_array == (np.array(color) / 127.5 - 1), axis=-1)] = 1

        for color in uarm_colors:
            out_uarm_mask_array[np.all(parsing_array == (np.array(color) / 127.5 - 1), axis=-1)] = 1
            ori_uarm_mask_array[np.all(ori_parsing_array == (np.array(color) / 127.5 - 1), axis=-1)] = 1

        for color in larm_colors:
            out_larm_mask_array[np.all(parsing_array == (np.array(color) / 127.5 - 1), axis=-1)] = 1
            ori_larm_mask_array[np.all(ori_parsing_array == (np.array(color) / 127.5 - 1), axis=-1)] = 1

        parts = []
        cloth_uarm, cloth_larm = cloth_rgb.copy(), cloth_rgb.copy()

        # torso
        cloth_rgb[out_torso_mask_array != 1] = 0
        valid_idx = [i for i in [1, 2, 5, 8, 9, 12] if cloth_val[i] != 0 and human_val[i] != 0]

        if len(valid_idx) >= 4:  # 至少需要 4 个点计算单应性矩阵
            points1 = cloth_pose[valid_idx]
            points2 = human_pose[valid_idx]
            M = cv2.findHomography(points1, points2, 0)[0]
            torso_warped = cv2.warpPerspective(cloth_rgb, M, (w, h))

            c_torso_gray = cv2.cvtColor(((torso_warped + 1) * 127.5).astype(np.uint8), cv2.COLOR_BGR2GRAY)
            _, mask_torso = cv2.threshold(c_torso_gray, 0, 255, cv2.THRESH_BINARY)
            mask_torso = cv2.morphologyEx(mask_torso, cv2.MORPH_OPEN, np.ones((9, 9), np.uint8))
            torso_warped = cv2.bitwise_and(torso_warped, torso_warped, mask=mask_torso)
        else:
            torso_warped = np.zeros_like(cloth_rgb)

        ori_other_mask_array = np.zeros(parsing_array.shape[:2], dtype=np.uint8)
        for color in uleg_colors + lleg_colors + other_colors:
            ori_other_mask_array[np.all(ori_parsing_array == (np.array(color) / 127.5 - 1), axis=-1)] = 1
        ori_arm_mask_array = (ori_uarm_mask_array == 1) | (ori_larm_mask_array == 1)
        ori_arm_other_mask_array = (ori_other_mask_array == 1) | ori_arm_mask_array
        ori_torso_other_mask_array = (ori_other_mask_array == 1) | (ori_torso_mask_array == 1)

        torso_warped[ori_arm_other_mask_array] = 0

        cloth_uarm[out_uarm_mask_array != 1] = 0
        cloth_larm[out_larm_mask_array != 1] = 0

        index = 0
        for cloth_box, human_box in zip(cloth_boxes, human_boxes):  # 0-3:ru, rl, lu, ll
            try:
                if index % 2 == 0:
                    cloth_sleeve = cloth_uarm
                    ori_arm_mask_array = ori_larm_mask_array
                else:
                    cloth_sleeve = cloth_larm
                    ori_arm_mask_array = ori_uarm_mask_array
                mask = np.zeros(cloth_sleeve.shape[:2], dtype=np.uint8)
                cv2.drawContours(mask, [cloth_box], -1, (255, 255, 255), -1, cv2.LINE_AA)
                arm = cv2.bitwise_and(cloth_sleeve, cloth_sleeve, mask=mask)
                parts.append(cv2.bitwise_and(cloth_sleeve, cloth_sleeve, mask=mask))

                M = cv2.findHomography(cloth_box, human_box, cv2.RANSAC, 5.0)[0]
                parts[-1] = cv2.warpPerspective(parts[-1], M, (w, h), cv2.INTER_LINEAR)
                img2gray = cv2.cvtColor(((parts[-1].copy() + 1) * 127.5).astype(np.uint8), cv2.COLOR_BGR2GRAY)
                ret, mask = cv2.threshold(img2gray, 0, 255, cv2.THRESH_BINARY)
                mask_inv = 255 - mask
                arms = cv2.bitwise_and(arms, arms, mask=mask_inv)
                arms = cv2.add(parts[-1], arms)
                arms[ori_arm_mask_array == 1] = 0
                arms[ori_torso_other_mask_array] = 0
                final_result = cv2.add(torso_warped, arms)
                torso_warped = final_result
            except:
                continue
            index += 1
    elif c_type == '1':
        # Measure human body length
        human_waist_length = get_dist(human_pose[9], human_pose[12])
        human_thigh_length = max(get_dist(human_pose[9], human_pose[10]), get_dist(human_pose[12], human_pose[13]))
        human_calf_length = max(get_dist(human_pose[10], human_pose[11]), get_dist(human_pose[13], human_pose[14]))

        # Get cloth shoulder length
        cloth_waist_length = get_dist(cloth_pose[9], cloth_pose[12])

        # Get body ratio
        thigh_ratio = human_thigh_length / human_waist_length
        calf_ratio = human_calf_length / human_waist_length

        # Start adjusting
        cloth_pose_adjusted = cloth_pose.copy()
        cloth_pose_adjusted[10] = cloth_pose_adjusted[9] + get_uintVec(cloth_pose[9], cloth_pose[
            10]) * cloth_waist_length * thigh_ratio  # right_knee
        cloth_pose_adjusted[11] = cloth_pose_adjusted[10] + get_uintVec(cloth_pose[10], cloth_pose[
            11]) * cloth_waist_length * calf_ratio  # right_ankle

        cloth_pose_adjusted[13] = cloth_pose_adjusted[12] + get_uintVec(cloth_pose[12], cloth_pose[
            13]) * cloth_waist_length * thigh_ratio  # left_knee
        cloth_pose_adjusted[14] = cloth_pose_adjusted[13] + get_uintVec(cloth_pose[13], cloth_pose[
            14]) * cloth_waist_length * calf_ratio  # left_ankle

        cloth_pose = cloth_pose_adjusted

        # legs
        cloth_boxes = []
        human_boxes = []
        limbs = [(9, 10), (10, 11), (12, 13), (13, 14)]
        for idx, limb in enumerate(limbs):
            if cloth_pose[limb[0]].sum() == 0 or cloth_pose[limb[1]].sum() == 0 or human_pose[limb[0]].sum() == 0 or \
                    human_pose[limb[1]].sum() == 0:
                continue
            # Only warped long pants or skirts
            cloth_boxes.append(get_box(cloth_pose[limb[0]], cloth_pose[limb[1]]).astype(np.int32))
            human_boxes.append(get_box(human_pose[limb[0]], human_pose[limb[1]]).astype(np.int32))

        out_uleg_mask_array = np.zeros(parsing_array.shape[:2], dtype=np.uint8)
        out_lleg_mask_array = np.zeros(parsing_array.shape[:2], dtype=np.uint8)
        ori_uleg_mask_array = np.zeros(parsing_array.shape[:2], dtype=np.uint8)
        ori_lleg_mask_array = np.zeros(parsing_array.shape[:2], dtype=np.uint8)

        for color in uleg_colors:
            out_uleg_mask_array[np.all(parsing_array == (np.array(color) / 127.5 - 1), axis=-1)] = 1
            ori_uleg_mask_array[np.all(ori_parsing_array == (np.array(color) / 127.5 - 1), axis=-1)] = 1

        for color in lleg_colors:
            out_lleg_mask_array[np.all(parsing_array == (np.array(color) / 127.5 - 1), axis=-1)] = 1
            ori_lleg_mask_array[np.all(ori_parsing_array == (np.array(color) / 127.5 - 1), axis=-1)] = 1

        # Remove lower part of lower for torso part
        lower_bound = int(cloth_pose[8, 1] + get_dist(human_pose[9], human_pose[12]))
        torso_rgb = cloth_rgb.copy()
        torso_rgb[lower_bound:] = 0
        cloth_rgb[:lower_bound] = 0

        try:
            cloth_box = get_box(cloth_pose[9], cloth_pose[12]).astype(np.int32)
            human_box = get_box(human_pose[9], human_pose[12]).astype(np.int32)
            M = cv2.findHomography(cloth_box, human_box, cv2.RANSAC, 5.0)[0]
            torso_warped = cv2.warpPerspective(torso_rgb, M, (w, h))
        except:
            torso_warped = torso_rgb

        ori_other_mask_array = np.zeros(parsing_array.shape[:2], dtype=np.uint8)
        for color in uarm_colors + larm_colors + other_colors:
            ori_other_mask_array[np.all(ori_parsing_array == (np.array(color) / 127.5 - 1), axis=-1)] = 1
        torso_warped[ori_other_mask_array == 1] = 0
        final_result = torso_warped

        cloth_lleg, cloth_uleg = cloth_rgb.copy(), cloth_rgb.copy()
        cloth_uleg[out_uleg_mask_array != 1] = 0
        cloth_lleg[out_lleg_mask_array != 1] = 0
        # If it is a trouser
        if sub_type == 0:
            parts = []
            legs = np.zeros_like(cloth_rgb)
            index = 0
            for cloth_box, human_box in zip(cloth_boxes, human_boxes):
                try:
                    if index % 2 == 0:
                        leg_rgb = cloth_uleg
                        ori_leg_mask_array = ori_lleg_mask_array
                    else:
                        leg_rgb = cloth_lleg
                        ori_leg_mask_array = ori_uleg_mask_array
                    mask = np.zeros(leg_rgb.shape[:2]).astype(np.uint8)
                    cv2.drawContours(mask, [cloth_box], -1, (255, 255, 255), -1, cv2.LINE_AA)
                    parts.append(cv2.bitwise_and(leg_rgb, leg_rgb, mask=mask))

                    M = cv2.findHomography(cloth_box, human_box, cv2.RANSAC, 5.0)[0]
                    parts[-1] = cv2.warpPerspective(parts[-1], M, (w, h), cv2.INTER_LINEAR)
                    img2gray = cv2.cvtColor(((parts[-1].copy() + 1) * 127.5).astype(np.uint8), cv2.COLOR_BGR2GRAY)
                    ret, mask = cv2.threshold(img2gray, 0, 255, cv2.THRESH_BINARY)
                    mask_inv = 255 - mask
                    legs = cv2.bitwise_and(legs, legs, mask=mask_inv)
                    legs = cv2.add(parts[-1], legs)
                    legs[ori_leg_mask_array == 1] = 0
                    final_result = cv2.add(torso_warped, legs)
                    torso_warped = final_result
                except:
                    continue
                index += 1
        # If it is a skirt
        else:
            human_pose_adjusted = human_pose.copy()
            # correct waist to horizontal
            human_pose_adjusted[9, 0] = human_pose_adjusted[8, 0] - get_dist(human_pose[8], human_pose[9])
            human_pose_adjusted[9, 1] = human_pose_adjusted[8, 1]
            human_pose_adjusted[12, 0] = human_pose_adjusted[8, 0] + get_dist(human_pose[8], human_pose[12])
            human_pose_adjusted[12, 1] = human_pose_adjusted[12, 1]

            try:
                cloth_box = get_box(cloth_pose[9], cloth_pose[12]).astype(np.int32)
                human_box_adjusted = get_box(human_pose_adjusted[9], human_pose_adjusted[12]).astype(np.int32)
                M = cv2.findHomography(cloth_box, human_box_adjusted, cv2.RANSAC, 5.0)[0]
                legs = cv2.warpPerspective(cloth_rgb, M, (w, h))
            except:
                legs = np.zeros_like(cloth_rgb)
            final_result = cv2.add(torso_warped, legs)

    return final_result
