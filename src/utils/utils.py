import torch
import torch.nn.functional as F


def apply_gaussian_blur(pose_map, kernel_size=5, sigma=2):
    """
    对 pose map 进行高斯模糊。

    Args:
        pose_map (torch.Tensor): 形状为 (B, 1, H, W) 的 pose map。
        kernel_size (int): 高斯核的大小，必须为奇数。
        sigma (float): 高斯模糊的标准差。

    Returns:
        torch.Tensor: 模糊化后的 pose map，形状同输入。
    """
    # 创建高斯核
    grid = torch.arange(kernel_size).float() - kernel_size // 2
    gauss_kernel_1d = torch.exp(-0.5 * (grid / sigma) ** 2)
    gauss_kernel_1d = gauss_kernel_1d / gauss_kernel_1d.sum()

    # 生成二维高斯核
    gauss_kernel_2d = torch.outer(gauss_kernel_1d, gauss_kernel_1d)
    gauss_kernel_2d = gauss_kernel_2d.unsqueeze(0).unsqueeze(0).to(pose_map.device)

    # 对 pose map 进行模糊
    blurred_pose_map = F.conv2d(
        pose_map, gauss_kernel_2d, padding=kernel_size // 2, groups=pose_map.size(1)
    )

    return blurred_pose_map
