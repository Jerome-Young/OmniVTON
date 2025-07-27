import torch 
from torch import nn
import torch.nn.functional as F
import numbers, math
from typing import Any, Callable, Dict, List, Optional, Union, Tuple


def l1(_crossattn_similarity, mask, token_idx = [1,2]):
    similarity = torch.cat(_crossattn_similarity,1)[1]
    similarity = similarity.mean(0).permute(2,0,1)
    # similarity = similarity.softmax(dim = 0)
    
    return (similarity[token_idx] * mask.cuda()).sum()


def bce(_crossattn_similarity, mask, token_idx = [1,2]):
    similarity = torch.cat(_crossattn_similarity,1)[1]
    similarity = similarity.mean(0).permute(2,0,1)
    # similarity = similarity.softmax(dim = 0)

    return -sum([
        F.binary_cross_entropy_with_logits(x - 1.0, mask.cuda())
        for x in similarity[token_idx]
    ])


def miloss(_probs):
    _probs_out, _probs_in = torch.cat(_probs, 1)[1].chunk(2)
    b, h, w, s = _probs_out.shape
    scores = F.cosine_similarity(_probs_out.view(b, -1, s), _probs_in.view(b, -1, s), dim=2)
    labels = torch.arange(b).cuda()
    loss = F.cross_entropy(scores, labels)
    return loss

def mseloss(feats):
    print(len(feats['up_feat1']))
    # print(torch.cat(feats['up_feat1'])[0].shape)
    _probs_out, _probs_in = torch.cat(feats, 1)[1].chunk(2)
    b, h, w, s = _probs_out.shape
    scores = F.cosine_similarity(_probs_out.view(b, -1, s), _probs_in.view(b, -1, s), dim=2)
    labels = torch.arange(b).cuda()
    loss = F.cross_entropy(scores, labels)
    return loss


def attnloss(scores):
    """输入scores为张量（非列表）"""
    return (1 - scores).sum() * 0.001  # 向量化操作


def softmax(_crossattn_similarity, mask, token_idx = [1,2]):
    similarity = torch.cat(_crossattn_similarity,1)[1]
    similarity = similarity.mean(0).permute(2,0,1)

    similarity = similarity[1:].softmax(dim = 0) # Comute the softmax to obtain probability values
    token_idx = [x - 1 for x in token_idx]

    score = similarity[token_idx].sum(dim = 0) # Sum up all relevant tokens to get pixel-wise probability of belonging to the correct class
    score = torch.log(score) # Obtain log-probabilities per-pixel
    return (score * mask.cuda()).sum() # Sum up log-probabilities (equivalent to multiplying P-values) for all pixels inside of the mask

class GaussianSmoothing(nn.Module):
    """
    Apply gaussian smoothing on a
    1d, 2d or 3d tensor. Filtering is performed seperately for each channel
    in the input using a depthwise convolution.
    Arguments:
        channels (int, sequence): Number of channels of the input tensors. Output will
            have this number of channels as well.
        kernel_size (int, sequence): Size of the gaussian kernel.
        sigma (float, sequence): Standard deviation of the gaussian kernel.
        dim (int, optional): The number of dimensions of the data.
            Default value is 2 (spatial).
    """
    def __init__(self, channels, kernel_size, sigma, dim=2):
        super(GaussianSmoothing, self).__init__()
        if isinstance(kernel_size, numbers.Number):
            kernel_size = [kernel_size] * dim
        if isinstance(sigma, numbers.Number):
            sigma = [sigma] * dim

        # The gaussian kernel is the product of the
        # gaussian function of each dimension.
        kernel = 1
        meshgrids = torch.meshgrid(
            [
                torch.arange(size, dtype=torch.float32)
                for size in kernel_size
            ]
        )
        for size, std, mgrid in zip(kernel_size, sigma, meshgrids):
            mean = (size - 1) / 2
            kernel *= 1 / (std * math.sqrt(2 * math.pi)) * \
                      torch.exp(-((mgrid - mean) / (2 * std)) ** 2)

        # Make sure sum of values in gaussian kernel equals 1.
        kernel = kernel / torch.sum(kernel)

        # Reshape to depthwise convolutional weight
        kernel = kernel.view(1, 1, *kernel.size())
        kernel = kernel.repeat(channels, *[1] * (kernel.dim() - 1))

        self.register_buffer('weight', kernel)
        self.groups = channels

        if dim == 1:
            self.conv = F.conv1d
        elif dim == 2:
            self.conv = F.conv2d
        elif dim == 3:
            self.conv = F.conv3d
        else:
            raise RuntimeError(
                'Only 1, 2 and 3 dimensions are supported. Received {}.'.format(dim)
            )

    def forward(self, input):
        """
        Apply gaussian filter to input.
        Arguments:
            input (torch.Tensor): Input to apply gaussian filter on.
        Returns:
            filtered (torch.Tensor): Filtered output.
        """
        return self.conv(input, weight=self.weight.to(input.dtype), groups=self.groups)


def compute_max_attention_per_index(attention_maps: torch.Tensor,
                                     indices_to_alter: List[int],
                                     smooth_attentions: bool = True,
                                     sigma: float = 0.5,
                                     kernel_size: int = 3) -> List[torch.Tensor]:
    """ Computes the maximum attention value for each of the tokens we wish to alter. """
    last_idx = -1
    attention_for_text = attention_maps[:, :, 1:last_idx]
    attention_for_text *= 100
    attention_for_text = torch.nn.functional.softmax(attention_for_text, dim=-1)
    # Shift indices since we removed the first token
    indices_to_alter = [index - 1 for index in indices_to_alter]
    # Extract the maximum values
    max_indices_list = []
    for i in indices_to_alter:
        image = attention_for_text[:, :, i]
        if smooth_attentions:
            smoothing = GaussianSmoothing(channels=1, kernel_size=kernel_size, sigma=sigma, dim=2).cuda()
            input = F.pad(image.unsqueeze(0).unsqueeze(0), (1, 1, 1, 1), mode='reflect')
            image = smoothing(input).squeeze(0).squeeze(0)
        max_indices_list.append(image.max())
    return max_indices_list


def compute_loss(attention_maps, indices_to_alter, return_losses: bool = False) -> torch.Tensor:
    """ Computes the attend-and-excite loss using the maximum attention value for each token. """
    max_attention_per_index = compute_max_attention_per_index(attention_maps, indices_to_alter)
    losses = [max(0, 1. - curr_max) for curr_max in max_attention_per_index]
    loss = max(losses)
    if return_losses:
        return loss, losses
    else:
        return loss