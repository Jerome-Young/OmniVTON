from ... import share

import xformers
import xformers.ops

import torch
from torch import nn, einsum
import torch.nn.functional as F
from einops import rearrange, repeat
import math

_ATTN_PRECISION = None


def forward_sd2(self, x, context=None, mask=None):
    h = self.heads
    q = self.to_q(x)
    context = x if context is None else context
    k = self.to_k(context)
    v = self.to_v(context)

    q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> (b h) n d', h=h), (q, k, v))

    if _ATTN_PRECISION == "fp32":  # force cast to fp32 to avoid overflowing
        with torch.autocast(enabled=False, device_type='cuda'):
            q, k = q.float(), k.float()
            sim = einsum('b i d, b j d -> b i j', q, k) * self.scale
    else:
        sim = einsum('b i d, b j d -> b i j', q, k) * self.scale
    del q, k

    if mask is not None:
        mask = rearrange(mask, 'b ... -> b (...)')
        max_neg_value = -torch.finfo(sim.dtype).max
        mask = repeat(mask, 'b j -> (b h) () j', h=h)
        sim.masked_fill_(~mask, max_neg_value)

    # attention, what we cannot get enough of
    sim = sim.softmax(dim=-1)

    out = einsum('b i j, b j d -> b i d', sim, v)
    out = rearrange(out, '(b h) n d -> b n (h d)', h=h)
    return self.to_out(out)


def forward_xformers(self, x, context=None, mask=None):
    q = self.to_q(x)
    context = x if context is None else context
    k = self.to_k(context)
    v = self.to_v(context)

    b, _, _ = q.shape
    q, k, v = map(
        lambda t: t.unsqueeze(3)
        .reshape(b, t.shape[1], self.heads, self.dim_head)
        .permute(0, 2, 1, 3)
        .reshape(b * self.heads, t.shape[1], self.dim_head)
        .contiguous(),
        (q, k, v),
    )

    out = xformers.ops.memory_efficient_attention(q, k, v, attn_bias=None, op=self.attention_op)

    if mask is not None:
        raise NotImplementedError
    out = (
        out.unsqueeze(0)
        .reshape(b, self.heads, out.shape[1], self.dim_head)
        .permute(0, 2, 1, 3)
        .reshape(b, out.shape[1], self.heads * self.dim_head)
    )
    return self.to_out(out)


forward = forward_xformers

import traceback


def get_batch_sim(q, k, v, num_heads, scale):
    b = q.shape[0] // num_heads
    q = rearrange(q, "(b h) n d -> h (b n) d", h=num_heads)
    k = rearrange(k, "(b h) n d -> h (b n) d", h=num_heads)
    v = rearrange(v, "(b h) n d -> h (b n) d", h=num_heads)

    sim = torch.einsum("h i d, h j d -> h i j", q, k) * scale
    return sim


def calc_mean_std(feat, eps: float = 1e-5):
    feat_std = (feat.var(dim=-2, keepdims=True) + eps).sqrt()
    feat_mean = feat.mean(dim=-2, keepdims=True)
    return feat_mean, feat_std


def adain(x, y):
    assert (x.size()[:2] == y.size()[:2])
    x_mean, x_std = calc_mean_std(x)
    y_mean, y_std = calc_mean_std(y)
    normalized_feat = (x - x_mean) / x_std
    return normalized_feat * y_std + y_mean


def efficient_projection(out_v, in_v):
    # Compute Gram matrix
    gram_matrix = torch.matmul(out_v.permute(0, 2, 1), out_v)  # (B, C, C)
    # Add small regularization term
    gram_matrix += 1e-6 * torch.eye(gram_matrix.shape[-1], device=gram_matrix.device)
    # Check if the matrix is positive definite
    try:
        L = torch.linalg.cholesky(gram_matrix.to(torch.float32))
        gram_inv = torch.cholesky_inverse(L)
    except torch._C._LinAlgError:
        # Fallback to pseudo-inverse if Cholesky fails
        gram_inv = torch.linalg.pinv(gram_matrix.to(torch.float32))
    # Compute projection matrix
    projection_matrix = torch.matmul(torch.matmul(out_v, gram_inv.to(torch.float16)),
                                     out_v.permute(0, 2, 1))  # (B, H, H)
    # Project in_v onto the orthogonal subspace
    P_sim = torch.matmul(projection_matrix, in_v)
    return P_sim


def compute_cosine_similarity(a, b):
    # 计算余弦相似度
    a_norm = F.normalize(a, p=2, dim=-1)  # 归一化向量
    b_norm = F.normalize(b, p=2, dim=-1)
    cosine_sim = torch.einsum("b i d, b i d -> b i", a_norm, b_norm)  # 计算点积
    return cosine_sim


def forward_and_save(self, x, context=None, in_mask=None, out_mask=None, outpainting=False):
    att_type = "self" if context is None else "cross"
    HW = x.shape[1]
    W = int(math.sqrt(HW / (4 / 3)))
    H = HW // W

    h = self.heads
    q = self.to_q(x)
    context = x if context is None else context
    k = self.to_k(context)
    v = self.to_v(context)

    q, k, v = map(lambda t: rearrange(t, "b n (h d) -> (b h) n d", h=h), (q, k, v))

    if att_type == "self" and out_mask is not None:
        out_mask_resized = F.interpolate(out_mask, size=(H, W), mode='nearest')
        out_mask = out_mask_resized.reshape(1, -1, 1)
        in_mask_resized = F.interpolate(in_mask, size=(H, W), mode='nearest')
        in_mask = in_mask_resized.reshape(1, -1, 1)

        qu, qc = q.chunk(2)
        ku, kc = k.chunk(2)
        vu, vc = v.chunk(2)

        # cloth
        ori_q = torch.cat((qu[:h], qc[:h]))
        ori_ku = torch.cat((ku[:h], ku[-h:]), 1)
        ori_kc = torch.cat((kc[:h], kc[-h:]), 1)
        ori_k = torch.cat((ori_ku, ori_kc))
        ori_sim = einsum("b i d, b j d -> b i j", ori_q, ori_k) * self.scale
        ori_sim = ori_sim.softmax(-1)

        if outpainting:
            ori_vu = torch.cat((vu[:h], vu[-h:]), 1)
            ori_vc = torch.cat((vc[:h], vc[-h:]), 1)
            ori_v = torch.cat((ori_vu, ori_vc))
            ori_output = einsum("b i j, b j d -> b i d", ori_sim, ori_v)
        else:
            ori_v = torch.cat((vu[:h], vc[:h]))
            ori_output = einsum("b i j, b j d -> b i d", ori_sim[:, :, :int(ori_k.shape[1] / 2)], ori_v)
        ori_output_u, ori_output_c = rearrange(ori_output, "(b h) n d -> b n (h d)", h=h).chunk(2)

        # person
        hyd_q = torch.cat((qu[-h:], qc[-h:]))

        hyd_ku = torch.cat((ku[-h:], ku[:h]), 1)
        hyd_kc = torch.cat((kc[-h:], kc[:h]), 1)
        hyd_k = torch.cat((hyd_ku, hyd_kc))

        if outpainting:
            hyd_vu = torch.cat((vu[-h:], vu[:h]), 1)
            hyd_vc = torch.cat((vc[-h:], vc[:h]), 1)
        else:
            hyd_vu = torch.cat((vu[-h:], vu[:h]*out_mask), 1)
            hyd_vc = torch.cat((vc[-h:], vc[:h]*out_mask), 1)
        hyd_v = torch.cat((hyd_vu, hyd_vc))

        hyd_sim = einsum("b i d, b j d -> b i j", hyd_q, hyd_k) * self.scale
        hyd_sim = hyd_sim.softmax(-1)
        hyd_output = einsum("b i j, b j d -> b i d", hyd_sim, hyd_v)
        hyd_output_u, hyd_output_c = rearrange(hyd_output, "(b h) n d -> b n (h d)", h=h).chunk(2)
        out = torch.cat([ori_output_u, hyd_output_u,
                         ori_output_c, hyd_output_c], dim=0)
    else:
        sim = einsum("b i d, b j d -> b i j", q, k) * self.scale
        sim = sim.softmax(dim=-1)
        out = einsum("b i j, b j d -> b i d", sim, v)
        out = rearrange(out, "(b h) n d -> b n (h d)", h=h)

    return self.to_out(out)