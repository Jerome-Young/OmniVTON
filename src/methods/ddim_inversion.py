import torch
from src.utils.iimage import IImage
from pytorch_lightning import seed_everything
from tqdm import tqdm
import pprint
import torch.nn.functional as F

from src.smplfusion import share, router, attentionpatch, transformerpatch
from src.utils import tokenize, scores


def init_guidance():
    # Setup model for guidance only!
    router.attention_forward = attentionpatch.default.forward_and_save
    router.basic_transformer_forward = transformerpatch.default.forward


def run(
        ddim,
        cprompt,
        pprompt,
        image,
        cloth,
        in_mask,
        out_mask,
        seed=0,
        negative_prompt='',
        positive_prompt='',
        num_steps=50,
        guidance_scale=7.5,
):
    image = image.padx(64)
    in_mask = in_mask.dilate(1).alpha().padx(64)
    out_mask = out_mask.dilate(1).alpha().padx(64)
    dt = 1000 // num_steps
    full_pprompt = pprompt
    if positive_prompt != '':
        full_cprompt = f'{cprompt}, {positive_prompt}'
        full_pprompt = f'{pprompt}, {positive_prompt}'

    c = ddim.encoder.encode(full_pprompt)
    u = ddim.encoder.encode(negative_prompt)
    context = torch.cat([u, c], dim=0)

    init_guidance()

    # Image condition
    unet_condition = ddim.get_inpainting_condition(image, in_mask)
    dtype = ddim.vae.encoder.conv_in.weight.dtype
    p_z0 = ddim.vae.encode(image.torch().cuda().to(dtype)).mean * ddim.config.scale_factor

    share.set_mask(in_mask)

    # Starting latent
    seed_everything(seed)
    zt = p_z0

    # Turn off gradients
    ddim.unet.requires_grad_(False)
    pbar = tqdm(range(19, 1000, dt))

    for timestep in share.DDIMIterator(pbar):

        _zt = zt if unet_condition is None else torch.cat([zt, unet_condition], 1)
        with torch.autocast('cuda'):
            eps_uncond, eps = ddim.unet(
                torch.cat([_zt, _zt]).to(dtype),
                timesteps=torch.tensor([timestep, timestep]).cuda(),
                context=context,
                time=timestep
            ).chunk(2)

        eps = (eps_uncond + guidance_scale * (eps - eps_uncond))
        z0 = (zt - share.schedule.sqrt_one_minus_alphas[timestep] * eps) / share.schedule.sqrt_alphas[timestep]

        next_timestep = timestep + dt
        if next_timestep > 999:
            continue
        zt = share.schedule.sqrt_alphas[next_timestep] * z0 + share.schedule.sqrt_one_minus_alphas[next_timestep] * eps

    print('ddim inversion success')

    return zt
