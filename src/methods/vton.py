import torch
from src.utils.iimage import IImage
from pytorch_lightning import seed_everything
from tqdm import tqdm
import pprint
import torch.nn.functional as F

from src.smplfusion import share, router, attentionpatch, transformerpatch
from src.utils import tokenize, scores

verbose = False


def init_guidance():
    # Setup model for guidance only!
    router.attention_forward = attentionpatch.default.forward_and_save
    router.basic_transformer_forward = transformerpatch.default.forward


def gaussian_lowpass_mixing(noise, pose_latent, alpha=1, sigma=0.1):
    noise_fft = torch.fft.fftshift(torch.fft.fft2(noise.to(dtype=torch.float32)))
    pose_fft = torch.fft.fftshift(torch.fft.fft2(pose_latent.to(dtype=torch.float32)))

    B, C, H, W = noise.shape
    y, x = torch.meshgrid(
        torch.linspace(-1, 1, H, device=noise.device),
        torch.linspace(-1, 1, W, device=noise.device),
    )
    d = torch.sqrt(x ** 2 + y ** 2)
    gaussian_mask = torch.exp(-(d ** 2) / (2 * sigma ** 2))
    gaussian_mask = gaussian_mask.view(1, 1, H, W)

    mixed_low = alpha * pose_fft * gaussian_mask + (1 - alpha) * noise_fft * gaussian_mask
    mixed_high = noise_fft * (1 - gaussian_mask)
    mixed_fft = mixed_low + mixed_high

    mixed_fft = torch.fft.ifftshift(mixed_fft)
    mixed_latent = torch.fft.ifft2(mixed_fft).real

    return mixed_latent


def run(
        ddim,
        cprompt,
        pprompt,
        image,
        cloth,
        in_mask,
        out_mask,
        warped_cloth=None,
        init_noise=None,
        seed=0,
        negative_prompt='',
        positive_prompt='',
        num_steps=50,
        guidance_scale=7.5,
        stage='vton'
):
    image = image.padx(64)
    cloth = cloth.padx(64)
    in_mask = in_mask.dilate(1).alpha().padx(64)
    out_mask = out_mask.dilate(1).alpha().padx(64)
    dt = 1000 // num_steps
    full_cprompt = cprompt
    if positive_prompt != '':
        full_cprompt = f'{cprompt}, {positive_prompt}'
        full_pprompt = f'{pprompt}, {positive_prompt}'

    c = ddim.encoder.encode(full_cprompt)
    u = ddim.encoder.encode(negative_prompt)
    context = torch.cat([u, u, c, c], dim=0)

    init_guidance()
    in_mask_tensor = (in_mask.torch().cuda() + 1) / 2
    out_mask_tensor = 1 - ((out_mask.torch().cuda() + 1) / 2)

    # Image condition
    out_unet_condition = ddim.get_inpainting_condition(cloth, out_mask)
    if warped_cloth is not None:
        warped_cloth = IImage(warped_cloth).padx(64)
        in_unet_condition = ddim.get_inpainting_condition(image, in_mask, warped_cloth)
    else:
        in_unet_condition = ddim.get_inpainting_condition(image, in_mask)

    outpainting = True
    if stage == 'vton':
        outpainting = False

    share.set_mask(out_mask)
    dtype = out_unet_condition.dtype

    # Starting latent
    seed_everything(seed)
    unet_condition = torch.cat([out_unet_condition, in_unet_condition])
    z0 = torch.randn((1, 4) + unet_condition.shape[2:]).cuda().to(dtype)

    if init_noise is not None:
        z0 = gaussian_lowpass_mixing(z0, init_noise).to(dtype)

    zt = z0.repeat(2, 1, 1, 1)
    # Setup unet for guidance
    ddim.unet.requires_grad_(False)
    pbar = tqdm(range(999, 0, -dt)) if verbose else range(999, 0, -dt)

    for timestep in share.DDIMIterator(pbar):
        zt = zt.detach()
        zt.requires_grad = False

        _zt = zt if unet_condition is None else torch.cat([zt, unet_condition], 1)
        with torch.autocast('cuda'):
            eps_uncond, eps = ddim.unet(
                torch.cat([_zt, _zt]).to(dtype),
                timesteps=torch.tensor([timestep, timestep, timestep, timestep]).cuda(),
                context=context,
                in_mask=in_mask_tensor,
                out_mask=out_mask_tensor,
                outpainting=outpainting,
            ).detach().chunk(2)

        # Unconditional guidance
        eps = (eps_uncond + guidance_scale * (eps - eps_uncond))
        z0 = (zt - share.schedule.sqrt_one_minus_alphas[timestep] * eps) / share.schedule.sqrt_alphas[timestep]

        # DDIM Step
        with torch.no_grad():
            zt = share.schedule.sqrt_alphas[timestep - dt] * z0 + share.schedule.sqrt_one_minus_alphas[
                timestep - dt] * eps

    with torch.no_grad():
        output_image = IImage(ddim.vae.decode(z0 / ddim.config.scale_factor))

    return output_image