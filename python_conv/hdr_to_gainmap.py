from loguru import logger

from einops import repeat

import numpy as np
import torch

from ioops import ImageMetadata
from metrics import psnr
from utils import DTYPE
import utils


def compute_gain(hdr_y_nits: torch.Tensor, sdr_y_nits: torch.Tensor,
                 hdr_offset: tuple[float, float, float],
                 sdr_offset: tuple[float, float, float],
                 max_stops: float = 2.3):
    hdr_offset = torch.tensor(hdr_offset, dtype=DTYPE,
                              device=hdr_y_nits.device)
    sdr_offset = torch.tensor(sdr_offset, dtype=DTYPE,
                              device=hdr_y_nits.device)
    gain = torch.log2((hdr_y_nits + hdr_offset) / (sdr_y_nits + sdr_offset))
    mask_low = sdr_y_nits < 2. / 255.0
    gain[mask_low] = torch.minimum(gain[mask_low],
                                   torch.tensor(max_stops, dtype=DTYPE,
                                                device=hdr_y_nits.device))
    return gain


def affine_map_gain(gainlog2: torch.Tensor,
                    min_gainlog2: tuple[float, float, float],
                    max_gainlog2: tuple[float, float, float],
                    norm_min: float, norm_max: float,
                    map_gamma: tuple[float, float, float]):
    norm_range = norm_max - norm_min
    map_gamma = torch.tensor(map_gamma, dtype=DTYPE, device=gainlog2.device)
    mapped_val = (gainlog2 - min_gainlog2[None, None, ...]) /\
        (max_gainlog2 - min_gainlog2[None, None, ...]) * norm_range + norm_min
    # REVISIT:
    # mapped_val = (gainlog2 - min_gainlog2[None, None, ...]) /\
    #     (max_gainlog2 - min_gainlog2[None, None, ...])
    mapped_val = torch.clip(mapped_val, norm_min, norm_max)
    mapped_val = torch.pow(mapped_val, map_gamma)
    return mapped_val


def recompute_hdr_luminance(sdr_luminance: torch.Tensor,
                            gain: torch.Tensor,
                            hdr_offset: tuple[float, float, float],
                            sdr_offset: tuple[float, float, float]
                            ) -> torch.Tensor:
    gain_factor = torch.exp2(gain)
    hdr_offset = torch.tensor(hdr_offset, dtype=DTYPE,
                              device=sdr_luminance.device)
    sdr_offset = torch.tensor(sdr_offset, dtype=DTYPE,
                              device=sdr_luminance.device)
    return torch.multiply(sdr_luminance + sdr_offset, gain_factor) - hdr_offset


def undo_affine_map_gain(gain: torch.Tensor,
                         map_gamma: tuple[float, float, float],
                         min_content_boost: tuple[float, float, float],
                         max_content_boost: tuple[float, float, float],
                         norm_min: float, norm_max: float
                         ) -> torch.Tensor:
    norm_range = norm_max - norm_min
    map_gamma = torch.tensor(map_gamma, dtype=DTYPE, device=gain.device)
    effective_gain = torch.pow(gain, 1.0 / map_gamma)
    unnormalized_gain = (effective_gain - norm_min) / norm_range
    log_min = torch.tensor(np.log2(min_content_boost),
                           dtype=DTYPE, device=gain.device)[None, None, ...]
    log_max = torch.tensor(np.log2(max_content_boost),
                           dtype=DTYPE, device=gain.device)[None, None, ...]
    # REVISIT:
    # return torch.lerp(log_min, log_max, effective_gain)
    return torch.lerp(log_min, log_max, unnormalized_gain)


def apply_gain(e: torch.Tensor,
               gain: torch.Tensor,
               map_gamma: tuple[float, float, float],
               min_content_boost: tuple[float, float, float],
               max_content_boost: tuple[float, float, float],
               hdr_offset: tuple[float, float, float],
               sdr_offset: tuple[float, float, float],
               norm_min: float, norm_max: float
               ) -> torch.Tensor:
    log_boost = undo_affine_map_gain(gain, map_gamma,
                                     min_content_boost, max_content_boost,
                                     norm_min, norm_max)
    return recompute_hdr_luminance(e, log_boost, hdr_offset, sdr_offset)


def plot_image_with_rgb_values(img: torch.Tensor):
    """
    Plot an image with interactive cursor showing RGB values.

    Args:
        img: torch.Tensor of shape (H, W, C) with values in [0, 1]
    """
    import matplotlib.pyplot as plt

    # Convert tensor to numpy array
    if torch.is_tensor(img):
        img_np = img.cpu().numpy()
    else:
        img_np = img

    # Create the figure and axis
    fig, ax = plt.subplots()
    im = ax.imshow(img_np)

    # Create text annotation that will be updated
    annot = ax.annotate("", xy=(0, 0), xytext=(10, 10), textcoords="offset points",
                        bbox=dict(boxstyle="round", fc="w",
                                  ec="0.5", alpha=0.9),
                        fontsize=9)
    annot.set_visible(False)

    def update_annot(event):
        if event.inaxes == ax:
            x, y = int(event.xdata), int(event.ydata)
            if 0 <= x < img_np.shape[1] and 0 <= y < img_np.shape[0]:
                rgb = img_np[y, x]
                annot.xy = (x, y)
                text = f'RGB: ({rgb[0]:.3f}, {rgb[1]:.3f}, {rgb[2]:.3f})'
                annot.set_text(text)
                annot.set_visible(True)
                fig.canvas.draw_idle()
        else:
            annot.set_visible(False)
            fig.canvas.draw_idle()

    fig.canvas.mpl_connect('motion_notify_event', update_annot)
    plt.show()


def generate_gainmap(img_hdr: torch.Tensor,
                     meta: ImageMetadata,
                     abs_clip: bool = True,
                     c3: bool = False
                     ) -> dict[str, torch.Tensor | ImageMetadata]:
    hdr_inv_oetf = utils.GetInvOETFFn(meta.oetf)
    hdr_ootf = utils.GetOOTFFn(meta.oetf)
    hdr_gamut_conv = utils.GetGamutConversionFn(meta.gamut,
                                                utils.Gamut.BT2100)
    hdr_luminance_fn = utils.GetLuminanceFn(meta.gamut)
    bt2100_luminance_fn = utils.GetLuminanceFn(utils.Gamut.BT2100)
    hdr_peak_nits = utils.GetReferenceDisplayPeakLuminanceInNits(meta.oetf)
    sdr_gamut_conv = utils.GetGamutConversionFn(src_gamut=utils.Gamut.BT2100,
                                                dst_gamut=utils.Gamut.BT709)
    sdr_inv_oetf = utils.GetInvOETFFn(utils.OETF.SRGB)
    sdr_oetf = utils.GetOETFFn(utils.OETF.SRGB)
    sdr_hdr_gamut_conv = utils.GetGamutConversionFn(
        src_gamut=utils.Gamut.BT709, dst_gamut=utils.Gamut.BT2100)

    height, width, channels = img_hdr.shape
    logger.debug(
        f"Image min/max/mean {img_hdr.to(DTYPE).min()}/" +
        f"{img_hdr.to(DTYPE).max()}/{img_hdr.to(DTYPE).mean()}")

    if meta.oetf != utils.OETF.LINEAR:
        img_hdr_norm = img_hdr.to(DTYPE) / float((1 << meta.bit_depth) - 1)
    else:
        img_hdr_norm = img_hdr.to(DTYPE)

    img_hdr_lin = hdr_inv_oetf(img_hdr_norm)
    img_hdr_lin = hdr_ootf(img_hdr_lin, hdr_luminance_fn)
    img_hdr_lin = hdr_gamut_conv(img_hdr_lin)
    img_hdr_lin[img_hdr_lin < 0] = 0.

    if abs_clip:
        clip_value = torch.tensor(meta.clip_percentile,
                                  dtype=DTYPE,
                                  device=img_hdr_lin.device)
    else:
        clip_value = torch.quantile(img_hdr_lin.reshape(-1),
                                    meta.clip_percentile)

    logger.debug(
        f"HDR {meta.clip_percentile}th-percentile clip value: {clip_value}")

    img_hdr_lin_tonemapped = torch.minimum(
        img_hdr_lin, clip_value) / clip_value

    img_hdr_lin_tonemapped = utils.ApplyToneMapping(
        img_hdr_lin_tonemapped,
        utils.ToneMapping.REINHARD,
        hdr_peak_nits / utils.SDR_WHITE_NITS,
        meta.oetf != utils.OETF.LINEAR)

    img_sdr_lin = sdr_gamut_conv(img_hdr_lin_tonemapped)

    def perceptual_gamut_compression(
            rgb: torch.Tensor,
            peak: float = 1.0,
            slope_rgb: tuple[float, float, float] = (0., 6., 6.)
    ) -> torch.Tensor:
        exc = torch.clamp(rgb - peak, 0.0)
        slope = torch.tensor(slope_rgb,
                             dtype=rgb.dtype,
                             device=rgb.device)

        comp = torch.clamp(exc * slope, 0.0, 1.0)

        yuv = utils.sRGB_RGBToYUV(rgb)
        # chroma = rgb - Y                                   # signed chroma
        # rgb_cmp = Y + chroma * (1.0 - comp)                # squash chroma
        yuv *= (1 - comp)
        rgb = utils.sRGB_YUVToRGB(yuv)
        return rgb.clamp(0.0, peak)

    # img_sdr_lin = perceptual_gamut_compression(img_sdr_lin)
    img_sdr = sdr_oetf(img_sdr_lin)
    img_sdr = torch.clamp(img_sdr, 0., 1.)

    img_sdr_lin = torch.clip(img_sdr * 255.0 + 0.5, 0, 255).to(torch.uint8)
    img_sdr_lin = img_sdr_lin.to(DTYPE) / 255.0

    img_sdr_lin = sdr_inv_oetf(img_sdr_lin)
    img_sdr_lin = sdr_hdr_gamut_conv(img_sdr_lin)

    if c3:
        img_sdr_lin *= utils.SDR_WHITE_NITS
        img_hdr_nits = img_hdr_lin * hdr_peak_nits
        gainmap = compute_gain(
            img_hdr_nits, img_sdr_lin,
            hdr_offset=meta.hdr_offset,
            sdr_offset=meta.sdr_offset,
            max_stops=2.3 if meta.oetf == utils.OETF.HLG else 5.6).to(DTYPE)
    else:
        sdr_y_nits = repeat(
            bt2100_luminance_fn(img_sdr_lin) * utils.SDR_WHITE_NITS,
            "h w c -> h w (c 3)")
        hdr_y_nits = repeat(bt2100_luminance_fn(img_hdr_lin) * hdr_peak_nits,
                            "h w c -> h w (c 3)")
        gainmap = compute_gain(
            hdr_y_nits, sdr_y_nits,
            hdr_offset=meta.hdr_offset,
            sdr_offset=meta.sdr_offset,
            max_stops=2.3 if meta.oetf == utils.OETF.HLG else 5.6).to(DTYPE)
    # DEPRECATE:
    # min_gain = gainmap.amin(dim=(0, 1))
    # max_gain = gainmap.amax(dim=(0, 1))
    min_gain, max_gain = torch.quantile(
        gainmap.reshape(-1, 3),
        torch.tensor([meta.min_max_quantile, 1. - meta.min_max_quantile],
                     dtype=DTYPE, device=gainmap.device))

    min_gain, max_gain = min_gain.to(DTYPE), max_gain.to(DTYPE)

    min_gain = torch.clip(min_gain, -14.3, 15.6)
    max_gain = torch.clip(max_gain, -14.3, 15.6)

    if meta.min_content_boost is not None:
        min_gain = torch.log2(torch.tensor(meta.min_content_boost,
                                           dtype=DTYPE, device=gainmap.device))
    else:
        meta.min_content_boost = np.exp2(min_gain).cpu().numpy().tolist()
    if meta.max_content_boost is not None:
        max_gain = torch.log2(torch.tensor(meta.max_content_boost,
                                           dtype=DTYPE, device=gainmap.device))
    else:
        meta.max_content_boost = torch.exp2(max_gain).cpu().numpy().tolist()

    max_gain[torch.abs(max_gain - min_gain) < 1.0e-8] += 0.1

    logger.debug(f"min/max gain (log2): {min_gain}/{max_gain}")
    logger.debug(
        f"min/max gain (exp2): {np.exp2(min_gain)}/{np.exp2(max_gain)}")

    gainmap = affine_map_gain(gainmap, min_gain, max_gain, meta.affine_min,
                              meta.affine_max, meta.map_gamma)

    sdr_meta = ImageMetadata.from_dict(meta.to_dict())
    sdr_meta.gamut = utils.Gamut.BT709
    sdr_meta.oetf = utils.OETF.SRGB
    sdr_meta.bit_depth = 8
    return {
        "gainmap": gainmap,
        "img_hdr_linear": img_hdr_lin,
        "img_sdr": img_sdr,
        "hdr_metadata": meta,
        "sdr_metadata": sdr_meta
    }


def gainmap_sdr_to_hdr(img_sdr: torch.Tensor,
                       gainmap: torch.Tensor,
                       meta: ImageMetadata) -> dict[str, torch.Tensor]:

    hdr_peak_nits = utils.GetReferenceDisplayPeakLuminanceInNits(
        utils.OETF.HLG)

    sdr_inv_oetf = utils.GetInvOETFFn(meta.oetf)
    sdr_hdr_gamut_conv = utils.GetGamutConversionFn(
        src_gamut=meta.gamut, dst_gamut=utils.Gamut.BT2100)

    img_sdr_norm = img_sdr.to(DTYPE) / \
        float((1 << meta.bit_depth) - 1)
    img_sdr_lin = sdr_inv_oetf(img_sdr_norm)
    img_sdr_lin = sdr_hdr_gamut_conv(img_sdr_lin)
    img_hdr_recon = apply_gain(
        img_sdr_lin * utils.SDR_WHITE_NITS, gainmap, meta.map_gamma,
        meta.min_content_boost, meta.max_content_boost,
        meta.hdr_offset, meta.sdr_offset,
        meta.affine_min, meta.affine_max)
    img_hdr_recon /= hdr_peak_nits
    img_hdr_recon = torch.clip(img_hdr_recon, 0., 1.)

    return {"img_hdr_lin": img_hdr_recon}


def compare_hdr_to_uhdr(img_hdr: torch.Tensor,
                        img_sdr: torch.Tensor,
                        gainmap: torch.Tensor,
                        hdr_meta: ImageMetadata,
                        sdr_meta: ImageMetadata,
                        c3: bool = False) -> dict[str, torch.Tensor]:
    hdr_inv_oetf = utils.GetInvOETFFn(hdr_meta.oetf)
    hdr_ootf = utils.GetOOTFFn(hdr_meta.oetf)
    hdr_luminance_fn = utils.GetLuminanceFn(hdr_meta.gamut)
    hdr_gamut_conv = utils.GetGamutConversionFn(src_gamut=hdr_meta.gamut,
                                                dst_gamut=utils.Gamut.BT2100)
    hdr_peak_nits = utils.GetReferenceDisplayPeakLuminanceInNits(hdr_meta.oetf)

    bt2100_luminance_fn = utils.GetLuminanceFn(utils.Gamut.BT2100)
    sdr_inv_oetf = utils.GetInvOETFFn(sdr_meta.oetf)
    sdr_hdr_gamut_conv = utils.GetGamutConversionFn(
        src_gamut=sdr_meta.gamut, dst_gamut=utils.Gamut.BT2100)

    if hdr_meta.oetf != utils.OETF.LINEAR:
        img_hdr_norm = img_hdr.to(DTYPE) / float((1 << hdr_meta.bit_depth) - 1)
    else:
        img_hdr_norm = img_hdr.to(DTYPE) / img_hdr.to(DTYPE).max()
    img_sdr_norm = img_sdr.to(DTYPE) / float((1 << sdr_meta.bit_depth) - 1)

    img_hdr_lin = hdr_inv_oetf(img_hdr_norm)
    img_hdr_lin = hdr_ootf(img_hdr_lin, hdr_luminance_fn)
    img_hdr_lin = hdr_gamut_conv(img_hdr_lin)
    img_hdr_lin = torch.clip(img_hdr_lin, 0., 1.)
    if c3:
        img_hdr_lum = img_hdr_lin
    else:
        img_hdr_lum = bt2100_luminance_fn(img_hdr_lin)

    img_sdr_lin = sdr_inv_oetf(img_sdr_norm)
    img_sdr_lin = sdr_hdr_gamut_conv(img_sdr_lin)
    img_hdr_recon = apply_gain(
        img_sdr_lin * utils.SDR_WHITE_NITS, gainmap, sdr_meta.map_gamma,
        sdr_meta.min_content_boost, sdr_meta.max_content_boost,
        sdr_meta.hdr_offset, sdr_meta.sdr_offset,
        sdr_meta.affine_min, sdr_meta.affine_max)
    img_hdr_recon /= hdr_peak_nits
    img_hdr_recon = torch.clip(img_hdr_recon, 0., 1.)

    if c3:
        img_sdr_lum = img_sdr_lin * utils.SDR_WHITE_NITS
    else:
        img_sdr_lum = bt2100_luminance_fn(img_sdr_lin) * utils.SDR_WHITE_NITS
    gain_log2 = undo_affine_map_gain(gainmap, sdr_meta.map_gamma,
                                     sdr_meta.min_content_boost,
                                     sdr_meta.max_content_boost,
                                     sdr_meta.affine_min, sdr_meta.affine_max)
    img_hdr_recon_lum = recompute_hdr_luminance(
        img_sdr_lum, gain_log2, sdr_meta.hdr_offset, sdr_meta.sdr_offset)
    img_hdr_recon_lum /= hdr_peak_nits

    psnr_lum = psnr(img_hdr_lum, img_hdr_recon_lum)
    psnr_img = psnr(img_hdr_lin, img_hdr_recon)

    logger.info(f"PSNR luminance: {psnr_lum}dB")
    logger.info(f"PSNR image: {psnr_img}dB")

    return {
        "psnr_lum": psnr_lum,
        "psnr_img": psnr_img,
    }


def reconstruct_hdr(img_sdr: torch.Tensor,
                    gainmap: torch.Tensor,
                    hdr_meta: ImageMetadata,
                    sdr_meta: ImageMetadata,
                    c3: bool = False) -> dict[str, torch.Tensor]:
    hdr_oetf = utils.GetOETFFn(hdr_meta.oetf)
    hdr_inv_ootf = utils.GetInvOOTFFn(hdr_meta.oetf)
    hdr_luminance_fn = utils.GetLuminanceFn(hdr_meta.gamut)
    hdr_peak_nits = utils.GetReferenceDisplayPeakLuminanceInNits(hdr_meta.oetf)

    sdr_inv_oetf = utils.GetInvOETFFn(sdr_meta.oetf)
    sdr_hdr_gamut_conv = utils.GetGamutConversionFn(
        src_gamut=sdr_meta.gamut, dst_gamut=utils.Gamut.BT2100)

    if hdr_meta.oetf != utils.OETF.LINEAR:
        img_sdr_norm = img_sdr.to(DTYPE) / float((1 << sdr_meta.bit_depth) - 1)

    img_sdr_lin = sdr_inv_oetf(img_sdr_norm)
    img_sdr_lin = sdr_hdr_gamut_conv(img_sdr_lin)
    img_hdr_recon = apply_gain(
        img_sdr_lin * utils.SDR_WHITE_NITS, gainmap, sdr_meta.map_gamma,
        sdr_meta.min_content_boost, sdr_meta.max_content_boost,
        sdr_meta.hdr_offset, sdr_meta.sdr_offset,
        sdr_meta.affine_min, sdr_meta.affine_max)
    img_hdr_recon /= hdr_peak_nits
    img_hdr_recon = torch.clip(img_hdr_recon, 0., 1.)
    img_hdr_recon = hdr_inv_ootf(img_hdr_recon, hdr_luminance_fn)
    img_hdr_recon = hdr_oetf(img_hdr_recon)

    return {
        "img_hdr_recon": img_hdr_recon,
    }
