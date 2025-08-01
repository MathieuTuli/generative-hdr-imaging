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
    mapped_val = torch.clamp(mapped_val, norm_min, norm_max)
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
    map_gamma = torch.tensor(map_gamma, dtype=DTYPE, device=gain.device)
    effective_gain = torch.pow(gain, 1.0 / map_gamma)
    norm_range = norm_max - norm_min
    unnormalized_gain = (effective_gain - norm_min) / norm_range
    log_min = torch.tensor(np.log2(min_content_boost),
                           dtype=DTYPE, device=gain.device)[None, None, ...]
    log_max = torch.tensor(np.log2(max_content_boost),
                           dtype=DTYPE, device=gain.device)[None, None, ...]
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


def generate_gainmap(img_hdr: torch.Tensor,
                     meta: ImageMetadata,
                     c3: bool = False,
                     img_sdr: torch.Tensor = None,
                     sdr_meta: ImageMetadata = None,
                     dst_gamut: utils.Gamut = utils.Gamut.BT2100
                     ) -> dict[str, torch.Tensor | ImageMetadata]:
    hdr_inv_oetf = utils.GetInvOETFFn(meta.oetf)
    hdr_ootf = utils.GetOOTFFn(meta.oetf)
    hdr_gamut_conv = utils.GetGamutConversionFn(src_gamut=meta.gamut,
                                                dst_gamut=dst_gamut)
    hdr_luminance_fn = utils.GetLuminanceFn(meta.gamut)
    bt2100_luminance_fn = utils.GetLuminanceFn(dst_gamut)
    hdr_peak_nits = utils.GetReferenceDisplayPeakLuminanceInNits(meta.oetf)
    sdr_gamut_conv = utils.GetGamutConversionFn(src_gamut=dst_gamut,
                                                dst_gamut=utils.Gamut.BT709)
    sdr_inv_oetf = utils.GetInvOETFFn(utils.OETF.SRGB)
    sdr_oetf = utils.GetOETFFn(utils.OETF.SRGB)
    sdr_hdr_gamut_conv = utils.GetGamutConversionFn(
        src_gamut=utils.Gamut.BT709, dst_gamut=dst_gamut)

    height, width, channels = img_hdr.shape
    logger.debug(
        f"Image min/max/mean {img_hdr.to(DTYPE).min()}/" +
        f"{img_hdr.to(DTYPE).max()}/{img_hdr.to(DTYPE).mean()}")

    if meta.oetf != utils.OETF.LINEAR:
        img_hdr_norm = img_hdr.to(DTYPE) / float((1 << meta.bit_depth) - 1)
        img_hdr_lin = hdr_inv_oetf(img_hdr_norm)
        img_hdr_lin = hdr_ootf(img_hdr_lin, hdr_luminance_fn)
    else:
        img_hdr_lin = img_hdr.to(DTYPE) / img_hdr.to(DTYPE).max()

    img_hdr_lin = hdr_gamut_conv(img_hdr_lin)
    img_hdr_lin = torch.clamp(img_hdr_lin, 0., 1.)

    if img_sdr is None:
        img_hdr_lin_biased = img_hdr_lin * 2 ** meta.hdr_exposure_bias
        img_hdr_lin_biased *= hdr_peak_nits / utils.SDR_WHITE_NITS

        img_sdr_lin = sdr_gamut_conv(img_hdr_lin_biased)
        img_sdr_lin = torch.clamp(img_sdr_lin, 0., 1.)
        img_sdr = sdr_oetf(img_sdr_lin)

        img_sdr_lin = torch.clamp(
            img_sdr * 255.0 + 0.5, 0, 255).to(torch.uint8)
        img_sdr_lin = img_sdr_lin.to(DTYPE) / 255.0

        img_sdr_lin = sdr_inv_oetf(img_sdr_lin)
        img_sdr_lin = sdr_hdr_gamut_conv(img_sdr_lin)
    else:
        img_sdr = img_sdr.to(DTYPE) / float((1 << sdr_meta.bit_depth) - 1)
        img_sdr_lin = sdr_inv_oetf(img_sdr)

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
                     dtype=DTYPE, device=gainmap.device), dim=0)

    min_gain, max_gain = min_gain.to(DTYPE), max_gain.to(DTYPE)

    min_gain = torch.clamp(min_gain, -14.3, 15.6)
    max_gain = torch.clamp(max_gain, -14.3, 15.6)

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

    meta.hdr_capacity_min = min_gain.min().item()
    meta.hdr_capacity_max = max_gain.max().item()

    max_gain[torch.abs(max_gain - min_gain) < 1.0e-8] += 0.1

    logger.debug(f"min/max gain (log2): {min_gain}/{max_gain}")
    logger.debug(
        f"min/max gain (exp2): {np.exp2(min_gain)}/{np.exp2(max_gain)}")

    gainmap_q = torch.clamp(gainmap * 255 + 0.5, 0, 255) / 255.
    gainmap_q = affine_map_gain(gainmap_q, min_gain, max_gain, meta.affine_min,
                                meta.affine_max, meta.map_gamma)
    gainmap = affine_map_gain(gainmap, min_gain, max_gain, meta.affine_min,
                              meta.affine_max, meta.map_gamma)

    sdr_meta = ImageMetadata.from_dict(meta.to_dict())
    sdr_meta.gamut = utils.Gamut.BT709
    sdr_meta.oetf = utils.OETF.SRGB
    sdr_meta.bit_depth = 8
    return {
        "gainmap": gainmap,
        "gainmap_q": gainmap_q,
        "img_hdr_linear": img_hdr_lin,
        "img_sdr": img_sdr,
        "hdr_metadata": meta,
        "sdr_metadata": sdr_meta
    }


def compare_hdr_to_uhdr(img_hdr: torch.Tensor,
                        img_sdr: torch.Tensor,
                        gainmap: torch.Tensor,
                        hdr_meta: ImageMetadata,
                        sdr_meta: ImageMetadata,
                        c3: bool = False,
                        dst_gamut: utils.Gamut = utils.Gamut.BT2100
                        ) -> dict[str, torch.Tensor]:
    hdr_inv_oetf = utils.GetInvOETFFn(hdr_meta.oetf)
    hdr_ootf = utils.GetOOTFFn(hdr_meta.oetf)
    hdr_oetf = utils.GetOETFFn(hdr_meta.oetf)
    hdr_inv_ootf = utils.GetInvOOTFFn(hdr_meta.oetf)
    hdr_luminance_fn = utils.GetLuminanceFn(hdr_meta.gamut)
    hdr_gamut_conv = utils.GetGamutConversionFn(src_gamut=hdr_meta.gamut,
                                                dst_gamut=dst_gamut)
    hdr_peak_nits = utils.GetReferenceDisplayPeakLuminanceInNits(hdr_meta.oetf)

    bt2100_luminance_fn = utils.GetLuminanceFn(dst_gamut)
    sdr_inv_oetf = utils.GetInvOETFFn(sdr_meta.oetf)
    sdr_hdr_gamut_conv = utils.GetGamutConversionFn(
        src_gamut=sdr_meta.gamut, dst_gamut=dst_gamut)

    if hdr_meta.oetf != utils.OETF.LINEAR:
        img_hdr_norm = img_hdr.to(DTYPE) / float((1 << hdr_meta.bit_depth) - 1)
        img_hdr_lin = hdr_inv_oetf(img_hdr_norm)
        img_hdr_lin = hdr_ootf(img_hdr_lin, hdr_luminance_fn)
        img_hdr_lin = hdr_gamut_conv(img_hdr_lin)
        img_hdr_lin = torch.clamp(img_hdr_lin, 0., 1.)
        img_hdr_norm = hdr_inv_ootf(img_hdr_lin, hdr_luminance_fn)
        img_hdr_norm = hdr_oetf(img_hdr_norm)
    else:
        img_hdr_lin = img_hdr.to(DTYPE) / img_hdr.to(DTYPE).max()
        img_hdr_lin = hdr_gamut_conv(img_hdr_lin)
        img_hdr_lin = torch.clamp(img_hdr_lin, 0., 1.)
        img_hdr_norm = img_hdr_lin.clone()

    img_sdr_norm = img_sdr.to(DTYPE) / float((1 << sdr_meta.bit_depth) - 1)

    if c3:
        img_hdr_lum = img_hdr_lin
    else:
        img_hdr_lum = bt2100_luminance_fn(img_hdr_lin)

    img_sdr_lin = sdr_inv_oetf(img_sdr_norm)
    img_sdr_lin = sdr_hdr_gamut_conv(img_sdr_lin)
    img_hdr_recon = reconstruct_hdr(
        img_sdr, gainmap, hdr_meta, sdr_meta, c3, dst_gamut)["img_hdr_recon"]

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
    psnr_img = psnr(img_hdr_norm, img_hdr_recon)

    if psnr_lum.isnan():
        logger.info("PSNR luminance was nan")
    if psnr_img.isnan():
        logger.info("PSNR image was nan")

    logger.info(f"PSNR luminance: {psnr_lum} dB")
    logger.info(f"PSNR image: {psnr_img} dB")

    return {
        "psnr_lum": psnr_lum,
        "psnr_img": psnr_img,
    }


def reconstruct_hdr(img_sdr: torch.Tensor,
                    gainmap: torch.Tensor,
                    hdr_meta: ImageMetadata,
                    sdr_meta: ImageMetadata,
                    c3: bool = False,
                    dst_gamut: utils.Gamut = utils.Gamut.BT2100
                    ) -> dict[str, torch.Tensor]:
    hdr_oetf = utils.GetOETFFn(hdr_meta.oetf)
    hdr_inv_ootf = utils.GetInvOOTFFn(hdr_meta.oetf)
    hdr_luminance_fn = utils.GetLuminanceFn(hdr_meta.gamut)
    hdr_peak_nits = utils.GetReferenceDisplayPeakLuminanceInNits(hdr_meta.oetf)

    sdr_inv_oetf = utils.GetInvOETFFn(sdr_meta.oetf)
    sdr_hdr_gamut_conv = utils.GetGamutConversionFn(
        src_gamut=sdr_meta.gamut, dst_gamut=dst_gamut)

    img_sdr_norm = img_sdr.to(DTYPE) / float((1 << sdr_meta.bit_depth) - 1)
    img_sdr_lin = sdr_inv_oetf(img_sdr_norm)
    img_sdr_lin = sdr_hdr_gamut_conv(img_sdr_lin)
    img_sdr_lin = torch.clamp(img_sdr_lin, 0., 1.)
    img_hdr_recon = apply_gain(
        img_sdr_lin * utils.SDR_WHITE_NITS, gainmap, sdr_meta.map_gamma,
        sdr_meta.min_content_boost, sdr_meta.max_content_boost,
        sdr_meta.hdr_offset, sdr_meta.sdr_offset,
        sdr_meta.affine_min, sdr_meta.affine_max)
    img_hdr_recon /= hdr_peak_nits
    img_hdr_recon = torch.clamp(img_hdr_recon, 0., 1.)
    img_hdr_recon = hdr_inv_ootf(img_hdr_recon, hdr_luminance_fn)
    img_hdr_recon = hdr_oetf(img_hdr_recon)

    return {
        "img_hdr_recon": img_hdr_recon,
    }
