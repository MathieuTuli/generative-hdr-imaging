from loguru import logger

import numpy as np
import torch

from ioops import ImageMetadata
from metrics import psnr
import utils


def compute_gain(hdr_y_nits: torch.Tensor, sdr_y_nits: torch.Tensor,
                 hdr_offset: float = 0.015625,
                 sdr_offset: float = 0.015625):
    gain = torch.log2((hdr_y_nits + hdr_offset) / (sdr_y_nits + sdr_offset))
    mask_low = sdr_y_nits < 2. / 255.0
    gain[mask_low] = torch.minimum(gain[mask_low],
                                   torch.tensor(2.3, dtype=torch.float32,
                                                device=hdr_y_nits.device))
    return gain


def affine_map_gain(gainlog2: torch.Tensor,
                    min_gainlog2: float,
                    max_gainlog2: float,
                    map_gamma: float):
    mapped_val = (gainlog2 - min_gainlog2) / (max_gainlog2 - min_gainlog2)
    mapped_val = torch.clip(mapped_val, 0., 1.)
    if (map_gamma != 1.0):
        mapped_val = torch.pow(mapped_val,
                               torch.tensor(map_gamma, dtype=torch.float32,
                                            device=gainlog2.device))
    return mapped_val


def recompute_hdr_luminance(sdr_luminance: torch.Tensor,
                            gain: torch.Tensor,
                            hdr_offset: float = 0.015625,
                            sdr_offset: float = 0.015625) -> torch.Tensor:
    gain_factor = torch.exp2(gain)
    return torch.multiply(sdr_luminance + sdr_offset, gain_factor) - hdr_offset


def undo_affine_map_gain(gain: torch.Tensor,
                         map_gamma: float,
                         min_content_boost: float,
                         max_content_boost: float) -> torch.Tensor:
    effective_gain = torch.power(
        gain, 1.0 / map_gamma) if map_gamma != 1.0 else gain
    log_min = np.log2(min_content_boost)
    log_max = np.log2(max_content_boost)
    return torch.multiply(log_min, 1.0 - effective_gain
                          ) + torch.multiply(log_max, effective_gain)


def apply_gain(e: torch.Tensor,
               gain: torch.Tensor,
               map_gamma: float,
               min_content_boost: float,
               max_content_boost: float,
               hdr_offset: float = 0.015625,
               sdr_offset: float = 0.015625) -> torch.Tensor:
    effective_gain = torch.power(
        gain, 1.0 / map_gamma) if map_gamma != 1.0 else gain
    log_min = np.log2(min_content_boost)
    log_max = np.log2(max_content_boost)
    log_boost = torch.multiply(log_min, 1.0 - effective_gain) + \
        torch.multiply(log_max, effective_gain)
    gain_factor = torch.exp2(log_boost)
    return torch.multiply(e + sdr_offset, gain_factor) - hdr_offset


def generate_gainmap(img_hdr: torch.Tensor,
                     meta: ImageMetadata,
                     ):
    hdr_inv_oetf = utils.GetInvOETFFn(meta.oetf)
    hdr_ootf = utils.GetOOTFFn(meta.oetf)
    hdr_gamut_conv = utils.GetGamutConversionFn(utils.Gamut.BT2100, meta.gamut)
    hdr_luminance_fn = utils.GetLuminanceFn(meta.gamut)
    bt2100_luminance_fn = utils.GetLuminanceFn(utils.Gamut.BT2100)
    hdr_peak_nits = utils.GetReferenceDisplayPeakLuminanceInNits(meta.oetf)
    sdr_gamut_conv = utils.GetGamutConversionFn(src_gamut=utils.Gamut.BT2100,
                                                dst_gamut=utils.Gamut.BT709)
    sdr_inv_oetf = utils.GetInvOETFFn(utils.OETF.SRGB)
    sdr_oetf = utils.GetOETFFn(utils.OETF.SRGB)
    sdr_hdr_gamut_conv = utils.GetGamutConversionFn(src_gamut=utils.Gamut.BT709,
                                                    dst_gamut=utils.Gamut.BT2100)

    height, width, channels = img_hdr.shape
    logger.debug(
        f"Image min/max/mean {img_hdr.to(torch.float32).min()}/" +
        f"{img_hdr.to(torch.float32).max()}/{img_hdr.to(torch.float32).mean()}")

    img_hdr_norm = img_hdr.to(torch.float32) / \
        float((1 << meta.bit_depth) - 1)

    img_hdr_linear = hdr_inv_oetf(img_hdr_norm)
    img_hdr_linear = hdr_ootf(img_hdr_linear, hdr_luminance_fn)
    img_hdr_linear = hdr_gamut_conv(img_hdr_linear)
    img_hdr_linear[img_hdr_linear < 0.] = 0.

    clip_value = torch.quantile(img_hdr_linear.reshape(-1),
                                meta.clip_percentile)
    logger.debug(
        f"HDR {meta.clip_percentile}th-percentile clip value: {clip_value}")

    img_sdr = sdr_gamut_conv(img_hdr_linear)
    img_sdr[img_sdr < 0.] = 0.
    img_sdr = torch.minimum(img_sdr, clip_value) / clip_value
    img_sdr = sdr_oetf(img_sdr)

    img_sdr_lin = torch.clip(img_sdr * 255.0 + 0.5, 0, 255).to(torch.uint8)
    img_sdr_lin = img_sdr_lin.to(torch.float32) / 255.0

    img_sdr_lin = sdr_inv_oetf(img_sdr_lin)
    img_sdr_lin = sdr_hdr_gamut_conv(img_sdr_lin)

    sdr_y_nits = bt2100_luminance_fn(img_sdr_lin) * utils.SDR_WHITE_NITS
    hdr_y_nits = bt2100_luminance_fn(img_hdr_linear) * hdr_peak_nits
    gainmap = compute_gain(hdr_y_nits, sdr_y_nits).to(torch.float32)
    min_gain = gainmap.min().item()
    max_gain = gainmap.max().item()

    min_gain = np.clip(min_gain, -14.3, 15.6)
    max_gain = np.clip(max_gain, -14.3, 15.6)

    if meta.min_content_boost is not None:
        min_gain = np.log2(meta.min_content_boost)
    else:
        meta.min_content_boost = np.exp2(min_gain)
    if meta.max_content_boost is not None:
        max_gain = np.log2(meta.max_content_boost)
    else:
        meta.max_content_boost = np.exp2(max_gain)

    if abs(max_gain - min_gain) < 1.0e-8:
        max_gain += 0.1

    logger.debug(f"max/min gain (log2): {max_gain}/{min_gain}")
    logger.debug(
        f"max/min gain (exp2): {np.exp2(max_gain)}/{np.exp2(min_gain)}")

    gainmap = affine_map_gain(gainmap, min_gain, max_gain, meta.map_gamma)

    sdr_meta = ImageMetadata.from_dict(meta.to_dict())
    sdr_meta.gamut = utils.Gamut.BT709
    sdr_meta.oetf = utils.OETF.SRGB
    sdr_meta.bit_depth = 8
    return {
        "gainmap": gainmap,
        "img_hdr_linear": img_hdr_linear,
        "img_sdr": img_sdr,
        "hdr_metadata": meta,
        "sdr_metadata": sdr_meta
    }


def gainmap_sdr_to_hdr(img_sdr: torch.Tensor,
                       gainmap: torch.Tensor,
                       meta: ImageMetadata):

    hdr_peak_nits = utils.GetReferenceDisplayPeakLuminanceInNits(
        utils.OETF.HLG)

    sdr_inv_oetf = utils.GetInvOETFFn(meta.oetf)
    sdr_hdr_gamut_conv = utils.GetGamutConversionFn(
        src_gamut=meta.gamut, dst_gamut=utils.Gamut.BT2100)

    img_sdr_norm = img_sdr.to(torch.float32) / \
        float((1 << meta.bit_depth) - 1)
    img_sdr_lin = sdr_inv_oetf(img_sdr_norm)
    img_sdr_lin = sdr_hdr_gamut_conv(img_sdr_lin)
    img_hdr_recon = apply_gain(
        img_sdr_lin, gainmap, meta.map_gamma,
        meta.min_content_boost, meta.max_content_boost,
        meta.hdr_offset, meta.sdr_offset)
    img_hdr_recon *= utils.SDR_WHITE_NITS / hdr_peak_nits
    img_hdr_recon = torch.clip(img_hdr_recon, 0., 1.)

    return {"img_hdr_lin": img_hdr_recon}


def compare_hdr_to_uhdr(img_hdr: torch.Tensor,
                        img_sdr: torch.Tensor,
                        gainmap: torch.Tensor,
                        hdr_meta: ImageMetadata,
                        sdr_meta: ImageMetadata):
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

    img_hdr_norm = img_hdr.to(torch.float32) / \
        float((1 << hdr_meta.bit_depth) - 1)
    img_sdr_norm = img_sdr.to(torch.float32) / \
        float((1 << sdr_meta.bit_depth) - 1)

    img_hdr_lin = hdr_inv_oetf(img_hdr_norm)
    img_hdr_lin = hdr_ootf(img_hdr_lin, hdr_luminance_fn)
    img_hdr_lin = hdr_gamut_conv(img_hdr_lin)
    img_hdr_lin = torch.clip(img_hdr_lin, 0., 1.)
    img_hdr_lum = bt2100_luminance_fn(img_hdr_lin)

    img_sdr_lin = sdr_inv_oetf(img_sdr_norm)
    img_sdr_lin = sdr_hdr_gamut_conv(img_sdr_lin)
    img_hdr_recon = apply_gain(
        img_sdr_lin, gainmap, sdr_meta.map_gamma,
        sdr_meta.min_content_boost, sdr_meta.max_content_boost,
        sdr_meta.hdr_offset, sdr_meta.sdr_offset)
    img_hdr_recon *= utils.SDR_WHITE_NITS / hdr_peak_nits
    img_hdr_recon = torch.clip(img_hdr_recon, 0., 1.)

    img_sdr_lum = bt2100_luminance_fn(img_sdr_lin) * utils.SDR_WHITE_NITS
    gain_log2 = undo_affine_map_gain(gainmap, sdr_meta.map_gamma,
                                     sdr_meta.min_content_boost,
                                     sdr_meta.max_content_boost)
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
