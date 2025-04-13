from PIL import Image
from loguru import logger

import numpy as np

from ioops import ImageMetadata
from metrics import psnr
import utils


def recompute_hdr_luminance(sdr_luminance: np.ndarray,
                            gain: np.ndarray,
                            hdr_offset: float = 0.015625,
                            sdr_offset: float = 0.015625) -> np.ndarray:
    gain_factor = np.exp2(gain)
    return np.multiply(sdr_luminance + sdr_offset, gain_factor) - hdr_offset


def undo_affine_map_gain(gain: np.ndarray,
                         map_gamma: float,
                         min_content_boost: float,
                         max_content_boost: float) -> np.ndarray:
    effective_gain = np.power(
        gain, 1.0 / map_gamma) if map_gamma != 1.0 else gain
    log_min = np.log2(min_content_boost)
    log_max = np.log2(max_content_boost)
    return np.multiply(log_min, 1.0 - effective_gain
                       ) + np.multiply(log_max, effective_gain)


def apply_gain(e: np.ndarray,
               gain: np.ndarray,
               map_gamma: float,
               min_content_boost: float,
               max_content_boost: float,
               hdr_offset: float = 0.015625,
               sdr_offset: float = 0.015625) -> np.ndarray:
    effective_gain = np.power(
        gain, 1.0 / map_gamma) if map_gamma != 1.0 else gain
    log_min = np.log2(min_content_boost)
    log_max = np.log2(max_content_boost)
    log_boost = np.multiply(log_min, 1.0 - effective_gain) + \
        np.multiply(log_max, effective_gain)
    gain_factor = np.exp2(log_boost)
    return np.multiply(e + sdr_offset, gain_factor) - hdr_offset


def gainmap_sdr_to_hdr(img_sdr: np.ndarray,
                       gainmap: np.ndarray,
                       meta: ImageMetadata):

    sdr_gamut_conv = utils.GetGamutConversionFn(
        src_gamut=meta.gamut,
        dst_gamut=utils.Gamut.BT2100)
    bt2100_luminance_fn = utils.GetLuminanceFn(utils.Gamut.BT2100)
    sdr_gamut_conv = utils.GetGamutConversionFn(meta.gamut)
    # REVISIT: right now hard coded, but for comparison, it should be the
    # input HDR transfer
    hdr_inv_ootf = utils.GetInvOOTFFn(utils.OETF.HLG)
    hdr_oetf = utils.GetOETFFn(utils.OETF.HLG)
    hdr_peak_nits = utils.GetReferenceDisplayPeakLuminanceInNits(
        utils.OETF.HLG)

    sdr_inv_oetf = utils.GetInvOETFFn(meta.oetf)
    sdr_gamut_conv = utils.GetGamutConversionFn(
        src_gamut=meta.gamut, dst_gamut=utils.Gamut.BT2100)

    img_sdr_lin = sdr_inv_oetf(img_sdr)
    img_sdr_lin = sdr_gamut_conv(img_sdr_lin)
    img_hdr_lin = apply_gain(
        img_sdr_lin, gainmap,
        meta.map_gamma,
        meta.min_content_boost,
        meta.max_content_boost,
        meta.hdr_offset,
        meta.sdr_offset
    )
    img_hdr_lin = img_hdr_lin * utils.SDR_WHITE_NITS / hdr_peak_nits
    img_hdr_lin = np.clip(img_hdr_lin, 0., 1.)
    img_hdr = hdr_inv_ootf(img_hdr_lin, bt2100_luminance_fn)
    img_hdr = hdr_oetf(img_hdr)

    return {"img_hdr": img_hdr}


def compare_hdr_to_uhdr(img_hdr: np.ndarray,
                        img_sdr: np.ndarray,
                        gainmap: np.ndarray,
                        hdr_meta: ImageMetadata,
                        sdr_meta: ImageMetadata):
    hdr_inv_oetf = utils.GetInvOETFFn(hdr_meta.oetf)
    hdr_ootf = utils.GetOOTFFn(hdr_meta.oetf)
    hdr_luminance_fn = utils.GetLuminanceFn(hdr_meta.gamut)
    hdr_gamut_conv = utils.GetGamutConversionFn(
        src_gamut=hdr_meta.gamut,
        dst_gamut=utils.Gamut.BT2100)
    hdr_peak_nits = utils.GetReferenceDisplayPeakLuminanceInNits(
        hdr_meta.oetf)

    bt2100_luminance_fn = utils.GetLuminanceFn(utils.Gamut.BT2100)
    sdr_inv_oetf = utils.GetInvOETFFn(sdr_meta.oetf)
    sdr_hdr_gamut_conv = utils.GetGamutConversionFn(
        src_gamut=sdr_meta.gamut, dst_gamut=utils.Gamut.BT2100)

    img_hdr_lin = hdr_inv_oetf(img_hdr)
    img_hdr_lin = hdr_ootf(img_hdr_lin, hdr_luminance_fn)
    img_hdr_lin = hdr_gamut_conv(img_hdr_lin)
    img_hdr_lin = np.clip(img_hdr_lin, 0., 1.)
    img_hdr_lum = bt2100_luminance_fn(img_hdr_lin)

    img_sdr_lin = sdr_inv_oetf(img_sdr)
    img_sdr_lin = sdr_hdr_gamut_conv(img_sdr_lin)
    img_hdr_recon = apply_gain(
        img_sdr_lin, gainmap, sdr_meta.map_gamma,
        sdr_meta.min_content_boost, sdr_meta.max_content_boost,
        sdr_meta.hdr_offset, sdr_meta.sdr_offset)
    img_hdr_recon * utils.SDR_WHITE_NITS / hdr_peak_nits
    img_hdr_recon = np.clip(img_hdr_recon, 0., 1.)

    img_sdr_lum = bt2100_luminance_fn(img_sdr_lin) * utils.SDR_WHITE_NITS
    gain_log2 = undo_affine_map_gain(gainmap, sdr_meta.map_gamma,
                                     sdr_meta.min_content_boost,
                                     sdr_meta.max_content_boost)
    img_hdr_recon_lum = recompute_hdr_luminance(
        img_sdr_lum, gain_log2, sdr_meta.hdr_offset, sdr_meta.sdr_offset)
    img_hdr_recon_lum /= hdr_peak_nits

    psnr_lum = psnr(img_hdr_lum, img_hdr_recon_lum)
    psnr_img = psnr(img_hdr_lin, img_hdr_recon)

    logger.info(f"PSNR luminance: {psnr_lum}")
    logger.info(f"PSNR image: {psnr_img}")
