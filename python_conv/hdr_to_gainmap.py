from loguru import logger

import numpy as np

from image_io import ImageMetadata
from utils import Gamut, OETF
import utils


def compute_gain(hdr_y_nits: float, sdr_y_nits: float,
                 hdr_offset: float = 0.015625,
                 sdr_offset: float = 0.015625):
    gain = np.log2((hdr_y_nits + hdr_offset) / (sdr_y_nits + sdr_offset))
    mask_low = sdr_y_nits < 2. / 255.0
    gain[mask_low] = np.minimum(gain[mask_low], 2.3)
    return gain


def affine_map_gain(gainlog2: float,
                    min_gainlog2: float,
                    max_gainlog2: float,
                    map_gamma: float):
    mapped_val = (gainlog2 - min_gainlog2) / (max_gainlog2 - min_gainlog2)
    mapped_val = np.clip(mapped_val, 0., 1.)
    if (map_gamma != 1.0):
        mapped_val = np.power(mapped_val, map_gamma)
    return mapped_val


def generate_gainmap(img_hdr: np.ndarray,
                     meta: ImageMetadata,
                     ):
    hdr_inv_oetf = utils.GetInvOETFFn(meta.oetf)
    hdr_ootf = utils.GetOOTFFn(meta.oetf)
    hdr_gamut_conv = utils.GetGamutConversionFn(Gamut.BT2100, meta.gamut)
    hdr_luminance_fn = utils.GetLuminanceFn(meta.gamut)
    bt2100_luminance_fn = utils.GetLuminanceFn(Gamut.BT2100)
    hdr_peak_nits = utils.GetReferenceDisplayPeakLuminanceInNits(meta.oetf)
    sdr_gamut_conv = utils.GetGamutConversionFn(src_gamut=Gamut.BT2100,
                                                dst_gamut=Gamut.BT709)
    sdr_inv_oetf = utils.GetInvOETFFn(OETF.SRGB)
    sdr_oetf = utils.GetOETFFn(OETF.SRGB)
    sdr_hdr_gamut_conv = utils.GetGamutConversionFn(src_gamut=Gamut.BT709,
                                                    dst_gamut=Gamut.BT2100)

    height, width, channels = img_hdr.shape
    logger.debug(
        f"Image min/max/mean {img_hdr.min()}/{img_hdr.max()}/{img_hdr.mean()}")

    img_hdr_norm = img_hdr.astype(np.float32) / \
        float((1 << meta.bit_depth) - 1)
    img_hdr_linear = np.full_like(img_hdr, -1, dtype=np.float32)
    img_sdr = np.full_like(img_hdr, -1, dtype=np.float32)
    gainmap = np.full_like(img_hdr, -1, dtype=np.float32)
    gainmap_affine = np.full_like(img_hdr, -1, dtype=np.uint8)

    img_hdr_linear = hdr_inv_oetf(img_hdr_norm)
    img_hdr_linear = hdr_ootf(img_hdr_linear, hdr_luminance_fn)
    img_hdr_linear = hdr_gamut_conv(img_hdr_linear)
    img_hdr_linear[img_hdr_linear < 0.] = 0.

    all_values = img_hdr_linear.reshape(-1)
    clip_value = np.percentile(all_values, meta.clip_percentile * 100)
    logger.debug(
        f"HDR {meta.clip_percentile}th-percentile clip value: {clip_value}")

    min_gain, max_gain = 255.0, -255.0
    img_sdr = sdr_gamut_conv(img_hdr_linear)
    img_sdr[img_sdr < 0.] = 0.
    img_sdr = np.minimum(img_sdr, clip_value) / clip_value
    img_sdr = sdr_oetf(img_sdr)

    img_sdr_lin = np.clip(img_sdr * 255.0 + 0.5, 0, 255).astype(np.uint8)
    img_sdr_lin = img_sdr_lin.astype(np.float32) / 255.0

    img_sdr_lin = sdr_inv_oetf(img_sdr_lin)
    img_sdr_lin = sdr_hdr_gamut_conv(img_sdr_lin)

    sdr_y_nits = bt2100_luminance_fn(img_sdr_lin) * utils.SDR_WHITE_NITS
    hdr_y_nits = bt2100_luminance_fn(img_hdr_linear) * hdr_peak_nits
    gainmap = compute_gain(hdr_y_nits, sdr_y_nits).astype(np.float32)
    min_gain = min(gainmap.min(), min_gain)
    max_gain = max(gainmap.max(), max_gain)

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
    mapped_gain = gainmap * 255.
    gainmap_affine = np.clip(mapped_gain + 0.5, 0, 255).astype(np.uint8)

    return {
        "gainmap": gainmap,
        "gainmap_affine": gainmap_affine,
        "img_hdr_linear": img_hdr_linear,
        "img_sdr": img_sdr,
        "metadata": meta
    }
