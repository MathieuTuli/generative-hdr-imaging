from dataclasses import dataclass
from pathlib import Path

import subprocess
import json

from loguru import logger

import numpy as np

import cv2


from utils import Gamut, OETF
import utils


@dataclass
class ImageMetadata:
    gamut: Gamut
    oetf: OETF
    bit_depth: int
    clip_percentile: float = 1.0
    hdr_offset: float = 0.015625
    sdr_offset: float = 0.015625
    min_content_boost: float = 1.0
    max_content_boost: float = 4.0
    map_gamma: float = 1.0
    hdr_capacity_min: float = 1.0
    hdr_capacity_max: float = 4.0


# @dataclass
# class Image:
#     width: int = 0
#     height: int = 0
#     bit_depth: int = 0
#     bytes_per_row: int = 0
#     color_type: int = 0
#     channels: int = 0
#     data: np.ndarray | None = None
#     metadata: ImageMetadata | None = None

def load_hdr_image(fname: Path):
    image = cv2.imread(str(fname), cv2.IMREAD_UNCHANGED)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    if image is None:
        raise ValueError(f"Failed to load image from path: {fname}")
    try:
        result = subprocess.run(
            ["exiftool", "-j", fname],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=True,
            text=True
        )
        # Parse the JSON output
        metadata_list = json.loads(result.stdout)
        metadata_raw = metadata_list[0] if metadata_list else {}
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"ExifTool returned an error: {e.stderr}")

    bit_depth = metadata_raw["BitDepth"]
    oetf = metadata_raw["TransferCharacteristics"]
    if oetf.find("HLG") > 0 or oetf.find("2020") > 0:
        oetf = OETF.HLG
    elif oetf.find("PQ") > 0 or oetf.find("2084") > 0:
        oetf = OETF.PQ
    elif oetf.find("709") > 0 or oetf.find("sRGB") > 0:
        oetf = OETF.SRGB
    else:
        raise ValueError("Unknown TransferCharacteristics {oetf}")
    gamut = metadata_raw["ColorPrimaries"]
    if gamut.find("709") > 0 or gamut.find("sRGB") > 0:
        gamut = Gamut.BT709
    elif gamut.find("P3") > 0 or gamut.find("SMPTE") > 0:
        gamut = Gamut.P3
    elif gamut.find("2100") > 0 or gamut.find("2020") > 0:
        gamut = Gamut.BT2100
    else:
        raise ValueError("Unknown ColorPrimaries {gamut}")
    metadata = ImageMetadata(gamut=gamut, oetf=oetf, bit_depth=bit_depth)
    return image, metadata


def compute_gain(hdr_y_nits: float, sdr_y_nits: float,
                 hdr_offset: float = 0.015625,
                 sdr_offset: float = 0.015625):
    gain = np.log2((hdr_y_nits + hdr_offset) / (sdr_y_nits + sdr_offset))
    if sdr_y_nits < 2. / 255.0:
        gain = np.minimum(gain, 2.3)
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


def generate_gainmap(fname: Path,
                     clip_percentile: float,
                     map_gamma: float,
                     min_content_boost: None | float = None,
                     max_content_boost: None | float = None,
                     ):
    img_hdr, meta = load_hdr_image(fname)
    meta.clip_percentile = clip_percentile
    meta.map_gamma = map_gamma

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
        float((1 << meta.bit_depth) - 1.)
    img_hdr_linear = np.full_like(img_hdr, -1, dtype=np.float32)
    img_sdr = np.full_like(img_hdr, -1, dtype=np.float32)
    gainmap = np.full_like(img_hdr, -1, dtype=np.float32)
    gainmap_affine = np.full_like(img_hdr, -1, dtype=np.uint8)

    for i, row in enumerate(img_hdr_norm):
        for j, pix in enumerate(row):
            pix_hdr_rgb = hdr_inv_oetf(pix)
            pix_hdr_rgb = hdr_ootf(pix_hdr_rgb, hdr_luminance_fn)
            pix_hdr_rgb = hdr_gamut_conv(pix_hdr_rgb)
            pix_hdr_rgb[pix_hdr_rgb < 0.] = 0.
            img_hdr_linear[i, j] = pix_hdr_rgb

    all_values = img_hdr_linear.reshape(-1)
    # clip_value = np.percentile(all_values, clip_percentile * 100)
    clip_value = sorted(all_values)[int(len(all_values) * clip_percentile)]
    logger.debug(
        f"HDR {clip_percentile}th-percentile clip value: {clip_value}")

    min_gain, max_gain = 255.0, -255.0
    for i, row in enumerate(img_hdr_linear):
        for j, pix in enumerate(row):
            pix_sdr_rgb = sdr_gamut_conv(pix)
            pix_hdr_rgb[pix_sdr_rgb < 0.] = 0.
            pix_sdr_rgb = np.minimum(pix_sdr_rgb, clip_value) / clip_value

            pix_srgb_gamma = sdr_oetf(pix_sdr_rgb)
            img_sdr[i, j] = pix_srgb_gamma

            pix_srgb_gamma = np.clip(
                pix_srgb_gamma * 255.0 + 0.5, 0, 255).astype(np.uint8)
            pix_srgb_gamma = pix_srgb_gamma.astype(np.float32) / 255.0

            pix_sdr_rgb = sdr_inv_oetf(pix_srgb_gamma)
            pix_sdr_rgb_bt2100 = sdr_hdr_gamut_conv(pix_sdr_rgb)

            sdr_y_nits = bt2100_luminance_fn(
                pix_sdr_rgb_bt2100) * utils.SDR_WHITE_NITS
            hdr_y_nits = bt2100_luminance_fn(pix) * hdr_peak_nits
            gain = compute_gain(hdr_y_nits, sdr_y_nits)
            min_gain = min(gain, min_gain)
            max_gain = max(gain, max_gain)
            gainmap[i, j] = np.array([gain, gain, gain], dtype=np.float32)

    min_gain = np.clip(min_gain, -14.3, 15.6)
    max_gain = np.clip(max_gain, -14.3, 15.6)

    # min_content_boost = 1.
    # max_content_boost = 8.0
    if min_content_boost is not None:
        min_gain = np.log2(min_content_boost)
    if max_content_boost is not None:
        max_gain = np.log2(max_content_boost)

    if abs(max_gain - min_gain) < 1.0e-8:
        max_gain += 0.1

    meta.min_content_boost = np.exp2(min_gain)
    meta.max_content_boost = np.exp2(max_gain)

    logger.debug(f"max/min gain (log2): {max_gain}/{min_gain}")
    logger.debug(
        f"max/min gain (exp2): {np.exp2(max_gain)}/{np.exp2(min_gain)}")

    for i, row in enumerate(gainmap):
        for j, pix in enumerate(row):
            gainmap[i, j] = affine_map_gain(pix, min_gain, max_gain, map_gamma)
        mapped_gain = gainmap[i, j] * 255.
        gainmap_affine[i, j] = np.clip(
            mapped_gain + 0.5, 0, 255).astype(np.uint8)

    return {
        "gainmap": gainmap,
        "gainmap_affine": gainmap_affine,
        "img_hdr_linear": img_hdr_linear,
        "img_sdr": img_sdr,
    }


if __name__ == "__main__":
    data = generate_gainmap(Path("images/test_hdr_png.png"),
                            clip_percentile=0.95, map_gamma=1.0,
                            min_content_boost=1.0,
                            max_content_boost=8.0)
    np.save("gainmap.npy", data["gainmap"][:, :, :1])
    cv2.imwrite("sdr.png", cv2.cvtColor(
        np.clip(data["img_sdr"] * 255. + 0.5, 0., 255).astype(np.uint8),
        cv2.COLOR_BGR2RGB))
