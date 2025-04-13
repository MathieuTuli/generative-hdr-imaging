from pathlib import Path

import numpy as np
import fire
import cv2

from hdr_to_gainmap import generate_gainmap
from image_io import load_hdr_image


def hdr_to_gainmap(
        fname: Path,
        clip_percentile: float = 0.95,
        hdr_offset: float = 0.015625,
        sdr_offset: float = 0.015625,
        min_content_boost: float = 1.0,
        max_content_boost: float = 4.0,
        map_gamma: float = 1.0,
        hdr_capacity_min: float = 1.0,
        hdr_capacity_max: float = 4.0):

    img_hdr, meta = load_hdr_image(fname)
    meta.clip_percentile = clip_percentile
    meta.hdr_offset = hdr_offset
    meta.sdr_offset = sdr_offset
    meta.min_content_boost = min_content_boost
    meta.max_content_boost = max_content_boost
    meta.map_gamma = map_gamma
    meta.hdr_capacity_min = hdr_capacity_min
    meta.hdr_capacity_max = hdr_capacity_max

    data = generate_gainmap(fname, clip_percentile, map_gamma, 1, 8)

    np.save("gainmap.npy", data["gainmap"][:, :, :1])
    cv2.imwrite("sdr.png",
                cv2.cvtColor(
                    np.clip(data["img_sdr"] * 255. + 0.5, 0, 255).astype(np.uint8),
                    cv2.COLOR_BGR2RGB))


if __name__ == '__main__':
    fcns = {
        "hdr_to_gainmap": hdr_to_gainmap,
    }
    fire.Fire(fcns)
