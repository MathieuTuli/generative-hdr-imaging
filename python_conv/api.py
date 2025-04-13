from pathlib import Path

import sys

from loguru import logger

import numpy as np
import fire

from ioops import (ImageMetadata, load_hdr_image,
                   save_png, save_npy, load_sdr_image)
from hdr_to_gainmap import generate_gainmap, compare_hdr_to_uhdr


def hdr_to_gainmap(
        fname: Path,
        outdir: Path,
        clip_percentile: float = 0.95,
        hdr_offset: float = 0.015625,
        sdr_offset: float = 0.015625,
        min_content_boost: float = 1.0,
        max_content_boost: float = 4.0,
        map_gamma: float = 1.0,
        hdr_capacity_min: float = 1.0,
        hdr_capacity_max: float = 4.0,
        debug: bool = False):
    if not debug:
        logger.remove()
        logger.add(sys.stderr, level="INFO")

    img_hdr, meta = load_hdr_image(fname)
    meta.clip_percentile = clip_percentile
    meta.hdr_offset = hdr_offset
    meta.sdr_offset = sdr_offset
    meta.min_content_boost = min_content_boost
    meta.max_content_boost = 8
    meta.map_gamma = map_gamma
    meta.hdr_capacity_min = hdr_capacity_min
    meta.hdr_capacity_max = hdr_capacity_max

    data = generate_gainmap(img_hdr, meta)

    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    save_npy(outdir / "gainmap.npy", data["gainmap"][:, :, :1])
    save_npy(outdir / "img_hdr_linear.npy", data["img_hdr_linear"])
    save_png(outdir / "img_sdr.png", data["img_sdr"])
    data["hdr_metadata"].save(outdir / "hdr_metadata.json")
    data["sdr_metadata"].save(outdir / "sdr_metadata.json")


def compare_reconstruction(
        hdr_path: Path,
        sdr_path: Path,
        gainmap_path: Path,
        sdr_metadata: Path,
        debug: bool = False
):
    if not debug:
        logger.remove()
        logger.add(sys.stderr, level="INFO")
    img_hdr, hdr_meta = load_hdr_image(hdr_path)
    img_sdr = load_sdr_image(sdr_path)
    sdr_meta = ImageMetadata.from_json(sdr_metadata)
    gainmap = np.load(gainmap_path)
    compare_hdr_to_uhdr(img_hdr, img_sdr, gainmap, hdr_meta, sdr_meta)


if __name__ == '__main__':
    fcns = {
        "hdr_to_gainmap": hdr_to_gainmap,
        "compare_reconstruction": compare_reconstruction,
    }
    fire.Fire(fcns)
