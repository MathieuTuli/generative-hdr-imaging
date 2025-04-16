from pathlib import Path

import multiprocessing
import sys

from loguru import logger

import torch
import fire

from ioops import (ImageMetadata, load_hdr_image,
                   save_png, save_tensor, load_image)
from hdr_to_gainmap import generate_gainmap, compare_hdr_to_uhdr


class App:
    def __init__(self, debug: bool = False):
        if not debug:
            logger.remove()
            logger.add(sys.stderr, level="INFO")

    def hdr_to_gainmap(
            self,
            fname: str,
            outdir: str,
            clip_percentile: float = 0.95,
            min_max_quantile: float = 0.02,
            affine_min: float = -1.,
            affine_max: float = 1.,
            hdr_offset: tuple[float, float, float] = (0.015625, 0.015625, 0.015625),  # noqa
            sdr_offset: tuple[float, float, float] = (0.015625,  0.015625,  0.015625),  # noqa
            min_content_boost: None | tuple[float, float, float] = None,
            max_content_boost: None | tuple[float, float, float] = None,
            map_gamma: tuple[float, float, float] = (1.0, 1.0, 1.0),
            hdr_capacity_min: float = 1.0,
            hdr_capacity_max: float = 4.0,
            c3: bool = False,
            cuda: bool = False,):
        if min_content_boost is not None:
            assert isinstance(min_content_boost, tuple), \
                f"Got {min_content_boost}"
        if max_content_boost is not None:
            assert isinstance(max_content_boost, tuple), \
                f"Got {max_content_boost}"
        assert isinstance(map_gamma, tuple), f"Got {map_gamma}"
        assert isinstance(hdr_offset, tuple), f"Got {hdr_offset}"
        assert isinstance(sdr_offset, tuple), f"Got {sdr_offset}"
        logger.info(f"Running hdr_to_gainmap fpr {fname}")
        if isinstance(fname, str):
            fname = Path(fname)

        img_hdr, meta = load_hdr_image(fname)
        if cuda:
            img_hdr = img_hdr.to("cuda")
        meta.clip_percentile = clip_percentile
        meta.min_max_quantile = min_max_quantile
        meta.affine_min = affine_min
        meta.affine_max = affine_max
        meta.hdr_offset = hdr_offset
        meta.sdr_offset = sdr_offset
        meta.min_content_boost = min_content_boost
        meta.max_content_boost = max_content_boost
        meta.map_gamma = map_gamma
        meta.hdr_capacity_min = hdr_capacity_min
        meta.hdr_capacity_max = hdr_capacity_max

        data = generate_gainmap(img_hdr, meta, c3)

        outdir = Path(outdir)
        outdir.mkdir(parents=True, exist_ok=True)
        save_tensor(outdir / f"{fname.stem}__gainmap.pt", data["gainmap"])
        save_png(outdir / f"{fname.stem}__gainmap.png", data["gainmap"])
        save_tensor(outdir / f"{fname.stem}__hdr_linear.pt", data["img_hdr_linear"])  # noqa
        save_png(outdir / f"{fname.stem}__sdr.png", data["img_sdr"])
        data["hdr_metadata"].save(outdir / f"{fname.stem}__hdr_metadata.json")
        data["sdr_metadata"].save(outdir / f"{fname.stem}__sdr_metadata.json")

    def hdr_to_gainmap_batched(
            self,
            indir: str,
            outdir: str,
            proc: int,
            clip_percentile: float = 0.95,
            min_max_quantile: float = 0.02,
            affine_min: float = -1.,
            affine_max: float = 1.,
            hdr_offset: tuple[float, float, float] = (0.015625, 0.015625, 0.015625),  # noqa
            sdr_offset: tuple[float, float, float] = (0.015625,  0.015625,  0.015625),  # noqa
            min_content_boost: None | tuple[float, float, float] = None,
            max_content_boost: None | tuple[float, float, float] = None,
            map_gamma: tuple[float, float, float] = (1.0, 1.0, 1.0),
            hdr_capacity_min: float = 1.0,
            hdr_capacity_max: float = 4.0,
            c3: bool = False,
            cuda: bool = False,):
        if min_content_boost is not None:
            assert isinstance(min_content_boost, tuple), \
                f"Got {min_content_boost}"
        if max_content_boost is not None:
            assert isinstance(max_content_boost, tuple), \
                f"Got {max_content_boost}"
        assert isinstance(map_gamma, tuple), f"Got {map_gamma}"
        assert isinstance(hdr_offset, tuple), f"Got {hdr_offset}"
        assert isinstance(sdr_offset, tuple), f"Got {sdr_offset}"

        fnames = list(Path(indir).iterdir())
        args = [(fname, outdir, clip_percentile, min_max_quantile,
                 affine_min, affine_max, hdr_offset, sdr_offset,
                 min_content_boost, max_content_boost, map_gamma,
                 hdr_capacity_min, hdr_capacity_max, c3, cuda)
                for fname in fnames]
        with multiprocessing.Pool(processes=proc) as pool:
            pool.starmap(self.hdr_to_gainmap, args)

    def compare_reconstruction(
            self,
            hdr_path: Path,
            sdr_path: Path,
            gainmap_path: Path,
            sdr_metadata: Path,
            c3: bool = False,
    ):
        img_hdr, hdr_meta = load_hdr_image(hdr_path)
        img_sdr = load_image(sdr_path)
        sdr_meta = ImageMetadata.from_json(sdr_metadata)
        gainmap = torch.load(gainmap_path, weights_only=True)
        compare_hdr_to_uhdr(img_hdr, img_sdr, gainmap, hdr_meta, sdr_meta, c3)


if __name__ == '__main__':
    fire.Fire(App)
