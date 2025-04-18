from pathlib import Path

import multiprocessing
import sys

from loguru import logger

import numpy as np
import torch
import fire

from ioops import (ImageMetadata, load_hdr_image,
                   save_png, save_tensor, load_image, save_json)
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
            min_max_quantile: float = 0.0,
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
        logger.info(f"Running hdr_to_gainmap for {fname}")
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
        save_png(outdir / f"{fname.stem}__gainmap.png",
                 (data["gainmap"] - affine_min) / (affine_max - affine_min))
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
            min_max_quantile: float = 0.0,
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
            sdr_metadata_path: Path,
            c3: bool = False,) -> dict[str,  torch.Tensor]:
        if isinstance(hdr_path, str):
            hdr_path = Path(hdr_path)
        if isinstance(sdr_path, str):
            sdr_path = Path(sdr_path)
        if isinstance(gainmap_path, str):
            gainmap_path = Path(gainmap_path)
        if isinstance(sdr_metadata_path, str):
            sdr_metadata_path = Path(sdr_metadata_path)

        assert hdr_path.stem.split("__")[0] == sdr_path.stem.split("__")[0]
        assert sdr_path.stem.split("__")[0] == gainmap_path.stem.split("__")[0]
        assert sdr_path.stem.split(
            "__")[0] == sdr_metadata_path.stem.split("__")[0]
        img_hdr, hdr_meta = load_hdr_image(hdr_path)
        img_sdr = load_image(sdr_path)
        sdr_meta = ImageMetadata.from_json(sdr_metadata_path)
        gainmap = torch.load(gainmap_path, weights_only=True)
        return compare_hdr_to_uhdr(
            img_hdr, img_sdr, gainmap, hdr_meta, sdr_meta, c3)

    def compare_reconstruction_batched(self,
                                       hdrs_glob_pattern: str,
                                       sdrs_glob_pattern: str,
                                       gainmaps_glob_pattern: str,
                                       metadatas_glob_pattern: str,
                                       output_path: str):
        output_path = Path(output_path)
        assert output_path.suffix == ".json", \
            "Expected output path to be .json"
        hdrs = sorted(Path().glob(hdrs_glob_pattern))
        sdrs = sorted(Path().glob(sdrs_glob_pattern))
        metas = sorted(Path().glob(metadatas_glob_pattern))
        gainmaps = sorted(Path().glob(gainmaps_glob_pattern))

        psnrs_lum, psnrs_img = list(), list()
        for hdr, sdr, gainmap, meta in zip(hdrs, sdrs, gainmaps, metas):
            ret = self.compare_reconstruction(hdr, sdr, gainmap, meta)
            psnrs_lum.append(ret["psnr_lum"])
            psnrs_img.append(ret["psnr_img"])

        results = {
            "hdrs_glob_pattern": hdrs_glob_pattern,
            "sdrs_glob_pattern": sdrs_glob_pattern,
            "metadatas_glob_pattern": metadatas_glob_pattern,
            "gainmaps_glob_pattern": gainmaps_glob_pattern,
            "mean_psnr_lum": np.mean(psnrs_lum).item(),
            "std_psnr_lum": np.std(psnrs_lum).item(),
            "mean_psnr_img": np.mean(psnrs_img).item(),
            "std_psnr_img": np.std(psnrs_img).item(),
        }
        logger.info(f"Mean PSNR Lum: {results['mean_psnr_lum']}")
        logger.info(f"STD PSNR Lum: {results['std_psnr_lum']}")
        logger.info(f"Mean PSNR Image: {results['mean_psnr_img']}")
        logger.info(f"STD PSNR Image: {results['std_psnr_img']}")

        save_json(output_path, results)


if __name__ == '__main__':
    fire.Fire(App)
