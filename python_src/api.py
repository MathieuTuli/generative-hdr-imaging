from os.path import commonpath, relpath
from pathlib import Path

import multiprocessing
import sys

from loguru import logger

import numpy as np
import torch
import fire

from ioops import (ImageMetadata, load_image_and_meta, get_fnames_from_glob,
                   save_png, save_tensor, load_tensor, load_image, save_json)
from hdr_to_gainmap import (generate_gainmap,
                            compare_hdr_to_uhdr,
                            reconstruct_hdr)
from utils import Gamut


class App:
    def __init__(self, suppress: bool = False, debug: bool = False):
        if suppress:
            logger.remove()
            logger.add(sys.stderr, level="WARNING")
        elif not debug:
            logger.remove()
            logger.add(sys.stderr, level="INFO")

    def hdr_to_gainmap(
            self,
            fname: str,
            hdr_exposure_bias: float,
            affine_min: float,
            affine_max: float,
            c3: bool,
            sdr_fname: str = None,
            dst_gamut: str = "BT2100",
            outdir: None | str = None,
            min_max_quantile: float = 0.0,
            hdr_offset: tuple[float, float, float] = (0.015625, 0.015625, 0.015625),  # noqa
            sdr_offset: tuple[float, float, float] = (0.015625,  0.015625,  0.015625),  # noqa
            min_content_boost: None | tuple[float, float, float] = None,
            max_content_boost: None | tuple[float, float, float] = None,
            map_gamma: tuple[float, float, float] = (1.0, 1.0, 1.0),
            hdr_capacity_min: float = 1.0,
            hdr_capacity_max: float = 4.0,
            save_torch: bool = False,
            cuda: bool = False,):
        """
        - outdir: None | str - if None - will return the data - used for
            in loop loading
        """
        if min_content_boost is not None:
            assert isinstance(min_content_boost, tuple), \
                f"Got {min_content_boost}"
        if max_content_boost is not None:
            assert isinstance(max_content_boost, tuple), \
                f"Got {max_content_boost}"
        assert isinstance(map_gamma, tuple), f"Got {map_gamma}"
        assert isinstance(hdr_offset, tuple), f"Got {hdr_offset}"
        assert isinstance(sdr_offset, tuple), f"Got {sdr_offset}"
        assert isinstance(hdr_exposure_bias, float), f"Got {hdr_exposure_bias}"
        assert isinstance(affine_min, float), f"Got {affine_min}"
        assert isinstance(affine_max, float), f"Got {affine_max}"
        assert isinstance(min_max_quantile, float), f"Got {min_max_quantile}"
        logger.info(f"Running hdr_to_gainmap for {fname}")
        if isinstance(fname, str):
            fname = Path(fname)

        img_hdr, meta = load_image_and_meta(fname)
        if cuda:
            img_hdr = img_hdr.to("cuda")
        meta.hdr_exposure_bias = hdr_exposure_bias
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

        img_sdr, sdr_meta = None, None
        if sdr_fname is not None:
            img_sdr, sdr_meta = load_image_and_meta(Path(sdr_fname), sdr=True)

        data = generate_gainmap(img_hdr=img_hdr, meta=meta, c3=c3,
                                img_sdr=img_sdr, sdr_meta=sdr_meta,
                                dst_gamut=Gamut[dst_gamut])

        if outdir is None:
            data["img_hdr"] = img_hdr
            data["hdr_meta"] = meta
            return data

        outdir = Path(outdir)
        outdir.mkdir(parents=True, exist_ok=True)
        save_tensor(
            outdir / f"{fname.stem}__gainmap", data["gainmap"],
            save_torch=save_torch)
        save_png(outdir / f"{fname.stem}__gainmap.png",
                 (data["gainmap"] - affine_min) / (affine_max - affine_min))
        save_tensor(outdir / f"{fname.stem}__hdr_linear",
                    data["img_hdr_linear"], save_torch=save_torch)
        save_png(outdir / f"{fname.stem}__sdr.png", data["img_sdr"])
        data["hdr_metadata"].save(
            outdir / f"{fname.stem}__hdr_metadata.json")
        data["sdr_metadata"].save(
            outdir / f"{fname.stem}__sdr_metadata.json")

    def hdr_to_gainmap_batched(
            self,
            input_glob_pattern: str,
            outdir: str,
            proc: int,
            sdr_path_glob_pattern: str = None,
            root_dir: str = None,
            hdr_exposure_bias: float = 0.0,
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
            dst_gamut: str = "BT2100",
            save_torch: bool = False,
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

        fnames = get_fnames_from_glob(input_glob_pattern, root_dir)
        sdr_fnames = None
        if sdr_path_glob_pattern is not None:
            sdr_fnames = get_fnames_from_glob(sdr_path_glob_pattern, root_dir)
        assert len(fnames) > 0, f"No files found from {input_glob_pattern}"

        # Convert all paths to absolute paths
        fnames = [Path(f).absolute() for f in fnames]

        common_parent = Path(commonpath([str(f) for f in fnames]))
        logger.info(f"Found common parent directory: {common_parent}")

        args = []
        for fname in fnames:
            rel_path = Path(relpath(fname.parent, common_parent))
            out_subdir = Path(outdir) / rel_path
            out_subdir.mkdir(parents=True, exist_ok=True)

            args.append((
                fname, hdr_exposure_bias, affine_min, affine_max, c3,
                sdr_fnames, dst_gamut, out_subdir, min_max_quantile,
                hdr_offset, sdr_offset, min_content_boost, max_content_boost,
                map_gamma, hdr_capacity_min, hdr_capacity_max, save_torch, cuda
            ))
        with multiprocessing.Pool(processes=proc) as pool:
            pool.starmap(self.hdr_to_gainmap, args)

    def reconstruct_hdr(
            self,
            hdr_path: Path,
            sdr_path: Path,
            gainmap_path: Path,
            sdr_metadata_path: Path,
            outdir: Path,
            c3: bool,
            dst_gamut: str = "BT2100",
    ) -> None:
        if isinstance(hdr_path, str):
            hdr_path = Path(hdr_path)
        if isinstance(sdr_path, str):
            sdr_path = Path(sdr_path)
        if isinstance(gainmap_path, str):
            gainmap_path = Path(gainmap_path)
        if isinstance(sdr_metadata_path, str):
            sdr_metadata_path = Path(sdr_metadata_path)
        if isinstance(outdir, str):
            outdir = Path(outdir)
        outdir.mkdir(exist_ok=True)

        assert hdr_path.stem == sdr_path.stem.split("__")[0], \
            f"{hdr_path.stem} vs. {sdr_path.stem}"
        assert sdr_path.stem.split("__")[0] == gainmap_path.stem.split("__")[0]
        assert sdr_path.stem.split(
            "__")[0] == sdr_metadata_path.stem.split("__")[0]
        _, hdr_meta = load_image_and_meta(hdr_path)
        img_sdr = load_image(sdr_path)
        sdr_meta = ImageMetadata.from_json(sdr_metadata_path)
        gainmap = load_tensor(gainmap_path)
        data = reconstruct_hdr(img_sdr=img_sdr, gainmap=gainmap,
                               hdr_meta=hdr_meta, sdr_meta=sdr_meta, c3=c3,
                               dst_gamut=Gamut[dst_gamut])
        save_png(outdir / f"{hdr_path.stem}__reconstruction_8bit.png",
                 data["img_hdr_recon"], uint16=False)  # noqa
        save_tensor(outdir / f"{hdr_path.stem}__reconstruction",
                    torch.clip(data["img_hdr_recon"] * 65535. + 0.5, 0,
                               65535).to(torch.uint16))

    def reconstruct_hdr_batched(self,
                                hdrs_glob_pattern: str,
                                sdrs_glob_pattern: str,
                                gainmaps_glob_pattern: str,
                                metadatas_glob_pattern: str,
                                outdir: str,
                                c3: bool,
                                dst_gamut: str = "BT2100"
                                ):
        logger.debug("Reconstructing for globs:\n" +
                     f"  hdrs_glob_pattern:  {hdrs_glob_pattern}\n" +
                     f"  sdrs_glob_pattern:  {sdrs_glob_pattern}\n" +
                     f"  gainmaps_glob_pattern:  {gainmaps_glob_pattern}\n" +
                     f"  metadatas_glob_pattern:  {metadatas_glob_pattern}\n")
        outdir = Path(outdir)
        outdir.mkdir(exist_ok=True)
        hdrs = sorted(get_fnames_from_glob(hdrs_glob_pattern))
        sdrs = sorted(get_fnames_from_glob(sdrs_glob_pattern))
        metas = sorted(get_fnames_from_glob(metadatas_glob_pattern))
        gainmaps = sorted(get_fnames_from_glob(gainmaps_glob_pattern))

        for hdr, sdr, gainmap, meta in zip(hdrs, sdrs, gainmaps, metas):
            self.reconstruct_hdr(hdr, sdr, gainmap, meta,
                                 outdir, c3, dst_gamut)

    def compare_reconstruction(
            self,
            hdr_path: Path,
            sdr_path: Path,
            gainmap_path: Path,
            sdr_metadata_path: Path,
            c3: bool,
            ret: bool = False,
            dst_gamut: str = "BT2100",) -> dict[str,  torch.Tensor]:
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
        img_hdr, hdr_meta = load_image_and_meta(hdr_path)
        img_sdr = load_image(sdr_path)
        sdr_meta = ImageMetadata.from_json(sdr_metadata_path)
        gainmap = load_tensor(gainmap_path)
        logger.info(f"Comparing for {gainmap_path}")
        data = compare_hdr_to_uhdr(img_hdr=img_hdr, img_sdr=img_sdr,
                                   gainmap=gainmap, hdr_meta=hdr_meta,
                                   sdr_meta=sdr_meta, c3=c3,
                                   dst_gamut=Gamut[dst_gamut])
        if ret:
            return data

    def compare_reconstruction_batched(self,
                                       hdrs_glob_pattern: str,
                                       sdrs_glob_pattern: str,
                                       gainmaps_glob_pattern: str,
                                       metadatas_glob_pattern: str,
                                       output_path: str,
                                       c3: bool,
                                       dst_gamut: str = "BT2100",):
        output_path = Path(output_path)
        assert output_path.suffix == ".json", \
            "Expected output path to be .json"
        logger.debug(f"hdrs_glob_pattern: {hdrs_glob_pattern}")
        logger.debug(f"sdrs_glob_pattern: {sdrs_glob_pattern}")
        logger.debug(f"gainmaps_glob_pattern: {gainmaps_glob_pattern}")
        logger.debug(f"metadatas_glob_pattern: {metadatas_glob_pattern}")
        logger.debug(f"output_path: {output_path}")
        gainmaps = sorted(get_fnames_from_glob(gainmaps_glob_pattern))
        valid = set([x.stem.split("__")[0] for x in gainmaps])

        hdrs = sorted([x for x in get_fnames_from_glob(hdrs_glob_pattern)
                       if x.stem in valid])
        sdrs = sorted([x for x in get_fnames_from_glob(sdrs_glob_pattern)
                       if x.stem.split("__")[0] in valid])
        metas = sorted([x for x in get_fnames_from_glob(metadatas_glob_pattern)
                        if x.stem.split("__")[0] in valid])

        psnrs_lum, psnrs_img = list(), list()
        results = {}
        results["results"] = list()
        for hdr, sdr, gainmap, meta in zip(hdrs, sdrs, gainmaps, metas):
            ret = self.compare_reconstruction(
                hdr, sdr, gainmap, meta, ret=True, c3=c3)
            psnrs_lum.append(ret["psnr_lum"])
            psnrs_img.append(ret["psnr_img"])
            results["results"].append(
                {"fname": str(gainmap),
                 "psnr_lum": ret["psnr_lum"].item(),
                 "psnr_img": ret["psnr_img"].item()}
            )

        results.update({
            "hdrs_glob_pattern": hdrs_glob_pattern,
            "sdrs_glob_pattern": sdrs_glob_pattern,
            "metadatas_glob_pattern": metadatas_glob_pattern,
            "gainmaps_glob_pattern": gainmaps_glob_pattern,
            "mean_psnr_lum": np.mean(psnrs_lum).item(),
            "std_psnr_lum": np.std(psnrs_lum).item(),
            "mean_psnr_img": np.mean(psnrs_img).item(),
            "std_psnr_img": np.std(psnrs_img).item(),
        })
        logger.info(f"Mean PSNR Lum: {results['mean_psnr_lum']}")
        logger.info(f"STD PSNR Lum: {results['std_psnr_lum']}")
        logger.info(f"Mean PSNR Image: {results['mean_psnr_img']}")
        logger.info(f"STD PSNR Image: {results['std_psnr_img']}")

        save_json(output_path, results)


if __name__ == '__main__':
    fire.Fire(App)
