from os.path import commonpath, relpath
from pathlib import Path

import multiprocessing
import sys

from loguru import logger

import numpy as np
import torch
import fire

from ioops import (ImageMetadata, load_hdr_image,
                   save_png, save_tensor, load_tensor, load_image, save_json)
from hdr_to_gainmap import (generate_gainmap,
                            compare_hdr_to_uhdr,
                            reconstruct_hdr
                            )


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
            outdir: None | str = None,
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
        logger.info(f"Running hdr_to_gainmap for {fname}")
        if isinstance(fname, str):
            fname = Path(fname)

        img_hdr, meta = load_hdr_image(fname)
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

        data = generate_gainmap(img_hdr=img_hdr, meta=meta, c3=c3)

        if outdir is None:
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

        if Path(input_glob_pattern).is_dir():
            fnames = list(Path(input_glob_pattern).iterdir())
        elif Path(input_glob_pattern).is_file():
            if root_dir is None:
                logger.warning(
                    "The root dir is set to None, but input is a file")
            else:
                root_dir = Path(root_dir)
                assert root_dir.exists() and root_dir.is_dir()
            with open(input_glob_pattern, "r") as f:
                fnames = [root_dir / x.strip() for x in f.readlines()]
        else:
            fnames = [x for x in Path(input_glob_pattern).parent.glob(
                    Path(input_glob_pattern).name)]
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
                fname, out_subdir, hdr_exposure_bias, min_max_quantile,
                affine_min, affine_max, hdr_offset, sdr_offset,
                min_content_boost, max_content_boost, map_gamma,
                hdr_capacity_min, hdr_capacity_max, c3, save_torch, cuda
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
            c3: bool = False,) -> None:
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
        _, hdr_meta = load_hdr_image(hdr_path)
        img_sdr = load_image(sdr_path)
        sdr_meta = ImageMetadata.from_json(sdr_metadata_path)
        gainmap = load_tensor(gainmap_path)
        data = reconstruct_hdr(img_sdr=img_sdr, gainmap=gainmap,
                               hdr_meta=hdr_meta, sdr_meta=sdr_meta, c3=c3)
        # save_png(outdir / f"{hdr_path.stem}__reconstruction.png", data["img_hdr_recon"], uint16=True)  # noqa
        save_tensor(outdir / f"{hdr_path.stem}__reconstruction",
                    torch.clip(data["img_hdr_recon"] * 65535. + 0.5, 0,
                               65535).to(torch.uint16))

    def reconstruct_hdr_batched(self,
                                hdrs_glob_pattern: str,
                                sdrs_glob_pattern: str,
                                gainmaps_glob_pattern: str,
                                metadatas_glob_pattern: str,
                                outdir: str,
                                c3: bool = False,):
        logger.debug("Reconstructing for globs:\n" +
                     f"  hdrs_glob_pattern:  {hdrs_glob_pattern}\n" +
                     f"  sdrs_glob_pattern:  {sdrs_glob_pattern}\n" +
                     f"  gainmaps_glob_pattern:  {gainmaps_glob_pattern}\n" +
                     f"  metadatas_glob_pattern:  {metadatas_glob_pattern}\n")
        outdir = Path(outdir)
        outdir.mkdir(exist_ok=True)
        hdrs = sorted(Path().glob(hdrs_glob_pattern))
        sdrs = sorted(Path().glob(sdrs_glob_pattern))
        metas = sorted(Path().glob(metadatas_glob_pattern))
        gainmaps = sorted(Path().glob(gainmaps_glob_pattern))

        for hdr, sdr, gainmap, meta in zip(hdrs, sdrs, gainmaps, metas):
            self.reconstruct_hdr(hdr, sdr, gainmap, meta, outdir, c3)

    def compare_reconstruction(
            self,
            hdr_path: Path,
            sdr_path: Path,
            gainmap_path: Path,
            sdr_metadata_path: Path,
            c3: bool = False,
            ret: bool = False) -> dict[str,  torch.Tensor]:
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
        gainmap = load_tensor(gainmap_path)
        logger.info(f"Comparing for {gainmap_path}")
        data = compare_hdr_to_uhdr(
            img_hdr=img_hdr, img_sdr=img_sdr,
            gainmap=gainmap, hdr_meta=hdr_meta, sdr_meta=sdr_meta, c3=c3)
        if ret:
            return data

    def compare_reconstruction_batched(self,
                                       hdrs_glob_pattern: str,
                                       sdrs_glob_pattern: str,
                                       gainmaps_glob_pattern: str,
                                       metadatas_glob_pattern: str,
                                       output_path: str,
                                       c3: bool = False):
        output_path = Path(output_path)
        assert output_path.suffix == ".json", \
            "Expected output path to be .json"
        logger.debug(f"hdrs_glob_pattern: {hdrs_glob_pattern}")
        logger.debug(f"sdrs_glob_pattern: {sdrs_glob_pattern}")
        logger.debug(f"gainmaps_glob_pattern: {gainmaps_glob_pattern}")
        logger.debug(f"metadatas_glob_pattern: {metadatas_glob_pattern}")
        logger.debug(f"output_path: {output_path}")
        gainmaps = sorted(Path(gainmaps_glob_pattern).parent.glob(
            Path(gainmaps_glob_pattern).name))
        valid = set([x.stem.split("__")[0] for x in gainmaps])

        hdrs = sorted(
            [x for x in Path(hdrs_glob_pattern).parent.glob(
                Path(hdrs_glob_pattern).name) if x.stem in valid]
        )
        sdrs = sorted(
            [x for x in Path(sdrs_glob_pattern).parent.glob(
                Path(sdrs_glob_pattern).name) if x.stem.split("__")[0] in valid]
        )
        metas = sorted(
            [x for x in Path(metadatas_glob_pattern).parent.glob(
                Path(metadatas_glob_pattern).name) if x.stem.split("__")[0] in valid]
        )

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
