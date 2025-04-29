from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any

import subprocess
import json

from einops import repeat
from loguru import logger
import imageio

import numpy as np
import torch
import cv2

from utils import Gamut, OETF
import utils


@dataclass
class ImageMetadata:
    gamut: Gamut
    oetf: OETF
    bit_depth: int
    hdr_exposure_bias: float = 0.0
    min_max_quantile: float = 0.0
    affine_min: float = -1
    affine_max: float = 1
    hdr_offset: tuple[float, float, float] = (0.015625, 0.015625, 0.015625)
    sdr_offset: tuple[float, float, float] = (0.015625,  0.015625,  0.015625)
    min_content_boost: tuple[float, float, float] = (1.0, 1.0, 1.0)
    max_content_boost: tuple[float, float, float] = (4.0, 4.0, 4.0)
    map_gamma: tuple[float, float, float] = (1.0, 1.0, 1.0)
    hdr_capacity_min: float = 0.0
    hdr_capacity_max: float = 2.0

    def save(self, filepath: Path | str) -> None:
        data = asdict(self)
        data['gamut'] = self.gamut.name
        data['oetf'] = self.oetf.name
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)

    @classmethod
    def from_json(cls, filepath: Path | str) -> 'ImageMetadata':
        with open(filepath) as f:
            data = json.load(f)
        data['gamut'] = Gamut[data['gamut']]
        data['oetf'] = OETF[data['oetf']]

        return cls(**data)

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data['gamut'] = self.gamut.name
        data['oetf'] = self.oetf.name
        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> 'ImageMetadata':
        data = data.copy()
        data['gamut'] = Gamut[data['gamut']]
        data['oetf'] = OETF[data['oetf']]
        return cls(**data)


def load_image(fname: Path):
    image = cv2.imread(str(fname), cv2.IMREAD_UNCHANGED)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    dtype_mapping = {
        'uint8': torch.uint8,
        'uint16': torch.uint16,
        'float32': torch.float32,
        'float64': torch.float64
    }
    dtype = dtype_mapping.get(str(image.dtype), utils.DTYPE)
    image = torch.tensor(image, dtype=dtype)
    return image


def load_hdr_image(fname: Path):
    image = load_image(fname)
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
        metadata_list = json.loads(result.stdout)
        metadata_raw = metadata_list[0] if metadata_list else {}
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"ExifTool returned an error: {e.stderr}")

    bit_depth = metadata_raw.get("BitDepth", 1)
    oetf = metadata_raw.get("TransferCharacteristics", "Linear")
    if oetf.find("HLG") >= 0 or oetf.find("2020") >= 0:
        oetf = OETF.HLG
    elif oetf.find("PQ") >= 0 or oetf.find("2084") >= 0:
        oetf = OETF.PQ
    elif oetf.find("709") >= 0 or oetf.find("sRGB") >= 0:
        oetf = OETF.SRGB
    elif oetf == "Linear":
        oetf = OETF.LINEAR
    else:
        raise ValueError(f"Unknown TransferCharacteristics {oetf}")
    gamut = metadata_raw.get("ColorPrimaries", "sRGB")
    if gamut.find("709") >= 0 or gamut.find("sRGB") >= 0:
        gamut = Gamut.BT709
    elif gamut.find("P3") >= 0 or gamut.find("SMPTE") >= 0:
        gamut = Gamut.P3
    elif gamut.find("2100") >= 0 or gamut.find("2020") >= 0:
        gamut = Gamut.BT2100
    else:
        raise ValueError(f"Unknown ColorPrimaries {gamut}")
    metadata = ImageMetadata(gamut=gamut, oetf=oetf, bit_depth=bit_depth)
    return image, metadata


def save_tensor(fname: Path, data: torch.Tensor, save_torch: bool = False):
    if save_torch:
        fname = fname.with_suffix(".pt")
        torch.save(data.cpu(), fname)
    else:
        fname = fname.with_suffix(".npy")
        np.save(fname, data.cpu().numpy())


def load_tensor(fname: Path) -> torch.Tensor:
    if fname.suffix == ".npy":
        arr = torch.from_numpy(np.load(fname)).to(utils.DTYPE)
        if len(arr.shape) == 2 or (len(arr.shape) == 3 and arr.shape[-1] == 1):
            arr = repeat(arr, "w h -> w h 3")
        return arr
    else:
        return torch.load(fname, weights_only=False, dtype=utils.DTYPE)


def save_png(fname: Path, data: np.ndarray, uint16: bool = False):
    if data.dtype not in {torch.uint8, torch.uint16}:
        if uint16:
            data = cv2.cvtColor(
                torch.clip(data * 65535. + 0.5, 0,
                           65535).to(torch.uint16).cpu().numpy(),
                cv2.COLOR_RGB2BGR).astype(np.uint16)
        else:
            data = cv2.cvtColor(
                torch.clip(data * 255. + 0.5, 0,
                           255).to(torch.uint8).cpu().numpy(),
                cv2.COLOR_RGB2BGR)
    if uint16:
        imageio.imwrite(str(fname), data)
    else:
        cv2.imwrite(str(fname), data)


def save_json(fname: Path, data: dict[str, Any]):
    fname.parent.mkdir(exist_ok=True, parents=True)
    if fname.exists():
        c = 0
        new_fname = None
        while True:
            new_fname = fname.with_stem(fname.stem + f"_{c}")
            if not new_fname.exists():
                break
            c += 1
        logger.info(f"Attempting to save json to {fname} which exists, incrementing file as {new_fname}")  # noqa
        fname = new_fname
    with fname.open("w") as f:
        json.dump(data, f, indent=2)
