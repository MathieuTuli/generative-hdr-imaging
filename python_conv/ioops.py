from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any

import subprocess
import json

import torch
import cv2

from utils import Gamut, OETF
import utils


@dataclass
class ImageMetadata:
    gamut: Gamut
    oetf: OETF
    bit_depth: int
    clip_percentile: float = 1.0
    hdr_offset: tuple[float, float, float] = (0.015625, 0.015625, 0.015625)
    sdr_offset: tuple[float, float, float] = (0.015625,  0.015625,  0.015625)
    min_content_boost: tuple[float, float, float] = (1.0, 1.0, 1.0)
    max_content_boost: tuple[float, float, float] = (4.0, 4.0, 4.0)
    map_gamma: tuple[float, float, float] = (1.0, 1.0, 1.0)
    hdr_capacity_min: float = 1.0
    hdr_capacity_max: float = 4.0

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


def save_tensor(fname: Path, data):
    torch.save(data, fname)


def save_png(fname: Path, data):
    if data.dtype not in {torch.uint8, torch.uint16}:
        data = cv2.cvtColor(
            torch.clip(data * 255. + 0.5, 0,
                       255).to(torch.uint8).cpu().numpy(),
            cv2.COLOR_BGR2RGB)
    cv2.imwrite(str(fname), data)
