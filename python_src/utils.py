from typing import Callable, List
from enum import Enum, auto
import torch

DTYPE = torch.float32

# --- Types ---
ColorTransformFn = Callable[[torch.Tensor], torch.Tensor]
LuminanceFn = Callable[[torch.Tensor], float]
SceneToDisplayLuminanceFn = Callable[[torch.Tensor, LuminanceFn], torch.Tensor]

# -- Control Variables ---
USE_SRGB_OETF_LUT = False
USE_HLG_OETF_LUT = False
USE_PQ_OETF_LUT = False
USE_SRGB_INVOETF_LUT = False
USE_HLG_INVOETF_LUT = False
USE_PQ_INVOETF_LUT = False
USE_APPLY_GAIN_LUT = False
USE_HLG_OOTF_APPROX = False

# --- LUT Precision Constants ---
SRGB_INV_OETF_PRECISION = 10
SRGB_INV_OETF_NUMENTRIES = 1 << SRGB_INV_OETF_PRECISION
HLG_OETF_PRECISION = 16
HLG_OETF_NUMENTRIES = 1 << HLG_OETF_PRECISION
HLG_INV_OETF_PRECISION = 12
HLG_INV_OETF_NUMENTRIES = 1 << HLG_INV_OETF_PRECISION
PQ_OETF_PRECISION = 16
PQ_OETF_NUMENTRIES = 1 << PQ_OETF_PRECISION
PQ_INV_OETF_PRECISION = 12
PQ_INV_OETF_NUMENTRIES = 1 << PQ_INV_OETF_PRECISION
GAIN_FACTOR_PRECISION = 10
GAIN_FACTOR_NUMENTRIES = 1 << GAIN_FACTOR_PRECISION

# --- Nominal Display Luminance Values ---
SDR_WHITE_NITS = 203.0
HLG_MAX_NITS = 1000.0
PQ_MAX_NITS = 1000.0


class LookUpTable:
    def __init__(self,
                 num_entries: int,
                 compute_func: Callable[[float], float]):
        self.table: list[float] = [
            compute_func(idx / (num_entries - 1)) for idx in range(num_entries)
        ]

    def get_table(self) -> List[float]:
        return self.table


# --- Global LUT Cache Variables ---
_kSrgbLut: LookUpTable | None = None
_kHlgLut: LookUpTable | None = None
_kHlgInvLut: LookUpTable | None = None
_kPqLut: LookUpTable | None = None
_kPqInvLut: LookUpTable | None = None


def safe_int(v: float) -> int:
    return int(v + 0.5)


class ToneMapping(Enum):
    BASE = auto()
    REINHARD = auto()
    GAMMA = auto()
    FILMIC = auto()
    ACES = auto()
    UNCHARTED2 = auto()
    DRAGO = auto()
    LOTTES = auto()
    HABLE = auto()

from typing import Literal, Union
from enum import Enum


class Gamut(Enum):
    BT709 = auto()
    P3 = auto()
    BT2100 = auto()


class OETF(Enum):
    SRGB = auto()
    HLG = auto()
    PQ = auto()
    LINEAR = auto()


def get_kSrgbLut() -> LookUpTable:
    global _kSrgbLut
    if _kSrgbLut is None:
        _kSrgbLut = LookUpTable(SRGB_INV_OETF_NUMENTRIES, sRGB_InvOETF)
    return _kSrgbLut


def get_kHlgLut() -> LookUpTable:
    global _kHlgLut
    if _kHlgLut is None:
        _kHlgLut = LookUpTable(HLG_OETF_NUMENTRIES, HLG_OETF)
    return _kHlgLut


def get_kHlgInvLut() -> LookUpTable:
    global _kHlgInvLut
    if _kHlgInvLut is None:
        _kHlgInvLut = LookUpTable(HLG_INV_OETF_NUMENTRIES, HLG_InvOETF)
    return _kHlgInvLut


def get_kPqLut() -> LookUpTable:
    global _kPqLut
    if _kPqLut is None:
        _kPqLut = LookUpTable(PQ_OETF_NUMENTRIES, PQ_OETF)
    return _kPqLut


def get_kPqInvLut() -> LookUpTable:
    global _kPqInvLut
    if _kPqInvLut is None:
        _kPqInvLut = LookUpTable(PQ_INV_OETF_NUMENTRIES, PQ_InvOETF)
    return _kPqInvLut

# ==============================================================================
# sRGB Transformations
# ==============================================================================


SRGB_R = 0.212639
SRGB_G = 0.715169
SRGB_B = 0.072192


def sRGBLuminance(e: torch.Tensor) -> float:
    return torch.matmul(e, torch.tensor([SRGB_R, SRGB_G, SRGB_B],
                                        dtype=DTYPE,
                                        device=e.device))[..., None]


SRGB_CB = 2.0 * (1.0 - SRGB_B)
SRGB_CR = 2.0 * (1.0 - SRGB_R)


def sRGB_RGBToYUV(e_gamma: torch.Tensor) -> torch.Tensor:
    y_gamma = sRGBLuminance(e_gamma)
    return torch.cat(
        [y_gamma,
         ((e_gamma[:, :, 2:3] - y_gamma) / SRGB_CB),
         ((e_gamma[:, :, 0:1] - y_gamma) / SRGB_CR)], dim=2)


SRGB_GCB = SRGB_B * SRGB_CB / SRGB_G
SRGB_GCR = SRGB_R * SRGB_CR / SRGB_G


def sRGB_YUVToRGB(e_gamma: torch.Tensor) -> torch.Tensor:
    y = e_gamma[..., 0]
    u = e_gamma[..., 1]
    v = e_gamma[..., 2]
    rgb = torch.empty_like(e_gamma)
    rgb[..., 0] = y + SRGB_CR * v
    rgb[..., 1] = y - SRGB_GCB * u - SRGB_GCR * v
    rgb[..., 2] = y + SRGB_CB * u
    return torch.clip(rgb, 0.0, 1.0)


def sRGB_InvOETF(e_gamma: torch.Tensor) -> torch.Tensor:
    result = torch.empty_like(e_gamma)
    mask_low = e_gamma <= 0.04045
    mask_high = ~mask_low
    result[mask_low] = e_gamma[mask_low] / 12.92
    result[mask_high] = ((e_gamma[mask_high] + 0.055) / 1.055) ** 2.4
    return result


# DEPRECATE:
def sRGB_InvOETFLUT(e_gamma: float) -> float:
    value = safe_int(e_gamma * (SRGB_INV_OETF_NUMENTRIES - 1))
    value = torch.clip(value, 0, SRGB_INV_OETF_NUMENTRIES - 1)
    return get_kSrgbLut().get_table()[value]


def sRGB_OETF(e: torch.Tensor) -> torch.Tensor:
    threshold = 0.0031308
    low_slope = 12.92
    high_offset = 0.055
    power_exponent = 1.0 / 2.4
    result = torch.empty_like(e)
    mask_low = e <= threshold
    mask_high = ~mask_low
    result[mask_low] = low_slope * e[mask_low]
    result[mask_high] = (1.0 + high_offset) * \
        (e[mask_high] ** power_exponent) - high_offset
    return result


# DEPRECATE:
def sRGB_OETFLUT(e: float) -> float:
    raise NotImplementedError


# ==============================================================================
# Display-P3 Transformations
# ==============================================================================

P3_R = 0.2289746
P3_G = 0.6917385
P3_B = 0.0792869


def P3Luminance(e: torch.Tensor) -> float:
    return torch.matmul(e, torch.tensor([P3_R, P3_G, P3_B],
                                        dtype=DTYPE,
                                        device=e.device))[..., None]


P3_YR = 0.299
P3_YG = 0.587
P3_YB = 0.114
P3_CB = 1.772
P3_CR = 1.402


def P3_RGBToYUV(e_gamma: torch.Tensor) -> torch.Tensor:
    y_gamma = (P3_YR * e_gamma[..., 0:1] +
               P3_YG * e_gamma[..., 1:2] +
               P3_YB * e_gamma[..., 2:3])

    yuv = torch.empty_like(e_gamma)
    yuv[..., 0:1] = y_gamma
    yuv[..., 1:2] = (e_gamma[..., 2:3] - y_gamma) / P3_CB  # U component
    yuv[..., 2:3] = (e_gamma[..., 0:1] - y_gamma) / P3_CR  # V component
    return yuv


P3_GCB = P3_YB * P3_CB / P3_YG
P3_GCR = P3_YR * P3_CR / P3_YG


def P3_YUVToRGB(e_gamma: torch.Tensor) -> torch.Tensor:
    rgb = torch.empty_like(e_gamma)
    rgb[..., 0] = e_gamma[..., 0] + P3_CR * e_gamma[..., 2]
    rgb[..., 1] = e_gamma[..., 0] - P3_GCB * \
        e_gamma[..., 1] - P3_GCR * e_gamma[..., 2]
    rgb[..., 2] = e_gamma[..., 0] + P3_CB * e_gamma[..., 1]
    return torch.clip(rgb, 0.0, 1.0)

# ==============================================================================
# BT.2100 Transformations
# ==============================================================================


BT2100_R = 0.2627
BT2100_G = 0.677998
BT2100_B = 0.059302


def Bt2100Luminance(e: torch.Tensor) -> float:
    return torch.matmul(e, torch.tensor([BT2100_R, BT2100_G, BT2100_B],
                                        dtype=DTYPE,
                                        device=e.device))[..., None]


BT2100_CB = 2.0 * (1.0 - BT2100_B)
BT2100_CR = 2.0 * (1.0 - BT2100_R)


def Bt2100_RGBToYUV(e_gamma: torch.Tensor) -> torch.Tensor:
    y_gamma = Bt2100Luminance(e_gamma)
    yuv = torch.empty_like(e_gamma)
    yuv[..., 0:1] = y_gamma
    yuv[..., 1:2] = (e_gamma[..., 2:3] - y_gamma) / BT2100_CB
    yuv[..., 2:3] = (e_gamma[..., 0:1] - y_gamma) / BT2100_CR
    return yuv


BT2100_GCB = BT2100_B * BT2100_CB / BT2100_G
BT2100_GCR = BT2100_R * BT2100_CR / BT2100_G


def Bt2100_YUVToRGB(e_gamma: torch.Tensor) -> torch.Tensor:
    rgb = torch.empty_like(e_gamma)
    rgb[..., 0] = e_gamma[..., 0] + BT2100_CR * e_gamma[..., 2]
    rgb[..., 1] = e_gamma[..., 0] - BT2100_GCB * \
        e_gamma[..., 1] - BT2100_GCR * e_gamma[..., 2]
    rgb[..., 2] = e_gamma[..., 0] + BT2100_CB * e_gamma[..., 1]
    return torch.clip(rgb, 0.0, 1.0)

# ==============================================================================
# HLG Transformations
# ==============================================================================


HLG_A = 0.17883277
HLG_B = 0.28466892
HLG_C = 0.55991073


def HLG_OETF(e: torch.Tensor) -> torch.Tensor:
    result = torch.empty_like(e)
    mask_low = e <= 1.0 / 12.0
    mask_high = ~mask_low
    result[mask_low] = torch.sqrt(3.0 * e[mask_low])
    result[mask_high] = HLG_A * torch.log(12.0 * e[mask_high] - HLG_B) + HLG_C
    return result


# DEPRECATE:
def HLG_OETFLUT(e: float) -> float:
    value = safe_int(e * (HLG_OETF_NUMENTRIES - 1))
    value = torch.clip(value, 0, HLG_OETF_NUMENTRIES - 1)
    return get_kHlgLut().get_table()[value]


def HLG_InvOETF(e_gamma: float) -> float:
    result = torch.empty_like(e_gamma)
    mask_low = e_gamma <= 0.5
    mask_high = ~mask_low
    result[mask_low] = (e_gamma[mask_low] ** 2) / 3.0
    result[mask_high] = (
        torch.exp((e_gamma[mask_high] - HLG_C) / HLG_A) + HLG_B) / 12.0
    return result


# DEPRECATE:
def HLG_InvOETFLUT(e_gamma: float) -> float:
    value = safe_int(e_gamma * (HLG_INV_OETF_NUMENTRIES - 1))
    value = torch.clip(value, 0, HLG_INV_OETF_NUMENTRIES - 1)
    return get_kHlgInvLut().get_table()[value]


OOTF_GAMMA = 1.2


def HLG_OOTF(e: torch.Tensor, luminance: LuminanceFn) -> torch.Tensor:
    y = luminance(e)
    y[torch.isclose(y, torch.zeros(1, dtype=DTYPE, device=y.device))] += 1e-7
    return e * (y ** (OOTF_GAMMA - 1.0))


def HLG_OOTFApprox(e: torch.Tensor,
                   luminance: LuminanceFn | None = None) -> torch.Tensor:
    return torch.pow(e, torch.tensor(OOTF_GAMMA, dtype=DTYPE,
                                     device=e.device))


def HLG_InvOOTF(e: torch.Tensor, luminance: LuminanceFn) -> torch.Tensor:
    y = luminance(e)
    y[torch.isclose(y, torch.zeros(1, dtype=DTYPE, device=y.device))] += 1e-7
    return e * (y ** ((1.0 / OOTF_GAMMA) - 1.0))


def HLG_InvOOTFApprox(e: torch.Tensor) -> torch.Tensor:
    return torch.pow(e, torch.tensor((1.0 / OOTF_GAMMA), dtype=DTYPE,
                                     device=e.device))

# ==============================================================================
# PQ Transformations
# ==============================================================================


PQ_M1 = 2610.0 / 16384.0
PQ_M2 = (2523.0 / 4096.0) * 128.0
PQ_C1 = 3424.0 / 4096.0
PQ_C2 = (2413.0 / 4096.0) * 32.0
PQ_C3 = (2392.0 / 4096.0) * 32.0


def PQ_OETF(e: torch.Tensor) -> torch.Tensor:
    result = torch.zeros_like(e)
    mask = e > 0.0
    e_masked = e[mask]
    e_m1 = e_masked ** PQ_M1
    numerator = PQ_C1 + PQ_C2 * e_m1
    denominator = 1.0 + PQ_C3 * e_m1
    result[mask] = (numerator / denominator) ** PQ_M2
    return result


# DEPRECATE:
def PQ_OETFLUT(e: float) -> float:
    value = safe_int(e * (PQ_OETF_NUMENTRIES - 1))
    value = torch.clip(value, 0, PQ_OETF_NUMENTRIES - 1)
    return get_kPqLut().get_table()[value]


def PQ_InvOETF(e_gamma: torch.Tensor) -> torch.Tensor:
    val = torch.pow(e_gamma, torch.tensor(1.0 / PQ_M2, dtype=DTYPE,
                                          device=e_gamma.device))
    numerator = torch.maximum(val - PQ_C1,
                              torch.tensor(0.0, dtype=DTYPE,
                                           device=e_gamma.device))
    denominator = PQ_C2 - PQ_C3 * val
    return torch.pow(numerator / denominator,
                     torch.tensor(1.0 / PQ_M1, dtype=DTYPE,
                                  device=e_gamma.device))


# DEPRECATE:
def PQ_InvOETFLUT(e_gamma: float) -> float:
    value = safe_int(e_gamma * (PQ_INV_OETF_NUMENTRIES - 1))
    value = torch.clip(value, 0, PQ_INV_OETF_NUMENTRIES - 1)
    return get_kPqInvLut().get_table()[value]


# ==============================================================================
# Gamut Conversions
# ==============================================================================


def ConvertGamut(e: torch.Tensor, coeffs: torch.Tensor) -> torch.Tensor:
    return torch.matmul(coeffs[None, None, ...].to(e.device),
                        e[..., None]).squeeze(-1)


# Conversion matrices (provided as lists of 9 floats)
BT709_TO_P3 = torch.tensor([[0.822462,  0.177537,  0.000001],
                            [0.033194,  0.966807, -0.000001],
                            [0.017083,  0.072398,  0.91052]],
                           dtype=DTYPE)

BT709_TO_BT2100 = torch.tensor([[0.627404,  0.329282,  0.043314],
                                [0.069097,  0.919541,  0.011362],
                                [0.016392,  0.088013,  0.895595]],
                               dtype=DTYPE)

P3_TO_BT709 = torch.tensor([[1.22494,   -0.22494,   0.0],
                            [-0.042057,  1.042057,  0.0],
                            [-0.019638, -0.078636,  1.098274]],
                           dtype=DTYPE)

P3_TO_BT2100 = torch.tensor([[0.753833,  0.198597,  0.04757],
                             [0.045744,  0.941777,  0.012479],
                             [-0.00121,  0.017601,  0.983608]],
                            dtype=DTYPE)

BT2100_TO_BT709 = torch.tensor([[1.660491,  -0.587641, -0.07285],
                                [-0.124551,  1.1329,   -0.008349],
                                [-0.018151, -0.100579,  1.11873]],
                               dtype=DTYPE)

BT2100_TO_P3 = torch.tensor([[1.343578,  -0.282179, -0.061399],
                             [-0.065298,  1.075788,  -0.01049],
                             [0.002822, -0.019598,  1.016777]],
                            dtype=DTYPE)


def Bt709ToP3(e: torch.Tensor) -> torch.Tensor:
    return ConvertGamut(e, BT709_TO_P3)


def Bt709ToBt2100(e: torch.Tensor) -> torch.Tensor:
    return ConvertGamut(e, BT709_TO_BT2100)


def P3ToBt709(e: torch.Tensor) -> torch.Tensor:
    return ConvertGamut(e, P3_TO_BT709)


def P3ToBt2100(e: torch.Tensor) -> torch.Tensor:
    return ConvertGamut(e, P3_TO_BT2100)


def Bt2100ToBt709(e: torch.Tensor) -> torch.Tensor:
    return ConvertGamut(e, BT2100_TO_BT709)


def Bt2100ToP3(e: torch.Tensor) -> torch.Tensor:
    return ConvertGamut(e, BT2100_TO_P3)


# YUV gamut conversion matrices
YUV_BT709_TO_BT601 = torch.tensor([[1.0,      0.101579,   0.196076],
                                   [0.0,      0.989854,  -0.110653],
                                   [0.0,     -0.072453,   0.983398]],
                                  dtype=DTYPE)

YUV_BT709_TO_BT2100 = torch.tensor([[1.0,     -0.016969,   0.096312],
                                    [0.0,      0.995306,  -0.051192],
                                    [0.0,      0.011507,   1.002637]],
                                   dtype=DTYPE)

YUV_BT601_TO_BT709 = torch.tensor([[1.0,     -0.118188,  -0.212685],
                                   [0.0,      1.018640,   0.114618],
                                   [0.0,      0.075049,   1.025327]],
                                  dtype=DTYPE)

YUV_BT601_TO_BT2100 = torch.tensor([[1.0,     -0.128245,  -0.115879],
                                    [0.0,      1.010016,   0.061592],
                                    [0.0,      0.086969,   1.029350]],
                                   dtype=DTYPE)

YUV_BT2100_TO_BT709 = torch.tensor([[1.0,      0.018149,  -0.095132],
                                    [0.0,      1.004123,   0.051267],
                                    [0.0,     -0.011524,   0.996782]],
                                   dtype=DTYPE)

YUV_BT2100_TO_BT601 = torch.tensor([[1.0,      0.117887,   0.105521],
                                    [0.0,      0.995211,  -0.059549],
                                    [0.0,     -0.084085,   0.976518]],
                                   dtype=DTYPE)


def YUV_Bt709ToBt601(e: torch.Tensor) -> torch.Tensor:
    return ConvertGamut(e, YUV_BT709_TO_BT601)


def YUV_Bt709ToBt2100(e: torch.Tensor) -> torch.Tensor:
    return ConvertGamut(e, YUV_BT709_TO_BT2100)


def YUV_Bt601ToBt709(e: torch.Tensor) -> torch.Tensor:
    return ConvertGamut(e, YUV_BT601_TO_BT709)


def YUV_Bt601ToBt2100(e: torch.Tensor) -> torch.Tensor:
    return ConvertGamut(e, YUV_BT601_TO_BT2100)


def YUV_Bt2100ToBt709(e: torch.Tensor) -> torch.Tensor:
    return ConvertGamut(e, YUV_BT2100_TO_BT709)


def YUV_Bt2100ToBt601(e: torch.Tensor) -> torch.Tensor:
    return ConvertGamut(e, YUV_BT2100_TO_BT601)

# ==============================================================================
# Function Selectors
# ==============================================================================


def IdentityConversion(e: torch.Tensor) -> torch.Tensor:
    return e


def IdentityOOTF(e: torch.Tensor, luminance: LuminanceFn) -> torch.Tensor:
    return e


def GetGamutConversionFn(src_gamut: Gamut,
                         dst_gamut: Gamut) -> ColorTransformFn:
    if src_gamut == Gamut.BT709:
        if dst_gamut == Gamut.BT709:
            return IdentityConversion
        elif dst_gamut == Gamut.P3:
            return Bt709ToP3
        elif dst_gamut == Gamut.BT2100:
            return Bt709ToBt2100
    elif src_gamut == Gamut.P3:
        if dst_gamut == Gamut.BT709:
            return P3ToBt709
        elif dst_gamut == Gamut.P3:
            return IdentityConversion
        elif dst_gamut == Gamut.BT2100:
            return P3ToBt2100
    elif src_gamut == Gamut.BT2100:
        if dst_gamut == Gamut.BT709:
            return Bt2100ToBt709
        elif dst_gamut == Gamut.P3:
            return Bt2100ToP3
        elif dst_gamut == Gamut.BT2100:
            return IdentityConversion
    raise ValueError(f"Unknown src/dst gammut {src_gamut}/{dst_gamut}")


def GetRGBToYUVFn(gamut: Gamut) -> ColorTransformFn:
    if gamut == Gamut.BT709:
        return sRGB_RGBToYUV
    elif gamut == Gamut.P3:
        return sRGB_RGBToYUV
    elif gamut == Gamut.BT2100:
        return Bt2100_RGBToYUV
    raise ValueError(f"Unknown gamut {gamut}")


def GetYUVToRGBFn(gamut: Gamut) -> ColorTransformFn:
    if gamut == Gamut.BT709:
        return sRGB_YUVToRGB
    elif gamut == Gamut.P3:
        return P3_YUVToRGB
    elif gamut == Gamut.BT2100:
        return Bt2100_YUVToRGB
    raise ValueError(f"Unknown gamut {gamut}")


def GetLuminanceFn(gamut: Gamut) -> LuminanceFn:
    if gamut == Gamut.BT709:
        return sRGBLuminance
    elif gamut == Gamut.P3:
        return P3Luminance
    elif gamut == Gamut.BT2100:
        return Bt2100Luminance
    raise ValueError(f"Unknown gamut {gamut}")


def GetOETFFn(transfer: OETF) -> ColorTransformFn:
    if transfer == OETF.LINEAR:
        return IdentityConversion
    elif transfer == OETF.HLG:
        return HLG_OETFLUT if USE_HLG_OETF_LUT else HLG_OETF
    elif transfer == OETF.PQ:
        return PQ_OETFLUT if USE_PQ_OETF_LUT else PQ_OETF
    elif transfer == OETF.SRGB:
        return sRGB_OETFLUT if USE_SRGB_OETF_LUT else sRGB_OETF
    raise ValueError(f"Unknown oetf {transfer}")


def GetInvOETFFn(transfer: OETF) -> ColorTransformFn:
    if transfer == OETF.LINEAR:
        return IdentityConversion
    elif transfer == OETF.HLG:
        return HLG_InvOETFLUT if USE_HLG_INVOETF_LUT else HLG_InvOETF
    elif transfer == OETF.PQ:
        return PQ_InvOETFLUT if USE_PQ_INVOETF_LUT else PQ_InvOETF
    elif transfer == OETF.SRGB:
        return sRGB_InvOETFLUT if USE_SRGB_INVOETF_LUT else sRGB_InvOETF
    raise ValueError(f"Unknown oetf {transfer}")


def GetInvOOTFFn(transfer: OETF) -> SceneToDisplayLuminanceFn:
    if transfer == OETF.LINEAR:
        return IdentityOOTF
    elif transfer == OETF.HLG:
        return HLG_InvOOTFApprox if USE_HLG_OOTF_APPROX else HLG_InvOOTF
    elif transfer == OETF.PQ:
        return IdentityOOTF
    elif transfer == OETF.SRGB:
        return IdentityOOTF
    raise ValueError(f"Unknown oetf {transfer}")


def GetOOTFFn(transfer: OETF) -> SceneToDisplayLuminanceFn:
    if transfer == OETF.LINEAR:
        return IdentityOOTF
    elif transfer == OETF.HLG:
        return HLG_OOTFApprox if USE_HLG_OOTF_APPROX else HLG_OOTF
    elif transfer == OETF.PQ:
        return IdentityOOTF
    elif transfer == OETF.SRGB:
        return IdentityOOTF
    raise ValueError(f"Unknown oetf {transfer}")


def GetReferenceDisplayPeakLuminanceInNits(transfer: OETF) -> float:
    if transfer == OETF.LINEAR:
        return PQ_MAX_NITS
    elif transfer == OETF.HLG:
        return HLG_MAX_NITS
    elif transfer == OETF.PQ:
        return PQ_MAX_NITS
    elif transfer == OETF.SRGB:
        return SDR_WHITE_NITS
    raise ValueError(f"Unknown oetf {transfer}")

# ==============================================================================
# Tone Mapping
# ==============================================================================


def perceptual_gamut_compression(rgb: torch.Tensor,
                                 peak: float = 1.0,
                                 slope: float = 6.0,
                                 knee: float = 0.90) -> torch.Tensor:
    yuv = sRGB_RGBToYUV(rgb)
    y = yuv[..., 0]

    exc = torch.clamp((y - knee*peak) / (peak*(1 - knee) + 1e-6), 0.0, 1.0)
    comp = torch.clamp(exc * slope, 0.0, 1.0)[..., None]

    yuv[..., 1:] *= 1.0 - comp
    return sRGB_YUVToRGB(yuv).clamp(0.0, peak)


def ApplyToneMapping(e: torch.Tensor,
                     mode: ToneMapping,
                     headroom: float,
                     is_normalized: bool) -> torch.Tensor:
    if is_normalized:
        e = e * headroom

    if mode == ToneMapping.BASE:
        e = e
    elif mode == ToneMapping.REINHARD:
        max_hdr = e.max()
        max_sdr = max_hdr * \
            (1. + (max_hdr / (headroom * headroom))) / (1. + max_hdr)
        e *= max_sdr
        e[e < 0.] = 0.
    elif mode == ToneMapping.GAMMA:
        e = e ** (torch.tensor(1.0 / 2.2, dtype=e.dtype, device=e.device))
    elif mode == ToneMapping.FILMIC:
        A = torch.tensor(2.51, dtype=e.dtype, device=e.device)
        B = torch.tensor(0.03, dtype=e.dtype, device=e.device)
        C = torch.tensor(2.43, dtype=e.dtype, device=e.device)
        D = torch.tensor(0.59, dtype=e.dtype, device=e.device)
        E = torch.tensor(0.14, dtype=e.dtype, device=e.device)
        e = (e * (A * e + B)) / (e * (C * e + D) + E)
    elif mode == ToneMapping.ACES:
        a = torch.tensor(2.51, dtype=e.dtype, device=e.device)
        b = torch.tensor(0.03, dtype=e.dtype, device=e.device)
        c = torch.tensor(2.43, dtype=e.dtype, device=e.device)
        d = torch.tensor(0.59, dtype=e.dtype, device=e.device)
        e_val = torch.tensor(0.14, dtype=e.dtype, device=e.device)
        adjusted = e * torch.tensor(0.6, dtype=e.dtype, device=e.device)
        e = (adjusted * (adjusted + b) * a) / \
            (adjusted * (adjusted * c + d) + e_val)
    elif mode == ToneMapping.UNCHARTED2:
        A = torch.tensor(0.15, dtype=e.dtype, device=e.device)
        B = torch.tensor(0.50, dtype=e.dtype, device=e.device)
        C = torch.tensor(0.10, dtype=e.dtype, device=e.device)
        D = torch.tensor(0.20, dtype=e.dtype, device=e.device)
        E = torch.tensor(0.02, dtype=e.dtype, device=e.device)
        F = torch.tensor(0.30, dtype=e.dtype, device=e.device)
        W = torch.tensor(11.2, dtype=e.dtype, device=e.device)

        def uncharted2_tonemap(e_val: torch.Tensor) -> torch.Tensor:
            return ((e_val * (A * e_val + C * B) + D * E) /
                    (e_val * (A * e_val + B) + D * F)) - E / F
        e = uncharted2_tonemap(e) / uncharted2_tonemap(W)
    elif mode == ToneMapping.DRAGO:
        bias = torch.tensor(0.85, dtype=e.dtype, device=e.device)
        Lwa = torch.tensor(1.0, dtype=e.dtype, device=e.device)
        e = torch.log(torch.ones_like(e) + e) / \
            torch.log(torch.ones_like(e) + Lwa)
        e = e ** bias
    elif mode == ToneMapping.LOTTES:
        a_val = torch.tensor(1.6, dtype=e.dtype, device=e.device)
        mid_in = torch.tensor(0.18, dtype=e.dtype, device=e.device)
        mid_out = torch.tensor(0.267, dtype=e.dtype, device=e.device)
        t = e * a_val
        e = t / (t + torch.ones_like(t))
        z = (mid_in * a_val) / (mid_in * a_val + torch.ones_like(mid_in))
        e = e * (mid_out / z)
    elif mode == ToneMapping.HABLE:
        # Shoulder strength
        A = torch.tensor(0.22, dtype=e.dtype, device=e.device)
        B = torch.tensor(0.30, dtype=e.dtype,
                         device=e.device)  # Linear strength
        C = torch.tensor(0.10, dtype=e.dtype, device=e.device)  # Linear angle
        D = torch.tensor(0.20, dtype=e.dtype, device=e.device)  # Toe strength
        E = torch.tensor(0.01, dtype=e.dtype, device=e.device)  # Toe numerator
        F = torch.tensor(0.30, dtype=e.dtype,
                         device=e.device)  # Toe denominator
        W = torch.tensor(11.2, dtype=e.dtype, device=e.device)

        def hable(e_val: torch.Tensor) -> torch.Tensor:
            return (e_val * (A * e_val + C * B) + D * E) / (e_val * (A * e_val + B) + D * F) - E / F
        e = hable(e) / hable(W)
    return e
