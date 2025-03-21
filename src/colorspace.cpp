#include "colorspace.hpp"
#include "imageops.hpp"
#include <cmath>

namespace colorspace {

// ----------------------------------------
// CONSTANTS
// ----------------------------------------
// NOTE: sRGB transformations

// See IEC 61966-2-1/Amd 1:2003, Equation F.7.
static const double SRGB_R = 0.212639, SRGB_G = 0.715169, SRGB_B = 0.072192;

double sRGBLuminance(Color e) {
    return SRGB_R * e.r + SRGB_G * e.g + SRGB_B * e.b;
}

// See ITU-R BT.709-6, Section 3.
// Uses the same coefficients for deriving luma signal as
// IEC 61966-2-1/Amd 1:2003 states for luminance, so we reuse the luminance
// function above.
static const double SRGB_CB = (2 * (1 - SRGB_B)), SRGB_CR = (2 * (1 - SRGB_R));

// these are gamma encoded, i.e. non-linear
Color sRGB_RGBToYUV(Color e_gamma) {
    double y_gamma = sRGBLuminance(e_gamma);
    return {{{y_gamma, (e_gamma.b - y_gamma) / SRGB_CB,
              (e_gamma.r - y_gamma) / SRGB_CR}}};
}

// See ITU-R BT.709-6, Section 3.
// Same derivation to BT.2100's YUV->RGB, below. Similar to srgbRgbToYuv, we
// can reuse the luminance coefficients since they are the same.
static const double SRGB_GCB = SRGB_B * SRGB_CB / SRGB_G;
static const double SRGB_GCR = SRGB_R * SRGB_CR / SRGB_G;

Color sRGB_YUVToRGB(Color e_gamma) {
    return {{{CLAMP(e_gamma.y + SRGB_CR * e_gamma.v),
              CLAMP(e_gamma.y - SRGB_GCB * e_gamma.u - SRGB_GCR * e_gamma.v),
              CLAMP(e_gamma.y + SRGB_CB * e_gamma.u)}}};
}

// See IEC 61966-2-1/Amd 1:2003, Equations F.5 and F.6.
double sRGB_InvOETF(double e_gamma) {
    if (e_gamma <= 0.04045) {
        return e_gamma / 12.92;
    } else {
        return pow((e_gamma + 0.055) / 1.055, 2.4);
    }
}

Color sRGB_InvOETF(Color e_gamma) {
    return {{{sRGB_InvOETF(e_gamma.r), sRGB_InvOETF(e_gamma.g),
              sRGB_InvOETF(e_gamma.b)}}};
}

double sRGB_InvOETFLUT(double e_gamma) {
    int32_t value =
        static_cast<int32_t>(e_gamma * (SRGB_INV_OETF_NUMENTRIES - 1) + 0.5);
    // TODO() : Remove once conversion modules have appropriate clamping in
    // place
    value = CLIP(value, 0, SRGB_INV_OETF_NUMENTRIES - 1);
    static LookUpTable kSrgbLut(SRGB_INV_OETF_NUMENTRIES,
                                static_cast<double (*)(double)>(sRGB_InvOETF));
    return kSrgbLut.getTable()[value];
}

Color sRGB_InvOETFLUT(Color e_gamma) {
    return {{{sRGB_InvOETFLUT(e_gamma.r), sRGB_InvOETFLUT(e_gamma.g),
              sRGB_InvOETFLUT(e_gamma.b)}}};
}

// See IEC 61966-2-1/Amd 1:2003, Equations F.10 and F.11.
double sRGB_OETF(double e) {
    constexpr double threshold = 0.0031308;
    constexpr double low_slope = 12.92;
    constexpr double high_offset = 0.055;
    constexpr double power_exponent = 1.0 / 2.4;
    if (e <= threshold) {
        return low_slope * e;
    }
    return (1.0 + high_offset) * std::pow(e, power_exponent) - high_offset;
}

Color sRGB_OETF(Color e) {
    return {{{sRGB_OETF(e.r), sRGB_OETF(e.g), sRGB_OETF(e.b)}}};
}

// NOTE: Display-P3 transformations

// See SMPTE EG 432-1, Equation G-7.
static const double P3_R = 0.2289746, P3_G = 0.6917385, P3_B = 0.0792869;

double P3Luminance(Color e) { return P3_R * e.r + P3_G * e.g + P3_B * e.b; }

// See ITU-R BT.601-7, Sections 2.5.1 and 2.5.2.
// Unfortunately, calculation of luma signal differs from calculation of
// luminance for Display-P3, so we can't reuse P3Luminance here.
static const double P3_YR = 0.299, P3_YG = 0.587, P3_YB = 0.114;
static const double P3_CB = 1.772, P3_CR = 1.402;

Color P3_RGBToYUV(Color e_gamma) {
    double y_gamma = P3_YR * e_gamma.r + P3_YG * e_gamma.g + P3_YB * e_gamma.b;
    return {{{y_gamma, (e_gamma.b - y_gamma) / P3_CB,
              (e_gamma.r - y_gamma) / P3_CR}}};
}

// See ITU-R BT.601-7, Sections 2.5.1 and 2.5.2.
// Same derivation to BT.2100's YUV->RGB, below. Similar to P3_RGBToYUV, we must
// use luma signal coefficients rather than the luminance coefficients.
static const double P3_GCB = P3_YB * P3_CB / P3_YG;
static const double P3_GCR = P3_YR * P3_CR / P3_YG;

Color P3_YUVToRGB(Color e_gamma) {
    return {{{CLAMP(e_gamma.y + P3_CR * e_gamma.v),
              CLAMP(e_gamma.y - P3_GCB * e_gamma.u - P3_GCR * e_gamma.v),
              CLAMP(e_gamma.y + P3_CB * e_gamma.u)}}};
}

// NOTE: BT.2100 transformations - according to ITU-R BT.2100-2

// See ITU-R BT.2100-2, Table 5, HLG Reference OOTF
static const double BT2100_R = 0.2627, BT2100_G = 0.677998, BT2100_B = 0.059302;

double Bt2100Luminance(Color e) {
    return BT2100_R * e.r + BT2100_G * e.g + BT2100_B * e.b;
}

// See ITU-R BT.2100-2, Table 6, Derivation of colour difference signals.
// BT.2100 uses the same coefficients for calculating luma signal and luminance,
// so we reuse the luminance function here.
static const double BT2100_CB = (2 * (1 - BT2100_B)),
                    BT2100_CR = (2 * (1 - BT2100_R));

Color Bt2100_RGBToYUV(Color e_gamma) {
    double y_gamma = Bt2100Luminance(e_gamma);
    return {{{y_gamma, (e_gamma.b - y_gamma) / BT2100_CB,
              (e_gamma.r - y_gamma) / BT2100_CR}}};
}

// See ITU-R BT.2100-2, Table 6, Derivation of colour difference signals.
//
// Similar to Bt2100_RGBToYUV above, we can reuse the luminance coefficients.
//
// Derived by inversing Bt2100_RGBToYUV. The derivation for R and B are  pretty
// straight forward; we just invert the formulas for U and V above. But deriving
// the formula for G is a bit more complicated:
//
// Start with equation for luminance:
//   Y = BT2100_R * R + BT2100_G * G + BT2100_B * B
// Solve for G:
//   G = (Y - BT2100_R * R - BT2100_B * B) / BT2100_B
// Substitute equations for R and B in terms YUV:
//   G = (Y - BT2100_R * (Y + BT2100_CR * V) - BT2100_B * (Y + BT2100_CB * U)) /
//   BT2100_B
// Simplify:
//   G = Y * ((1 - BT2100_R - BT2100_B) / BT2100_G)
//     + U * (BT2100_B * BT2100_CB / BT2100_G)
//     + V * (BT2100_R * BT2100_CR / BT2100_G)
//
// We then get the following coeficients for calculating G from YUV:
//
// Coef for Y = (1 - BT2100_R - BT2100_B) / BT2100_G = 1
// Coef for U = BT2100_B * BT2100_CB / BT2100_G = BT2100_GCB = ~0.1645
// Coef for V = BT2100_R * BT2100_CR / BT2100_G = BT2100_GCR = ~0.5713

static const double BT2100_GCB = BT2100_B * BT2100_CB / BT2100_G;
static const double BT2100_GCR = BT2100_R * BT2100_CR / BT2100_G;

Color Bt2100_YUVToRGB(Color e_gamma) {
    return {
        {{CLAMP(e_gamma.y + BT2100_CR * e_gamma.v),
          CLAMP(e_gamma.y - BT2100_GCB * e_gamma.u - BT2100_GCR * e_gamma.v),
          CLAMP(e_gamma.y + BT2100_CB * e_gamma.u)}}};
}

// See ITU-R BT.2100-2, Table 5, HLG Reference OETF.
static const double HLG_A = 0.17883277, HLG_B = 0.28466892, HLG_C = 0.55991073;

double HLG_OETF(double e) {
    if (e <= 1.0 / 12.0) {
        return sqrt(3.0 * e);
    } else {
        return HLG_A * log(12.0 * e - HLG_B) + HLG_C;
    }
}

Color HLG_OETF(Color e) {
    return {{{HLG_OETF(e.r), HLG_OETF(e.g), HLG_OETF(e.b)}}};
}

double HLG_OETFLUT(double e) {
    int32_t value = static_cast<int32_t>(e * (HLG_OETF_NUMENTRIES - 1) + 0.5);
    // TODO() : Remove once conversion modules have appropriate clamping in
    // place
    value = CLIP(value, 0, HLG_OETF_NUMENTRIES - 1);
    static LookUpTable kHlgLut(HLG_OETF_NUMENTRIES,
                               static_cast<double (*)(double)>(HLG_OETF));
    return kHlgLut.getTable()[value];
}

Color HLG_OETFLUT(Color e) {
    return {{{HLG_OETFLUT(e.r), HLG_OETFLUT(e.g), HLG_OETFLUT(e.b)}}};
}

// See ITU-R BT.2100-2, Table 5, HLG Reference EOTF.
double HLG_InvOETF(double e_gamma) {
    if (e_gamma <= 0.5) {
        return pow(e_gamma, 2.0) / 3.0;
    } else {
        return (exp((e_gamma - HLG_C) / HLG_A) + HLG_B) / 12.0;
    }
}

Color HLG_InvOETF(Color e_gamma) {
    return {{{HLG_InvOETF(e_gamma.r), HLG_InvOETF(e_gamma.g),
              HLG_InvOETF(e_gamma.b)}}};
}

double HLG_InvOETFLUT(double e_gamma) {
    int32_t value =
        static_cast<int32_t>(e_gamma * (HLG_INV_OETF_NUMENTRIES - 1) + 0.5);
    // TODO() : Remove once conversion modules have appropriate clamping in
    // place
    value = CLIP(value, 0, HLG_INV_OETF_NUMENTRIES - 1);
    static LookUpTable kHlgInvLut(HLG_INV_OETF_NUMENTRIES,
                                  static_cast<double (*)(double)>(HLG_InvOETF));
    return kHlgInvLut.getTable()[value];
}

Color HLG_InvOETFLUT(Color e_gamma) {
    return {{{HLG_InvOETFLUT(e_gamma.r), HLG_InvOETFLUT(e_gamma.g),
              HLG_InvOETFLUT(e_gamma.b)}}};
}

// See ITU-R BT.2100-2, Table 5, Note 5f
// Gamma = 1.2 + 0.42 * log(kHlgMaxNits / 1000)
static const double OOTF_GAMMA = 1.2;

// See ITU-R BT.2100-2, Table 5, HLG Reference OOTF
Color HLG_OOTF(Color e, LuminanceFn luminance) {
    double y = luminance(e);
    return e * std::pow(y, OOTF_GAMMA - 1.0);
}

Color HLG_OOTFApprox(Color e, [[maybe_unused]] LuminanceFn luminance) {
    return {{{std::pow(e.r, OOTF_GAMMA), std::pow(e.g, OOTF_GAMMA),
              std::pow(e.b, OOTF_GAMMA)}}};
}

// See ITU-R BT.2100-2, Table 5, Note 5i
Color HLG_InvOOTF(Color e, LuminanceFn luminance) {
    double y = luminance(e);
    return e * std::pow(y, (1.0 / OOTF_GAMMA) - 1.0);
}

Color HLG_InvOOTFApprox(Color e) {
    return {{{std::pow(e.r, 1.0 / OOTF_GAMMA), std::pow(e.g, 1.0 / OOTF_GAMMA),
              std::pow(e.b, 1.0 / OOTF_GAMMA)}}};
}

// See ITU-R BT.2100-2, Table 4, Reference PQ OETF.
static const double PQ_M1 = 2610.0 / 16384.0, PQ_M2 = 2523.0 / 4096.0 * 128.0;
static const double PQ_C1 = 3424.0 / 4096.0, PQ_C2 = 2413.0 / 4096.0 * 32.0,
                    PQ_C3 = 2392.0 / 4096.0 * 32.0;

double PQ_OETF(double e) {
    if (e <= 0.0)
        return 0.0;
    return pow((PQ_C1 + PQ_C2 * pow(e, PQ_M1)) / (1 + PQ_C3 * pow(e, PQ_M1)),
               PQ_M2);
}

Color PQ_OETF(Color e) {
    return {{{PQ_OETF(e.r), PQ_OETF(e.g), PQ_OETF(e.b)}}};
}

double PQ_OETFLUT(double e) {
    int32_t value = static_cast<int32_t>(e * (PQ_OETF_NUMENTRIES - 1) + 0.5);
    // TODO() : Remove once conversion modules have appropriate clamping in
    // place
    value = CLIP(value, 0, PQ_OETF_NUMENTRIES - 1);
    static LookUpTable kPqLut(PQ_OETF_NUMENTRIES,
                              static_cast<double (*)(double)>(PQ_OETF));
    return kPqLut.getTable()[value];
}

Color PQ_OETFLUT(Color e) {
    return {{{PQ_OETFLUT(e.r), PQ_OETFLUT(e.g), PQ_OETFLUT(e.b)}}};
}

double PQ_InvOETF(double e_gamma) {
    double val = pow(e_gamma, (1 / PQ_M2));
    return pow((((std::max)(val - PQ_C1, 0.0)) / (PQ_C2 - PQ_C3 * val)),
               1 / PQ_M1);
}

Color PQ_InvOETF(Color e_gamma) {
    return {{{PQ_InvOETF(e_gamma.r), PQ_InvOETF(e_gamma.g),
              PQ_InvOETF(e_gamma.b)}}};
}

double PQ_InvOETFLUT(double e_gamma) {
    int32_t value =
        static_cast<int32_t>(e_gamma * (PQ_INV_OETF_NUMENTRIES - 1) + 0.5);
    // TODO() : Remove once conversion modules have appropriate clamping in
    // place
    value = CLIP(value, 0, PQ_INV_OETF_NUMENTRIES - 1);
    static LookUpTable kPqInvLut(PQ_INV_OETF_NUMENTRIES,
                                 static_cast<double (*)(double)>(PQ_InvOETF));
    return kPqInvLut.getTable()[value];
}

Color PQ_InvOETFLUT(Color e_gamma) {
    return {{{PQ_InvOETFLUT(e_gamma.r), PQ_InvOETFLUT(e_gamma.g),
              PQ_InvOETFLUT(e_gamma.b)}}};
}

// ----------------------------------------
// CONVERSION
// ----------------------------------------
// Display P3 also uses sRGB

// Apply a color transformation matrix to a vector of RGB values
std::vector<double> ApplyColorMatrix(const std::vector<double> &rgb,
                                     const std::array<double, 9> &matrix) {
    std::vector<double> result(3, 0.0);

    // Matrix multiplication: result = matrix * rgb
    result[0] = matrix[0] * rgb[0] + matrix[1] * rgb[1] + matrix[2] * rgb[2];
    result[1] = matrix[3] * rgb[0] + matrix[4] * rgb[1] + matrix[5] * rgb[2];
    result[2] = matrix[6] * rgb[0] + matrix[7] * rgb[1] + matrix[8] * rgb[2];

    return result;
}

double Rec2020HLGToLinear(double x) {
    /* Follows ITU-R BT.2100-2 */
    const double a = 0.17883277;
    const double b = 1.0 - 4.0 * a;               // 0.28466892;
    const double c = 0.5 - a * std::log(4.0 * a); // 0.55991073;

    const double epsilon = 1e-6;
    ASSERT(-epsilon <= x && x <= (1.0 + epsilon),
           "Input should be in range [0, 1] for Rec2020, from %f", x);
    // x = std::max(0.0, std::min(1.0, x));

    if (x <= 0.5) {
        return (x * x) / 3.0;
    } else {
        return (std::exp((x - c) / a) + b) / 12.0;
    }
}

double Rec2020GammaToLinear(double x) {
    const double alpha = 1.09929682680944;
    const double beta = 0.018053968510807;
    const double epsilon = 1e-6;
    ASSERT(-epsilon <= x && x <= (1.0 + epsilon),
           "Input should be in range [0, 1] for Rec2020, from %f", x);

    if (x < beta * 4.5) {
        return x / 4.5;
    } else {
        return std::pow((x + alpha - 1) / alpha, 1 / 0.45);
    }
}

double P3PQToLinear(double x) {
    const double epsilon = 1e-6;
    ASSERT(-epsilon <= x && x <= (1.0 + epsilon),
           "Input should be in range [0, 1], from %f", x);
    // SMPTE ST 2084 constants
    const double m1 = 2610.0 / 16384;
    const double m2 = 128.0 * 2523.0 / 4096;
    const double c1 = 3424.0 / 4096.0;
    const double c2 = 32.0 * 2413.0 / 4096.0;
    const double c3 = 32.0 * 2392.0 / 4096.0;

    double xpow = std::pow(x, 1.0 / m2);
    double num = std::max(xpow - c1, 0.0);
    double den = c2 - c3 * xpow;

    // REVISIT:
    // Result is in nits (cd/m²), normalized by 10000 nits
    // For typical display, you may need to adjust this scale factor
    return std::pow(num / den, 1.0 / m1);
}

double sRGBToLinear(double x) {
    const double epsilon = 1e-6;
    ASSERT(-epsilon <= x && x <= (1.0 + epsilon),
           "Input should be in range [0, 1], from %f", x);
    if (x <= 0.04045) {
        return x / 12.92;
    } else {
        return std::pow((x + 0.055) / 1.055, 2.4);
    }
}

double LinearToRec2020HLG(double x) {
    /* Follows ITU-R BT.2100-2 */
    const double a = 0.17883277;
    const double b = 1.0 - 4.0 * a;               // 0.28466892;
    const double c = 0.5 - a * std::log(4.0 * a); // 0.55991073;

    const double epsilon = 1e-6;
    ASSERT(-epsilon <= x && x <= (1.0 + epsilon),
           "Input should be in range [0, 1] for Rec2020, from %f", x);
    // x = std::max(0.0, std::min(1.0, x));

    if (x <= 1.0 / 12.0) {
        return std::sqrt(3 * x);
    } else {
        return a * std::log(12.0 * x - b) + c;
    }
}

double LinearToRec2020Gamma(double x) {
    const double alpha = 1.09929682680944;
    const double beta = 0.018053968510807;
    const double epsilon = 1e-6;
    ASSERT(-epsilon <= x && x <= (1.0 + epsilon),
           "Input should be in range [0, 1] for Rec2020, from %f", x);

    if (x < beta) {
        return x * 4.5;
    } else {
        return alpha * std::pow(x, 0.45) - (alpha - 1);
    }
}

double LinearToP3PQ(double x) {
    const double epsilon = 1e-6;
    ASSERT(-epsilon <= x && x <= (1.0 + epsilon),
           "Input should be in range [0, 1], from %f", x);
    // REVISIT:
    // assumes normalization by 10_000 nits
    const double m1 = 2610.0 / 16384;
    const double m2 = 128.0 * 2523.0 / 4096;
    const double c1 = 3424.0 / 4096.0;
    const double c2 = 32.0 * 2413.0 / 4096.0;
    const double c3 = 32.0 * 2392.0 / 4096.0;

    // Convert from [0,1] range to absolute nits (assuming 10000 nits max)
    double y = x; // * 10000.0;

    // Clamp input to avoid NaN/infinity issues
    // y = std::max(0.0, y);

    double ym1 = std::pow(y, m1);
    double num = c1 + c2 * ym1;
    double den = 1.0 + c3 * ym1;

    return std::pow(num / den, m2);
}

double LinearTosRGB(double x) {
    const double epsilon = 1e-6;
    ASSERT(-epsilon <= x && x <= (1.0 + epsilon),
           "Input should be in range [0, 1], from %f", x);
    if (x <= 0.0031308) {
        return 12.92 * x;
    } else {
        return 1.055 * std::pow(x, 1.0 / 2.4) - 0.055;
    }
}

// ----------------------------------------
// YUV CONVERSION
// ----------------------------------------

// See ITU-R BT.2100-2, Table 6, Derivation of colour difference signals.
// BT.2100 uses the same coefficients for calculating luma signal and luminance,
// so we reuse the luminance function here.
std::vector<double> LinearBt2100ToYUV(const std::vector<double> &rgb) {
    const double bt2100r = 0.2627, bt2100g = 0.677998, bt2100b = 0.059302;
    const double bt2100Cb = (2 * (1 - bt2100b)), bt2100Cr = (2 * (1 - bt2100r));

    double y_gamma = rgb[0] * bt2100r + rgb[1] * bt2100g + rgb[2] * bt2100b;
    return {y_gamma, (rgb[2] - y_gamma) / bt2100Cb,
            (rgb[0] - y_gamma) / bt2100Cr};
}

// See IEC 61966-2-1/Amd 1:2003, Equation F.7.
// See ITU-R BT.709-6, Section 3.
// Uses the same coefficients for deriving luma signal as
// IEC 61966-2-1/Amd 1:2003 states for luminance, so we reuse the luminance
// function above.
std::vector<double> LinearsRGBToYUV(const std::vector<double> &rgb) {
    const double sRGBr = 0.212639, sRGBg = 0.715169, sRGBb = 0.072192;
    const double sRGBCb = (2 * (1 - sRGBb)), sRGBCr = (2 * (1 - sRGBr));

    double y_gamma = rgb[0] * sRGBr + rgb[1] * sRGBg + rgb[2] * sRGBb;
    return {y_gamma, (rgb[2] - y_gamma) / sRGBCb, (rgb[0] - y_gamma) / sRGBCr};
}

double ApplyToneMapping(double x, ToneMapping mode, double target_nits = 100.0,
                        double max_nits = 100.0) {
    if (mode == ToneMapping::BASE) {
        x *= target_nits / max_nits;
    } else if (mode == ToneMapping::REINHARD) {
        x = x / (1.0 + x);
    } else if (mode == ToneMapping::GAMMA) {
        // REVISIT: 2.2
        x = std::pow(x, 1.0 / 2.2);
    } else if (mode == ToneMapping::FILMIC) {
        const double A = 2.51;
        const double B = 0.03;
        const double C = 2.43;
        const double D = 0.59;
        const double E = 0.14;
        x = (x * (A * x + B)) / (x * (C * x + D) + E);
    } else if (mode == ToneMapping::ACES) {
        const double a = 2.51;
        const double b = 0.03;
        const double c = 2.43;
        const double d = 0.59;
        const double e = 0.14;
        // REVISIT: Exposure adjustment for ACES
        double adjusted = x * 0.6;
        x = (adjusted * (adjusted + b) * a) /
            (adjusted * (adjusted * c + d) + e);
    } else if (mode == ToneMapping::UNCHARTED2) {
        const double A = 0.15;
        const double B = 0.50;
        const double C = 0.10;
        const double D = 0.20;
        const double E = 0.02;
        const double F = 0.30;
        const double W = 11.2;

        auto uncharted2_tonemap = [=](double x) -> double {
            return ((x * (A * x + C * B) + D * E) / (x * (A * x + B) + D * F)) -
                   E / F;
        };

        x = uncharted2_tonemap(x) / uncharted2_tonemap(W);
    } else if (mode == ToneMapping::DRAGO) {
        const double bias = 0.85;
        const double Lwa = 1.0;

        x = std::log(1 + x) / std::log(1 + Lwa);
        x = std::pow(x, bias);
    } else if (mode == ToneMapping::LOTTES) {
        const double a = 1.6;

        const double mid_in = 0.18;
        const double mid_out = 0.267;

        const double t = x * a;
        x = t / (t + 1);

        const double z = (mid_in * a) / (mid_in * a + 1);
        x = x * (mid_out / z);
    } else if (mode == ToneMapping::HABLE) {
        const double A = 0.22; // Shoulder strength
        const double B = 0.30; // Linear strength
        const double C = 0.10; // Linear angle
        const double D = 0.20; // Toe strength
        const double E = 0.01; // Toe numerator
        const double F = 0.30; // Toe denominator
        const double W = 11.2;

        std::function<double(double)> hable = [=](double x) -> double {
            return (x * (A * x + C * B) + D * E) / (x * (A * x + B) + D * F) -
                   E / F;
        };

        x = hable(x) / hable(W);
    } else if (mode == ToneMapping::REINHARD) {
        x = x / (1.0 + x);
    }
    return x;
}

// DEPRECATE:
// std::unique_ptr<imageops::PNGImage>
// HDRToYUV(const std::unique_ptr<imageops::PNGImage> &hdr_image, double clip_low,
//          double clip_high, utils::Error &error, ToneMapping mode) {
//     if (!hdr_image || !hdr_image->row_pointers) {
//         error = {true, "Invalid input HDR image"};
//         return nullptr;
//     }
// 
//     auto raw_image = std::make_unique<imageops::PNGImage>();
//     raw_image->width = hdr_image->width;
//     raw_image->height = hdr_image->height;
//     raw_image->color_type = PNG_COLOR_TYPE_RGB;
//     raw_image->bit_depth = hdr_image->bit_depth;
//     // REVISIT: is this right?
//     raw_image->bytes_per_row = hdr_image->bytes_per_row;
// 
//     raw_image->row_pointers =
//         (png_bytep *)malloc(sizeof(png_bytep) * raw_image->height);
//     for (size_t y = 0; y < raw_image->height; y++) {
//         raw_image->row_pointers[y] =
//             (png_byte *)malloc(raw_image->bytes_per_row);
//     }
// 
//     const size_t channels = 3;
//     for (size_t y = 0; y < hdr_image->height; y++) {
//         png_bytep hdr_row = hdr_image->row_pointers[y];
//         png_bytep raw_row = raw_image->row_pointers[y];
// 
//         for (size_t x = 0; x < hdr_image->width; x++) {
//             size_t raw_idx = x * channels;
// 
//             uint16_t values[3];
//             for (size_t i = 0; i < 3; i++) {
//                 // *2 because input is 16-bit
//                 size_t idx = x * channels * 2 + i * 2;
//                 // PNG stores in network byte order (big-endian)
//                 values[i] = (hdr_row[idx] << 8) | hdr_row[idx + 1];
//             }
// 
//             double r = (static_cast<double>(values[0]) / 65535.0);
//             double g = (static_cast<double>(values[1]) / 65535.0);
//             double b = (static_cast<double>(values[2]) / 65535.0);
// 
//             r = Rec2020HLGToLinear(r);
//             g = Rec2020HLGToLinear(g);
//             b = Rec2020HLGToLinear(b);
//             // REVISIT: what to clip to?
//             r = CLIP(r, 0.0, 1.0);
//             g = CLIP(g, 0.0, 1.0);
//             b = CLIP(b, 0.0, 1.0);
// 
//             raw_row[raw_idx + 0] = r;
//             raw_row[raw_idx + 1] = g;
//             raw_row[raw_idx + 2] = b;
//         }
//     }
// 
//     return raw_image;
// }
// 
// std::unique_ptr<imageops::PNGImage>
// HDRToSDR(const std::unique_ptr<imageops::PNGImage> &hdr_image, double clip_low,
//          double clip_high, utils::Error &error,
//          ToneMapping tone_mapping = ToneMapping::BASE) {
//     if (!hdr_image || !hdr_image->row_pointers) {
//         error = {true, "Invalid input HDR image"};
//         return nullptr;
//     }
// 
//     std::unique_ptr<imageops::PNGImage> sdr_image =
//         std::make_unique<imageops::PNGImage>();
//     sdr_image->width = hdr_image->width;
//     sdr_image->height = hdr_image->height;
//     sdr_image->color_type = PNG_COLOR_TYPE_RGB;
//     sdr_image->bit_depth = 8;
//     sdr_image->bytes_per_row = hdr_image->width * 3;
// 
//     sdr_image->row_pointers =
//         (png_bytep *)malloc(sizeof(png_bytep) * sdr_image->height);
//     for (size_t y = 0; y < sdr_image->height; y++) {
//         sdr_image->row_pointers[y] =
//             (png_byte *)malloc(sdr_image->bytes_per_row);
//     }
// 
//     const size_t channels = 3;
// 
//     // REVISIT:
//     // double max_value = 0.0;
//     // if (params.auto_clip) {
//     //     std::vector<double> all_values;
//     //     all_values.reserve(hdr_image->width * hdr_image->height * channels);
//     //     for (size_t y = 0; y < hdr_image->height; y++) {
//     //         png_bytep row = hdr_image->row_pointers[y];
//     //         for (size_t x = 0; x < hdr_image->width; x++) {
//     //             for (size_t c = 0; c < channels; c++) {
//     //                 size_t idx = x * channels + c;
//     //                 // Convert 16-bit value to double [0,1]
//     //                 uint16_t value = (row[idx * 2] << 8) | row[idx * 2 + 1];
//     //                 all_values.push_back(value / 65535.0);
//     //             }
//     //         }
//     //     }
//     //     // Sort to find 99.9th percentile
//     //     std::sort(all_values.begin(), all_values.end());
//     //     size_t percentile_idx = static_cast<size_t>(all_values.size() *
//     //     0.999); max_value = all_values[percentile_idx];
//     // } else {
//     //     max_value = params.clip_high > 0 ? params.clip_high : 1.0;
//     // }
// 
//     for (size_t y = 0; y < hdr_image->height; y++) {
//         png_bytep hdr_row = hdr_image->row_pointers[y];
//         png_bytep sdr_row = sdr_image->row_pointers[y];
// 
//         for (size_t x = 0; x < hdr_image->width; x++) {
//             size_t sdr_idx = x * channels;
// 
//             uint16_t values[3];
//             for (size_t i = 0; i < 3; i++) {
//                 // *2 because input is 16-bit
//                 size_t idx = x * channels * 2 + i * 2;
//                 // PNG stores in network byte order (big-endian)
//                 values[i] = (hdr_row[idx] << 8) | hdr_row[idx + 1];
//             }
// 
//             double r = (static_cast<double>(values[0]) / 65535.0);
//             double g = (static_cast<double>(values[1]) / 65535.0);
//             double b = (static_cast<double>(values[2]) / 65535.0);
// 
//             r = Rec2020HLGToLinear(r);
//             g = Rec2020HLGToLinear(g);
//             b = Rec2020HLGToLinear(b);
//             // REVISIT: what to clip to?
//             r = CLIP(r, 0.0, 1.0);
//             g = CLIP(g, 0.0, 1.0);
//             b = CLIP(b, 0.0, 1.0);
//             r = ApplyToneMapping(r, tone_mapping);
//             g = ApplyToneMapping(g, tone_mapping);
//             b = ApplyToneMapping(b, tone_mapping);
//             // LinearRec2020ToLinearsRGB(r, g, b);
//             std::vector<double> rgb =
//                 XYZToLinearsRGB(LinearRec2020ToXYZ({r, g, b}));
//             r = LinearTosRGB(CLIP(rgb[0], 0.0, 1.0));
//             g = LinearTosRGB(CLIP(rgb[1], 0.0, 1.0));
//             b = LinearTosRGB(CLIP(rgb[2], 0.0, 1.0));
// 
//             sdr_row[sdr_idx + 0] =
//                 static_cast<uint8_t>(CLIP(r * 255.0, 0.0, 255.0));
//             sdr_row[sdr_idx + 1] =
//                 static_cast<uint8_t>(CLIP(g * 255.0, 0.0, 255.0));
//             sdr_row[sdr_idx + 2] =
//                 static_cast<uint8_t>(CLIP(b * 255.0, 0.0, 255.0));
//         }
//     }
// 
//     return sdr_image;
// }
} // namespace colorspace
