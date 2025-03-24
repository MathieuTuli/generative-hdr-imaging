#include "colorspace.hpp"
#include "imageops.hpp"
#include <cmath>

namespace colorspace {
// NOTE: some helpful tips
// Color e: linear space
// Color e_gamma: non-linear, gamma-encoded space

// NOTE: sRGB transformations

// See IEC 61966-2-1/Amd 1:2003, Equation F.7.
static const float SRGB_R = 0.212639f, SRGB_G = 0.715169f, SRGB_B = 0.072192f;

float sRGBLuminance(Color e) {
    return SRGB_R * e.r + SRGB_G * e.g + SRGB_B * e.b;
}

// See ITU-R BT.709-6, Section 3.
// Uses the same coefficients for deriving luma signal as
// IEC 61966-2-1/Amd 1:2003 states for luminance, so we reuse the luminance
// function above.
static const float SRGB_CB = (2.f * (1.f - SRGB_B)), SRGB_CR = (2 * (1 - SRGB_R));

// these are gamma encoded, i.e. non-linear
Color sRGB_RGBToYUV(Color e_gamma) {
    float y_gamma = sRGBLuminance(e_gamma);
    return {{{y_gamma, (e_gamma.b - y_gamma) / SRGB_CB,
              (e_gamma.r - y_gamma) / SRGB_CR}}};
}

// See ITU-R BT.709-6, Section 3.
// Same derivation to BT.2100's YUV->RGB, below. Similar to srgbRgbToYuv, we
// can reuse the luminance coefficients since they are the same.
static const float SRGB_GCB = SRGB_B * SRGB_CB / SRGB_G;
static const float SRGB_GCR = SRGB_R * SRGB_CR / SRGB_G;

Color sRGB_YUVToRGB(Color e_gamma) {
    return {{{CLAMP(e_gamma.y + SRGB_CR * e_gamma.v),
              CLAMP(e_gamma.y - SRGB_GCB * e_gamma.u - SRGB_GCR * e_gamma.v),
              CLAMP(e_gamma.y + SRGB_CB * e_gamma.u)}}};
}

// See IEC 61966-2-1/Amd 1:2003, Equations F.5 and F.6.
float sRGB_InvOETF(float e_gamma) {
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

float sRGB_InvOETFLUT(float e_gamma) {
    int32_t value =
        static_cast<int32_t>(e_gamma * (SRGB_INV_OETF_NUMENTRIES - 1) + 0.5);
    // TODO() : Remove once conversion modules have appropriate clamping in
    // place
    value = CLIP(value, 0, SRGB_INV_OETF_NUMENTRIES - 1);
    static LookUpTable kSrgbLut(SRGB_INV_OETF_NUMENTRIES,
                                static_cast<float (*)(float)>(sRGB_InvOETF));
    return kSrgbLut.getTable()[value];
}

Color sRGB_InvOETFLUT(Color e_gamma) {
    return {{{sRGB_InvOETFLUT(e_gamma.r), sRGB_InvOETFLUT(e_gamma.g),
              sRGB_InvOETFLUT(e_gamma.b)}}};
}

// See IEC 61966-2-1/Amd 1:2003, Equations F.10 and F.11.
float sRGB_OETF(float e) {
    constexpr float threshold = 0.0031308;
    constexpr float low_slope = 12.92;
    constexpr float high_offset = 0.055;
    constexpr float power_exponent = 1.0 / 2.4;
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
static const float P3_R = 0.2289746f, P3_G = 0.6917385f, P3_B = 0.0792869f;

float P3Luminance(Color e) { return P3_R * e.r + P3_G * e.g + P3_B * e.b; }

// See ITU-R BT.601-7, Sections 2.5.1 and 2.5.2.
// Unfortunately, calculation of luma signal differs from calculation of
// luminance for Display-P3, so we can't reuse P3Luminance here.
static const float P3_YR = 0.299f, P3_YG = 0.587f, P3_YB = 0.114f;
static const float P3_CB = 1.772f, P3_CR = 1.402f;

Color P3_RGBToYUV(Color e_gamma) {
    float y_gamma = P3_YR * e_gamma.r + P3_YG * e_gamma.g + P3_YB * e_gamma.b;
    return {{{y_gamma, (e_gamma.b - y_gamma) / P3_CB,
              (e_gamma.r - y_gamma) / P3_CR}}};
}

// See ITU-R BT.601-7, Sections 2.5.1 and 2.5.2.
// Same derivation to BT.2100's YUV->RGB, below. Similar to P3_RGBToYUV, we must
// use luma signal coefficients rather than the luminance coefficients.
static const float P3_GCB = P3_YB * P3_CB / P3_YG;
static const float P3_GCR = P3_YR * P3_CR / P3_YG;

Color P3_YUVToRGB(Color e_gamma) {
    return {{{CLAMP(e_gamma.y + P3_CR * e_gamma.v),
              CLAMP(e_gamma.y - P3_GCB * e_gamma.u - P3_GCR * e_gamma.v),
              CLAMP(e_gamma.y + P3_CB * e_gamma.u)}}};
}

// NOTE: BT.2100 transformations - according to ITU-R BT.2100-2

// See ITU-R BT.2100-2, Table 5, HLG Reference OOTF
static const float BT2100_R = 0.2627f, BT2100_G = 0.677998f, BT2100_B = 0.059302f;

float Bt2100Luminance(Color e) {
    return BT2100_R * e.r + BT2100_G * e.g + BT2100_B * e.b;
}

// See ITU-R BT.2100-2, Table 6, Derivation of colour difference signals.
// BT.2100 uses the same coefficients for calculating luma signal and luminance,
// so we reuse the luminance function here.
static const float BT2100_CB = (2.f * (1.f - BT2100_B)),
                   BT2100_CR = (2.f * (1.f - BT2100_R));

Color Bt2100_RGBToYUV(Color e_gamma) {
    float y_gamma = Bt2100Luminance(e_gamma);
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

static const float BT2100_GCB = BT2100_B * BT2100_CB / BT2100_G;
static const float BT2100_GCR = BT2100_R * BT2100_CR / BT2100_G;

Color Bt2100_YUVToRGB(Color e_gamma) {
    return {
        {{CLAMP(e_gamma.y + BT2100_CR * e_gamma.v),
          CLAMP(e_gamma.y - BT2100_GCB * e_gamma.u - BT2100_GCR * e_gamma.v),
          CLAMP(e_gamma.y + BT2100_CB * e_gamma.u)}}};
}

// See ITU-R BT.2100-2, Table 5, HLG Reference OETF.
static const float HLG_A = 0.17883277f, HLG_B = 0.28466892f, HLG_C = 0.55991073f;

float HLG_OETF(float e) {
    if (e <= 1.0 / 12.0) {
        return sqrt(3.0 * e);
    } else {
        return HLG_A * log(12.0 * e - HLG_B) + HLG_C;
    }
}

Color HLG_OETF(Color e) {
    return {{{HLG_OETF(e.r), HLG_OETF(e.g), HLG_OETF(e.b)}}};
}

float HLG_OETFLUT(float e) {
    int32_t value = static_cast<int32_t>(e * (HLG_OETF_NUMENTRIES - 1) + 0.5);
    // TODO() : Remove once conversion modules have appropriate clamping in
    // place
    value = CLIP(value, 0, HLG_OETF_NUMENTRIES - 1);
    static LookUpTable kHlgLut(HLG_OETF_NUMENTRIES,
                               static_cast<float (*)(float)>(HLG_OETF));
    return kHlgLut.getTable()[value];
}

Color HLG_OETFLUT(Color e) {
    return {{{HLG_OETFLUT(e.r), HLG_OETFLUT(e.g), HLG_OETFLUT(e.b)}}};
}

// See ITU-R BT.2100-2, Table 5, HLG Reference EOTF.
float HLG_InvOETF(float e_gamma) {
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

float HLG_InvOETFLUT(float e_gamma) {
    int32_t value =
        static_cast<int32_t>(e_gamma * (HLG_INV_OETF_NUMENTRIES - 1) + 0.5);
    // TODO() : Remove once conversion modules have appropriate clamping in
    // place
    value = CLIP(value, 0, HLG_INV_OETF_NUMENTRIES - 1);
    static LookUpTable kHlgInvLut(HLG_INV_OETF_NUMENTRIES,
                                  static_cast<float (*)(float)>(HLG_InvOETF));
    return kHlgInvLut.getTable()[value];
}

Color HLG_InvOETFLUT(Color e_gamma) {
    return {{{HLG_InvOETFLUT(e_gamma.r), HLG_InvOETFLUT(e_gamma.g),
              HLG_InvOETFLUT(e_gamma.b)}}};
}

// See ITU-R BT.2100-2, Table 5, Note 5f
// Gamma = 1.2 + 0.42 * log(kHlgMaxNits / 1000)
static const float OOTF_GAMMA = 1.2f;

// See ITU-R BT.2100-2, Table 5, HLG Reference OOTF
Color HLG_OOTF(Color e, LuminanceFn luminance) {
    float y = luminance(e);
    return e * std::pow(y, OOTF_GAMMA - 1.0);
}

Color HLG_OOTFApprox(Color e, [[maybe_unused]] LuminanceFn luminance) {
    return {{{std::pow(e.r, OOTF_GAMMA), std::pow(e.g, OOTF_GAMMA),
              std::pow(e.b, OOTF_GAMMA)}}};
}

// See ITU-R BT.2100-2, Table 5, Note 5i
Color HLG_InvOOTF(Color e, LuminanceFn luminance) {
    float y = luminance(e);
    return e * std::pow(y, (1.0 / OOTF_GAMMA) - 1.0);
}

Color HLG_InvOOTFApprox(Color e, [[maybe_unused]] LuminanceFn luminance) {
    return {{{std::pow(e.r, 1.0f / OOTF_GAMMA), std::pow(e.g, 1.0f / OOTF_GAMMA),
              std::pow(e.b, 1.0f / OOTF_GAMMA)}}};
}

// See ITU-R BT.2100-2, Table 4, Reference PQ OETF.
static const float PQ_M1 = 2610.0f / 16384.0f, PQ_M2 = 2523.0f / 4096.0f * 128.0f;
static const float PQ_C1 = 3424.0f / 4096.0f, PQ_C2 = 2413.0f / 4096.0f * 32.0f,
                   PQ_C3 = 2392.0f / 4096.0f * 32.0f;

float PQ_OETF(float e) {
    if (e <= 0.0)
        return 0.0;
    return pow((PQ_C1 + PQ_C2 * pow(e, PQ_M1)) / (1 + PQ_C3 * pow(e, PQ_M1)),
               PQ_M2);
}

Color PQ_OETF(Color e) {
    return {{{PQ_OETF(e.r), PQ_OETF(e.g), PQ_OETF(e.b)}}};
}

float PQ_OETFLUT(float e) {
    int32_t value = static_cast<int32_t>(e * (PQ_OETF_NUMENTRIES - 1) + 0.5);
    // TODO() : Remove once conversion modules have appropriate clamping in
    // place
    value = CLIP(value, 0, PQ_OETF_NUMENTRIES - 1);
    static LookUpTable kPqLut(PQ_OETF_NUMENTRIES,
                              static_cast<float (*)(float)>(PQ_OETF));
    return kPqLut.getTable()[value];
}

Color PQ_OETFLUT(Color e) {
    return {{{PQ_OETFLUT(e.r), PQ_OETFLUT(e.g), PQ_OETFLUT(e.b)}}};
}

float PQ_InvOETF(float e_gamma) {
    float val = pow(e_gamma, (1 / PQ_M2));
    return pow((((std::max)(val - PQ_C1, 0.0f)) / (PQ_C2 - PQ_C3 * val)),
               1.f / PQ_M1);
}

Color PQ_InvOETF(Color e_gamma) {
    return {{{PQ_InvOETF(e_gamma.r), PQ_InvOETF(e_gamma.g),
              PQ_InvOETF(e_gamma.b)}}};
}

float PQ_InvOETFLUT(float e_gamma) {
    int32_t value =
        static_cast<int32_t>(e_gamma * (PQ_INV_OETF_NUMENTRIES - 1) + 0.5);
    // TODO() : Remove once conversion modules have appropriate clamping in
    // place
    value = CLIP(value, 0, PQ_INV_OETF_NUMENTRIES - 1);
    static LookUpTable kPqInvLut(PQ_INV_OETF_NUMENTRIES,
                                 static_cast<float (*)(float)>(PQ_InvOETF));
    return kPqInvLut.getTable()[value];
}

Color PQ_InvOETFLUT(Color e_gamma) {
    return {{{PQ_InvOETFLUT(e_gamma.r), PQ_InvOETFLUT(e_gamma.g),
              PQ_InvOETFLUT(e_gamma.b)}}};
}

////////////////////////////////////////////////////////////////////////////////
// Color space conversions
// Sample, See,
// https://registry.khronos.org/DataFormat/specs/1.3/dataformat.1.3.html#_bt_709_bt_2020_primary_conversion_example

const std::array<float, 9> BT709_TO_P3 = {0.822462f, 0.177537f, 0.000001f,
                                          0.033194f, 0.966807f, -0.000001f,
                                          0.017083f, 0.072398f, 0.91052f};
const std::array<float, 9> BT709_TO_BT2100 = {0.627404f, 0.329282f, 0.043314f,
                                              0.069097f, 0.919541f, 0.011362f,
                                              0.016392f, 0.088013f, 0.895595f};
const std::array<float, 9> P3_TO_BT709 = {1.22494f,   -0.22494f,  0.0f,
                                          -0.042057f, 1.042057f,  0.0f,
                                          -0.019638f, -0.078636f, 1.098274f};
const std::array<float, 9> P3_TO_BT2100 = {0.753833f, 0.198597f, 0.04757f,
                                           0.045744f, 0.941777f, 0.012479f,
                                           -0.00121f, 0.017601f, 0.983608f};
const std::array<float, 9> BT2100_TO_BT709 = {
    1.660491f,  -0.587641f, -0.07285f,  -0.124551f, 1.1329f,
    -0.008349f, -0.018151f, -0.100579f, 1.11873f};
const std::array<float, 9> BT2100_TO_P3 = {1.343578f,  -0.282179f, -0.061399f,
                                           -0.065298f, 1.075788f,  -0.01049f,
                                           0.002822f,  -0.019598f, 1.016777f};
Color ConvertGamut(Color e, const std::array<float, 9> &coeffs) {
    return {{{coeffs[0] * e.r + coeffs[1] * e.g + coeffs[2] * e.b,
              coeffs[3] * e.r + coeffs[4] * e.g + coeffs[5] * e.b,
              coeffs[6] * e.r + coeffs[7] * e.g + coeffs[8] * e.b}}};
}
Color Bt709ToP3(Color e) { return ConvertGamut(e, BT709_TO_P3); }
Color Bt709ToBt2100(Color e) { return ConvertGamut(e, BT709_TO_BT2100); }
Color P3ToBt709(Color e) { return ConvertGamut(e, P3_TO_BT709); }
Color P3ToBt2100(Color e) { return ConvertGamut(e, P3_TO_BT2100); }
Color Bt2100ToBt709(Color e) { return ConvertGamut(e, BT2100_TO_BT709); }
Color Bt2100ToP3(Color e) { return ConvertGamut(e, BT2100_TO_P3); }

// All of these conversions are derived from the respective input YUV->RGB
// conversion followed by the RGB->YUV for the receiving encoding. They are
// consistent with the RGB<->YUV functions in gainmapmath.cpp, given that we use
// BT.709 encoding for sRGB and BT.601 encoding for Display-P3, to match
// DataSpace.

// Yuv Bt709 -> Yuv Bt601
// Y' = (1.0 * Y) + ( 0.101579 * U) + ( 0.196076 * V)
// U' = (0.0 * Y) + ( 0.989854 * U) + (-0.110653 * V)
// V' = (0.0 * Y) + (-0.072453 * U) + ( 0.983398 * V)
const std::array<float, 9> YUV_BT709_TO_BT601 = {1.0f, 0.101579f,  0.196076f,
                                                 0.0f, 0.989854f,  -0.110653f,
                                                 0.0f, -0.072453f, 0.983398f};

// Yuv Bt709 -> Yuv Bt2100
// Y' = (1.0 * Y) + (-0.016969 * U) + ( 0.096312 * V)
// U' = (0.0 * Y) + ( 0.995306 * U) + (-0.051192 * V)
// V' = (0.0 * Y) + ( 0.011507 * U) + ( 1.002637 * V)
const std::array<float, 9> YUV_BT709_TO_BT2100 = {1.0f, -0.016969f, 0.096312f,
                                                  0.0f, 0.995306f,  -0.051192f,
                                                  0.0f, 0.011507f,  1.002637f};

// Yuv Bt601 -> Yuv Bt709
// Y' = (1.0 * Y) + (-0.118188 * U) + (-0.212685 * V)
// U' = (0.0 * Y) + ( 1.018640 * U) + ( 0.114618 * V)
// V' = (0.0 * Y) + ( 0.075049 * U) + ( 1.025327 * V)
const std::array<float, 9> YUV_BT601_TO_BT709 = {1.0f, -0.118188f, -0.212685f,
                                                 0.0f, 1.018640f,  0.114618f,
                                                 0.0f, 0.075049f,  1.025327f};

// Yuv Bt601 -> Yuv Bt2100
// Y' = (1.0 * Y) + (-0.128245 * U) + (-0.115879 * V)
// U' = (0.0 * Y) + ( 1.010016 * U) + ( 0.061592 * V)
// V' = (0.0 * Y) + ( 0.086969 * U) + ( 1.029350 * V)
const std::array<float, 9> YUV_BT601_TO_BT2100 = {1.0f, -0.128245f, -0.115879,
                                                  0.0f, 1.010016f,  0.061592f,
                                                  0.0f, 0.086969f,  1.029350f};

// Yuv Bt2100 -> Yuv Bt709
// Y' = (1.0 * Y) + ( 0.018149 * U) + (-0.095132 * V)
// U' = (0.0 * Y) + ( 1.004123 * U) + ( 0.051267 * V)
// V' = (0.0 * Y) + (-0.011524 * U) + ( 0.996782 * V)
const std::array<float, 9> YUV_BT2100_TO_BT709 = {1.0f, 0.018149f,  -0.095132f,
                                                  0.0f, 1.004123f,  0.051267f,
                                                  0.0f, -0.011524f, 0.996782f};

// Yuv Bt2100 -> Yuv Bt601
// Y' = (1.0 * Y) + ( 0.117887 * U) + ( 0.105521 * V)
// U' = (0.0 * Y) + ( 0.995211 * U) + (-0.059549 * V)
// V' = (0.0 * Y) + (-0.084085 * U) + ( 0.976518 * V)
const std::array<float, 9> YUV_BT2100_TO_BT601 = {1.0f, 0.117887f,  0.105521f,
                                                  0.0f, 0.995211f,  -0.059549f,
                                                  0.0f, -0.084085f, 0.976518f};

Color YUVColorGamutConversion(Color e_gamma,
                              const std::array<float, 9> &coeffs) {
    const float y = e_gamma.y * std::get<0>(coeffs) +
                    e_gamma.u * std::get<1>(coeffs) +
                    e_gamma.v * std::get<2>(coeffs);
    const float u = e_gamma.y * std::get<3>(coeffs) +
                    e_gamma.u * std::get<4>(coeffs) +
                    e_gamma.v * std::get<5>(coeffs);
    const float v = e_gamma.y * std::get<6>(coeffs) +
                    e_gamma.u * std::get<7>(coeffs) +
                    e_gamma.v * std::get<8>(coeffs);
    return {{{y, u, v}}};
}
Color YUV_Bt709ToBt601(Color e) {
    return YUVColorGamutConversion(e, YUV_BT709_TO_BT601);
}
Color YUV_Bt709ToBt2100(Color e) {
    return YUVColorGamutConversion(e, YUV_BT709_TO_BT2100);
}
Color YUV_Bt601ToBt709(Color e) {
    return YUVColorGamutConversion(e, YUV_BT601_TO_BT709);
}
Color YUV_Bt601ToBt2100(Color e) {
    return YUVColorGamutConversion(e, YUV_BT601_TO_BT2100);
}
Color YUV_Bt2100ToBt709(Color e) {
    return YUVColorGamutConversion(e, YUV_BT2100_TO_BT709);
}
Color YUV_Bt2100ToBt601(Color e) {
    return YUVColorGamutConversion(e, YUV_BT2100_TO_BT601);
}

////////////////////////////////////////////////////////////////////////////////
// function selectors

// TODO: confirm we always want to convert like this before calculating
// luminance.
ColorTransformFn GetGamutConversionFn(Gamut dst_gamut, Gamut src_gamut) {
    switch (dst_gamut) {
    case Gamut::BT709:
        switch (src_gamut) {
        case Gamut::BT709:
            return IdentityConversion;
        case Gamut::P3:
            return P3ToBt709;
        case Gamut::BT2100:
            return Bt2100ToBt709;
        }
        break;
    case Gamut::P3:
        switch (src_gamut) {
        case Gamut::BT709:
            return Bt709ToP3;
        case Gamut::P3:
            return IdentityConversion;
        case Gamut::BT2100:
            return Bt2100ToP3;
        }
        break;
    case Gamut::BT2100:
        switch (src_gamut) {
        case Gamut::BT709:
            return Bt709ToBt2100;
        case Gamut::P3:
            return P3ToBt2100;
        case Gamut::BT2100:
            return IdentityConversion;
        }
        break;
    }
    return nullptr;
}

ColorTransformFn GetRGBToYUVFn(Gamut gamut) {
    switch (gamut) {
    case Gamut::BT709:
        return sRGB_RGBToYUV;
    case Gamut::P3:
        return sRGB_RGBToYUV;
    case Gamut::BT2100:
        return Bt2100_RGBToYUV;
    }
    return nullptr;
}

ColorTransformFn GetYUVToRGBFn(Gamut gamut) {
    switch (gamut) {
    case Gamut::BT709:
        return sRGB_YUVToRGB;
    case Gamut::P3:
        return P3_YUVToRGB;
    case Gamut::BT2100:
        return Bt2100_YUVToRGB;
    }
    return nullptr;
}

LuminanceFn GetLuminanceFn(Gamut gamut) {
    switch (gamut) {
    case Gamut::BT709:
        return sRGBLuminance;
    case Gamut::P3:
        return P3Luminance;
    case Gamut::BT2100:
        return Bt2100Luminance;
    }
    return nullptr;
}

ColorTransformFn GetOETFFn(OETF transfer) {
    switch (transfer) {
    case OETF::LINEAR:
        return IdentityConversion;
    case OETF::HLG:
#if USE_HLG_OETF_LUT
        return HLG_OETFLUT;
#else
        return HLG_OETF;
#endif
    case OETF::PQ:
#if USE_PQ_OETF_LUT
        return PQ_OETFLUT;
#else
        return PQ_OETF;
#endif
    case OETF::SRGB:
#if USE_SRGB_OETF_LUT
        return sRGB_OETFLUT;
#else
        return sRGB_OETF;
#endif
    }
    return nullptr;
}

ColorTransformFn GetInvOETFFn(OETF transfer) {
    switch (transfer) {
    case OETF::LINEAR:
        return IdentityConversion;
    case OETF::HLG:
#if USE_HLG_INVOETF_LUT
        return HLG_InvOETFLUT;
#else
        return HLG_InvOETF;
#endif
    case OETF::PQ:
#if USE_PQ_INVOETF_LUT
        return PQ_InvOETFLUT;
#else
        return PQ_InvOETF;
#endif
    case OETF::SRGB:
#if USE_SRGB_INVOETF_LUT
        return sRGB_InvOETFLUT;
#else
        return sRGB_InvOETF;
#endif
    }
    return nullptr;
}

SceneToDisplayLuminanceFn GetOOTFFn(OETF transfer) {
    switch (transfer) {
    case OETF::LINEAR:
        return IdentityOOTF;
    case OETF::HLG:
        return HLG_OOTFApprox;
    case OETF::PQ:
        return IdentityOOTF;
    case OETF::SRGB:
        return IdentityOOTF;
    }
    return nullptr;
}

float GetReferenceDisplayPeakLuminanceInNits(OETF transfer) {
    switch (transfer) {
    case OETF::LINEAR:
        return PQ_MAX_NITS;
    case OETF::HLG:
        return HLG_MAX_NITS;
    case OETF::PQ:
        return PQ_MAX_NITS;
    case OETF::SRGB:
        return SDR_WHITE_NITS;
    }
    return -1.0f;
}

float ApplyToneMapping(float x, ToneMapping mode, float target_nits = 100.0,
                       float max_nits = 100.0) {
    if (mode == ToneMapping::BASE) {
        x *= target_nits / max_nits;
    } else if (mode == ToneMapping::REINHARD) {
        x = x / (1.0 + x);
    } else if (mode == ToneMapping::GAMMA) {
        // REVISIT: 2.2
        x = std::pow(x, 1.0 / 2.2);
    } else if (mode == ToneMapping::FILMIC) {
        const float A = 2.51f;
        const float B = 0.03f;
        const float C = 2.43f;
        const float D = 0.59f;
        const float E = 0.14f;
        x = (x * (A * x + B)) / (x * (C * x + D) + E);
    } else if (mode == ToneMapping::ACES) {
        const float a = 2.51f;
        const float b = 0.03f;
        const float c = 2.43f;
        const float d = 0.59f;
        const float e = 0.14f;
        // REVISIT: Exposure adjustment for ACES
        float adjusted = x * 0.6;
        x = (adjusted * (adjusted + b) * a) /
            (adjusted * (adjusted * c + d) + e);
    } else if (mode == ToneMapping::UNCHARTED2) {
        const float A = 0.15f;
        const float B = 0.50f;
        const float C = 0.10f;
        const float D = 0.20f;
        const float E = 0.02f;
        const float F = 0.30f;
        const float W = 11.2f;

        auto uncharted2_tonemap = [=](float x) -> float {
            return ((x * (A * x + C * B) + D * E) / (x * (A * x + B) + D * F)) -
                   E / F;
        };

        x = uncharted2_tonemap(x) / uncharted2_tonemap(W);
    } else if (mode == ToneMapping::DRAGO) {
        const float bias = 0.85f;
        const float Lwa = 1.0f;

        x = std::log(1 + x) / std::log(1 + Lwa);
        x = std::pow(x, bias);
    } else if (mode == ToneMapping::LOTTES) {
        const float a = 1.6;

        const float mid_in = 0.18f;
        const float mid_out = 0.267f;

        const float t = x * a;
        x = t / (t + 1);

        const float z = (mid_in * a) / (mid_in * a + 1);
        x = x * (mid_out / z);
    } else if (mode == ToneMapping::HABLE) {
        const float A = 0.22f; // Shoulder strength
        const float B = 0.30f; // Linear strength
        const float C = 0.10f; // Linear angle
        const float D = 0.20f; // Toe strength
        const float E = 0.01f; // Toe numerator
        const float F = 0.30f; // Toe denominator
        const float W = 11.2f;

        std::function<float(float)> hable = [=](float x) -> float {
            return (x * (A * x + C * B) + D * E) / (x * (A * x + B) + D * F) -
                   E / F;
        };

        x = hable(x) / hable(W);
    } else if (mode == ToneMapping::REINHARD) {
        x = x / (1.0 + x);
    }
    return x;
}

Color ApplyToneMapping(Color rgb, ToneMapping mode, float target_nits = 100.0,
                       float max_nits = 100.0) {
    return {{{ApplyToneMapping(rgb.r, mode, target_nits, max_nits),
              ApplyToneMapping(rgb.g, mode, target_nits, max_nits),
              ApplyToneMapping(rgb.b, mode, target_nits, max_nits)}}};
}

} // namespace colorspace
