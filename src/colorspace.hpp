#ifndef COLORSPACE_HPP
#define COLORSPACE_HPP

#include "utils.h"
#include <functional>
#include <iostream>
#include <memory>

#define USE_SRGB_INVOETF_LUT 1
#define USE_HLG_OETF_LUT 1
#define USE_PQ_OETF_LUT 1
#define USE_HLG_INVOETF_LUT 1
#define USE_PQ_INVOETF_LUT 1
#define USE_APPLY_GAIN_LUT 1

struct Image;

namespace colorspace {

enum class ToneMapping {
    BASE,
    REINHARD,
    GAMMA,
    FILMIC,
    ACES,
    UNCHARTED2,
    DRAGO,
    LOTTES,
    HABLE
};

enum class Gamut {
    BT709,
    P3,
    BT2100,
};

enum class OETF { SRGB, HLG, PQ, LINEAR };

struct Color {
    union {
        struct {
            float r;
            float g;
            float b;
        };
        struct {
            float y;
            float u;
            float v;
        };
    };
};

inline Color operator+=(Color &lhs, const Color &rhs) {
    lhs.r += rhs.r;
    lhs.g += rhs.g;
    lhs.b += rhs.b;
    return lhs;
}

inline Color operator-=(Color &lhs, const Color &rhs) {
    lhs.r -= rhs.r;
    lhs.g -= rhs.g;
    lhs.b -= rhs.b;
    return lhs;
}

inline Color operator+(const Color &lhs, const Color &rhs) {
    Color temp = lhs;
    return temp += rhs;
}

inline Color operator-(const Color &lhs, const Color &rhs) {
    Color temp = lhs;
    return temp -= rhs;
}

inline Color operator+=(Color &lhs, const float rhs) {
    lhs.r += rhs;
    lhs.g += rhs;
    lhs.b += rhs;
    return lhs;
}

inline Color operator-=(Color &lhs, const float rhs) {
    lhs.r -= rhs;
    lhs.g -= rhs;
    lhs.b -= rhs;
    return lhs;
}

inline Color operator*=(Color &lhs, const float rhs) {
    lhs.r *= rhs;
    lhs.g *= rhs;
    lhs.b *= rhs;
    return lhs;
}

inline Color operator/=(Color &lhs, const float rhs) {
    lhs.r /= rhs;
    lhs.g /= rhs;
    lhs.b /= rhs;
    return lhs;
}

inline Color operator+(const Color &lhs, const float rhs) {
    Color temp = lhs;
    return temp += rhs;
}

inline Color operator-(const Color &lhs, const float rhs) {
    Color temp = lhs;
    return temp -= rhs;
}

inline Color operator*(const Color &lhs, const float rhs) {
    Color temp = lhs;
    return temp *= rhs;
}

inline Color operator/(const Color &lhs, const float rhs) {
    Color temp = lhs;
    return temp /= rhs;
}

class LookUpTable {
  public:
    LookUpTable(size_t numEntries, std::function<float(float)> compute_func) {
        for (size_t idx = 0; idx < numEntries; idx++) {
            float value =
                static_cast<float>(idx) / static_cast<float>(numEntries - 1);
            table.push_back(compute_func(value));
        }
    }
    const std::vector<float> &getTable() const { return table; }

  private:
    std::vector<float> table;
};

typedef Color (*ColorTransformFn)(Color);
typedef float (*LuminanceFn)(Color);
typedef Color (*SceneToDisplayLuminanceFn)(Color, LuminanceFn);

constexpr int32_t SRGB_INV_OETF_PRECISION = 10;
constexpr int32_t SRGB_INV_OETF_NUMENTRIES = 1 << SRGB_INV_OETF_PRECISION;
constexpr int32_t HLG_OETF_PRECISION = 16;
constexpr int32_t HLG_OETF_NUMENTRIES = 1 << HLG_OETF_PRECISION;
constexpr int32_t HLG_INV_OETF_PRECISION = 12;
constexpr int32_t HLG_INV_OETF_NUMENTRIES = 1 << HLG_INV_OETF_PRECISION;
constexpr int32_t PQ_OETF_PRECISION = 16;
constexpr int32_t PQ_OETF_NUMENTRIES = 1 << PQ_OETF_PRECISION;
constexpr int32_t PQ_INV_OETF_PRECISION = 12;
constexpr int32_t PQ_INV_OETF_NUMENTRIES = 1 << PQ_INV_OETF_PRECISION;
constexpr int32_t GAIN_FACTOR_PRECISION = 10;
constexpr int32_t GAIN_FACTOR_NUMENTRIES = 1 << GAIN_FACTOR_PRECISION;

// nominal {SDR, HLG, PQ} peak display luminance
// This aligns with the suggested default reference diffuse white from ISO/TS
// 22028-5 sdr white
static const float SDR_WHITE_NITS = 203.0;
// hlg peak white. 75% of hlg peak white maps to reference diffuse white
static const float HLG_MAX_NITS = 1000.0;
// pq peak white. 58% of pq peak white maps to reference diffuse white
static const float PQ_MAX_NITS = 10000.0;

// srgb
float sRGBLuminance(Color e);
Color sRGB_RGBToYUV(Color e_gamma);
Color sRGB_YUVToRGB(Color e_gamma);
float sRGB_InvOETF(float e_gamma);
Color sRGB_InvOETF(Color e_gamma);
float sRGB_InvOETFLUT(float e_gamma);
Color sRGB_InvOETFLUT(Color e_gamma);
float sRGB_OETF(float e);
Color sRGB_OETF(Color e);
// p3
float P3Luminance(Color e);
Color P3_RGBToYUV(Color e_gamma);
Color P3_YUVToRGB(Color e_gamma);
// bt2100
float Bt2100Luminance(Color e);
Color Bt2100_RGBToYUV(Color e_gamma);
Color Bt2100_YUVToRGB(Color e_gamma);
// hlg
float HLG_OETF(float e);
Color HLG_OETF(Color e);
float HLG_OETFLUT(float e);
Color HLG_OETFLUT(Color e);
float HLG_InvOETF(float e_gamma);
Color HLG_InvOETF(Color e_gamma);
float HLG_InvOETFLUT(float e_gamma);
Color HLG_InvOETFLUT(Color e_gamma);
Color HLG_OOTF(Color e, LuminanceFn luminance);
Color HLG_OOTFApprox(Color e, [[maybe_unused]] LuminanceFn luminance);
Color HLG_InvOOTF(Color e, LuminanceFn luminance);
Color HLG_InvOOTFApprox(Color e, [[maybe_unused]] LuminanceFn luminance);
// pq
float PQ_OETF(float e);
Color PQ_OETF(Color e);
float PQ_OETFLUT(float e);
Color PQ_OETFLUT(Color e);
float PQ_InvOETF(float e_gamma);
Color PQ_InvOETF(Color e_gamma);
float PQ_InvOETFLUT(float e_gamma);
Color PQ_InvOETFLUT(Color e_gamma);

Color ConvertGamut(Color e, const std::array<float, 9> &coeffs);
Color Bt709ToP3(Color e);
Color Bt709ToBt2100(Color e);
Color P3ToBt709(Color e);
Color P3ToBt2100(Color e);
Color Bt2100ToBt709(Color e);
Color Bt2100ToP3(Color e);
Color YUVColorGamutConversion(Color e_gamma,
                              const std::array<float, 9> &coeffs);
Color YUV_Bt709ToBt601(Color e);
Color YUV_Bt709ToBt2100(Color e);
Color YUV_Bt601ToBt709(Color e);
Color YUV_Bt601ToBt2100(Color e);
Color YUV_Bt2100ToBt709(Color e);
Color YUV_Bt2100ToBt601(Color e);

inline Color IdentityConversion(Color e) { return e; }
inline Color IdentityOOTF(Color e, [[maybe_unused]] LuminanceFn) { return e; }
ColorTransformFn GetGamutConversionFn(Gamut dst_gamut, Gamut src_gamut);
ColorTransformFn GetRGBToYUVFn(Gamut gamut);
ColorTransformFn GetYUVToRGBFn(Gamut gamut);
LuminanceFn GetLuminanceFn(Gamut gamut);
ColorTransformFn GetInvOETFFn(OETF transfer);
ColorTransformFn GetOETFFn(OETF transfer);
SceneToDisplayLuminanceFn GetOOTFFn(OETF transfer);
float GetReferenceDisplayPeakLuminanceInNits(OETF transfer);

float ApplyToneMapping(float x, ToneMapping mode, float target_nits,
                       float max_nits);
Color ApplyToneMapping(Color rgb, ToneMapping mode, float target_nits,
                       float max_nits);

static inline float ClipNegatives(float value) {
    return (value < 0.0f) ? 0.0f : value;
}

static inline Color ClipNegatives(Color e) {
    return {{{ClipNegatives(e.r), ClipNegatives(e.g), ClipNegatives(e.b)}}};
}

static inline float Clip(float value, float low, float high) {
    return ((value) < (low)) ? (low) : ((value) > (high)) ? (high) : (value);
}

static inline uint8_t Clip(uint8_t value, uint8_t low, uint8_t high) {
    return ((value) < (low)) ? (low) : ((value) > (high)) ? (high) : (value);
}

static inline Color Clip(Color e, float low, float high) {
    return {
        {{Clip(e.r, low, high), Clip(e.g, low, high), Clip(e.b, low, high)}}};
}

static inline Color Clamp(Color e) { return Clip(e, 0.f, 1.f); }

} // namespace colorspace
#endif
