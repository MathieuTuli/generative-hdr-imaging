#ifndef COLORSPACE_HPP
#define COLORSPACE_HPP

#include <functional>
#include <iostream>
#include <memory>
#include <utils.h>

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

enum class ColorSpace {
    BT709,
    P3,
    BT2100,
};

enum class OETF {
    SRGB,
    HLG,
    PQ,
};

struct Color {
    union {
        struct {
            double r;
            double g;
            double b;
        };
        struct {
            double y;
            double u;
            double v;
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
    LookUpTable(size_t numEntries, std::function<double(double)> compute_func) {
        for (size_t idx = 0; idx < numEntries; idx++) {
            double value =
                static_cast<double>(idx) / static_cast<double>(numEntries - 1);
            table.push_back(compute_func(value));
        }
    }
    const std::vector<double> &getTable() const { return table; }

  private:
    std::vector<double> table;
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
static const double SDR_WHITE_NITS = 203.0;
// hlg peak white. 75% of hlg peak white maps to reference diffuse white
static const double HLG_MAX_NITS = 1000.0;
// pq peak white. 58% of pq peak white maps to reference diffuse white
static const double PQ_MAX_NITS = 10000.0;

// srgb
double sRGBLuminance(Color e);
Color sRGB_RGBToYUV(Color e_gamma);
Color sRGB_YUVToRGB(Color e_gamma);
double sRGB_InvOETF(double e_gamma);
Color sRGB_InvOETF(Color e_gamma);
double sRGB_InvOETFLUT(double e_gamma);
Color sRGB_InvOETFLUT(Color e_gamma);
double sRGB_OETF(double e);
Color sRGB_OETF(Color e);
// p3
double P3Luminance(Color e);
Color P3_RGBToYUV(Color e_gamma);
Color P3_YUVToRGB(Color e_gamma);
// bt2100
double Bt2100Luminance(Color e);
Color Bt2100_RGBToYUV(Color e_gamma);
Color Bt2100_YUVToRGB(Color e_gamma);
// hlg
double HLG_OETF(double e);
Color HLG_OETF(Color e);
double HLG_OETFLUT(double e);
Color HLG_OETFLUT(Color e);
double HLG_InvOETF(double e_gamma);
Color HLG_InvOETF(Color e_gamma);
double HLG_InvOETFLUT(double e_gamma);
Color HLG_InvOETFLUT(Color e_gamma);
Color HLG_OOTF(Color e, LuminanceFn luminance);
Color HLG_OOTFApprox(Color e, [[maybe_unused]] LuminanceFn luminance);
Color HLG_InvOOTF(Color e, LuminanceFn luminance);
Color HLG_InvOOTFApprox(Color e);
// pq
double PQ_OETF(double e);
Color PQ_OETF(Color e);
double PQ_OETFLUT(double e);
Color PQ_OETFLUT(Color e);
double PQ_InvOETF(double e_gamma);
Color PQ_InvOETF(Color e_gamma);
double PQ_InvOETFLUT(double e_gamma);
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

double ApplyToneMapping(double x, ToneMapping mode, double target_nits,
                        double max_nits);
Color ApplyToneMapping(Color rgb, ToneMapping mode, double target_nits,
                       double max_nits);
std::unique_ptr<Image> HDRToSDR(const std::unique_ptr<Image> &hdr_image,
                                double clip_low, double clip_high,
                                utils::Error &error, ToneMapping mode);

} // namespace colorspace
#endif
