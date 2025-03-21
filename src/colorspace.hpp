#ifndef COLORSPACE_HPP
#define COLORSPACE_HPP

#include <iostream>
#include <utils.h>

struct PNGImage;

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
    SRGB,
    DISPLAYP3,
    P3PQ,
    REC2020HLG,
    REC2020GAMMA,
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

inline Color operator+=(Color& lhs, const Color& rhs) {
  lhs.r += rhs.r;
  lhs.g += rhs.g;
  lhs.b += rhs.b;
  return lhs;
}

inline Color operator-=(Color& lhs, const Color& rhs) {
  lhs.r -= rhs.r;
  lhs.g -= rhs.g;
  lhs.b -= rhs.b;
  return lhs;
}

inline Color operator+(const Color& lhs, const Color& rhs) {
  Color temp = lhs;
  return temp += rhs;
}

inline Color operator-(const Color& lhs, const Color& rhs) {
  Color temp = lhs;
  return temp -= rhs;
}

inline Color operator+=(Color& lhs, const float rhs) {
  lhs.r += rhs;
  lhs.g += rhs;
  lhs.b += rhs;
  return lhs;
}

inline Color operator-=(Color& lhs, const float rhs) {
  lhs.r -= rhs;
  lhs.g -= rhs;
  lhs.b -= rhs;
  return lhs;
}

inline Color operator*=(Color& lhs, const float rhs) {
  lhs.r *= rhs;
  lhs.g *= rhs;
  lhs.b *= rhs;
  return lhs;
}

inline Color operator/=(Color& lhs, const float rhs) {
  lhs.r /= rhs;
  lhs.g /= rhs;
  lhs.b /= rhs;
  return lhs;
}

inline Color operator+(const Color& lhs, const float rhs) {
  Color temp = lhs;
  return temp += rhs;
}

inline Color operator-(const Color& lhs, const float rhs) {
  Color temp = lhs;
  return temp -= rhs;
}

inline Color operator*(const Color& lhs, const float rhs) {
  Color temp = lhs;
  return temp *= rhs;
}

inline Color operator/(const Color& lhs, const float rhs) {
  Color temp = lhs;
  return temp /= rhs;
}

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

// ----------------------------------------
// CONVERSION
// ----------------------------------------
// 1. HLG ->
// 2. LINEAR ->
// 3. CLIP/COMPRESSION/HDR-SDR TONEMAP ->
// 4. COLOR SPACE ->
// 5. sRGB TONEMAP/TRANSFER ->
// 6. QUANT

double Bt2100HLGToLinear(double x);
double Bt2100PQToLinear(double x);
double Rec2020ToLinear(double x);
double sRGBToLinear(double x);

double LinearToBt2100HLG(double x);
double LinearToBt2100PQ(double x);
double LinearToRec2020(double x);
double LinearTosRGB(double x);

std::vector<double> LinearsRGBToYUV(const std::vector<double> &rgb);
std::vector<double> LinearBt2100ToYUV(const std::vector<double> &rgb);

double ApplyToneMapping(double x, ToneMapping mode, double target_nits,
                        double max_nits);

std::unique_ptr<PNGImage> HDRToYUV(const std::unique_ptr<PNGImage> &hdr_image,
                                   double clip_low, double clip_high,
                                   utils::Error &error, ToneMapping mode);
std::unique_ptr<PNGImage> HDRToSDR(const std::unique_ptr<PNGImage> &hdr_image,
                                   double clip_low, double clip_high,
                                   utils::Error &error, ToneMapping mode);

} // namespace colorspace
#endif
