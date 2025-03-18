#ifndef IMAGE_PROCESSOR_HPP
#define IMAGE_PROCESSOR_HPP

/*
// #include <array>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <vector>

#define BOUND(x, min, max) ((x) < (min)) ? (min) : ((x) > (max)) ? (max) : (x)

// Raw image data structure to hold unprocessed bytes
struct RawImageData {
    size_t width{0};
    size_t height{0};
    size_t channels{0};
    size_t bits_per_channel{0};
    bool is_float{false}; // true for float formats like EXR
    bool is_big_endian{false};
    std::vector<uint8_t> data; // Raw bytes
};


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

// XX: color conversions

extern const std::array<float, 9> kBt709ToP3;
extern const std::array<float, 9> kBt709ToBt2100;
extern const std::array<float, 9> kP3ToBt709;
extern const std::array<float, 9> kP3ToBt2100;
extern const std::array<float, 9> kBt2100ToBt709;
extern const std::array<float, 9> kBt2100ToP3;

inline Color identity_conversion(Color e) { return e; }
Color bt709_to_p3(Color e);
Color bt709_to_bt2100(Color e);
Color p3_to_bt709(Color e);
Color p3_to_bt2100(Color e);
Color bt2100_to_p3(Color e);
Color bt2100_to_bt709(Color e);

// XX: gain map
// Calculate the 8-bit unsigned integer gain value for the given SDR and HDR
// luminances in linear space and gainmap metadata fields.
uint8_t encodeGain(float y_sdr, float y_hdr,
                   uhdr_gainmap_metadata_ext_t *metadata, int index);
uint8_t encodeGain(float y_sdr, float y_hdr,
                   uhdr_gainmap_metadata_ext_t *metadata,
                   float log2MinContentBoost, float log2MaxContentBoost,
                   int index);
float computeGain(float sdr, float hdr);
uint8_t affineMapGain(float gainlog2, float mingainlog2, float maxgainlog2,
                      float gamma);

// Calculates the linear luminance in nits after applying the given gain
// value, with the given hdr ratio, to the given sdr input in the range [0, 1].
Color applyGain(Color e, float gain, uhdr_gainmap_metadata_ext_t *metadata);
Color applyGain(Color e, float gain, uhdr_gainmap_metadata_ext_t *metadata,
                float gainmapWeight);
Color applyGainLUT(Color e, float gain, GainLUT &gainLUT,
                   uhdr_gainmap_metadata_ext_t *metadata);

// Apply gain in R, G and B channels, with the given hdr ratio, to the given sdr
// input in the range [0, 1].
Color applyGain(Color e, Color gain, uhdr_gainmap_metadata_ext_t *metadata);
Color applyGain(Color e, Color gain, uhdr_gainmap_metadata_ext_t *metadata,
                float gainmapWeight);
Color applyGainLUT(Color e, Color gain, GainLUT &gainLUT,
                   uhdr_gainmap_metadata_ext_t *metadata);

// XX: common utils
static inline float clipNegatives(float value) {
    return (value < 0.0f) ? 0.0f : value;
}

static inline Color clipNegatives(Color e) {
    return {{{clipNegatives(e.r), clipNegatives(e.g), clipNegatives(e.b)}}};
}

// maximum limit of normalized pixel value in float representation
static const float kMaxPixelFloat = 1.0f;

static inline float clampPixelFloat(float value) {
    return (value < 0.0f)             ? 0.0f
           : (value > kMaxPixelFloat) ? kMaxPixelFloat
                                      : value;
}

static inline Color clampPixelFloat(Color e) {
    return {
        {{clampPixelFloat(e.r), clampPixelFloat(e.g), clampPixelFloat(e.b)}}};
}

// maximum limit of pixel value for linear hdr intent raw resource
static const float kMaxPixelFloatHdrLinear = 10000.0f / 203.0f;

static inline float clampPixelFloatLinear(float value) {
  return BOUND(value, 0.0f, kMaxPixelFloatHdrLinear);
}

static inline Color clampPixelFloatLinear(Color e) {
  return {{{clampPixelFloatLinear(e.r), clampPixelFloatLinear(e.g),
clampPixelFloatLinear(e.b)}}};
}

static float mapNonFiniteFloats(float val) {
  if (std::isinf(val)) {
    return val > 0 ? kMaxPixelFloatHdrLinear : 0.0f;
  }
  // nan
  return 0.0f;
}

static inline Color sanitizePixel(Color e) {
  float r = std::isfinite(e.r) ? clampPixelFloatLinear(e.r) :
mapNonFiniteFloats(e.r); float g = std::isfinite(e.g) ?
clampPixelFloatLinear(e.g) : mapNonFiniteFloats(e.g); float b =
std::isfinite(e.b) ? clampPixelFloatLinear(e.b) : mapNonFiniteFloats(e.b);
  return {{{r, g, b}}};
}
*/

#endif
