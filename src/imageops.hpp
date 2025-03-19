#ifndef API_HPP
#define API_HPP
#include "utils.h"
#include <cstdint>
#include <memory>
#include <png.h>
#include <string>
#include <vector>

namespace imageops {

// ----------------------------------------
// DATA CONTAINERS
// ----------------------------------------
enum class ToneMapping {
    REINHARD,
    GAMMA,
    FILMIC,
    ACES,
    UNCHARTED2,
    DRAGO,
    LOTTES,
    HABLE
};

struct PNGImage {
    size_t width{0};
    size_t height{0};
    uint8_t color_type{0};
    uint8_t bit_depth{0};
    size_t bytes_per_row;
    png_bytep *row_pointers;
};

struct ImageMetadata {
    float gamma{2.2f}; // Gamma value from gAMA chunk
    float white_point[2]{0.3127f,
                         0.3290f}; // CIE xy chromaticity from cHRM chunk
    float primaries[6]{
        // RGB primaries from cHRM chunk
        0.64f, 0.33f, // Red x,y
        0.30f, 0.60f, // Green x,y
        0.15f, 0.06f  // Blue x,y
    };
    float luminance{1.0f};                // From sRGB or iCCP chunk if present
    std::string color_space{"sRGB"};      // From sRGB or iCCP chunk
    std::string transfer_function{"HLG"}; // From sRGB or iCCP chunk
    bool has_transparency{false};         // From tRNS chunk
    std::string rendering_intent{"perceptual"}; // From sRGB chunk
};

enum class HDRFormat {
    UNKNOWN = -1,
    PNG_HDR = 0,
    AVIF = 1,
};

struct HDRProcessingParams {
    float exposure{0.0f};
    float saturation{1.0f};
    float contrast{1.0f};
    float gamma{2.2f};
    bool force_conversion{false};
};

struct SDRConversionParams {
    double max_nits{1000.0f};
    double target_nits{1000.0f};
    double gamma{2.2f};
    double exposure{1.0f};
    bool auto_clip{true};
    double clip_low{0.0f};
    double clip_high;
    bool preserve_highlights{true};
    double knee_point{0.75f};
    ToneMapping tone_mapping{ToneMapping::GAMMA};
};


// ----------------------------------------
// BASIC IO
// ----------------------------------------
bool HasExtension(const std::string &filename, const std::string &ext);
bool ValidateHeader(const std::string &filename, utils::Error &error);
HDRFormat DetectFormat(const std::string &filename);

// ----------------------------------------
// INVOLVED IO
// ----------------------------------------
void LoadAVIF(const std::string &filename, utils::Error &error);
std::unique_ptr<PNGImage> LoadHDRPNG(const std::string &filename,
                                     utils::Error &error);
ImageMetadata ReadAVIFMetadata(const std::string &filename,
                               utils::Error &error);
ImageMetadata ReadHDRPNGMetadata(const std::string &filename,
                                 utils::Error &error);

bool WritetoPNG(const std::unique_ptr<PNGImage> &image,
                const std::string &filename, utils::Error &error);

// ----------------------------------------
// CONVERSION
// ----------------------------------------
// HLG -> LINEAR -> CLIP/COMPRESSION/HDR-SDR TONEMAP -> COLOR SPACE -> SRGB TONEMAP -> QUANT
double LineartoHLG(double x);
double HLGtoLinear(double x);
#define CLIP(x, min, max) ((x) < (min)) ? (min) : ((x) > (max)) ? (max) : (x)
double DynamicRangeCompression(double x);

void Rec2020toSRGB(double &r, double &g, double &b);

double SRGBTransfer(double x);

std::unique_ptr<PNGImage> HDRtoSDR(const std::unique_ptr<PNGImage> &hdr_image,
                                   const SDRConversionParams &params,
                                   utils::Error &error);


} // namespace imageops
#endif
