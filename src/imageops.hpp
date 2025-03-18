#ifndef API_HPP
#define API_HPP
#include "utils.h"
#include <cstdint>
#include <memory>
#include <png.h>
#include <string>
#include <vector>

namespace imageops {

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

struct RawImageData {
    size_t width{0};
    size_t height{0};
    size_t channels{0};
    size_t bits_per_channel{0};
    bool is_float{false}; // true for float formats like EXR
    bool is_big_endian{false};
    std::vector<uint8_t> data; // Raw bytes
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
    float luminance{1.0f};           // From sRGB or iCCP chunk if present
    std::string color_space{"sRGB"}; // From sRGB or iCCP chunk
    std::string transfer_function{"HLG"}; // From sRGB or iCCP chunk
    bool has_transparency{false};    // From tRNS chunk
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

SDRConversionParams UpdateSDRParams(const SDRConversionParams &base_params,
                                    const ImageMetadata &metadata);

bool HasExtension(const std::string &filename, const std::string &ext);
bool ValidateHeader(const std::string &filename, utils::Error &error);
HDRFormat DetectFormat(const std::string &filename);

void LoadAVIF(const std::string &filename, utils::Error &error);
std::unique_ptr<PNGImage> LoadHDRPNG(const std::string &filename,
                                     utils::Error &error);
ImageMetadata ReadAVIFMetadata(const std::string &filename,
                               utils::Error &error);
ImageMetadata ReadHDRPNGMetadata(const std::string &filename,
                                 utils::Error &error);

// const ImageMetadata &metadata,
bool WritetoRAW(const std::unique_ptr<RawImageData> &image,
                const std::string &filename, utils::Error &error);
bool WritetoPNG(const std::unique_ptr<PNGImage> &image,
                const std::string &filename, utils::Error &error);

std::unique_ptr<RawImageData>
HDRPNGtoRAW(const std::unique_ptr<PNGImage> &image,
            const HDRProcessingParams &params, utils::Error &error);

std::unique_ptr<PNGImage> HDRtoSDR(const std::unique_ptr<PNGImage> &hdr_image,
                                   const SDRConversionParams &params,
                                   utils::Error &error);

inline size_t GetPixelCount(const PNGImage &img) {
    return img.width * img.height;
}

// inline size_t GetDataSize(const PNGImage &img) {
//     return img.row_pointers.size() *
//            (img.row_pointers.empty() ? 0 : img.row_pointers[0].size());
// }
//
// // Get pointer to pixel data for RawImageData
// inline const float *GetPixel(const RawImageData &image, size_t x, size_t y) {
//     if (!image.is_float || x >= image.width || y >= image.height) {
//         return nullptr;
//     }
//     return reinterpret_cast<const float *>(image.data.data() +
//                                            (y * image.width + x) *
//                                                image.channels *
//                                                sizeof(float));
// }
//
// // Get pointer to pixel data for PNGImage
// inline const uint8_t *GetPixel(const PNGImage &image, size_t x, size_t y) {
//     if (x >= image.width || y >= image.height ||
//         y >= image.row_pointers.size()) {
//         return nullptr;
//     }
//     const auto &row = image.row_pointers[y];
//     size_t pixel_size = (image.bit_depth / 8) *
//                         (image.color_type & PNG_COLOR_MASK_COLOR ? 3 : 1);
//     if (image.color_type & PNG_COLOR_MASK_ALPHA) {
//         pixel_size += image.bit_depth / 8;
//     }
//     if (x * pixel_size >= row.size()) {
//         return nullptr;
//     }
//     return row.data() + x * pixel_size;
// }
} // namespace imageops
#endif
