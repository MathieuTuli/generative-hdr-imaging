#ifndef API_HPP
#define API_HPP
#include "utils.h"
#include <cstdint>
#include <memory>
#include <png.h>
#include <string>
#include <vector>

namespace imageops {

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
    float exposure{1.0f};
    float gamma{2.2f};
    std::string colorSpace{"Linear"};
};

enum class HDRFormat {
    UNKNOWN = -1,
    PNG_HDR = 0,
    AVIF = 1,
};

// HDR processing functions
struct HDRProcessingParams {
    float exposure{0.0f};
    float saturation{1.0f};
    float contrast{1.0f};
    float gamma{2.2f};
};

struct SDRConversionParams {
    float max_nits{100.0f};
    float target_gamma{2.2f};
    bool preserve_highlights{true};
    float knee_point{0.75f};
};

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
                const std::string &filename,
                utils::Error &error);
bool WritetoPNG(const std::unique_ptr<PNGImage> &image,
                const std::string &filename, utils::Error &error);

std::unique_ptr<RawImageData>
HDRPNGtoRAW(const std::unique_ptr<PNGImage> &image,
            const HDRProcessingParams &params, utils::Error &error);

std::unique_ptr<RawImageData>
RAWtoSDR(const std::unique_ptr<RawImageData> &raw_image,
         const SDRConversionParams &params, utils::Error &error);

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
