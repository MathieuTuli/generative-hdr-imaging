#ifndef API_HPP
#define API_HPP
#include "colorspace.hpp"
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
//
struct ImageMetadata {
    colorspace::Gamut gamut;
    colorspace::OETF oetf;
    float clip_low{0.0};
    float clip_high{1.0};
    float offset_hdr{0.015625};
    float offset_sdr{0.015625};
    float min_content_boost{1.0};
    float max_content_boost{4.0};
    float map_gamma{1.0};
};

struct Image {
    size_t width{0};
    size_t height{0};
    uint8_t bit_depth{0};
    size_t bytes_per_row;
    size_t color_type;
    size_t channels;
    png_bytep *row_pointers;
    ImageMetadata metadata;
};

enum class HDRFormat {
    UNKNOWN = -1,
    HDRPNG = 0,
    AVIF = 1,
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
std::unique_ptr<Image> LoadImage(const std::string &filename,
                                 utils::Error &error);
void LoadAVIF(const std::string &filename, utils::Error &error);
std::unique_ptr<Image> LoadHDRPNG(const std::string &filename,
                                  utils::Error &error);
ImageMetadata ReadMetadata(const std::string &filename,
                                 utils::Error &error);

bool WriteToPNG(const std::unique_ptr<Image> &image,
                const std::string &filename, utils::Error &error);
bool WriteToNumpy(const std::vector<float> &data, int width, int height,
                  int channels, const std::string &dtype_str,
                  const std::string &output_path, utils::Error &error);
} // namespace imageops
#endif
