#ifndef IMAGEIO_HPP
#define IMAGEIO_HPP

#include <string>
#include "utils.h"
#include <vector>

namespace imageio {

struct HDRImage {
    size_t width{0};
    size_t height{0};
    size_t channels{0}; // Usually 3 or 4 for HDR
    std::vector<float> data;
};

struct ImageMetadata {
    float exposure{1.0f};
    float gamma{2.2f};
    std::string colorSpace{"Linear"};
};

enum class HDRFormat {
    UNKNOWN = -1,
    AVIF = 0,
    EXR = 1,
    PNG_HDR = 2,
};

bool HasExtension(const std::string &filename, const std::string &ext);
bool ValidateHeader(const std::string &filename, utils::Error &error);
HDRFormat DetectFormat(const std::string &filename);

HDRImage LoadImage(const std::string &filename, utils::Error &error);
HDRImage LoadAVIF(const std::string &filename, utils::Error &error);
HDRImage LoadEXR(const std::string &filename, utils::Error &error);
HDRImage LoadPNGHDR(const std::string &filename, utils::Error &error);

ImageMetadata ReadMetadata(const std::string &filename, HDRFormat format,
                           utils::Error &error);
ImageMetadata ReadAVIFMetadata(const std::string &filename,
                               utils::Error &error);
ImageMetadata ReadEXRMetadata(const std::string &filename, utils::Error &error);
ImageMetadata ReadPNGHDRMetadata(const std::string &filename,
                                 utils::Error &error);

bool SaveImage(const HDRImage &image, const std::string &filename,
               HDRFormat format, const ImageMetadata &metadata,
               utils::Error &error);

bool SaveAVIF(const HDRImage &image, const std::string &filename,
              const ImageMetadata &metadata, utils::Error &error);

bool SaveEXR(const HDRImage &image, const std::string &filename,
             const ImageMetadata &metadata, utils::Error &error);

bool SavePNGHDR(const HDRImage &image, const std::string &filename,
                const ImageMetadata &metadata, utils::Error &error);

inline size_t GetPixelCount(const HDRImage &img) {
    return img.width * img.height;
}

inline size_t GetDataSize(const HDRImage &img) {
    return GetPixelCount(img) * img.channels;
}

inline float *GetPixel(HDRImage &img, size_t x, size_t y) {
    return &img.data[y * img.width * img.channels + x * img.channels];
}

inline const float *GetPixel(const HDRImage &img, size_t x, size_t y) {
    return &img.data[y * img.width * img.channels + x * img.channels];
}
} // namespace imageio

#endif
