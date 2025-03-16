#ifndef IMAGEIO_HPP
#define IMAGEIO_HPP


#include <string>
#include <vector>
#include <stdexcept>

namespace imageio {

// Simple data structures
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

// Supported HDR formats
enum class HDRFormat { 
    AVIF, 
    EXR, 
    PNG_HDR, 
    UNKNOWN 
};

// Error handling
struct ImageError {
    bool success{false};
    std::string message;
};

// Utility functions
bool has_extension(const std::string& filename, const std::string& ext);
bool validate_header(const std::string& filename, ImageError& error);
HDRFormat detect_format(const std::string& filename);

// Loading functions
HDRImage load_image(const std::string& filename, ImageError& error);
HDRImage load_avif(const std::string& filename, ImageError& error);
HDRImage load_exr(const std::string& filename, ImageError& error);
HDRImage load_png_hdr(const std::string& filename, ImageError& error);

// Metadata reading functions
ImageMetadata read_metadata(const std::string& filename, HDRFormat format, ImageError& error);
ImageMetadata read_avif_metadata(const std::string& filename, ImageError& error);
ImageMetadata read_exr_metadata(const std::string& filename, ImageError& error);
ImageMetadata read_png_hdr_metadata(const std::string& filename, ImageError& error);

// Saving functions
bool save_image(const HDRImage& image, 
               const std::string& filename,
               HDRFormat format,
               const ImageMetadata& metadata,
               ImageError& error);

bool save_avif(const HDRImage& image,
              const std::string& filename,
              const ImageMetadata& metadata,
              ImageError& error);

bool save_exr(const HDRImage& image,
             const std::string& filename,
             const ImageMetadata& metadata,
             ImageError& error);

bool save_png_hdr(const HDRImage& image,
                 const std::string& filename,
                 const ImageMetadata& metadata,
                 ImageError& error);

// Utility functions for working with HDRImage
inline size_t get_pixel_count(const HDRImage& img) {
    return img.width * img.height;
}

inline size_t get_data_size(const HDRImage& img) {
    return get_pixel_count(img) * img.channels;
}

inline float* get_pixel(HDRImage& img, size_t x, size_t y) {
    return &img.data[y * img.width * img.channels + x * img.channels];
}

inline const float* get_pixel(const HDRImage& img, size_t x, size_t y) {
    return &img.data[y * img.width * img.channels + x * img.channels];
}

} // namespace imageio

#endif
