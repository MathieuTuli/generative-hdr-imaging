#include "imageops.hpp"
#include "utils.h"
#include <algorithm>
#include <cmath>
#include <filesystem>
#include <memory>
#include <png.h>
#include <vector>

namespace imageops {

bool HasExtension(const std::string &filename, const std::string &ext) {
    std::filesystem::path path(filename);
    std::string file_ext = path.extension().string();
    std::transform(file_ext.begin(), file_ext.end(), file_ext.begin(),
                   ::tolower);
    std::string compare_ext = ext[0] == '.' ? ext : "." + ext;
    std::transform(compare_ext.begin(), compare_ext.end(), compare_ext.begin(),
                   ::tolower);
    return file_ext == compare_ext;
}

bool ValidateHeader(const std::string &filename, utils::Error &error) {
    if (!std::filesystem::exists(filename)) {
        error = {true, "File does not exist: " + filename};
        return false;
    }
    // REVISIT: Add specific format validation here
    return true;
}

HDRFormat DetectFormat(const std::string &filename) {
    if (HasExtension(filename, ".avif"))
        return HDRFormat::AVIF;
    if (HasExtension(filename, ".hdr"))
        return imageops::HDRFormat::PNG_HDR;
    return HDRFormat::UNKNOWN;
}

// RawImageData LoadImage(const std::string &filename, utils::Error &error) {
//     HDRFormat format = DetectFormat(filename);
//     switch (format) {
//     case HDRFormat::AVIF:
//         return LoadAVIF(filename, error);
//     case HDRFormat::PNG_HDR:
//         return LoadHDRPNG(filename, error);
//     default:
//         error = {true, "Unsupported format"};
//         return HDRImage{};
//     }
// }

void LoadAVIF(const std::string &filename, utils::Error &error) {
    // TODO: Implement AVIF loading using libavif
    error = {true, "AVIF loading not implemented"};
}

std::unique_ptr<PNGImage> LoadHDRPNG(const std::string &filename,
                                     utils::Error &error) {
    FILE *fp = fopen(filename.c_str(), "rb");
    if (!fp) {
        error = {true, "Failed to open file: " + filename};
        return nullptr;
    }

    png_structp png = png_create_read_struct(PNG_LIBPNG_VER_STRING, nullptr,
                                             nullptr, nullptr);
    if (!png) {
        fclose(fp);
        error = {true, "Failed to create PNG read struct"};
        return nullptr;
    }

    png_infop info = png_create_info_struct(png);
    if (!info) {
        png_destroy_read_struct(&png, nullptr, nullptr);
        fclose(fp);
        error = {true, "Failed to create PNG info struct"};
        return nullptr;
    }

    if (setjmp(png_jmpbuf(png))) {
        png_destroy_read_struct(&png, &info, nullptr);
        fclose(fp);
        error = {true, "Error during PNG read"};
        return nullptr;
    }

    png_init_io(png, fp);
    png_read_info(png, info);

    std::unique_ptr<PNGImage> image = std::make_unique<PNGImage>();

    image->width = png_get_image_width(png, info);
    image->height = png_get_image_height(png, info);
    image->color_type = png_get_color_type(png, info);
    image->bit_depth = png_get_bit_depth(png, info);
    image->bytes_per_row = png_get_rowbytes(png, info);

    png_read_update_info(png, info);

    image->row_pointers =
        (png_bytep *)malloc(sizeof(png_bytep) * image->height);

    for (size_t i = 0; i < image->height; i++) {
        image->row_pointers[i] = (png_byte *)malloc(image->bytes_per_row);
    }
    png_read_image(png, image->row_pointers);

    png_destroy_read_struct(&png, &info, nullptr);
    fclose(fp);

    return image;
}

ImageMetadata ReadMetadata(const std::string &filename, HDRFormat format,
                           utils::Error &error) {
    switch (format) {
    case HDRFormat::AVIF:
        return ReadAVIFMetadata(filename, error);
    case HDRFormat::PNG_HDR:
        return ReadHDRPNGMetadata(filename, error);
    default:
        error = {true, "Unsupported format for metadata"};
        return ImageMetadata{};
    }
}

ImageMetadata ReadAVIFMetadata(const std::string &filename,
                               utils::Error &error) {
    // TODO: Implement AVIF metadata reading
    error = {true, "AVIF metadata reading not implemented"};
    return ImageMetadata{};
}

ImageMetadata ReadHDRPNGMetadata(const std::string &filename,
                                 utils::Error &error) {
    // TODO: Implement HDR PNG metadata reading
    error = {true, "HDR PNG metadata reading not implemented"};
    return ImageMetadata{};
}

bool WritetoRAW(const std::unique_ptr<RawImageData> &image,
                const std::string &filename, utils::Error &error) {
    if (!image) {
        error = {true, "Invalid image pointer"};
        return false;
    }

    FILE *fp = fopen(filename.c_str(), "wb");
    if (!fp) {
        error = {true, "Could not open file for writing: " + filename};
        return false;
    }
    return false;
}

bool WritetoPNG(const std::unique_ptr<PNGImage> &image,
                const std::string &filename, utils::Error &error) {
    if (!image) {
        error = {true, "Invalid image pointer"};
        return false;
    }

    FILE *fp = fopen(filename.c_str(), "wb");
    if (!fp) {
        error = {true, "Could not open file for writing: " + filename};
        return false;
    }

    png_structp png =
        png_create_write_struct(PNG_LIBPNG_VER_STRING, NULL, NULL, NULL);
    if (!png) {
        fclose(fp);
        error = {true, "Failed to create PNG write struct"};
        return false;
    }

    png_infop info = png_create_info_struct(png);
    if (!info) {
        png_destroy_read_struct(&png, nullptr, nullptr);
        fclose(fp);
        error = {true, "Failed to create PNG info struct"};
        return false;
    }

    if (setjmp(png_jmpbuf(png))) {
        png_destroy_read_struct(&png, &info, nullptr);
        fclose(fp);
        error = {true, "Error during PNG read"};
        return false;
    }

    png_init_io(png, fp);

    // Output is 8bit depth, RGBA format.
    png_set_IHDR(png, info, image->width, image->height, 8, PNG_COLOR_TYPE_RGBA,
                 PNG_INTERLACE_NONE, PNG_COMPRESSION_TYPE_DEFAULT,
                 PNG_FILTER_TYPE_DEFAULT);
    png_write_info(png, info);

    // To remove the alpha channel for PNG_COLOR_TYPE_RGB format,
    // Use png_set_filler().
    // png_set_filler(png, 0, PNG_FILLER_AFTER);

    if (!image->row_pointers) {
        png_destroy_read_struct(&png, &info, nullptr);
        fclose(fp);
        error = {true, "Error during PNG write, empty row pointer"};
        return false;
    }

    png_write_image(png, image->row_pointers);
    png_write_end(png, NULL);

    // for (int y = 0; y < image->height; y++) {
    //     free(image->row_pointers[y]);
    // }
    // free(image->row_pointers);

    fclose(fp);
    png_destroy_write_struct(&png, &info);
    return true;
}

std::unique_ptr<RawImageData>
HDRPNGtoRAW(const std::unique_ptr<PNGImage> &image,
            const HDRProcessingParams &params, utils::Error &error) {
    if (!image) {
        error = {true, "Invalid input image"};
        return nullptr;
    }

    std::unique_ptr<RawImageData> raw_image = std::make_unique<RawImageData>();
    raw_image->width = image->width;
    raw_image->height = image->height;
    raw_image->channels = 3; // RGB
    raw_image->bits_per_channel = image->bit_depth;
    raw_image->is_float = (image->bit_depth > 8); // Use float for >8 bit
    raw_image->is_big_endian = false;

    // Calculate size and allocate buffer
    size_t pixel_count = image->width * image->height;
    size_t bytes_per_pixel =
        (raw_image->bits_per_channel * raw_image->channels) / 8;
    raw_image->data.resize(pixel_count * bytes_per_pixel);

    // Copy and convert data
    size_t dst_pos = 0;

    for (size_t y = 0; y < image->height; y++) {
        const png_bytep src_row = image->row_pointers[y];
        for (size_t x = 0; x < image->bytes_per_row; x += bytes_per_pixel) {
            // Copy RGB components
            for (size_t c = 0; c < raw_image->channels; c++) {
                raw_image->data[dst_pos++] = src_row[x + c];
            }
        }
    }

    return raw_image;
}
} // namespace imageops
