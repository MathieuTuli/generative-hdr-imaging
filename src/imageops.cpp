#include "imageops.hpp"
#include "npy.hpp"
#include "utils.h"
#include <ExifTool.h>
#include <algorithm>
#include <cassert>
#include <cmath>
#include <filesystem>
#include <iostream>
#include <memory>
#include <png.h>

namespace imageops {

// ----------------------------------------
// BASIC IO
// ----------------------------------------
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
    if (HasExtension(filename, ".png"))
        return imageops::HDRFormat::HDRPNG;
    return HDRFormat::UNKNOWN;
}

// ----------------------------------------
// INVOLVED IO
// ----------------------------------------
std::unique_ptr<Image> LoadImage(const std::string &filename,
                                 utils::Error &error) {
    // First validate the file exists and is readable
    if (!ValidateHeader(filename, error)) {
        return nullptr;
    }

    // Detect format based on file extension
    HDRFormat format = DetectFormat(filename);

    switch (format) {
    case HDRFormat::HDRPNG:
        return LoadHDRPNG(filename, error);
    case HDRFormat::AVIF:
        // LoadAVIF(filename, error); // Currently returns error
        error = {true, "Avif not supported yet"};
        return nullptr;
    default:
        error = {true, "Unsupported image format for file: " + filename};
        return nullptr;
    }
}

void LoadAVIF(const std::string &filename, utils::Error &error) {
    // TODO: Implement AVIF loading using libavif
    error = {true, "AVIF loading not implemented"};
}

std::unique_ptr<Image> LoadHDRPNG(const std::string &filename,
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
    if (png_get_bit_depth(png, info) != 16) {
        png_destroy_read_struct(&png, &info, nullptr);
        fclose(fp);
        error = {true, "Input png file is not 16-bit depth"};
        return nullptr;
    }

    std::unique_ptr<Image> image = std::make_unique<Image>();

    image->width = png_get_image_width(png, info);
    image->height = png_get_image_height(png, info);
    // REVISIT:
    // image->color_type = png_get_color_type(png, info);
    image->bit_depth = png_get_bit_depth(png, info);
    image->bytes_per_row = png_get_rowbytes(png, info);

    png_read_update_info(png, info);
    image->metadata = ReadMetadata(filename, error);
    if (error.raise)
        return nullptr;

    image->row_pointers =
        (png_bytep *)malloc(sizeof(png_bytep) * image->height);

    for (size_t i = 0; i < image->height; i++) {
        image->row_pointers[i] = (png_byte *)malloc(image->bytes_per_row);
    }
    png_read_image(png, image->row_pointers);

    uint16_t min_val = UINT16_MAX;
    uint16_t max_val = 0;
    float sum = 0.0;
    const size_t channels = 3;
    const size_t total_pixels = image->width * image->height * channels;

    for (size_t y = 0; y < image->height; y++) {
        png_bytep row = image->row_pointers[y];
        for (size_t x = 0; x < image->width; x++) {
            for (size_t c = 0; c < channels; c++) {
                size_t idx = x * channels * 2 + c * 2; // *2 because 16-bit
                uint16_t value = (row[idx] << 8) | row[idx + 1]; // big-endian
                min_val = std::min(min_val, value);
                max_val = std::max(max_val, value);
                sum += value;
            }
        }
    }

    float mean = sum / total_pixels;

    std::cout << "Image value range:" << std::endl;
    std::cout << "  Min: " << min_val << " (" << (min_val / 65535.0) << ")"
              << std::endl;
    std::cout << "  Max: " << max_val << " (" << (max_val / 65535.0) << ")"
              << std::endl;
    std::cout << "  Mean: " << mean << " (" << (mean / 65535.0) << ")"
              << std::endl;

    png_destroy_read_struct(&png, &info, nullptr);
    fclose(fp);

    return image;
}

ImageMetadata ReadMetadata(const std::string &filename, utils::Error &error) {
    ImageMetadata metadata;
    std::unique_ptr<ExifTool> et(new ExifTool());

    if (!et) {
        error = {true, "Failed to create ExifTool instance"};
        return metadata;
    }

    TagInfo *info = et->ImageInfo(filename.c_str());

    char *err = et->GetError();
    if (err) {
        error = {true, std::string("ExifTool error: ") + err};
        return metadata;
    }

    if (!info) {
        if (et->LastComplete() <= 0) {
            error = {true, "Error executing exiftool!"};
            return metadata;
        }
    } else {
        for (TagInfo *i = info; i; i = i->next) {
            std::string name(i->name);
            std::string value(i->value);

            if (name == "TransferCharacteristics") {
                if (value.find("HLG") != std::string::npos) {
                    metadata.oetf = colorspace::OETF::HLG;
                } else if (value.find("PQ") != std::string::npos ||
                           value.find("2084") != std::string::npos) {
                    metadata.oetf = colorspace::OETF::PQ;
                } else if (value.find("2020") != std::string::npos) {
                    metadata.oetf = colorspace::OETF::HLG;
                } else if (value.find("709") != std::string::npos) {
                    metadata.oetf = colorspace::OETF::SRGB;
                } else {
                    error = {true, "Unknown transfer function: " + value};
                }
            } else if (name == "ColorPrimaries") {
                if (value.find("2100") != std::string::npos ||
                    value.find("2020") != std::string::npos) {
                    metadata.gamut = colorspace::Gamut::BT2100;
                } else if (value.find("P3") != std::string::npos ||
                           value.find("SMPTE") != std::string::npos) {
                    metadata.gamut = colorspace::Gamut::BT709;
                } else if (value.find("709") != std::string::npos ||
                           value.find("sRGB") != std::string::npos) {
                    metadata.gamut = colorspace::Gamut::BT709;
                } else {
                    error = {true, "Unknown color space: " + value};
                }
            }
            std::cout << i->name << " = " << i->value << std::endl;
        }
        delete info;
    }
    return metadata;
}

bool WriteToPNG(const std::unique_ptr<Image> &image,
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
        png_destroy_write_struct(&png, nullptr);
        fclose(fp);
        error = {true, "Failed to create PNG info struct"};
        return false;
    }

    if (setjmp(png_jmpbuf(png))) {
        png_destroy_write_struct(&png, &info);
        fclose(fp);
        error = {true, "Error during PNG write"};
        return false;
    }

    png_init_io(png, fp);

    // Set to 8-bit depth, RGB format
    png_set_IHDR(png, info, image->width, image->height, 8, image->color_type,
                 PNG_INTERLACE_NONE, PNG_COMPRESSION_TYPE_DEFAULT,
                 PNG_FILTER_TYPE_DEFAULT);

    png_write_info(png, info);

    if (!image->row_pointers) {
        png_destroy_write_struct(&png, &info);
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

bool WriteToNumpy(const std::vector<float> &data, int width, int height,
                  int channels, const std::string &dtype_str,
                  const std::string &output_path, utils::Error &error) {

    if (data.size() != static_cast<size_t>(width * height * channels)) {
        error = {true,
                 "Error: Data size doesn't match the specified dimensions"};
        return false;
    }

    std::vector<long unsigned> shape = {static_cast<long unsigned>(height),
                                        static_cast<long unsigned>(width),
                                        static_cast<long unsigned>(channels)};

    try {
        // Create the appropriate NumPy array based on dtype
        if (dtype_str == "float64" || dtype_str == "float") {
            // For float64, we can use the original data directly
            npy::SaveArrayAsNumpy(output_path, false, shape.size(),
                                  shape.data(), data);
        } else if (dtype_str == "float32") {
            // Convert to float32
            std::vector<float> float_data(data.size());
            for (size_t i = 0; i < data.size(); ++i) {
                float_data[i] = static_cast<float>(data[i]);
            }
            npy::SaveArrayAsNumpy(output_path, false, shape.size(),
                                  shape.data(), float_data);
        } else if (dtype_str == "uint8") {
            // Convert to uint8 with clamping
            std::vector<uint8_t> uint8_data(data.size());
            for (size_t i = 0; i < data.size(); ++i) {
                // Clamp values between 0 and 255 for uint8
                uint8_data[i] = static_cast<uint8_t>(
                    std::max(0.0f, std::min(255.0f, data[i])));
            }
            npy::SaveArrayAsNumpy(output_path, false, shape.size(),
                                  shape.data(), uint8_data);
        } else if (dtype_str == "int32" || dtype_str == "int") {
            // Convert to int32
            std::vector<int32_t> int32_data(data.size());
            for (size_t i = 0; i < data.size(); ++i) {
                int32_data[i] = static_cast<int32_t>(data[i]);
            }
            npy::SaveArrayAsNumpy(output_path, false, shape.size(),
                                  shape.data(), int32_data);
        } else {
            error = {true, "Error: Unsupported dtype: " + dtype_str};
            return false;
        }
    } catch (const std::exception &e) {
        error = {true, "Error: Exception occurred while saving NumPy array: " +
                           std::string(e.what())};
        return false;
    }
    return true;
}
} // namespace imageops
