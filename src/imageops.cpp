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
    // DEPRECATE:
    // if (png_get_bit_depth(png, info) != 16) {
    //     png_destroy_read_struct(&png, &info, nullptr);
    //     fclose(fp);
    //     error = {true, "Input png file is not 16-bit depth"};
    //     return nullptr;
    // }

    std::unique_ptr<Image> image = std::make_unique<Image>();

    image->width = png_get_image_width(png, info);
    image->height = png_get_image_height(png, info);
    image->color_type = png_get_color_type(png, info);
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

    uint32_t max_value = (1 << image->bit_depth) - 1;
    uint32_t min_val = max_value;
    uint32_t max_val = 0;
    double sum = 0.0; // Using double for better precision with large sums
    const size_t channels = 3;
    const size_t total_pixels = image->width * image->height * channels;
    const size_t bytes_per_sample =
        (image->bit_depth + 7) / 8; // Round up to nearest byte

    for (size_t y = 0; y < image->height; y++) {
        png_bytep row = image->row_pointers[y];
        for (size_t x = 0; x < image->width; x++) {
            for (size_t c = 0; c < channels; c++) {
                uint32_t value = 0;
                size_t idx =
                    x * channels * bytes_per_sample + c * bytes_per_sample;

                // Read value based on bit depth
                if (bytes_per_sample == 1) {
                    value = row[idx];
                } else if (bytes_per_sample == 2) {
                    // PNG stores 16-bit values in big-endian (network byte order)
                    // regardless of the host's endianness
                    value = (row[idx] << 8) | row[idx + 1]; 
                }

                min_val = std::min(min_val, value);
                max_val = std::max(max_val, value);
                sum += value;
            }
        }
    }

    double mean = sum / total_pixels;
    double scale = 1.0 / max_value; // Normalize to [0,1] range

    std::cout << "Image value range (" << image->bit_depth
              << "-bit):" << std::endl;
    std::cout << "  Min: " << min_val << " (" << (min_val * scale) << ")"
              << std::endl;
    std::cout << "  Max: " << max_val << " (" << (max_val * scale) << ")"
              << std::endl;
    std::cout << "  Mean: " << mean << " (" << (mean * scale) << ")"
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

    std::cout << "----------------------------------------" << std::endl;
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
                } else if (value.find("709") != std::string::npos ||
                           value.find("sRGB") != std::string::npos) {
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
    std::cout << "----------------------------------------" << std::endl;
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
    png_set_IHDR(png, info, image->width, image->height, image->bit_depth,
                 image->color_type, PNG_INTERLACE_NONE,
                 PNG_COMPRESSION_TYPE_DEFAULT, PNG_FILTER_TYPE_DEFAULT);

    // Add color space and transfer function metadata as text chunks
    std::string oetf_str, gamut_str;

    switch (image->metadata.oetf) {
    case colorspace::OETF::HLG:
        oetf_str = "HLG";
        break;
    case colorspace::OETF::PQ:
        oetf_str = "PQ";
        break;
    case colorspace::OETF::SRGB:
        oetf_str = "sRGB";
        break;
    default:
        oetf_str = "Unknown";
    }

    switch (image->metadata.gamut) {
    case colorspace::Gamut::BT2100:
        gamut_str = "BT.2100";
        break;
    case colorspace::Gamut::BT709:
        gamut_str = "BT.709";
        break;
    default:
        gamut_str = "Unknown";
    }

    std::string combined_oetf = gamut_str + " " + oetf_str;

    // Create text chunks
    png_text text[2];
    text[0].compression = PNG_TEXT_COMPRESSION_NONE;
    text[0].key = const_cast<char *>("TransferCharacteristics");
    text[0].text = const_cast<char *>(combined_oetf.c_str());
    text[0].text_length = combined_oetf.length();

    text[1].compression = PNG_TEXT_COMPRESSION_NONE;
    text[1].key = const_cast<char *>("ColorPrimaries");
    text[1].text = const_cast<char *>(gamut_str.c_str());
    text[1].text_length = gamut_str.length();

    png_set_text(png, info, text, 2);
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
} // namespace imageops
