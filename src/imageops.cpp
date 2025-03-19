#include "imageops.hpp"
#include "utils.h"
#include <algorithm>
#include <cassert>
#include <cmath>
#include <filesystem>
#include <functional>
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
    if (HasExtension(filename, ".hdr"))
        return imageops::HDRFormat::PNG_HDR;
    return HDRFormat::UNKNOWN;
}

// ----------------------------------------
// INVOLVED IO
// ----------------------------------------

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
    if (png_get_bit_depth(png, info) != 16) {
        png_destroy_read_struct(&png, &info, nullptr);
        fclose(fp);
        error = {true, "Input png file is not 16-bit depth"};
        return nullptr;
    }

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
    ImageMetadata metadata;
    // Set default values for P3 HLG
    metadata.color_space = "Display P3";
    metadata.transfer_function = "HLG";
    metadata.gamma = 1.0f;        // HLG has its own EOTF
    metadata.luminance = 1000.0f; // Typical HLG peak luminance
    // P3 chromaticity coordinates
    metadata.primaries[0] = 0.680f;    // P3 Red x
    metadata.primaries[1] = 0.320f;    // P3 Red y
    metadata.primaries[2] = 0.265f;    // P3 Green x
    metadata.primaries[3] = 0.690f;    // P3 Green y
    metadata.primaries[4] = 0.150f;    // P3 Blue x
    metadata.primaries[5] = 0.060f;    // P3 Blue y
    metadata.white_point[0] = 0.3127f; // D65 x
    metadata.white_point[1] = 0.3290f; // D65 y
    FILE *fp = fopen(filename.c_str(), "rb");
    if (!fp) {
        error = {true, "Failed to open file: " + filename};
        return metadata;
    }

    png_structp png = png_create_read_struct(PNG_LIBPNG_VER_STRING, nullptr,
                                             nullptr, nullptr);
    if (!png) {
        fclose(fp);
        error = {true, "Failed to create PNG read struct"};
        return metadata;
    }

    png_infop info = png_create_info_struct(png);
    if (!info) {
        png_destroy_read_struct(&png, nullptr, nullptr);
        fclose(fp);
        error = {true, "Failed to create PNG info struct"};
        return metadata;
    }

    if (setjmp(png_jmpbuf(png))) {
        png_destroy_read_struct(&png, &info, nullptr);
        fclose(fp);
        error = {true, "Error during PNG read"};
        return metadata;
    }

    png_init_io(png, fp);
    png_read_info(png, info);

    // Read gamma
    double file_gamma;
    if (png_get_gAMA(png, info, &file_gamma)) {
        metadata.gamma = static_cast<float>(file_gamma);
    }

    // Read chromaticity
    double wx, wy, rx, ry, gx, gy, bx, by;
    if (png_get_cHRM(png, info, &wx, &wy, &rx, &ry, &gx, &gy, &bx, &by)) {
        metadata.white_point[0] = static_cast<float>(wx);
        metadata.white_point[1] = static_cast<float>(wy);
        metadata.primaries[0] = static_cast<float>(rx);
        metadata.primaries[1] = static_cast<float>(ry);
        metadata.primaries[2] = static_cast<float>(gx);
        metadata.primaries[3] = static_cast<float>(gy);
        metadata.primaries[4] = static_cast<float>(bx);
        metadata.primaries[5] = static_cast<float>(by);
    }

    // Check for sRGB chunk
    int srgb_intent;
    if (png_get_sRGB(png, info, &srgb_intent)) {
        metadata.color_space = "sRGB";
        switch (srgb_intent) {
        case PNG_sRGB_INTENT_PERCEPTUAL:
            metadata.rendering_intent = "perceptual";
            break;
        case PNG_sRGB_INTENT_RELATIVE:
            metadata.rendering_intent = "relative-colorimetric";
            break;
        case PNG_sRGB_INTENT_SATURATION:
            metadata.rendering_intent = "saturation";
            break;
        case PNG_sRGB_INTENT_ABSOLUTE:
            metadata.rendering_intent = "absolute-colorimetric";
            break;
        }
    }

    // Check for ICC profile
    png_charp name;
    png_bytep profile;
    png_uint_32 proflen;
    int compression_type;
    if (png_get_iCCP(png, info, &name, &compression_type, &profile, &proflen)) {
        metadata.color_space = std::string(name);
    }

    // Check for transparency
    png_bytep trans_alpha;
    int num_trans;
    png_color_16p trans_color;
    metadata.has_transparency =
        png_get_tRNS(png, info, &trans_alpha, &num_trans, &trans_color) != 0;

    png_destroy_read_struct(&png, &info, nullptr);
    fclose(fp);
    return metadata;
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
    png_set_IHDR(png, info, image->width, image->height, 8, PNG_COLOR_TYPE_RGB,
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

// ----------------------------------------
// CONVERSION
// ----------------------------------------
double LineartoHLG(double x) {
    /* Follows ITU-R BT.2100-2 */
    const double a = 0.17883277;
    const double b = 1.0 - 4.0 * a;               // 0.28466892;
    const double c = 0.5 - a * std::log(4.0 * a); // 0.55991073;

    const double epsilon =
        1e-6; // Define a small epsilon for floating point comparison
    ASSERT(-epsilon <= x && x <= (1.0 + epsilon),
           "Input should be in range [0, 1] for Rec2020, from %f", x);
    // x = std::max(0.0, std::min(1.0, x));

    if (x <= 1.0 / 12.0) {
        return std::sqrt(3 * x);
    } else {
        return a * std::log(12.0 * x - b) + c;
    }
}

double HLGtoLinear(double x) {
    /* Follows ITU-R BT.2100-2 */
    const double a = 0.17883277;
    const double b = 1.0 - 4.0 * a;               // 0.28466892;
    const double c = 0.5 - a * std::log(4.0 * a); // 0.55991073;

    const double epsilon =
        1e-6; // Define a small epsilon for floating point comparison
    ASSERT(-epsilon <= x && x <= (1.0 + epsilon),
           "Input should be in range [0, 1] for Rec2020, from %f", x);
    // x = std::max(0.0, std::min(1.0, x));

    if (x <= 0.5) {
        return (x * x) / 3.0;
    } else {
        return (std::exp((x - c) / a) + b) / 12.0;
    }
}
double ApplyToneMapping(double x, ToneMapping mode, double target_nits = 100.0,
                        double max_nits = 100.0) {
    if (mode == ToneMapping::BASE) {
        x *= target_nits / max_nits;
    } else if (mode == ToneMapping::REINHARD) {
        x = x / (1.0 + x);
    } else if (mode == ToneMapping::GAMMA) {
        // REVISIT: 2.2
        x = std::pow(x, 1.0 / 2.2);
    } else if (mode == ToneMapping::FILMIC) {
        const double A = 2.51;
        const double B = 0.03;
        const double C = 2.43;
        const double D = 0.59;
        const double E = 0.14;
        x = (x * (A * x + B)) / (x * (C * x + D) + E);
    } else if (mode == ToneMapping::ACES) {
        const double a = 2.51;
        const double b = 0.03;
        const double c = 2.43;
        const double d = 0.59;
        const double e = 0.14;
        // REVISIT: Exposure adjustment for ACES
        double adjusted = x * 0.6;
        x = (adjusted * (adjusted + b) * a) /
            (adjusted * (adjusted * c + d) + e);
    } else if (mode == ToneMapping::UNCHARTED2) {
        const double A = 0.15;
        const double B = 0.50;
        const double C = 0.10;
        const double D = 0.20;
        const double E = 0.02;
        const double F = 0.30;
        const double W = 11.2;

        auto uncharted2_tonemap = [=](double x) -> double {
            return ((x * (A * x + C * B) + D * E) / (x * (A * x + B) + D * F)) -
                   E / F;
        };

        x = uncharted2_tonemap(x) / uncharted2_tonemap(W);
    } else if (mode == ToneMapping::DRAGO) {
        const double bias = 0.85;
        const double Lwa = 1.0;

        x = std::log(1 + x) / std::log(1 + Lwa);
        x = std::pow(x, bias);
    } else if (mode == ToneMapping::LOTTES) {
        const double a = 1.6;

        const double mid_in = 0.18;
        const double mid_out = 0.267;

        const double t = x * a;
        x = t / (t + 1);

        const double z = (mid_in * a) / (mid_in * a + 1);
        x = x * (mid_out / z);
    } else if (mode == ToneMapping::HABLE) {
        const double A = 0.22; // Shoulder strength
        const double B = 0.30; // Linear strength
        const double C = 0.10; // Linear angle
        const double D = 0.20; // Toe strength
        const double E = 0.01; // Toe numerator
        const double F = 0.30; // Toe denominator
        const double W = 11.2;

        std::function<double(double)> hable = [=](double x) -> double {
            return (x * (A * x + C * B) + D * E) / (x * (A * x + B) + D * F) -
                   E / F;
        };

        x = hable(x) / hable(W);
    } else if (mode == ToneMapping::REINHARD) {
        x = x / (1.0 + x);
    }
    return x;
}

void LinearRec2020toLinearsRGB(double &r, double &g, double &b) {
    // using D65 white point
    // standard RGB primaries and white point
    // const double matrix[3][3] = {{1.6605, -0.5876, -0.0728},
    //                              {-0.1246, 1.1329, -0.0083},
    //                              {-0.0182, -0.1006, 1.1187}};
    const double rec2020_to_xyz[3][3] = {{0.6370, 0.1446, 0.1689},
                                         {0.2627, 0.6780, 0.0593},
                                         {0.0000, 0.0281, 1.0610}};

    double x = rec2020_to_xyz[0][0] * r + rec2020_to_xyz[0][1] * g +
               rec2020_to_xyz[0][2] * b;
    double y = rec2020_to_xyz[1][0] * r + rec2020_to_xyz[1][1] * g +
               rec2020_to_xyz[1][2] * b;
    double z = rec2020_to_xyz[2][0] * r + rec2020_to_xyz[2][1] * g +
               rec2020_to_xyz[2][2] * b;

    const double xyz_to_sRGB[3][3] = {{3.2404542, -1.5371385, -0.4985314},
                                      {-0.9692660, 1.8760108, 0.0415560},
                                      {0.0556434, -0.2040259, 1.0572252}};

    double new_r =
        xyz_to_sRGB[0][0] * x + xyz_to_sRGB[0][1] * y + xyz_to_sRGB[0][2] * z;
    double new_g =
        xyz_to_sRGB[1][0] * x + xyz_to_sRGB[1][1] * y + xyz_to_sRGB[1][2] * z;
    double new_b =
        xyz_to_sRGB[2][0] * x + xyz_to_sRGB[2][1] * y + xyz_to_sRGB[2][2] * z;

    r = CLIP(new_r, 0.0, 1.0);
    g = CLIP(new_g, 0.0, 1.0);
    b = CLIP(new_b, 0.0, 1.0);
}

double LineartosRGB(double x) {
    const double epsilon =
        1e-6; // Define a small epsilon for floating point comparison
    ASSERT(-epsilon <= x && x <= (1.0 + epsilon),
           "Input should be in range [0, 1], from %f", x);
    if (x <= 0.0031308) {
        return 12.92 * x;
    } else {
        return 1.055 * std::pow(x, 1.0 / 2.4) - 0.055;
    }
}

double sRGBtoLinear(double x) {
    const double epsilon =
        1e-6; // Define a small epsilon for floating point comparison
    ASSERT(-epsilon <= x && x <= (1.0 + epsilon),
           "Input should be in range [0, 1], from %f", x);
    if (x <= 0.04045) {
        return x / 12.92;
    } else {
        return std::pow((x + 0.055) / 1.055, 2.4);
    }
}

std::unique_ptr<PNGImage>
HDRtoSDR(const std::unique_ptr<PNGImage> &hdr_image, double clip_low,
         double clip_high, utils::Error &error,
         ToneMapping tone_mapping = ToneMapping::BASE) {
    if (!hdr_image || !hdr_image->row_pointers) {
        error = {true, "Invalid input HDR image"};
        return nullptr;
    }

    auto sdr_image = std::make_unique<PNGImage>();
    sdr_image->width = hdr_image->width;
    sdr_image->height = hdr_image->height;
    sdr_image->color_type = PNG_COLOR_TYPE_RGB;
    sdr_image->bit_depth = 8;
    sdr_image->bytes_per_row = hdr_image->width * 3;

    sdr_image->row_pointers =
        (png_bytep *)malloc(sizeof(png_bytep) * sdr_image->height);
    for (size_t y = 0; y < sdr_image->height; y++) {
        sdr_image->row_pointers[y] =
            (png_byte *)malloc(sdr_image->bytes_per_row);
    }

    const size_t channels = 3;

    /*
    double max_value = 0.0;
    if (params.auto_clip) {
        std::vector<double> all_values;
        all_values.reserve(hdr_image->width * hdr_image->height * channels);

        for (size_t y = 0; y < hdr_image->height; y++) {
            png_bytep row = hdr_image->row_pointers[y];
            for (size_t x = 0; x < hdr_image->width; x++) {
                for (size_t c = 0; c < channels; c++) {
                    size_t idx = x * channels + c;
                    // Convert 16-bit value to float [0,1]
                    uint16_t value = (row[idx * 2] << 8) | row[idx * 2 + 1];
                    all_values.push_back(value / 65535.0);
                }
            }
        }

        // Sort to find 99.9th percentile
        std::sort(all_values.begin(), all_values.end());
        size_t percentile_idx = static_cast<size_t>(all_values.size() * 0.999);
        max_value = all_values[percentile_idx];
    } else {
        max_value = params.clip_high > 0 ? params.clip_high : 1.0;
    }
    */

    for (size_t y = 0; y < hdr_image->height; y++) {
        png_bytep hdr_row = hdr_image->row_pointers[y];
        png_bytep sdr_row = sdr_image->row_pointers[y];

        for (size_t x = 0; x < hdr_image->width; x++) {
            size_t sdr_idx = x * channels;

            uint16_t values[3];
            for (size_t i = 0; i < 3; i++) {
                // *2 because input is 16-bit
                size_t idx = x * channels * 2 + i * 2;
                // PNG stores in network byte order (big-endian)
                values[i] = (hdr_row[idx] << 8) | hdr_row[idx + 1];
            }

            double r = (values[0] / 65535.0);
            double g = (values[1] / 65535.0);
            double b = (values[2] / 65535.0);

            r = HLGtoLinear(r);
            g = HLGtoLinear(g);
            b = HLGtoLinear(b);
            // REVISIT: what to clip to?
            r = CLIP(r, 0.0, 1.0);
            g = CLIP(g, 0.0, 1.0);
            b = CLIP(b, 0.0, 1.0);
            r = ApplyToneMapping(r, tone_mapping);
            g = ApplyToneMapping(g, tone_mapping);
            b = ApplyToneMapping(b, tone_mapping);
            LinearRec2020toLinearsRGB(r, g, b);
            r = LineartosRGB(r);
            g = LineartosRGB(g);
            b = LineartosRGB(b);

            sdr_row[sdr_idx + 0] =
                static_cast<uint8_t>(CLIP(r * 255.0, 0.0, 255.0));
            sdr_row[sdr_idx + 1] =
                static_cast<uint8_t>(CLIP(g * 255.0, 0.0, 255.0));
            sdr_row[sdr_idx + 2] =
                static_cast<uint8_t>(CLIP(b * 255.0, 0.0, 255.0));
        }
    }

    return sdr_image;
}
} // namespace imageops
