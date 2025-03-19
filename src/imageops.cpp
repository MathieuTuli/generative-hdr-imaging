#include "imageops.hpp"
#include "utils.h"
#include <algorithm>
#include <cassert>
#include <cmath>
#include <filesystem>
#include <iostream>
#include <memory>
#include <png.h>
#include <vector>

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
double DynamicRangeCompression(double x) {
    // REVISIT:
    x = CLIP(x, 0, 1.0);
    double target_nits = 100.0;
    double max_nits = 100.0;


    x *= target_nits / max_nits;

    // REVISIT: Improved tone mapping
    // REINHARD, ACES, or HABLE
    // x = x / (x + 1.0);
    return x;
}


// Helper function for Rec.2020 to sRGB color space conversion
void Rec2020toSRGB(double &r, double &g, double &b) {
    // Rec.2020 to sRGB conversion matrix (D65 white point)
    // These values are derived from the standard RGB primaries and white point
    const double matrix[3][3] = {{1.6605, -0.5876, -0.0728},
                                 {-0.1246, 1.1329, -0.0083},
                                 {-0.0182, -0.1006, 1.1187}};

    // Apply color space conversion
    double new_r = matrix[0][0] * r + matrix[0][1] * g + matrix[0][2] * b;
    double new_g = matrix[1][0] * r + matrix[1][1] * g + matrix[1][2] * b;
    double new_b = matrix[2][0] * r + matrix[2][1] * g + matrix[2][2] * b;

    // Clamp values to [0,1] range after conversion
    r = std::max(0.0, std::min(1.0, new_r));
    g = std::max(0.0, std::min(1.0, new_g));
    b = std::max(0.0, std::min(1.0, new_b));
}

double SRGBTransfer(double x) {
    if (x <= 0.0031308f) {
        return 12.92 * x;
    } else {
        return 1.055 * std::pow(x, 1.0 / 2.4f) - 0.055f;
    }
}

std::unique_ptr<PNGImage> HDRtoSDR(const std::unique_ptr<PNGImage> &hdr_image,
                                   const SDRConversionParams &params,
                                   utils::Error &error) {
    if (!hdr_image || !hdr_image->row_pointers) {
        error = {true, "Invalid input HDR image"};
        return nullptr;
    }

    // Validate input parameters for HLG P3 conversion
    const double target_nits =
        params.target_nits > 0 ? params.target_nits : 100.0;
    const double max_nits = params.max_nits > 0 ? params.max_nits : 100.0;
    const double luminance_scale = target_nits / max_nits;

    // Create output image with same dimensions but 8-bit depth
    auto sdr_image = std::make_unique<PNGImage>();
    sdr_image->width = hdr_image->width;
    sdr_image->height = hdr_image->height;
    sdr_image->color_type = PNG_COLOR_TYPE_RGB;
    sdr_image->bit_depth = 8; // Output will be 8-bit
    sdr_image->bytes_per_row =
        hdr_image->width *
        3; // (hdr_image->color_type == PNG_COLOR_TYPE_RGBA ? 4 : 3);

    // Allocate memory for output image
    sdr_image->row_pointers =
        (png_bytep *)malloc(sizeof(png_bytep) * sdr_image->height);
    for (size_t y = 0; y < sdr_image->height; y++) {
        sdr_image->row_pointers[y] =
            (png_byte *)malloc(sdr_image->bytes_per_row);
    }

    // Process each pixel
    const double gamma = params.gamma > 0 ? params.gamma : 2.2; // Default gamma
    const double exposure =
        params.exposure > 0 ? params.exposure : 1.0; // Default exposure
    const size_t channels =
        (hdr_image->color_type == PNG_COLOR_TYPE_RGBA) ? 4 : 3;

    // Pre-calculate exposure adjustment
    const double exposure_scale = std::pow(2.0, exposure);

    // Find max value for auto-clipping if needed
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

    // Process each pixel
    for (size_t y = 0; y < hdr_image->height; y++) {
        png_bytep in_row = hdr_image->row_pointers[y];
        png_bytep out_row = sdr_image->row_pointers[y];

        for (size_t x = 0; x < hdr_image->width; x++) {
            for (size_t c = 0; c < channels; c++) {
                size_t in_idx =
                    x * channels * 2 + c * 2; // *2 because input is 16-bit
                size_t out_idx = x * channels + c;

                // Read 16-bit values for all channels first
                uint16_t values[3];
                for (size_t i = 0; i < 3; i++) {
                    size_t idx = x * channels * 2 + i * 2;
                    // Note: PNG stores in network byte order (big-endian)
                    values[i] = (in_row[idx] << 8) | in_row[idx + 1];
                }

                // Convert to normalized [0,1] range
                double r = (values[0] / 65535.0);
                double g = (values[1] / 65535.0);
                double b = (values[2] / 65535.0);

                // Apply exposure before HLG EOTF
                r *= exposure_scale;
                g *= exposure_scale;
                b *= exposure_scale;

                // Clamp values to valid HLG input range [0,1]
                r = std::min(1.0, std::max(0.0, r));
                g = std::min(1.0, std::max(0.0, g));
                b = std::min(1.0, std::max(0.0, b));

                // Apply HLG EOTF to get linear values with more conservative
                // scaling
                r = HLGtoLinear(r) *
                    0.8; // Reduce intensity by 20% to prevent highlight blowout
                g = HLGtoLinear(g) * 0.8;
                b = HLGtoLinear(b) * 0.8;

                // More conservative display scaling
                const double display_scale = std::min(1.0, max_nits / 1000.0);
                r *= display_scale;
                g *= display_scale;
                b *= display_scale;

                // Convert from Rec.2020 to sRGB color space with highlight
                // protection
                double max_rgb_pre = std::max({r, g, b});
                Rec2020toSRGB(r, g, b);
                double max_rgb_post = std::max({r, g, b});

                // Preserve relative highlight ratios
                if (max_rgb_post > max_rgb_pre && max_rgb_pre > 0) {
                    double scale = max_rgb_pre / max_rgb_post;
                    r *= scale;
                    g *= scale;
                    b *= scale;
                }

                // Apply luminance scaling with highlight preservation
                if (params.preserve_highlights && max_nits > target_nits) {
                    double max_rgb = std::max({r, g, b});
                    if (max_rgb > params.knee_point) {
                        // Smooth roll-off for highlights
                        double ratio = (max_rgb - params.knee_point) /
                                       (1.0 - params.knee_point);
                        double scale =
                            params.knee_point +
                            (1.0 - params.knee_point) *
                                (1.0 -
                                 std::exp(-ratio *
                                          4.0)); // Adjustable roll-off speed
                        r *= scale / max_rgb;
                        g *= scale / max_rgb;
                        b *= scale / max_rgb;
                    }
                }

                // Scale to target display luminance
                r *= target_nits / max_nits;
                g *= target_nits / max_nits;
                b *= target_nits / max_nits;

                // Select current channel value for further processing
                double linearized = (c == 0) ? r : (c == 1) ? g : b;

                // 3. Clip dynamic range
                double clipped =
                    std::min(std::max(linearized, params.clip_low), max_value);
                if (max_value != 1.0) {
                    clipped = (clipped - params.clip_low) /
                              (max_value - params.clip_low);
                }

                // 4. Apply tone mapping
                double tone_mapped;
                switch (params.tone_mapping) {
                case ToneMapping::REINHARD: {
                    // Simple Reinhard operator
                    tone_mapped = clipped / (1.0 + clipped);
                    break;
                }
                case ToneMapping::GAMMA: {
                    // Simple gamma correction
                    tone_mapped = std::pow(clipped, 1.0 / gamma);
                    break;
                }
                case ToneMapping::FILMIC: {
                    // John Hable's filmic curve parameters
                    const double A = 2.51;
                    const double B = 0.03;
                    const double C = 2.43;
                    const double D = 0.59;
                    const double E = 0.14;
                    tone_mapped = (clipped * (A * clipped + B)) /
                                  (clipped * (C * clipped + D) + E);
                    break;
                }
                case ToneMapping::ACES: {
                    // ACES approximation (Krzysztof Narkowicz)
                    const double a = 2.51;
                    const double b = 0.03;
                    const double c = 2.43;
                    const double d = 0.59;
                    const double e = 0.14;
                    double adjusted =
                        clipped * 0.6; // Exposure adjustment for ACES
                    tone_mapped = (adjusted * (adjusted + b) * a) /
                                  (adjusted * (adjusted * c + d) + e);
                    break;
                }
                case ToneMapping::UNCHARTED2: {
                    // Uncharted 2 tone mapping (John Hable)
                    const double A = 0.15;
                    const double B = 0.50;
                    const double C = 0.10;
                    const double D = 0.20;
                    const double E = 0.02;
                    const double F = 0.30;
                    const double W = 11.2;

                    auto uncharted2_tonemap = [=](double x) -> double {
                        return ((x * (A * x + C * B) + D * E) /
                                (x * (A * x + B) + D * F)) -
                               E / F;
                    };

                    tone_mapped =
                        uncharted2_tonemap(clipped) / uncharted2_tonemap(W);
                    break;
                }
                case ToneMapping::DRAGO: {
                    // Drago logarithmic mapping
                    const double bias = 0.85;
                    const double Lwa = 1.0; // World adaptation luminance

                    tone_mapped = std::log(1 + clipped) / std::log(1 + Lwa);
                    tone_mapped = std::pow(tone_mapped, bias);
                    break;
                }
                case ToneMapping::LOTTES: {
                    // Timothy Lottes' optimized tone mapping
                    const double a = 1.6;

                    const double midIn = 0.18;
                    const double midOut = 0.267;

                    // Lottes curve
                    const double t = clipped * a;
                    tone_mapped = t / (t + 1);

                    // Scale to preserve mids
                    const double z = (midIn * a) / (midIn * a + 1);
                    tone_mapped = tone_mapped * (midOut / z);
                    break;
                }
                case ToneMapping::HABLE: {
                    // John Hable's filmic tone mapping (Unreal 3/4)
                    const double A = 0.22; // Shoulder strength
                    const double B = 0.30; // Linear strength
                    const double C = 0.10; // Linear angle
                    const double D = 0.20; // Toe strength
                    const double E = 0.01; // Toe numerator
                    const double F = 0.30; // Toe denominator
                    const double W = 11.2;

                    auto hable = [=](double x) -> double {
                        return (x * (A * x + C * B) + D * E) /
                                   (x * (A * x + B) + D * F) -
                               E / F;
                    };

                    tone_mapped = hable(clipped) / hable(W);
                    break;
                }
                default: {
                    // Default to Reinhard if unknown tone mapping specified
                    tone_mapped = clipped / (1.0 + clipped);
                    break;
                }
                }

                // 5. Quantize to 8-bit
                out_row[out_idx] = static_cast<uint8_t>(
                    std::min(std::max(tone_mapped * 255.0, 0.0), 255.0));
            }
        }
    }

    return sdr_image;
}
} // namespace imageops
