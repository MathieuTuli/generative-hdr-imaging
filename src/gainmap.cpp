#include "gainmap.hpp"
#include "utils.h"
#include <iostream>

namespace gainmap {
std::unique_ptr<imageops::PNGImage>
ComputeGainMap(std::vector<double> hdr_yuv, std::vector<double> sdr_yuv,
               int width, int height, double offset_hdr, double offset_sdr,
               double min_content_boost, double max_content_boost,
               double map_gamma, utils::Error error) {

    std::unique_ptr<imageops::PNGImage> gainmap =
        std::make_unique<imageops::PNGImage>();
    gainmap->width = width;
    gainmap->height = height;
    gainmap->color_type = PNG_COLOR_TYPE_GRAY;
    gainmap->bit_depth = 8;
    gainmap->bytes_per_row = width * 3;

    gainmap->row_pointers =
        (png_bytep *)malloc(sizeof(png_bytep) * gainmap->height);
    for (size_t y = 0; y < gainmap->height; y++) {
        gainmap->row_pointers[y] = (png_byte *)malloc(gainmap->bytes_per_row);
    }

    for (size_t y = 0; y < gainmap->height; y++) {
        png_bytep gainmap_row = gainmap->row_pointers[y];
        for (size_t x = 0; x < gainmap->width; x++) {
            size_t idx = x * 3;
            double pixel_gain =
                (hdr_yuv[idx] + offset_hdr) / (sdr_yuv[idx] + offset_sdr);
            double map_min_log2 = std::log2(min_content_boost);
            double map_max_log2 = std::log2(max_content_boost);
            double log_recovery = (std::log2(pixel_gain) - map_min_log2) /
                                  (map_max_log2 - map_min_log2);
            log_recovery = CLIP(log_recovery, 0.0, 1.0);
            double recovery = std::max(std::pow(log_recovery, map_gamma), 0.0) * 10.0;

            uint8_t quantized_recovery = static_cast<uint8_t>(CLIP(recovery * 255.0, 0.0, 255.0));
            gainmap_row[x * 3] = quantized_recovery;
            gainmap_row[x * 3 + 1] = quantized_recovery;
            gainmap_row[x * 3 + 2] = quantized_recovery;
        }
    }

    return gainmap;
}

std::unique_ptr<imageops::PNGImage>
HDRToGainMap(const std::unique_ptr<imageops::PNGImage> &hdr_image,
             double offset_hdr, double offset_sdr, double min_content_boost,
             double max_content_boost, double map_gamma, utils::Error &error) {
    if (!hdr_image || !hdr_image->row_pointers) {
        error = {true, "Invalid input HDR image"};
        return nullptr;
    }

    // Create SDR version using HDRToSDR
    auto sdr_image =
        HDRToSDR(hdr_image, 0.0, 1.0, error, imageops::ToneMapping::BASE);
    if (!sdr_image) {
        error = {true, "Failed to create SDR image"};
        return nullptr;
    }

    const size_t width = hdr_image->width;
    const size_t height = hdr_image->height;
    const size_t channels = 3;

    // Convert HDR and SDR to YUV arrays
    std::vector<double> hdr_yuv;
    std::vector<double> sdr_yuv;
    hdr_yuv.reserve(width * height * channels);
    sdr_yuv.reserve(width * height * channels);

    // Process HDR image
    for (size_t y = 0; y < height; y++) {
        png_bytep hdr_row = hdr_image->row_pointers[y];
        for (size_t x = 0; x < width; x++) {
            // Convert 16-bit HDR values to linear RGB
            double rgb[3];
            for (size_t c = 0; c < channels; c++) {
                size_t idx = x * channels * 2 + c * 2;
                uint16_t value = (hdr_row[idx] << 8) | hdr_row[idx + 1];
                rgb[c] = imageops::Rec2020HLGToLinear(value / 65535.0);
            }

            // Convert to YUV
            auto yuv = imageops::XYZToRec2020YUV(
                imageops::LinearRec2020ToXYZ({rgb[0], rgb[1], rgb[2]}));
            hdr_yuv.insert(hdr_yuv.end(), yuv.begin(), yuv.end());
        }
    }

    // Process SDR image
    for (size_t y = 0; y < height; y++) {
        png_bytep sdr_row = sdr_image->row_pointers[y];
        for (size_t x = 0; x < width; x++) {
            // Convert 8-bit SDR values to linear RGB
            double rgb[3];
            for (size_t c = 0; c < channels; c++) {
                size_t idx = x * channels + c;
                rgb[c] = imageops::sRGBToLinear(sdr_row[idx] / 255.0);
            }

            // Convert to YUV
            auto yuv = imageops::XYZToRec2020YUV(
                imageops::LinearsRGBToXYZ({rgb[0], rgb[1], rgb[2]}));
            sdr_yuv.insert(sdr_yuv.end(), yuv.begin(), yuv.end());
        }
    }

    // Create gainmap using gainmap::compute_gain_map
    return ComputeGainMap(hdr_yuv, sdr_yuv, width, height, offset_hdr,
                          offset_sdr, min_content_boost, max_content_boost,
                          map_gamma, error);
}
} // namespace gainmap
