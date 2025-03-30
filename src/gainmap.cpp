#include "gainmap.hpp"
#include "npy.hpp"
#include "spdlog/spdlog.h"
#include "utils.h"
#include <cmath>
#include <fstream>
#include <iostream>
#include <json.hpp>

namespace gainmap {

/**
 * @brief Read a pixel from a PNG row and convert it to normalized RGB values
 * @param row The PNG row data
 * @param x The x coordinate of the pixel
 * @param channels Number of color channels (typically 3 for RGB)
 * @param bit_depth Bit depth of the image (8 or 16)
 * @return colorspace::Color with normalized RGB values in range [0,1]
 */
float ComputePSNR(const std::vector<colorspace::Color> &original,
                  const std::vector<colorspace::Color> &reconstructed) {
    if (original.size() != reconstructed.size() || original.empty()) {
        spdlog::error("Invalid vector sizes for PSNR computation");
        return -1.0f;
    }

    double mse_r = 0.0;
    double mse_g = 0.0;
    double mse_b = 0.0;

    for (size_t i = 0; i < original.size(); ++i) {
        double diff_r = original[i].r - reconstructed[i].r;
        double diff_g = original[i].g - reconstructed[i].g;
        double diff_b = original[i].b - reconstructed[i].b;

        mse_r += diff_r * diff_r;
        mse_g += diff_g * diff_g;
        mse_b += diff_b * diff_b;
    }

    double mse =
        (mse_r + mse_g + mse_b) / (static_cast<float>(original.size()) * 3.0f);

    if (mse < 1e-10) {
        return 100.0f;
    }

    float psnr = 20.0f * log10f(1.0f / sqrtf(mse));

    return psnr;
}
colorspace::Color ReadPixelFromRow(png_bytep row, size_t x, size_t channels,
                                   uint8_t bit_depth) {
    // round up to nearest byte
    size_t bytes_per_channel = (bit_depth + 7) / 8;
    size_t idx = x * channels * bytes_per_channel;

    // read values based on bit depth
    uint32_t r_value = 0, g_value = 0, b_value = 0;
    if (bytes_per_channel == 1) {
        // 8-bit values
        r_value = row[idx];
        g_value = row[idx + 1];
        b_value = row[idx + 2];
    } else if (bytes_per_channel == 2) {
        // 10-bit, 12-bit, or 16-bit values stored in 2 bytes
        // PNG stores 16-bit values in big-endian (network byte order)
        // regardless of the host's endianness
        r_value = (row[idx] << 8) | row[idx + 1];
        g_value = (row[idx + 2] << 8) | row[idx + 3];
        b_value = (row[idx + 4] << 8) | row[idx + 5];
    }

    // create a mask for the actual bit depth
    uint32_t max_value = (1 << bit_depth) - 1;

    // mask and normalize to [0,1]
    colorspace::Color rgb_gamma;
    rgb_gamma.r =
        static_cast<float>(r_value & max_value) / static_cast<float>(max_value);
    rgb_gamma.g =
        static_cast<float>(g_value & max_value) / static_cast<float>(max_value);
    rgb_gamma.b =
        static_cast<float>(b_value & max_value) / static_cast<float>(max_value);

    return rgb_gamma;
}

float ComputeGain(float hdr_y_nits, float sdr_y_nits,
                  float hdr_offset = 0.015625f, float sdr_offset = 0.015625f) {
    float gain = log2f((hdr_y_nits + hdr_offset) / (sdr_y_nits + sdr_offset));
    if (sdr_y_nits < 2.f / 255.0f) {
        // If sdr is zero and hdr is non zero, it can result in very large gain
        // values. In compression - decompression process, if the same sdr pixel
        // increases to 1, the hdr recovered pixel will blow out. Dont allow
        // dark pixels to signal large gains.
        gain = (std::min)(gain, 2.3f);
    }
    return gain;
}

float AffineMapGain(float gainlog2, float min_gainlog2, float max_gainlog2,
                    float map_gamma) {
    float mapped_val =
        (gainlog2 - min_gainlog2) / (max_gainlog2 - min_gainlog2);
    if (map_gamma != 1.0f)
        mapped_val = pow(mapped_val, map_gamma);
    return mapped_val;
}
float RecomputeHDRLuminance(float sdr_luminance, float gain,
                            float hdr_offset = 0.015625f,
                            float sdr_offset = 0.015625f) {
    float gain_factor = exp2f(gain);
    return (sdr_luminance + sdr_offset) * gain_factor - hdr_offset;
}

float UndoAffineMapGain(float gain, float map_gamma, float min_content_boost,
                        float max_content_boost) {
    float effective_gain =
        (map_gamma != 1.0f) ? powf(gain, 1.0f / map_gamma) : gain;
    float log_min = log2f(min_content_boost);
    float log_max = log2f(max_content_boost);
    float log_boost =
        log_min * (1.0f - effective_gain) + log_max * effective_gain;
    return log_boost;
}

colorspace::Color ApplyGain(colorspace::Color e, float gain, float map_gamma,
                            float min_content_boost, float max_content_boost,
                            float hdr_offset = 0.015625f,
                            float sdr_offset = 0.015625f) {
    if (map_gamma != 1.0f)
        gain = pow(gain, 1.0f / map_gamma);
    float log_boost = log2f(min_content_boost) * (1.0f - gain) +
                      log2f(max_content_boost) * gain;
    float gain_factor = exp2f(log_boost);
    return ((e + sdr_offset) * gain_factor) - hdr_offset;
}

// The pipeline is as follows:
// HDR -> linear (inv OETF)
// -> Rec2020 [save]
// linear -> gamma (Bt100 HLG OETF)
// gamma -> YUV [save]

// Rec2020 (linear) -> Bt709
// -> gamut map (clip)
// -> 99% clip
// -> sRGB OETF
// -> ToneMap
// -> Quantize [save]
// -> Inv ToneMap
// sRGB -> linear (inv sRGB OETF)
// -> Rec2020
// linear -> gamma (Bt100 HLG OETF)
// gamma -> YUV [save]
void HDRToGainMap(const std::unique_ptr<imageops::Image> &hdr_image,
                  float clip_percentile, float map_gamma,
                  const std::string &file_stem, const std::string &output_dir,
                  utils::Error &error) {
    if (!hdr_image || !hdr_image->row_pointers) {
        error = {true, "Invalid input HDR image"};
        return;
    }

    colorspace::Gamut hdr_gamut = hdr_image->metadata.gamut;
    colorspace::ColorTransformFn hdr_inv_oetf =
        colorspace::GetInvOETFFn(hdr_image->metadata.oetf);
    colorspace::SceneToDisplayLuminanceFn hdr_ootf =
        colorspace::GetOOTFFn(hdr_image->metadata.oetf);
    colorspace::ColorTransformFn hdr_gamut_conv =
        colorspace::GetGamutConversionFn(colorspace::Gamut::BT2100, hdr_gamut);
    // DEPRECATE:
    // colorspace::ColorTransformFn hdr_rgb2yuv =
    //     colorspace::GetRGBToYUVFn(hdr_gamut);
    // colorspace::ColorTransformFn hdr_yuv2rgb =
    //     colorspace::GetRGBToYUVFn(hdr_gamut);
    colorspace::LuminanceFn hdr_luminance_fn =
        colorspace::GetLuminanceFn(hdr_gamut);
    colorspace::LuminanceFn bt2100_luminance_fn =
        colorspace::GetLuminanceFn(colorspace::Gamut::BT2100);
    float hdr_peak_nits = colorspace::GetReferenceDisplayPeakLuminanceInNits(
        hdr_image->metadata.oetf);
    colorspace::ColorTransformFn sdr_gamut_conv =
        colorspace::GetGamutConversionFn(colorspace::Gamut::BT709,
                                         colorspace::Gamut::BT2100);
    colorspace::ColorTransformFn sdr_inv_oetf =
        colorspace::GetInvOETFFn(colorspace::OETF::SRGB);
    colorspace::ColorTransformFn sdr_oetf =
        colorspace::GetOETFFn(colorspace::OETF::SRGB);
    colorspace::ColorTransformFn sdr_hdr_gamut_conv =
        colorspace::GetGamutConversionFn(colorspace::Gamut::BT2100,
                                         colorspace::Gamut::BT709);

    const size_t width = hdr_image->width;
    const size_t height = hdr_image->height;
    const size_t channels = 3;

    std::vector<colorspace::Color> hdr_linear_image;
    std::vector<colorspace::Color> sdr_image;
    std::vector<float> gainmap;
    std::vector<uint8_t> affine_gainmap;
    hdr_linear_image.reserve(width * height * channels);
    sdr_image.reserve(width * height * channels);
    // TODO: we only do single-channel gainmaps for now
    gainmap.reserve(width * height * 1);
    affine_gainmap.reserve(width * height * 1);

    float min_gain = 255.f;
    float max_gain = -255.f;
    for (size_t y = 0; y < height; y++) {
        png_bytep hdr_row = hdr_image->row_pointers[y];
        for (size_t x = 0; x < width; x++) {

            colorspace::Color hdr_rgb_gamma =
                ReadPixelFromRow(hdr_row, x, channels, hdr_image->bit_depth);

            colorspace::Color hdr_rgb = hdr_inv_oetf(hdr_rgb_gamma);
            hdr_rgb = hdr_ootf(hdr_rgb, hdr_luminance_fn);
            hdr_rgb = hdr_gamut_conv(hdr_rgb);
            hdr_rgb = colorspace::ClipNegatives(hdr_rgb);
            hdr_linear_image.push_back(hdr_rgb);
        }
    }
    // NOTE: apply 99.9% clipping to sdr rgb values
    std::vector<float> all_values;
    all_values.reserve(width * height * channels);
    for (const colorspace::Color &color : hdr_linear_image) {
        all_values.push_back(color.r);
        all_values.push_back(color.g);
        all_values.push_back(color.b);
    }

    std::sort(all_values.begin(), all_values.end());
    size_t percentile_idx =
        static_cast<size_t>(all_values.size() * clip_percentile);
    float clip_value = all_values[percentile_idx];
    spdlog::debug("HDR {}th-percentile clip value: {}", clip_percentile,
                  clip_value);

    std::vector<colorspace::Color> test_base, test_recon;
    for (size_t i = 0; i < hdr_linear_image.size(); i++) {
        colorspace::Color hdr_rgb = hdr_linear_image[i];
        colorspace::Color sdr_rgb = sdr_gamut_conv(hdr_rgb);
        colorspace::Clamp(sdr_rgb);

        // NOTE: apply 99% clipping
        sdr_rgb.r = std::min(sdr_rgb.r, clip_value) / clip_value;
        sdr_rgb.g = std::min(sdr_rgb.g, clip_value) / clip_value;
        sdr_rgb.b = std::min(sdr_rgb.b, clip_value) / clip_value;

        colorspace::Color srgb_gamma = sdr_oetf(sdr_rgb);
        // TODO: apply tone map
        sdr_image.push_back(srgb_gamma);

        // add 0.5f for proper rounding and clamp to [0, 255]
        uint8_t r = static_cast<uint8_t>(
            std::min(255.f, std::max(0.f, srgb_gamma.r * 255.f + 0.5f)));
        uint8_t g = static_cast<uint8_t>(
            std::min(255.f, std::max(0.f, srgb_gamma.g * 255.f + 0.5f)));
        uint8_t b = static_cast<uint8_t>(
            std::min(255.f, std::max(0.f, srgb_gamma.b * 255.f + 0.5f)));

        srgb_gamma.r = static_cast<float>(r) / 255.f;
        srgb_gamma.g = static_cast<float>(g) / 255.f;
        srgb_gamma.b = static_cast<float>(b) / 255.f;

        sdr_rgb = sdr_inv_oetf(srgb_gamma);

        colorspace::Color sdr_rgb_bt2100 = sdr_hdr_gamut_conv(sdr_rgb);

        float hdr_y_nits, sdr_y_nits;

        // NOTE: libultrahdr has a version based on max rgb vs. luminance
        // we use luminance
        sdr_y_nits =
            bt2100_luminance_fn(sdr_rgb_bt2100) * colorspace::SDR_WHITE_NITS;
        hdr_y_nits = bt2100_luminance_fn(hdr_rgb) * hdr_peak_nits;
        float gain = ComputeGain(hdr_y_nits, sdr_y_nits);
        min_gain = std::min(gain, min_gain);
        max_gain = std::max(gain, max_gain);
        gainmap.push_back(gain);
    }

    // generate map
    // NOTE: from LibUltraHDR
    // gain coefficient range [-14.3, 15.6] is capable of representing hdr pels
    // from sdr pels. Allowing further excursion might not offer any benefit and
    // on the downside can cause bigger error during affine map and inverse
    // affine map.
    min_gain = (std::clamp)(min_gain, -14.3f, 15.6f);
    max_gain = (std::clamp)(max_gain, -14.3f, 15.6f);

    // TODO: if min/max content boost are given
    float min_content_boost = 1.0f;
    float max_content_boost = hdr_peak_nits / colorspace::SDR_WHITE_NITS;
    min_gain = std::min(min_gain, log2f(min_content_boost));
    max_gain = std::max(max_gain, log2f(max_content_boost));
    if (fabs(max_gain - min_gain) < 1.0e-8) {
        max_gain += 0.1f; // to avoid div by zero during affine transform
    }
    spdlog::debug("Max/min gain (log2): {}/{}", max_gain, min_gain);
    spdlog::debug("Max/min gain (exp2): {}/{}", exp2f(max_gain),
                  exp2f(min_gain));

    for (size_t i = 0; i < gainmap.size(); i++) {
        // REVISIT:
        // gainmap[i] = AffineMapGain(gainmap[i], min_gain, max_gain,
        // map_gamma);
        gainmap[i] = AffineMapGain(gainmap[i], min_gain, max_gain, map_gamma);
        float mapped_gain = gainmap[i] * 255.f;

        affine_gainmap[i] =
            static_cast<uint8_t>(colorspace::Clip(mapped_gain + 0.5f, 0, 255));
    }

    // Save HDR linear image as NPY
    {
        const size_t shape[] = {height, width, channels};
        bool fortran_order = false;
        std::vector<float> hdr_linear_flat;
        hdr_linear_flat.reserve(width * height * channels);
        for (const auto &color : hdr_linear_image) {
            hdr_linear_flat.push_back(color.r);
            hdr_linear_flat.push_back(color.g);
            hdr_linear_flat.push_back(color.b);
        }
        std::string hdr_linear_path =
            output_dir + "/" + file_stem + "_hdr_linear.npy";
        npy::SaveArrayAsNumpy(hdr_linear_path, fortran_order, 3, shape,
                              hdr_linear_flat);
    }

    // DEPRECATE:
    // Save input HDR image as PNG
    {
        std::string hdr_path = output_dir + "/" + file_stem + "__input_hdr.png";
        imageops::WriteToPNG(hdr_image, hdr_path, error);
        if (error.raise)
            return;
    }

    // Save SDR image as PNG
    {
        std::unique_ptr<imageops::Image> sdr_png =
            std::make_unique<imageops::Image>();
        sdr_png->width = width;
        sdr_png->height = height;
        sdr_png->bit_depth = 8;
        sdr_png->color_type = PNG_COLOR_TYPE_RGB;
        sdr_png->channels = channels;
        sdr_png->metadata = hdr_image->metadata;
        sdr_png->metadata.oetf = colorspace::OETF::SRGB;
        sdr_png->metadata.gamut = colorspace::Gamut::BT709;

        // Allocate row pointers
        sdr_png->row_pointers = (png_bytep *)malloc(sizeof(png_bytep) * height);
        for (size_t y = 0; y < height; y++) {
            sdr_png->row_pointers[y] = (png_byte *)malloc(width * channels);
        }

        // Fill pixel data
        for (size_t y = 0; y < height; y++) {
            png_bytep row = sdr_png->row_pointers[y];
            for (size_t x = 0; x < width; x++) {
                size_t idx = y * width + x;
                const auto &color = sdr_image[idx];
                size_t pixel_idx = x * channels;
                row[pixel_idx] = static_cast<uint8_t>(
                    std::min(255.f, std::max(0.f, color.r * 255.f + 0.5f)));
                row[pixel_idx + 1] = static_cast<uint8_t>(
                    std::min(255.f, std::max(0.f, color.g * 255.f + 0.5f)));
                row[pixel_idx + 2] = static_cast<uint8_t>(
                    std::min(255.f, std::max(0.f, color.b * 255.f + 0.5f)));
            }
        }

        std::string sdr_path = output_dir + "/" + file_stem + "__sdr.png";
        imageops::WriteToPNG(sdr_png, sdr_path, error);
        if (error.raise)
            return;
    }

    // Save gainmap as NPY
    {
        const size_t shape[] = {height, width, 1}; // Single channel
        bool fortran_order = false;
        std::string gainmap_npy_path =
            output_dir + "/" + file_stem + "__gainmap.npy";
        npy::SaveArrayAsNumpy(gainmap_npy_path, fortran_order, 3, shape,
                              gainmap);
    }

    // Save affine gainmap as PNG
    {
        std::unique_ptr<imageops::Image> gainmap_png =
            std::make_unique<imageops::Image>();
        gainmap_png->width = width;
        gainmap_png->height = height;
        gainmap_png->bit_depth = 8;
        gainmap_png->color_type = PNG_COLOR_TYPE_GRAY;
        gainmap_png->channels = 1;

        // Allocate row pointers
        gainmap_png->row_pointers =
            (png_bytep *)malloc(sizeof(png_bytep) * height);
        for (size_t y = 0; y < height; y++) {
            gainmap_png->row_pointers[y] = (png_byte *)malloc(width);
        }

        // Fill pixel data
        for (size_t y = 0; y < height; y++) {
            png_bytep row = gainmap_png->row_pointers[y];
            for (size_t x = 0; x < width; x++) {
                size_t idx = y * width + x;
                row[x] = affine_gainmap[idx];
            }
        }

        std::string gainmap_png_path =
            output_dir + "/" + file_stem + "__gainmap.png";
        imageops::WriteToPNG(gainmap_png, gainmap_png_path, error);
    }

    // Save metadata as JSON
    {
        nlohmann::json metadata;
        metadata["max_gain"] = max_gain;
        metadata["max_content_boost"] = exp2f(max_gain);
        metadata["min_gain"] = min_gain;
        metadata["min_content_boost"] = exp2f(min_gain);
        metadata["map_gamma"] = map_gamma;
        metadata["hdr_offset"] = 0.015625f;
        metadata["sdr_offset"] = 0.015625f;
        metadata["clip_percentile"] = clip_percentile;
        metadata["hdr_capacity_min"] = log2f(1.0f);
        metadata["hdr_capacity_max"] =
            hdr_peak_nits / colorspace::SDR_WHITE_NITS;
        metadata["hdr_peak_nits"] = hdr_peak_nits;

        std::string metadata_path =
            output_dir + "/" + file_stem + "__metadata.json";
        std::ofstream metadata_file(metadata_path);
        if (!metadata_file.is_open()) {
            error = {true, "Failed to open metadata.json for writing"};
            return;
        }
        metadata_file << std::setw(4) << metadata << std::endl;
    }
}

// Steps
// - load sRGB sdr image, quantized
// - load gainmap, floating values, before quantization
// - inv tonemap (CURRENTLY NOT DOING)
// - inv srgb oetf to get linear sRGB
// -
void GainmapSDRToHDR(const std::unique_ptr<imageops::Image> &sdr_image,
                     const std::vector<float> gainmap,
                     const std::string &metadata, const std::string &file_stem,
                     const std::string &output_dir, utils::Error &error) {
    if (!sdr_image || !sdr_image->row_pointers) {
        error = {true, "Invalid input SDR image"};
        return;
    }

    std::ifstream metadata_file(metadata);
    if (!metadata_file.is_open()) {
        error = {true, "Failed to open metadata file: " + metadata};
        return;
    }

    nlohmann::json json_metadata;
    try {
        metadata_file >> json_metadata;
    } catch (const nlohmann::json::exception &e) {
        error = {true,
                 "Failed to parse metadata JSON: " + std::string(e.what())};
        return;
    }

    try {
        sdr_image->metadata.clip_percentile =
            json_metadata["clip_percentile"].get<float>();
        sdr_image->metadata.hdr_offset =
            json_metadata["hdr_offset"].get<float>();
        sdr_image->metadata.sdr_offset =
            json_metadata["sdr_offset"].get<float>();
        sdr_image->metadata.map_gamma = json_metadata["map_gamma"].get<float>();
        sdr_image->metadata.hdr_capacity_min =
            json_metadata["hdr_capacity_min"].get<float>();
        sdr_image->metadata.hdr_capacity_max =
            json_metadata["hdr_capacity_max"].get<float>();
        sdr_image->metadata.min_content_boost =
            json_metadata["min_content_boost"].get<float>();
        sdr_image->metadata.max_content_boost =
            json_metadata["max_content_boost"].get<float>();
    } catch (const nlohmann::json::exception &e) {
        error = {true, "Failed to read required fields from metadata: " +
                           std::string(e.what())};
        return;
    }

    colorspace::ColorTransformFn sdr_gamut_conv =
        colorspace::GetGamutConversionFn(colorspace::Gamut::BT2100,
                                         sdr_image->metadata.gamut);

    const size_t channels = 3;
    std::vector<colorspace::Color> hdr_image;
    for (size_t y = 0; y < sdr_image->height; y++) {
        png_bytep sdr_row = sdr_image->row_pointers[y];
        for (size_t x = 0; x < sdr_image->width; x++) {

            colorspace::Color sdr_rgb_gamma =
                ReadPixelFromRow(sdr_row, x, channels, sdr_image->bit_depth);

            colorspace::Color sdr_rgb = colorspace::sRGB_InvOETF(sdr_rgb_gamma);
            sdr_rgb = sdr_gamut_conv(sdr_rgb);

            colorspace::Color hdr_rgb = ApplyGain(
                sdr_rgb, gainmap[y * sdr_image->width + x],
                sdr_image->metadata.map_gamma,
                sdr_image->metadata.min_content_boost,
                sdr_image->metadata.max_content_boost,
                sdr_image->metadata.hdr_offset, sdr_image->metadata.sdr_offset);

            hdr_rgb =
                hdr_rgb * colorspace::SDR_WHITE_NITS / colorspace::HLG_MAX_NITS;
            hdr_rgb = Clamp(hdr_rgb);
            hdr_rgb = colorspace::HLG_InvOOTFApprox(hdr_rgb);
            colorspace::Color hdr_rgb_gamma = colorspace::HLG_OETF(hdr_rgb);
            hdr_image.push_back(hdr_rgb_gamma);
        }
    }

    // save hdr image as 16-bit png
    {
        std::unique_ptr<imageops::Image> hdr_png =
            std::make_unique<imageops::Image>();
        hdr_png->width = sdr_image->width;
        hdr_png->height = sdr_image->height;
        hdr_png->bit_depth = 16;
        hdr_png->color_type = PNG_COLOR_TYPE_RGB;
        hdr_png->channels = channels;
        hdr_png->metadata = sdr_image->metadata;
        hdr_png->metadata.gamut = colorspace::Gamut::BT2100;
        hdr_png->metadata.oetf = colorspace::OETF::HLG;

        // Allocate row pointers for 16-bit data
        hdr_png->row_pointers =
            (png_bytep *)malloc(sizeof(png_bytep) * hdr_png->height);
        for (size_t y = 0; y < hdr_png->height; y++) {
            hdr_png->row_pointers[y] = (png_byte *)malloc(
                hdr_png->width * channels * 2); // 2 bytes per channel
        }

        // Fill pixel data - convert float [0,1] to 16-bit integers
        for (size_t y = 0; y < hdr_png->height; y++) {
            png_bytep row = hdr_png->row_pointers[y];
            for (size_t x = 0; x < hdr_png->width; x++) {
                const auto &color = hdr_image[y * hdr_png->width + x];
                size_t pixel_idx = x * channels * 2; // 2 bytes per channel

                // Convert [0,1] float to 16-bit value and split into bytes
                uint16_t r = static_cast<uint16_t>(
                    std::min(65535.f, std::max(0.f, color.r * 65535.f + 0.5f)));
                uint16_t g = static_cast<uint16_t>(
                    std::min(65535.f, std::max(0.f, color.g * 65535.f + 0.5f)));
                uint16_t b = static_cast<uint16_t>(
                    std::min(65535.f, std::max(0.f, color.b * 65535.f + 0.5f)));

                // PNG requires 16-bit values to be stored in big-endian
                // (network byte order) regardless of the host's endianness
                row[pixel_idx] = (r >> 8) & 0xFF;     // MSB
                row[pixel_idx + 1] = r & 0xFF;        // LSB
                row[pixel_idx + 2] = (g >> 8) & 0xFF; // MSB
                row[pixel_idx + 3] = g & 0xFF;        // LSB
                row[pixel_idx + 4] = (b >> 8) & 0xFF; // MSB
                row[pixel_idx + 5] = b & 0xFF;        // LSB
            }
        }

        std::string hdr_path =
            output_dir + "/" + file_stem + "__reconstructed_hdr.png";
        imageops::WriteToPNG(hdr_png, hdr_path, error);
        if (error.raise)
            return;
    }
}

void CompareHDRToUHDR(const std::unique_ptr<imageops::Image> &hdr_image,
                      const std::unique_ptr<imageops::Image> &sdr_image,
                      const std::vector<float> affine_gainmap,
                      const std::string &metadata, const std::string &file_stem,
                      const std::string &output_dir, utils::Error &error) {
    if (!hdr_image || !hdr_image->row_pointers) {
        error = {true, "Invalid input HDR image"};
        return;
    }
    if (!sdr_image || !hdr_image->row_pointers) {
        error = {true, "Invalid input SDR image"};
        return;
    }

    std::ifstream metadata_file(metadata);
    if (!metadata_file.is_open()) {
        error = {true, "Failed to open metadata file: " + metadata};
        return;
    }

    nlohmann::json json_metadata;
    try {
        metadata_file >> json_metadata;
    } catch (const nlohmann::json::exception &e) {
        error = {true,
                 "Failed to parse metadata JSON: " + std::string(e.what())};
        return;
    }

    float hdr_offset, sdr_offset, map_gamma, min_content_boost,
        max_content_boost, hdr_peak_nits;
    try {
        hdr_offset = json_metadata["hdr_offset"].get<float>();
        sdr_offset = json_metadata["sdr_offset"].get<float>();
        map_gamma = json_metadata["map_gamma"].get<float>();
        min_content_boost = json_metadata["min_content_boost"].get<float>();
        max_content_boost = json_metadata["max_content_boost"].get<float>();
        hdr_peak_nits = json_metadata["hdr_peak_nits"].get<float>();
    } catch (const nlohmann::json::exception &e) {
        error = {true, "Failed to read required fields from metadata: " +
                           std::string(e.what())};
        return;
    }

    colorspace::Gamut hdr_gamut = hdr_image->metadata.gamut;
    colorspace::ColorTransformFn hdr_inv_oetf =
        colorspace::GetInvOETFFn(hdr_image->metadata.oetf);
    colorspace::SceneToDisplayLuminanceFn hdr_ootf =
        colorspace::GetOOTFFn(hdr_image->metadata.oetf);
    colorspace::LuminanceFn hdr_luminance_fn =
        colorspace::GetLuminanceFn(hdr_gamut);
    colorspace::ColorTransformFn hdr_gamut_conv =
        colorspace::GetGamutConversionFn(colorspace::Gamut::BT2100, hdr_gamut);
    colorspace::ColorTransformFn hdr_rgb2yuv =
        colorspace::GetRGBToYUVFn(hdr_gamut);

    colorspace::LuminanceFn bt2100_luminance_fn =
        colorspace::GetLuminanceFn(colorspace::Gamut::BT2100);

    colorspace::ColorTransformFn sdr_inv_oetf =
        colorspace::GetInvOETFFn(sdr_image->metadata.oetf);
    colorspace::ColorTransformFn sdr_hdr_gamut_conv =
        colorspace::GetGamutConversionFn(colorspace::Gamut::BT2100,
                                         sdr_image->metadata.gamut);

    const size_t width = hdr_image->width;
    const size_t height = hdr_image->height;
    const size_t channels = 3;
    std::vector<colorspace::Color> original_image, reconstructed_image;

    for (size_t y = 0; y < height; y++) {
        png_bytep hdr_row = hdr_image->row_pointers[y];
        png_bytep sdr_row = sdr_image->row_pointers[y];
        for (size_t x = 0; x < width; x++) {

            colorspace::Color hdr_rgb_gamma =
                ReadPixelFromRow(hdr_row, x, channels, hdr_image->bit_depth);
            colorspace::Color hdr_rgb = hdr_inv_oetf(hdr_rgb_gamma);
            hdr_rgb = hdr_ootf(hdr_rgb, hdr_luminance_fn);
            hdr_rgb = hdr_gamut_conv(hdr_rgb);
            hdr_rgb = colorspace::Clamp(hdr_rgb);
            colorspace::Color hdr_yuv = hdr_rgb2yuv(hdr_rgb);
            float hdr_luminance = bt2100_luminance_fn(hdr_rgb);
            original_image.push_back(
                colorspace::Color{{{hdr_luminance, 0.0f, 0.0f}}});

            colorspace::Color sdr_rgb_gamma =
                ReadPixelFromRow(sdr_row, x, channels, sdr_image->bit_depth);
            colorspace::Color sdr_rgb = sdr_inv_oetf(sdr_rgb_gamma);
            sdr_rgb = sdr_hdr_gamut_conv(sdr_rgb);
            float sdr_luminance =
                bt2100_luminance_fn(sdr_rgb) * colorspace::SDR_WHITE_NITS;
            float gain_log2 =
                UndoAffineMapGain(affine_gainmap[y * width + x], map_gamma,
                                  min_content_boost, max_content_boost);
            float recon_nits = RecomputeHDRLuminance(sdr_luminance, gain_log2,
                                                     hdr_offset, sdr_offset);

            // colorspace::Color recon_hdr_rgb = ApplyGain(
            //     sdr_rgb, gainmap[y * sdr_image->width + x],
            //     sdr_image->metadata.map_gamma,
            //     sdr_image->metadata.min_content_boost,
            //     sdr_image->metadata.max_content_boost,
            //     sdr_image->metadata.hdr_offset,
            //     sdr_image->metadata.sdr_offset);

            // recon_hdr_rgb = recon_hdr_rgb * colorspace::SDR_WHITE_NITS /
            //                 colorspace::HLG_MAX_NITS;
            // recon_hdr_rgb = Clamp(recon_hdr_rgb);
            reconstructed_image.push_back(
                colorspace::Color{{{recon_nits / hdr_peak_nits, 0.0f, 0.0f}}});
        }
    }

    float psnr = ComputePSNR(original_image, reconstructed_image);
    spdlog::info("PSNR between original HDR and reconstructed HDR: {:.8f} dB",
                 psnr);
}

} // namespace gainmap
