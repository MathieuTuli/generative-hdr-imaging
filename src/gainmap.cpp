#include "gainmap.hpp"
#include "npy.hpp"
#include "utils.h"
#include <cmath>
#include <fstream>
#include <iostream>
#include <nlohmann/json.hpp>

namespace gainmap {
float ComputeGain(float hdr_y_nits, float sdr_y_nits,
                  float hdr_offset = 0.015625f, float sdr_offset = 0.015625f) {
    float gain = log2((hdr_y_nits + hdr_offset) / (sdr_y_nits + sdr_offset));
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
                    float gamma) {
    float mapped_val =
        (gainlog2 - min_gainlog2) / (max_gainlog2 - min_gainlog2);
    if (gamma != 1.0f)
        mapped_val = pow(mapped_val, gamma);
    return mapped_val;
}

// The pipeline is as follows:
// HDR -> linear (inv OETF)
// -> Rec2020 [save]
// linear -> gamma (Bt100 HLG OETF)
// gamma -> YUV [save]

// Rec2020 (linear) -> Bt709
// -> gammut map (clip)
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
                  float clip_percentile, float map_gamma, utils::Error &error) {
    if (!hdr_image || !hdr_image->row_pointers) {
        error = {true, "Invalid input HDR image"};
        return;
    }

    colorspace::Gamut hdr_gamut = hdr_image->metadata.gamut;
    colorspace::ColorTransformFn hdr_inv_oetf =
        colorspace::GetInvOETFFn(hdr_image->metadata.oetf);
    colorspace::SceneToDisplayLuminanceFn hdr_ootf =
        colorspace::GetOOTFFn(hdr_image->metadata.oetf);
    colorspace::ColorTransformFn hdr_gammut_conv =
        colorspace::GetGamutConversionFn(colorspace::Gamut::BT2100, hdr_gamut);
    colorspace::ColorTransformFn hdr_rgb2yuv =
        colorspace::GetRGBToYUVFn(hdr_gamut);
    colorspace::ColorTransformFn hdr_yuv2rgb =
        colorspace::GetRGBToYUVFn(hdr_gamut);
    colorspace::LuminanceFn hdr_luminance_fn =
        colorspace::GetLuminanceFn(hdr_gamut);
    colorspace::LuminanceFn bt2100_luminance_fn =
        colorspace::GetLuminanceFn(colorspace::Gamut::BT2100);
    float hdr_peaknits = colorspace::GetReferenceDisplayPeakLuminanceInNits(
        hdr_image->metadata.oetf);
    colorspace::ColorTransformFn sdr_gammut_conv =
        colorspace::GetGamutConversionFn(

            colorspace::Gamut::BT709, colorspace::Gamut::BT2100);
    colorspace::ColorTransformFn sdr_inv_oetf =
        colorspace::GetInvOETFFn(colorspace::OETF::SRGB);
    colorspace::ColorTransformFn sdr_oetf =
        colorspace::GetOETFFn(colorspace::OETF::SRGB);
    colorspace::ColorTransformFn sdr_hdr_gamut_conv =
        colorspace::GetGamutConversionFn(colorspace::Gamut::BT709,
                                         colorspace::Gamut::BT2100);

    std::cout << "Initalized conversion functions" << std::endl;
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
    std::cout << "Processing hdr..." << std::endl;
    for (size_t y = 0; y < height; y++) {
        png_bytep hdr_row = hdr_image->row_pointers[y];
        for (size_t x = 0; x < width; x++) {

            colorspace::Color hdr_rgb_gamma;
            // round up to nearest byte
            size_t bytes_per_channel = (hdr_image->bit_depth + 7) / 8;
            size_t idx = x * channels * bytes_per_channel;

            // read values based on bit depth
            uint32_t r_value = 0, g_value = 0, b_value = 0;
            if (bytes_per_channel == 1) {
                // 8-bit values
                r_value = hdr_row[idx];
                g_value = hdr_row[idx + 1];
                b_value = hdr_row[idx + 2];
            } else if (bytes_per_channel == 2) {
                // 10-bit, 12-bit, or 16-bit values stored in 2 bytes
                r_value = (hdr_row[idx] << 8) | hdr_row[idx + 1];
                g_value = (hdr_row[idx + 2] << 8) | hdr_row[idx + 3];
                b_value = (hdr_row[idx + 4] << 8) | hdr_row[idx + 5];
            }

            // create a mask for the actual bit depth
            uint32_t max_value = (1 << hdr_image->bit_depth) - 1;

            // mask and normalize to [0, 1]
            hdr_rgb_gamma.r =
                static_cast<float>(r_value & max_value) / max_value;
            hdr_rgb_gamma.g =
                static_cast<float>(g_value & max_value) / max_value;
            hdr_rgb_gamma.b =
                static_cast<float>(b_value & max_value) / max_value;

            colorspace::Color hdr_rgb = hdr_inv_oetf(hdr_rgb_gamma);
            hdr_rgb = hdr_ootf(hdr_rgb, hdr_luminance_fn);
            hdr_rgb = hdr_gammut_conv(hdr_rgb);
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
    std::cout << "HDR " << clip_percentile
              << "th-percentile clip value: " << clip_value << std::endl;

    for (size_t i = 0; i < hdr_linear_image.size(); i++) {
        colorspace::Color hdr_rgb = hdr_linear_image[i];
        colorspace::Color sdr_rgb = sdr_gammut_conv(hdr_rgb);
        colorspace::Clamp(sdr_rgb);

        // NOTE: apply 99% clipping
        sdr_rgb.r = std::min(sdr_rgb.r, clip_value) / clip_value;
        sdr_rgb.g = std::min(sdr_rgb.g, clip_value) / clip_value;
        sdr_rgb.b = std::min(sdr_rgb.b, clip_value) / clip_value;

        colorspace::Color srgb_gamma = sdr_oetf(sdr_rgb);
        // TODO: apply tone map
        sdr_image.push_back(srgb_gamma);

        uint8_t r = static_cast<uint8_t>(srgb_gamma.r * 255.f);
        uint8_t g = static_cast<uint8_t>(srgb_gamma.g * 255.f);
        uint8_t b = static_cast<uint8_t>(srgb_gamma.b * 255.f);

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
        hdr_y_nits = bt2100_luminance_fn(hdr_rgb) * hdr_peaknits;
        float gain = ComputeGain(hdr_y_nits, sdr_y_nits);
        min_gain = std::min(gain, min_gain);
        max_gain = std::max(gain, max_gain);
        gainmap.push_back(gain);
    }
    std::cout << "Gainmap computed." << std::endl;

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
    float max_content_boost = hdr_peaknits / colorspace::SDR_WHITE_NITS;
    min_gain = std::min(min_gain, min_content_boost);
    max_gain = std::max(max_gain, max_content_boost);
    if (fabs(max_gain - min_gain) < 1.0e-8) {
        max_gain += 0.1f; // to avoid div by zero during affine transform
    }
    std::cout << "Max/min gain (log2): " << max_gain << "/" << min_gain
              << std::endl;
    std::cout << "Max/min gain (exp2): " << exp2(max_gain) << "/"
              << exp2(min_gain) << std::endl;

    std::cout << "Computing affine gainmap..." << std::endl;
    for (size_t i = 0; i < gainmap.size(); i++) {
        gainmap[i] = AffineMapGain(gainmap[i], min_gain, max_gain, map_gamma);

        float mapped_gain = gainmap[i] * 255.f;
        affine_gainmap[i] =
            static_cast<uint8_t>(colorspace::Clip(mapped_gain + 0.5f, 0, 255));
    }

    std::cout << "Saving images and npy..." << std::endl;
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
        npy::SaveArrayAsNumpy("hdr_linear.npy", fortran_order, 3, shape,
                              hdr_linear_flat);
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
                row[pixel_idx] = static_cast<uint8_t>(color.r * 255.f);
                row[pixel_idx + 1] = static_cast<uint8_t>(color.g * 255.f);
                row[pixel_idx + 2] = static_cast<uint8_t>(color.b * 255.f);
            }
        }

        imageops::WriteToPNG(sdr_png, "sdr.png", error);
        if (error.raise)
            return;
    }

    // Save gainmap as NPY
    {
        const size_t shape[] = {height, width, 1}; // Single channel
        bool fortran_order = false;
        npy::SaveArrayAsNumpy("gainmap.npy", fortran_order, 3, shape, gainmap);
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

        imageops::WriteToPNG(gainmap_png, "gainmap.png", error);
    }

    // Save metadata as JSON
    {
        nlohmann::json metadata;
        metadata["max_gain_log2"] = max_gain;
        metadata["max_gain_exp2"] = exp2(max_gain);
        metadata["min_gain_log2"] = min_gain;
        metadata["min_gain_exp2"] = exp2(min_gain);
        metadata["map_gamma"] = map_gamma;
        metadata["hdr_offset"] = 0.015625f;
        metadata["sdr_offset"] = 0.015625f;
        metadata["hdr_capacity_min"] = 1.0f;
        metadata["hdr_capacity_max"] =
            hdr_peaknits / colorspace::SDR_WHITE_NITS;

        std::ofstream metadata_file("metadata.json");
        if (!metadata_file.is_open()) {
            error = {true, "Failed to open metadata.json for writing"};
            return;
        }
        metadata_file << std::setw(4) << metadata << std::endl;
    }
}
} // namespace gainmap
