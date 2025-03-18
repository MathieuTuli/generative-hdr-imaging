#include "../src/imageops.hpp"
#include "../src/utils.h"
#include <catch2/catch_test_macros.hpp>

TEST_CASE("PNG", "[imageops]") {
    utils::Error error;

    std::unique_ptr<imageops::PNGImage> image =
        imageops::LoadHDRPNG("./assets/test_hdr_png.png", error);
    if (error.raise) {
        INFO(error.message);
    }
    REQUIRE_FALSE(error.raise);
    bool ret = imageops::WritetoPNG(image, "./assets/test_hdr_png_raw.png", error);
    if (error.raise) {
        INFO(error.message);
    }
    REQUIRE_FALSE((error.raise && ret));

    // imageops::ImageMetadata metadata;
    // imageops::HDRProcessingParams hdr_params{
    //     .exposure = 0.0f, .saturation = 1.0f, .contrast = 1.0f, .gamma
    //     = 2.2f};
    // std::unique_ptr<imageops::RawImageData> raw_img =
    //     imageops::HDRPNGtoRAW(image, hdr_params, error);
    // if (error.raise) {
    //     INFO(error.message);
    // }
    // REQUIRE_FALSE(error.raise);
    // imageops::WritetoRAW(raw_img, "./assets/test_hdr_png_raw.png", metadata,
    //                      error);
    // if (error.raise) {
    //     INFO(error.message);
    // }
    // REQUIRE_FALSE(error.raise);
}
