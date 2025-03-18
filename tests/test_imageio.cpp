#include "../src/imageops.hpp"
#include "../src/utils.h"
#include <catch2/catch_test_macros.hpp>

TEST_CASE("PNG", "[imageops]") {
    utils::Error error;

    std::unique_ptr<imageops::PNGImage> image =
        imageops::LoadHDRPNG("./images/test_hdr_png.png", error);
    if (error.raise) {
        INFO(error.message);
    }
    std::vector<std::pair<imageops::ToneMapping, std::string>> tone_mappers = {
        {imageops::ToneMapping::REINHARD, "reinhard"},
        {imageops::ToneMapping::GAMMA, "gamma"},
        {imageops::ToneMapping::FILMIC, "filmic"},
        {imageops::ToneMapping::ACES, "aces"},
        {imageops::ToneMapping::UNCHARTED2, "uncharted2"},
        {imageops::ToneMapping::DRAGO, "drago"},
        {imageops::ToneMapping::LOTTES, "lottes"},
        {imageops::ToneMapping::HABLE, "hable"}};

    for (const auto &[tone_mapper, name] : tone_mappers) {
        imageops::SDRConversionParams sdr_params{.tone_mapping = tone_mapper};
        imageops::ImageMetadata metadata = imageops::ReadHDRPNGMetadata("./images/test_hdr_png.png", error);
        // imageops::UpdateSDRParams(sdr_params, metadata);
        std::unique_ptr<imageops::PNGImage> sdr_image =
            imageops::HDRtoSDR(image, sdr_params, error);
        REQUIRE_FALSE(error.raise);

        std::string output_path = "./images/test_hdr_png_" + name + ".png";
        bool ret = imageops::WritetoPNG(sdr_image, output_path, error);
        if (error.raise) {
            INFO("Failed to write " + name + " tone mapping: " + error.message);
        }
        REQUIRE_FALSE(error.raise);
        REQUIRE(ret);
    }
}
