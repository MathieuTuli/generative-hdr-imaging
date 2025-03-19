#include "../src/imageops.hpp"
#include "../src/utils.h"
#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

TEST_CASE("Helpers", "[imageops]") {
    double base = 10.0;
    double x;
    for (int i = 0; i < 10; i++){
        x = static_cast<double>(i + 1) / base;
        double hlg2l = imageops::HLGtoLinear(x);
        double l2hlg = imageops::LineartoHLG(hlg2l);
        INFO("Base " << x << "| HLG2L " << hlg2l << " | L2HLG " << l2hlg);
        REQUIRE_THAT(l2hlg, Catch::Matchers::WithinRel(x, 1e-6));
        l2hlg = imageops::LineartoHLG(x);
        hlg2l = imageops::HLGtoLinear(l2hlg);
        INFO("Base " << x << "| L2HLG " << l2hlg << "| HLG2L " << hlg2l);
        REQUIRE_THAT(hlg2l, Catch::Matchers::WithinRel(x, 1e-6));
    }

    SECTION("Invalid inputs") {
        REQUIRE_THROWS_AS(imageops::HLGtoLinear(-1.0), std::runtime_error);
        REQUIRE_THROWS_AS(imageops::LineartoHLG(-1.0), std::runtime_error);

        REQUIRE_THROWS_AS(imageops::HLGtoLinear(std::numeric_limits<double>::quiet_NaN()), std::runtime_error);
        REQUIRE_THROWS_AS(imageops::LineartoHLG(std::numeric_limits<double>::quiet_NaN()), std::runtime_error);

        REQUIRE_THROWS_AS(imageops::HLGtoLinear(std::numeric_limits<double>::infinity()), std::runtime_error);
        REQUIRE_THROWS_AS(imageops::LineartoHLG(std::numeric_limits<double>::infinity()), std::runtime_error);
    }
}

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
