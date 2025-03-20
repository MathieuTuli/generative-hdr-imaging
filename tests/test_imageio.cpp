#include "../src/imageops.hpp"
#include "../src/utils.h"
#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

TEST_CASE("Helpers", "[imageops]") {
    double base = 10.0;
    double x;
    for (int i = 0; i < 10; i++){
        x = static_cast<double>(i + 1) / base;

        double hlg2l = imageops::Rec2020HLGToLinear(x);
        double l2hlg = imageops::LinearToRec2020HLG(hlg2l);
        REQUIRE_THAT(l2hlg, Catch::Matchers::WithinRel(x, 1e-6));
        l2hlg = imageops::LinearToRec2020HLG(x);
        hlg2l = imageops::Rec2020HLGToLinear(l2hlg);
        REQUIRE_THAT(hlg2l, Catch::Matchers::WithinRel(x, 1e-6));

        double srgb2l = imageops::sRGBToLinear(x);
        double l2srgb = imageops::LinearTosRGB(srgb2l);
        REQUIRE_THAT(l2srgb, Catch::Matchers::WithinRel(x, 1e-6));
        l2srgb = imageops::LinearTosRGB(x);
        srgb2l = imageops::sRGBToLinear(l2srgb);
        REQUIRE_THAT(srgb2l, Catch::Matchers::WithinRel(x, 1e-6));
    }

    SECTION("Invalid inputs") {
        REQUIRE_THROWS_AS(imageops::Rec2020HLGToLinear(-1.0), std::runtime_error);
        REQUIRE_THROWS_AS(imageops::LinearToRec2020HLG(-1.0), std::runtime_error);

        REQUIRE_THROWS_AS(imageops::Rec2020HLGToLinear(std::numeric_limits<double>::quiet_NaN()), std::runtime_error);
        REQUIRE_THROWS_AS(imageops::LinearToRec2020HLG(std::numeric_limits<double>::quiet_NaN()), std::runtime_error);

        REQUIRE_THROWS_AS(imageops::Rec2020HLGToLinear(std::numeric_limits<double>::infinity()), std::runtime_error);
        REQUIRE_THROWS_AS(imageops::LinearToRec2020HLG(std::numeric_limits<double>::infinity()), std::runtime_error);

        REQUIRE_THROWS_AS(imageops::sRGBToLinear(-1.0), std::runtime_error);
        REQUIRE_THROWS_AS(imageops::LinearTosRGB(-1.0), std::runtime_error);

        REQUIRE_THROWS_AS(imageops::sRGBToLinear(std::numeric_limits<double>::quiet_NaN()), std::runtime_error);
        REQUIRE_THROWS_AS(imageops::LinearTosRGB(std::numeric_limits<double>::quiet_NaN()), std::runtime_error);

        REQUIRE_THROWS_AS(imageops::sRGBToLinear(std::numeric_limits<double>::infinity()), std::runtime_error);
        REQUIRE_THROWS_AS(imageops::LinearTosRGB(std::numeric_limits<double>::infinity()), std::runtime_error);
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
        {imageops::ToneMapping::BASE, "base"},
        {imageops::ToneMapping::REINHARD, "reinhard"},
        {imageops::ToneMapping::GAMMA, "gamma"},
        {imageops::ToneMapping::FILMIC, "filmic"},
        {imageops::ToneMapping::ACES, "aces"},
        {imageops::ToneMapping::UNCHARTED2, "uncharted2"},
        {imageops::ToneMapping::DRAGO, "drago"},
        {imageops::ToneMapping::LOTTES, "lottes"},
        {imageops::ToneMapping::HABLE, "hable"}};

    for (const auto &[tone_mapper, name] : tone_mappers) {
        std::unique_ptr<imageops::PNGImage> sdr_image =
            imageops::HDRToSDR(image, 0.0, 1.0, error, tone_mapper);
        REQUIRE_FALSE(error.raise);

        std::string output_path = "./images/test_hdr_png_" + name + ".png";
        bool ret = imageops::WriteToPNG(sdr_image, output_path, error);
        if (error.raise) {
            INFO("Failed to write " + name + " tone mapping: " + error.message);
        }
        REQUIRE_FALSE(error.raise);
        REQUIRE(ret);
    }
}
