#include <iostream>
#include "../src/imageops.hpp"
#include "../src/colorspace.hpp"
#include "../src/gainmap.hpp"
#include "../src/utils.h"
#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include "npy.hpp"

TEST_CASE("Color Space Conversions", "[imageops]") {
    double base = 10.0;
    double x;
    
    // Test transfer functions
    SECTION("Transfer Functions") {
        for (int i = 0; i < 10; i++) {
            x = static_cast<double>(i + 1) / base;
            // // Test HLG transfer function
            // double hlg2l = colorspace::Bt2100HLGToLinear(x);
            // double l2hlg = colorspace::LinearToBt2100HLG(hlg2l);
            // REQUIRE_THAT(l2hlg, Catch::Matchers::WithinAbs(x, 1e-6));
            // l2hlg = colorspace::LinearToBt2100HLG(x);
            // hlg2l = colorspace::Bt2100HLGToLinear(l2hlg);
            // REQUIRE_THAT(hlg2l, Catch::Matchers::WithinAbs(x, 1e-6));

            // // Test sRGB transfer function
            // double srgb2l = colorspace::sRGBToLinear(x);
            // double l2srgb = colorspace::LinearTosRGB(srgb2l);
            // REQUIRE_THAT(l2srgb, Catch::Matchers::WithinAbs(x, 1e-6));
            // l2srgb = colorspace::LinearTosRGB(x);
            // srgb2l = colorspace::sRGBToLinear(l2srgb);
            // REQUIRE_THAT(srgb2l, Catch::Matchers::WithinAbs(x, 1e-6));

            // // Test Rec2020 Gamma transfer function
            // double gamma2l = colorspace::Rec2020ToLinear(x);
            // double l2gamma = colorspace::LinearToRec2020(gamma2l);
            // REQUIRE_THAT(l2gamma, Catch::Matchers::WithinAbs(x, 1e-6));
            // l2gamma = colorspace::LinearToRec2020(x);
            // gamma2l = colorspace::Rec2020ToLinear(l2gamma);
            // REQUIRE_THAT(gamma2l, Catch::Matchers::WithinAbs(x, 1e-6));

            // // Test P3 PQ transfer function
            // double pq2l = colorspace::Bt2100PQToLinear(x);
            // double l2pq = colorspace::LinearToBt2100PQ(pq2l);
            // REQUIRE_THAT(l2pq, Catch::Matchers::WithinAbs(x, 1e-6));
            // l2pq = colorspace::LinearToBt2100PQ(x);
            // pq2l = colorspace::Bt2100PQToLinear(l2pq);
            // REQUIRE_THAT(pq2l, Catch::Matchers::WithinAbs(x, 1e-6));
        }
    }
}

TEST_CASE("Image Ops", "[hdrtosdr]") {
    utils::Error error;

    std::unique_ptr<imageops::Image> image =
        imageops::LoadHDRPNG("./images/test_hdr_png.png", error);
    if (error.raise) {
        INFO(error.message);
    }
    REQUIRE_FALSE(error.raise);
    std::vector<std::pair<colorspace::ToneMapping, std::string>> tone_mappers = {
        {colorspace::ToneMapping::BASE, "base"},
        {colorspace::ToneMapping::REINHARD, "reinhard"},
        {colorspace::ToneMapping::GAMMA, "gamma"},
        {colorspace::ToneMapping::FILMIC, "filmic"},
        {colorspace::ToneMapping::ACES, "aces"},
        {colorspace::ToneMapping::UNCHARTED2, "uncharted2"},
        {colorspace::ToneMapping::DRAGO, "drago"},
        {colorspace::ToneMapping::LOTTES, "lottes"},
        {colorspace::ToneMapping::HABLE, "hable"}};

    for (const auto &[tone_mapper, name] : tone_mappers) {
        std::cout << "Running " << name << std::endl;
        // DEPRECATE:
        // std::unique_ptr<imageops::Image> sdr_image =
        //    colorspace::HDRToSDR(image, 0.0, 1.0, error, tone_mapper);
        // REQUIRE_FALSE(error.raise);
        // std::string output_path = "./images/test_hdr_png_" + name + ".png";
        // bool ret = imageops::WriteToPNG(sdr_image, output_path, error);
        // if (error.raise) {
        //     INFO("Failed to write " + name + " tone mapping: " + error.message);
        // }
        // REQUIRE_FALSE(error.raise);
        // REQUIRE(ret);
    }
}

// TEST_CASE("Gainmap", "[gainmap]") {
//     utils::Error error;
// 
//     std::unique_ptr<imageops::Image> image =
//         imageops::LoadHDRPNG("./images/test_hdr_png.png", error);
//     if (error.raise) {
//         INFO(error.message);
//     }
//     REQUIRE_FALSE(error.raise);
// 
//     std::unique_ptr<imageops::Image> gainmap = gainmap::HDRToGainMap(image, 0.015625, 0.015625, 1.0, 4.0, 1.0, error);
//     std::string output_path = "./images/gainmap.png";
//     bool ret = imageops::WriteToPNG(gainmap, output_path, error);
//     if (error.raise) {
//         INFO("Failed to write gainmap: " + error.message);
//     }
//     REQUIRE_FALSE(error.raise);
//     REQUIRE(ret);
// }
