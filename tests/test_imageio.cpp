#include <iostream>
#include "../src/imageops.hpp"
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

            // Test HLG transfer function
            double hlg2l = imageops::Rec2020HLGToLinear(x);
            double l2hlg = imageops::LinearToRec2020HLG(hlg2l);
            REQUIRE_THAT(l2hlg, Catch::Matchers::WithinAbs(x, 1e-6));
            l2hlg = imageops::LinearToRec2020HLG(x);
            hlg2l = imageops::Rec2020HLGToLinear(l2hlg);
            REQUIRE_THAT(hlg2l, Catch::Matchers::WithinAbs(x, 1e-6));

            // Test sRGB transfer function
            double srgb2l = imageops::sRGBToLinear(x);
            double l2srgb = imageops::LinearTosRGB(srgb2l);
            REQUIRE_THAT(l2srgb, Catch::Matchers::WithinAbs(x, 1e-6));
            l2srgb = imageops::LinearTosRGB(x);
            srgb2l = imageops::sRGBToLinear(l2srgb);
            REQUIRE_THAT(srgb2l, Catch::Matchers::WithinAbs(x, 1e-6));

            // Test Rec2020 Gamma transfer function
            double gamma2l = imageops::Rec2020GammaToLinear(x);
            double l2gamma = imageops::LinearToRec2020Gamma(gamma2l);
            REQUIRE_THAT(l2gamma, Catch::Matchers::WithinAbs(x, 1e-6));
            l2gamma = imageops::LinearToRec2020Gamma(x);
            gamma2l = imageops::Rec2020GammaToLinear(l2gamma);
            REQUIRE_THAT(gamma2l, Catch::Matchers::WithinAbs(x, 1e-6));

            // Test P3 PQ transfer function
            double pq2l = imageops::P3PQToLinear(x);
            double l2pq = imageops::LinearToP3PQ(pq2l);
            REQUIRE_THAT(l2pq, Catch::Matchers::WithinAbs(x, 1e-6));
            l2pq = imageops::LinearToP3PQ(x);
            pq2l = imageops::P3PQToLinear(l2pq);
            REQUIRE_THAT(pq2l, Catch::Matchers::WithinAbs(x, 1e-6));
        }
    }

    // Test RGB<->XYZ conversions
    SECTION("RGB-XYZ Conversions") {
        for (int i = 0; i < 10; i++) {
            x = static_cast<double>(i + 1) / base;
            std::vector<double> rgb = {x, x, x};

            // Test Rec2020 RGB<->XYZ
            std::vector<double> xyz = imageops::LinearRec2020ToXYZ(rgb);
            std::vector<double> rgb_back = imageops::XYZToLinearRec2020(xyz);
            for (size_t j = 0; j < 3; j++) {
                REQUIRE_THAT(rgb_back[j], Catch::Matchers::WithinAbs(rgb[j], 1e-6));
            }

            // REVISIT: for some reason this is way less precise that counterparts?
            // Test P3 RGB<->XYZ
            xyz = imageops::LinearP3ToXYZ(rgb);
            rgb_back = imageops::XYZToLinearP3(xyz);
            for (size_t j = 0; j < 3; j++) {
                REQUIRE_THAT(rgb_back[j], Catch::Matchers::WithinAbs(rgb[j], 1e-3));
            }

            // Test sRGB<->XYZ
            xyz = imageops::LinearsRGBToXYZ(rgb);
            rgb_back = imageops::XYZToLinearsRGB(xyz);
            for (size_t j = 0; j < 3; j++) {
                REQUIRE_THAT(rgb_back[j], Catch::Matchers::WithinAbs(rgb[j], 1e-6));
            }
        }
    }

    // Test RGB<->XYZ conversions
    SECTION("REC2020-RGB-XYZ Conversions") {
        for (int i = 0; i < 10; i++) {
            x = static_cast<double>(i + 1) / base;
            std::vector<double> rgb = {x, x, x};

            std::vector<double> xyz = imageops::LinearRec2020ToXYZ(rgb);
            xyz = imageops::XYZToLinearsRGB(xyz);
            xyz = imageops::LinearsRGBToXYZ(xyz);
            std::vector<double> rgb_back = imageops::XYZToLinearRec2020(xyz);
            for (size_t j = 0; j < 3; j++) {
                REQUIRE_THAT(rgb_back[j], Catch::Matchers::WithinAbs(rgb[j], 1e-6));
            }
        }
    }

    // // Test XYZ->YUV conversions
    SECTION("XYZ-YUV Conversions") {
        for (int i = 0; i < 10; i++) {
            x = static_cast<double>(i + 1) / base;
            std::vector<double> rgb = {x, x, x};

            std::vector<double> xyz = imageops::LinearRec2020ToXYZ(rgb);
            std::vector<double> yuv1 = imageops::XYZToRec2020YUV(xyz);

            xyz = imageops::XYZToLinearsRGB(xyz);
            xyz = imageops::LinearsRGBToXYZ(xyz);
            std::vector<double> yuv2 = imageops::XYZToRec2020YUV(xyz);

            for (size_t j = 0; j < 3; j++) {
                // double diff = std::abs(yuv1[j] - yuv2[j]);
                // double relDiff = diff / std::max(std::abs(yuv1[j]), std::abs(yuv2[j]));
                REQUIRE_THAT(yuv1[j], Catch::Matchers::WithinAbs(yuv2[j], 1e-6));
            }
        }
    }
    std::vector<double> yuv = imageops::XYZToBt709YUV({0.1, 0.1, 0.1});
    std::cout << "BT709 " << yuv[0] << "," << yuv[1] << "," << yuv[2] << std::endl;
    yuv = imageops::XYZToRec2020YUV({0.1, 0.1, 0.1});
    std::cout << "REC2020 " << yuv[0] << "," << yuv[1] << "," << yuv[2] << std::endl;
}

TEST_CASE("PNG", "[hdrtosdr]") {
    utils::Error error;

    std::unique_ptr<imageops::PNGImage> image =
        imageops::LoadHDRPNG("./images/test_hdr_png.png", error);
    if (error.raise) {
        INFO(error.message);
    }
    REQUIRE_FALSE(error.raise);
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
        std::cout << "Running " << name << std::endl;
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

TEST_CASE("Gainmap", "[gainmap]") {
    utils::Error error;

    std::unique_ptr<imageops::PNGImage> image =
        imageops::LoadHDRPNG("./images/test_hdr_png.png", error);
    if (error.raise) {
        INFO(error.message);
    }
    REQUIRE_FALSE(error.raise);

    std::unique_ptr<imageops::PNGImage> gainmap = gainmap::HDRToGainMap(image, 0.015625, 0.015625, 1.0, 4.0, 1.0, error);
    std::string output_path = "./images/gainmap.png";
    bool ret = imageops::WriteToPNG(gainmap, output_path, error);
    if (error.raise) {
        INFO("Failed to write gainmap: " + error.message);
    }
    REQUIRE_FALSE(error.raise);
    REQUIRE(ret);
}
