#include "imageio.hpp"
#include "imageops.hpp"
#include <iostream>
#include <string>
#include <getopt.h>
#include "utils.h"

#define PROGRAM_VERSION "1.0.0"

enum class ConversionMode { HDR_TO_RAW, HDR_TO_SDR, RAW_TO_SDR };

void PrintUsage(const char *programName) {
    std::cout << "Usage: " << programName << " [OPTIONS]\n"
              << "\nOptions:\n"
              << "  -r, --hdr2raw       Convert HDR to RAW\n"
              << "  -s, --hdr2sdr       Convert HDR to SDR\n"
              << "  -w, --raw2sdr       Convert RAW to SDR\n"
              << "  -i, --input=FILE    Input image file\n"
              << "  -o, --output=FILE   Output image file\n"
              << "  -h, --help          Display this help and exit\n"
              << "  -v, --version       Output version information and exit\n"
              << "\nExample:\n"
              << "  " << programName << " -i input.exr -o output.png --hdr2sdr\n";
}

void PrintVersion(const char *programName) {
    std::cout << programName << " version " << PROGRAM_VERSION << "\n"
              << "Copyright (C) 2025 RadiantFlow\n"
              << "License: MIT License\n";
}

int main(int argc, char *argv[]) {
    std::string input_file;
    std::string output_file;
    ConversionMode mode;
    bool mode_set = false;

    static struct option long_options[] = {
        {"help",     no_argument,       0, 'h'},
        {"version",  no_argument,       0, 'v'},
        {"input",    required_argument, 0, 'i'},
        {"output",   required_argument, 0, 'o'},
        {"hdr2raw",  no_argument,       0, 'r'},
        {"hdr2sdr",  no_argument,       0, 's'},
        {"raw2sdr",  no_argument,       0, 'w'},
        {0, 0, 0, 0}
    };

    int option_index = 0;
    int c;

    while ((c = getopt_long(argc, argv, "hvi:o:rsw", long_options, &option_index)) != -1) {
        switch (c) {
        case 'h':
            PrintUsage(argv[0]);
            return 0;
        case 'v':
            PrintVersion(argv[0]);
            return 0;
        case 'i':
            input_file = optarg;
            break;
        case 'o':
            output_file = optarg;
            break;
        case 'r':
            if (mode_set) {
                std::cerr << "Error: Only one mode can be specified\n";
                return 1;
            }
            mode = ConversionMode::HDR_TO_RAW;
            mode_set = true;
            break;
        case 's':
            if (mode_set) {
                std::cerr << "Error: Only one mode can be specified\n";
                return 1;
            }
            mode = ConversionMode::HDR_TO_SDR;
            mode_set = true;
            break;
        case 'w':
            if (mode_set) {
                std::cerr << "Error: Only one mode can be specified\n";
                return 1;
            }
            mode = ConversionMode::RAW_TO_SDR;
            mode_set = true;
            break;
        case '?':
            return 1;
        default:
            PrintUsage(argv[0]);
            return 1;
        }
    }



    // Validate required arguments
    if (!mode_set) {
        std::cerr << "Error: MODE is required\n";
        PrintUsage(argv[0]);
        return 1;
    }
    if (input_file.empty()) {
        std::cerr << "Error: Input file is required (--input/-i)\n";
        PrintUsage(argv[0]);
        return 1;
    }
    if (output_file.empty()) {
        std::cerr << "Error: Output file is required (--output/-o)\n";
        PrintUsage(argv[0]);
        return 1;
    }

    utils::Error error;
    imageio::HDRImage input_image;
    imageio::HDRImage output_image;

    std::cout << "Loading image: " << input_file << std::endl;
    input_image = imageio::LoadImage(input_file, error);
    if (error.raise) {
        std::cerr << "Failed to load image: " << error.message << std::endl;
        return 1;
    }

    imageops::HDRProcessingParams hdr_params{
        .exposure = 0.0f, .saturation = 1.0f, .contrast = 1.0f, .gamma = 2.2f};

    imageops::SDRConversionParams sdr_params{.max_nits = 100.0f,
                                             .target_gamma = 2.2f,
                                             .preserve_highlights = true,
                                             .knee_point = 0.75f};

    switch (mode) {
    case ConversionMode::HDR_TO_RAW:
        std::cout << "Converting HDR to RAW..." << std::endl;
        if (!imageops::HDRtoRAW(input_image, hdr_params, error)) {
            std::cerr << "Failed to convert HDR to RAW: " << error.message << std::endl;
            return 1;
        }
        output_image = input_image;
        break;

    case ConversionMode::HDR_TO_SDR:
        std::cout << "Converting HDR to SDR..." << std::endl;
        if (!imageops::HDRtoRAW(input_image, hdr_params, error)) {
            std::cerr << "Failed to process HDR image: " << error.message << std::endl;
            return 1;
        }
        output_image = imageops::HDRtoSDR(input_image, sdr_params, error);
        if (error.raise) {
            std::cerr << "Failed to convert to SDR: " << error.message << std::endl;
            return 1;
        }
        break;

    case ConversionMode::RAW_TO_SDR:
        std::cout << "Converting RAW to SDR..." << std::endl;
        output_image = imageops::RAWtoSDR(input_image, sdr_params, error);
        if (error.raise) {
            std::cerr << "Failed to convert RAW to SDR: " << error.message << std::endl;
            return 1;
        }
        break;
    }

    std::cout << "Saving output image: " << output_file << std::endl;
    if (!imageio::SaveImage(output_image, output_file,
                            imageio::HDRFormat::PNG_HDR,
                            imageio::ImageMetadata{}, error)) {
        std::cerr << "Failed to save image: " << error.message << std::endl;
        return 1;
    }

    std::cout << "Conversion completed successfully!" << std::endl;
    return 0;
}
