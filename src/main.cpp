#include "gainmap.hpp"
#include "imageops.hpp"
#include "utils.h"
#include <getopt.h>
#include <iostream>
#include <string>

#define PROGRAM_VERSION "1.0.0"

enum class ConversionMode { HDR_TO_YUV, HDR_TO_SDR, HDR_TO_UHDR };

void PrintUsage(const char *programName) {
    std::cout
        << "Usage: " << programName << " [OPTIONS]\n"
        << "\nOptions:\n"
        << "  -u, --hdr2uhdr       Convert HDR to SDR + Gainmap\n"
        << "  -i, --input=FILE    Input image file\n"
        << "  -o, --output=FILE   Output image file\n"
        << "  -p, --percentile=N  Clip percentile for gainmap (default: 0.95)\n"
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
    float clip_percentile = 0.95f;

    static struct option long_options[] = {
        {"help", no_argument, 0, 'h'},
        {"version", no_argument, 0, 'v'},
        {"input", required_argument, 0, 'i'},
        {"output", required_argument, 0, 'o'},
        {"hdr2raw", no_argument, 0, 'r'},
        {"hdr2sdr", no_argument, 0, 's'},
        {"percentile", required_argument, 0, 'p'},
        {0, 0, 0, 0}};

    int option_index = 0;
    int c;

    while ((c = getopt_long(argc, argv, "hvi:o:up:", long_options,
                            &option_index)) != -1) {
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
        case 'u':
            if (mode_set) {
                std::cerr << "Error: Only one mode can be specified\n";
                return 1;
            }
            mode = ConversionMode::HDR_TO_UHDR;
            mode_set = true;
            break;
        case 'p':
            clip_percentile = std::stof(optarg);
            if (clip_percentile <= 0.0f || clip_percentile >= 1.0f) {
                std::cerr << "Error: Percentile must be between 0 and 1\n";
                return 1;
            }
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
    std::cout << "Loading image: " << input_file << std::endl;
    std::unique_ptr<imageops::Image> hdr_image =
        imageops::LoadImage(input_file, error);
    if (error.raise) {
        std::cerr << "Failed to load image: " << error.message << std::endl;
        return 1;
    }

    std::unique_ptr<imageops::Image> output_image;
    switch (mode) {
    case ConversionMode::HDR_TO_YUV:
        std::cout << "Converting HDR to YUV..." << std::endl;
        // TODO:
        if (error.raise) {
            std::cerr << "Failed to convert to SDR: " << error.message
                      << std::endl;
            return 1;
        }
        break;

    case ConversionMode::HDR_TO_SDR:
        std::cout << "Converting HDR to SDR..." << std::endl;
        // TODO:
        if (error.raise) {
            std::cerr << "Failed to convert to SDR: " << error.message
                      << std::endl;
            return 1;
        }
        break;
    case ConversionMode::HDR_TO_UHDR:
        std::cout << "Converting HDR to SDR + Gainmap..." << std::endl;
        gainmap::HDRToGainMap(hdr_image, clip_percentile, 1.0f, error);
        if (error.raise) {
            std::cerr << "Failed to convert to SDR + Gainmap: " << error.message
                      << std::endl;
            return 1;
        }
        break;
    }

    // std::cout << "Saving output image: " << output_file << std::endl;
    // if (!imageops::WriteToPNG(output_image, output_file, error)) {
    //     std::cerr << "Failed to save image: " << error.message << std::endl;
    //     return 1;
    // }

    std::cout << "Conversion completed successfully!" << std::endl;
    return 0;
}
