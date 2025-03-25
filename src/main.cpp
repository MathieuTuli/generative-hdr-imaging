#include "gainmap.hpp"
#include "imageops.hpp"
#include "npy.hpp"
#include "utils.h"
#include <filesystem>
#include <getopt.h>
#include <iostream>
#include <string>

#define PROGRAM_VERSION "1.0.0"

enum class ConversionMode { HDR_TO_UHDR, UHDR_TO_HDR };

void PrintUsage(const char *programName) {
    std::cout
        << "Usage: " << programName << " [OPTIONS]\n"
        << "\nOptions:\n"
        << "  -u, --hdr2uhdr       Convert HDR to SDR + Gainmap\n"
        << "  -z, --uhdr2hdr       Convert SDR + Gainmap to HDR\n"
        << "  -i, --input=FILE    Input image file\n"
        << "  -g, --gainmap=FILE  Gainmap file (required for uhdr2hdr)\n"
        << "  -m, --metadata=FILE  Metadata JSON file (required for uhdr2hdr)\n"
        << "  -o, --output=STEM   Output file stem (without extension)\n"
        << "  -d, --outdir=DIR    Output directory (default: current dir)\n"
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
    std::string metadata_file;
    std::string gainmap_file;
    std::string output_dir = ".";
    std::string output_stem;
    ConversionMode mode;
    bool mode_set = false;
    float clip_percentile = 0.95f;

    static struct option long_options[] = {
        {"help", no_argument, 0, 'h'},
        {"version", no_argument, 0, 'v'},
        {"metadata", required_argument, 0, 'm'},
        {"input", required_argument, 0, 'i'},
        {"output", required_argument, 0, 'o'},
        {"outdir", required_argument, 0, 'd'},
        {"hdr2raw", no_argument, 0, 'r'},
        {"hdr2sdr", no_argument, 0, 's'},
        {"percentile", required_argument, 0, 'p'},
        {"gainmap", required_argument, 0, 'm'},
        {0, 0, 0, 0}};

    int option_index = 0;
    int c;

    while ((c = getopt_long(argc, argv, "hvi:o:d:zug:m:p:", long_options,
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
            output_stem = optarg;
            break;
        case 'd':
            output_dir = optarg;
            break;
        case 'u':
            if (mode_set) {
                std::cerr << "Error: Only one mode can be specified\n";
                return 1;
            }
            mode = ConversionMode::HDR_TO_UHDR;
            mode_set = true;
            break;
        case 'z':
            if (mode_set) {
                std::cerr << "Error: Only one mode can be specified\n";
                return 1;
            }
            mode = ConversionMode::UHDR_TO_HDR;
            mode_set = true;
            break;
        case 'g':
            gainmap_file = optarg;
            break;
        case 'm':
            metadata_file = optarg;
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

    if (mode == ConversionMode::UHDR_TO_HDR && gainmap_file.empty() && metadata_file.empty()) {
        std::cerr << "Error: Gainmap file is required for SDR to HDR "
                     "conversion (--gainmap/-m)\n";
        PrintUsage(argv[0]);
        return 1;
    }
    if (input_file.empty()) {
        std::cerr << "Error: Input file is required (--input/-i)\n";
        PrintUsage(argv[0]);
        return 1;
    }
    if (output_stem.empty()) {
        std::cerr << "Error: Output stem is required (--output/-o)\n";
        PrintUsage(argv[0]);
        return 1;
    }

    // Create output directory if it doesn't exist
    std::filesystem::path dir_path(output_dir);
    if (!std::filesystem::exists(dir_path)) {
        if (!std::filesystem::create_directories(dir_path)) {
            std::cerr << "Error: Failed to create output directory: "
                      << output_dir << std::endl;
            return 1;
        }
    }

    utils::Error error;
    std::cout << "Loading image: " << input_file << std::endl;
    std::unique_ptr<imageops::Image> image =
        imageops::LoadImage(input_file, error);
    if (error.raise) {
        std::cerr << "Failed to load image: " << error.message << std::endl;
        return 1;
    }

    switch (mode) {
        case ConversionMode::HDR_TO_UHDR:
            std::cout << "Converting HDR to SDR + Gainmap..." << std::endl;
            gainmap::HDRToGainMap(image, clip_percentile, 1.0f, output_stem,
                                  output_dir, error);
            if (error.raise) {
                std::cerr << "Failed to convert to SDR + Gainmap: "
                          << error.message << std::endl;
                return 1;
            }
            break;
        case ConversionMode::UHDR_TO_HDR:
            std::cout << "Converting SDR + Gainmap to HDR..." << std::endl;

            // Load the gainmap NPY file
            std::vector<unsigned long> shape;
            bool fortran_order;
            std::vector<float> gainmap_data;

            try {
                npy::LoadArrayFromNumpy(gainmap_file, shape, fortran_order,
                                        gainmap_data);

                if (shape.size() != 3 || shape[2] != 1) {
                    std::cerr << "Invalid gainmap shape. Expected [height, "
                                 "width, 1], got [";
                    for (size_t i = 0; i < shape.size(); ++i) {
                        std::cerr << shape[i]
                                  << (i < shape.size() - 1 ? ", " : "");
                    }
                    std::cerr << "]" << std::endl;
                    return 1;
                }

                if (shape[0] != image->height || shape[1] != image->width) {
                    std::cerr << "Gainmap dimensions (" << shape[1] << "x"
                              << shape[0] << ") don't match input image ("
                              << image->width << "x" << image->height << ")"
                              << std::endl;
                    return 1;
                }
            } catch (const std::runtime_error &e) {
                std::cerr << "Failed to load gainmap NPY file: " << e.what()
                          << std::endl;
                return 1;
            }

            gainmap::GainmapSdrToHDR(image, gainmap_data, metadata_file,
                                     output_stem, output_dir, error);
            if (error.raise) {
                std::cerr << "Failed to convert SDR + Gainmap to HDR: "
                          << error.message << std::endl;
                return 1;
            }

            break;
    }

    std::cout << "Conversion completed successfully!" << std::endl;
    return 0;
}
