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
enum class InputMode { SINGLE_FILE, DIRECTORY };

// Helper function to get the stem of a filepath
std::string GetStem(const std::string &filepath) {
    std::filesystem::path path(filepath);
    return path.stem().string();
}

// Process a single HDR to UHDR conversion
bool ProcessHdrToUhdr(const std::string &input_file,
                      const std::string &output_stem,
                      const std::string &output_dir, float clip_percentile) {
    utils::Error error;
    std::cout << "Processing: " << input_file << std::endl;

    std::unique_ptr<imageops::Image> image =
        imageops::LoadImage(input_file, error);
    if (error.raise) {
        std::cerr << "Failed to load image: " << error.message << std::endl;
        return false;
    }

    gainmap::HDRToGainMap(image, clip_percentile, 1.0f, output_stem, output_dir,
                          error);
    if (error.raise) {
        std::cerr << "Failed to convert to SDR + Gainmap: " << error.message
                  << std::endl;
        return false;
    }
    return true;
}

// Process a single UHDR to HDR conversion
bool ProcessUhdrToHdr(const std::string &input_file,
                      const std::string &gainmap_file,
                      const std::string &metadata_file,
                      const std::string &output_stem,
                      const std::string &output_dir) {
    utils::Error error;
    std::cout << "Processing: " << input_file << std::endl;

    std::unique_ptr<imageops::Image> image =
        imageops::LoadImage(input_file, error);
    if (error.raise) {
        std::cerr << "Failed to load image: " << error.message << std::endl;
        return false;
    }

    // Load the gainmap NPY file
    std::vector<unsigned long> shape;
    bool fortran_order;
    std::vector<float> gainmap_data;

    try {
        npy::LoadArrayFromNumpy(gainmap_file, shape, fortran_order,
                                gainmap_data);

        if (shape.size() != 3 || shape[2] != 1) {
            std::cerr
                << "Invalid gainmap shape. Expected [height, width, 1], got [";
            for (size_t i = 0; i < shape.size(); ++i) {
                std::cerr << shape[i] << (i < shape.size() - 1 ? ", " : "");
            }
            std::cerr << "]" << std::endl;
            return false;
        }

        if (shape[0] != image->height || shape[1] != image->width) {
            std::cerr << "Gainmap dimensions (" << shape[1] << "x" << shape[0]
                      << ") don't match input image (" << image->width << "x"
                      << image->height << ")" << std::endl;
            return false;
        }
    } catch (const std::runtime_error &e) {
        std::cerr << "Failed to load gainmap NPY file: " << e.what()
                  << std::endl;
        return false;
    }

    gainmap::GainmapSdrToHDR(image, gainmap_data, metadata_file, output_stem,
                             output_dir, error);
    if (error.raise) {
        std::cerr << "Failed to convert SDR + Gainmap to HDR: " << error.message
                  << std::endl;
        return false;
    }
    return true;
}

void PrintUsage(const char *programName) {
    std::cout
        << "Usage: " << programName << " [OPTIONS]\n"
        << "\nOptions:\n"
        << "  -u, --hdr2uhdr       Convert HDR to SDR + Gainmap\n"
        << "  -z, --uhdr2hdr       Convert SDR + Gainmap to HDR\n"
        << "  -i, --input=PATH    Input image file or directory\n"
        << "  -g, --gainmap=FILE  Gainmap file (required for uhdr2hdr)\n"
        << "  -m, --metadata=FILE  Metadata JSON file (required for uhdr2hdr)\n"
        << "  -o, --output=STEM   Output file stem (only used with single file "
           "input)\n"
        << "  -d, --outdir=DIR    Output directory (default: current dir)\n"
        << "  -p, --percentile=N  Clip percentile for gainmap (default: 0.95)\n"
        << "  -h, --help          Display this help and exit\n"
        << "  -v, --version       Output version information and exit\n"
        << "\nExamples:\n"
        << "  Single file:\n"
        << "    " << programName << " -i input.exr -o output --hdr2uhdr\n"
        << "  Directory:\n"
        << "    " << programName << " -i input_dir -d output_dir --hdr2uhdr\n";
}

void PrintVersion(const char *programName) {
    std::cout << programName << " version " << PROGRAM_VERSION << "\n"
              << "Copyright (C) 2025 RadiantFlow\n"
              << "License: MIT License\n";
}

int main(int argc, char *argv[]) {
    std::string input_path;
    std::string metadata_file;
    std::string gainmap_file;
    std::string output_dir = ".";
    std::string output_stem;
    ConversionMode mode;
    InputMode input_mode;
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

#ifdef __LITTLE_ENDIAN__
    std::cout << "__LITTLE_ENDIAN__ is defined" << std::endl;
    ;
    std::cout << "__LITTLE_ENDIAN__ = " << __LITTLE_ENDIAN__ << std::endl;
#else
    std::cout << "__LITTLE_ENDIAN__ is not defined" << std::endl;
#endif

#ifdef __BYTE_ORDER__
    std::cout << "__BYTE_ORDER__ is defined" << std::endl;
    std::cout << "__BYTE_ORDER__ = " << __BYTE_ORDER__ << std::endl;
#else
    std::cout << "__BYTE_ORDER__ is not defined" << std::endl;
#endif

#ifdef __ORDER_LITTLE_ENDIAN__
    std::cout << "__ORDER_LITTLE_ENDIAN__ is defined" << std::endl;
    std::cout << "__ORDER_LITTLE_ENDIAN__ = " << __ORDER_LITTLE_ENDIAN__
              << std::endl;
#else
    std::cout << "__ORDER_LITTLE_ENDIAN__ is not defined" << std::endl;
#endif

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
            input_path = optarg;
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

    if (mode == ConversionMode::UHDR_TO_HDR && gainmap_file.empty() &&
        metadata_file.empty()) {
        std::cerr << "Error: Gainmap file is required for SDR to HDR "
                     "conversion (--gainmap/-m)\n";
        PrintUsage(argv[0]);
        return 1;
    }
    if (input_path.empty()) {
        std::cerr << "Error: Input path is required (--input/-i)\n";
        PrintUsage(argv[0]);
        return 1;
    }

    // Determine if input is a file or directory
    std::filesystem::path input_fs_path(input_path);
    if (!std::filesystem::exists(input_fs_path)) {
        std::cerr << "Error: Input path does not exist: " << input_path
                  << std::endl;
        return 1;
    }

    input_mode = std::filesystem::is_directory(input_fs_path)
                     ? InputMode::DIRECTORY
                     : InputMode::SINGLE_FILE;

    if (input_mode == InputMode::SINGLE_FILE && output_stem.empty()) {
        std::cerr << "Error: Output stem is required for single file mode "
                     "(--output/-o)\n";
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

    if (input_mode == InputMode::SINGLE_FILE) {
        bool success = false;
        if (mode == ConversionMode::HDR_TO_UHDR) {
            success = ProcessHdrToUhdr(input_path, output_stem, output_dir,
                                       clip_percentile);
        } else {
            success = ProcessUhdrToHdr(input_path, gainmap_file, metadata_file,
                                       output_stem, output_dir);
        }
        if (!success) {
            return 1;
        }
    } else {
        // Process directory
        std::cout << "Processing directory: " << input_path << std::endl;
        int processed = 0;
        int failed = 0;

        for (const auto &entry :
             std::filesystem::directory_iterator(input_path)) {
            if (!entry.is_regular_file())
                continue;

            std::string current_file = entry.path().string();
            std::string current_stem = GetStem(current_file);

            // Skip non-image files and metadata/gainmap files
            std::string ext = entry.path().extension().string();
            if (ext != ".exr" && ext != ".hdr" && ext != ".png" &&
                ext != ".jpg" && ext != ".jpeg") {
                continue;
            }

            bool success = false;
            if (mode == ConversionMode::HDR_TO_UHDR) {
                success = ProcessHdrToUhdr(current_file, current_stem,
                                           output_dir, clip_percentile);
            } else {
                // For UHDR_TO_HDR mode, construct gainmap and metadata paths
                std::string current_gainmap =
                    std::filesystem::path(current_file).parent_path().string() +
                    "/" + current_stem + "_gainmap.npy";
                std::string current_metadata =
                    std::filesystem::path(current_file).parent_path().string() +
                    "/" + current_stem + "_metadata.json";

                if (!std::filesystem::exists(current_gainmap) ||
                    !std::filesystem::exists(current_metadata)) {
                    std::cerr << "Skipping " << current_file
                              << ": Missing gainmap or metadata files"
                              << std::endl;
                    failed++;
                    continue;
                }

                success = ProcessUhdrToHdr(current_file, current_gainmap,
                                           current_metadata, current_stem,
                                           output_dir);
            }

            if (success) {
                processed++;
            } else {
                failed++;
            }
        }

        std::cout << "\nBatch processing completed:\n"
                  << "Successfully processed: " << processed << " files\n"
                  << "Failed: " << failed << " files\n";

        if (processed == 0) {
            std::cerr << "No valid images found in directory" << std::endl;
            return 1;
        }
    }

    std::cout << "Conversion completed successfully!" << std::endl;
    return 0;
}
