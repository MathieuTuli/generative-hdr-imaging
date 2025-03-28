#include "gainmap.hpp"
#include "imageops.hpp"
#include "npy.hpp"
#include "spdlog/spdlog.h"
#include "utils.h"
#include <filesystem>
#include <fmt/format.h>
#include <getopt.h>
#include <string>

#define PROGRAM_VERSION "1.0.0"

enum class ConversionMode { HDR_TO_UHDR, UHDR_TO_HDR, COMPARE_HDR_UHDR };
enum class InputMode { SINGLE_FILE, DIRECTORY };

// Helper function to get the stem of a filepath
std::string GetStem(const std::string &filepath) {
    std::filesystem::path path(filepath);
    return path.stem().string();
}

// Process a single HDR to UHDR conversion
bool ProcessHDRToUHDR(const std::string &input_file,
                      const std::string &output_stem,
                      const std::string &output_dir, float clip_percentile) {
    utils::Error error;
    spdlog::info("Converting HDR to UDR: {}", input_file);

    std::unique_ptr<imageops::Image> image =
        imageops::LoadImage(input_file, error);
    if (error.raise) {
        spdlog::error("Failed to load image: " + error.message);
        return false;
    }

    gainmap::HDRToGainMap(image, clip_percentile, 1.0f, output_stem, output_dir,
                          error);
    if (error.raise) {
        spdlog::error("Failed to convert to SDR + Gainmap: " + error.message);
        return false;
    }
    return true;
}

// Process a single UHDR to HDR conversion
bool ProcessUHDRToHDR(const std::string &input_file,
                      const std::string &gainmap_file,
                      const std::string &metadata_file,
                      const std::string &output_stem,
                      const std::string &output_dir) {
    utils::Error error;
    spdlog::info("Converting UHDR to HDR: " + input_file);

    std::unique_ptr<imageops::Image> image =
        imageops::LoadImage(input_file, error);
    if (error.raise) {
        spdlog::error("Failed to load image: " + error.message);
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
            std::string shape_str;
            for (size_t i = 0; i < shape.size(); ++i) {
                shape_str += std::to_string(shape[i]);
                if (i < shape.size() - 1) {
                    shape_str += ", ";
                }
            }
            spdlog::error(
                "Invalid gainmap shape. Expected [height, width, 1], got [{}]",
                shape_str);
            return false;
        }

        if (shape[0] != image->height || shape[1] != image->width) {
            spdlog::error(
                "Gainmap dimensions ({}x{}) don't match input image ({}x{})",
                shape[1], shape[0], image->width, image->height);
            return false;
        }
    } catch (const std::runtime_error &e) {
        spdlog::error("Failed to load gainmap NPY file: " +
                      std::string(e.what()));
        return false;
    }

    gainmap::GainmapSDRToHDR(image, gainmap_data, metadata_file, output_stem,
                             output_dir, error);
    if (error.raise) {
        spdlog::error("Failed to convert SDR + Gainmap to HDR: {}",
                      error.message);
        return false;
    }
    return true;
}

void PrintUsage(const char *programName) {
    fmt::print(
        "Usage: {} [OPTIONS]\n"
        "\nOptions:\n"
        "  -u, --HDR2uHDR       Convert HDR to SDR + Gainmap\n"
        "  -z, --uHDR2HDR       Convert SDR + Gainmap to HDR\n"
        "  -c, --compare        Compare original HDR with reconstructed HDR\n"
        "  -i, --input=PATH     Input image file or directory\n"
        "  -g, --gainmap=FILE   Gainmap file (required for uHDR2HDR)\n"
        "  -s, --sdr=FILE      SDR image file (required for compare mode)\n"
        "  -m, --metadata=FILE  Metadata JSON file (required for uHDR2HDR)\n"
        "  -o, --output=STEM    Output file stem (only used with single file "
        "input)\n"
        "  -d, --outdir=DIR     Output directory (default: current dir)\n"
        "  -p, --percentile=N   Clip percentile for gainmap (default: 0.95)\n"
        "  -h, --help           Display this help and exit\n"
        "  -v, --version        Output version information and exit\n"
        "  -q, --debug          Set debug log level\n"
        "\nExamples:\n"
        "  Single file:\n"
        "    {} -i input.exr -o output --HDR2uHDR\n"
        "  Directory:\n"
        "    {} -i input_dir -d output_dir --HDR2uHDR\n",
        programName, programName, programName);
}

void PrintVersion(const char *programName) {
    fmt::print(
        "{} version {}\nCopyright (C) 2025 RadiantFlow\nLicense: MIT License\n",
        programName, PROGRAM_VERSION);
}

int main(int argc, char *argv[]) {
    std::string input_path;
    std::string metadata_file;
    std::string gainmap_file;
    std::string sdr_path;
    std::string output_dir = ".";
    std::string output_stem;
    ConversionMode mode;
    InputMode input_mode;
    bool mode_set = false;
    float clip_percentile = 0.95f;

    static struct option long_options[] = {
        {"debug", no_argument, 0, 'q'},
        {"help", no_argument, 0, 'h'},
        {"version", no_argument, 0, 'v'},
        {"metadata", required_argument, 0, 'm'},
        {"input", required_argument, 0, 'i'},
        {"output", required_argument, 0, 'o'},
        {"outdir", required_argument, 0, 'd'},
        {"HDR2raw", no_argument, 0, 'r'},
        {"HDR2sdr", no_argument, 0, 's'},
        {"percentile", required_argument, 0, 'p'},
        {"gainmap", required_argument, 0, 'm'},
        {"compare", no_argument, 0, 'c'},
        {"sdr", required_argument, 0, 's'},
        {0, 0, 0, 0}};

    int option_index = 0;
    int c;

    while ((c = getopt_long(argc, argv, "qhvi:o:d:zug:m:p:cs:", long_options,
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
        case 's':
            sdr_path = optarg;
            break;
        case 'o':
            output_stem = optarg;
            break;
        case 'q':
            spdlog::set_level(spdlog::level::debug);
            break;
        case 'd':
            output_dir = optarg;
            break;
        case 'u':
            if (mode_set) {
                spdlog::error("Error: Only one mode can be specified");
                return 1;
            }
            mode = ConversionMode::HDR_TO_UHDR;
            mode_set = true;
            break;
        case 'z':
            if (mode_set) {
                spdlog::error("Error: Only one mode can be specified");
                return 1;
            }
            mode = ConversionMode::UHDR_TO_HDR;
            mode_set = true;
            break;
        case 'c':
            if (mode_set) {
                spdlog::error("Error: Only one mode can be specified");
                return 1;
            }
            mode = ConversionMode::COMPARE_HDR_UHDR;
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
                spdlog::error("Error: Percentile must be between 0 and 1");
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
        spdlog::error("Error: MODE is required");
        PrintUsage(argv[0]);
        return 1;
    }

    if ((mode == ConversionMode::UHDR_TO_HDR ||
         mode == ConversionMode::COMPARE_HDR_UHDR) &&
        (gainmap_file.empty() || metadata_file.empty())) {
        spdlog::error(
            "Error: Gainmap and metadata files are required for SDR to HDR "
            "conversion and comparison modes (--gainmap/-g and --metadata/-m)");
        PrintUsage(argv[0]);
        return 1;
    }

    if (mode == ConversionMode::COMPARE_HDR_UHDR && sdr_path.empty()) {
        spdlog::error(
            "Error: SDR image path is required for comparison mode (--sdr/-s)");
        PrintUsage(argv[0]);
        return 1;
    }

    if (input_path.empty()) {
        spdlog::error("Error: Input path is required (--input/-i)");
        PrintUsage(argv[0]);
        return 1;
    }

    // Determine if input is a file or directory
    std::filesystem::path input_fs_path(input_path);
    if (!std::filesystem::exists(input_fs_path)) {
        spdlog::error("Error: Input path does not exist: {}", input_path);
        return 1;
    }

    input_mode = std::filesystem::is_directory(input_fs_path)
                     ? InputMode::DIRECTORY
                     : InputMode::SINGLE_FILE;

    if (input_mode == InputMode::SINGLE_FILE && output_stem.empty() &&
        mode != ConversionMode::COMPARE_HDR_UHDR) {
        spdlog::error("Error: Output stem is required for single file mode "
                      "(--output/-o)");
        PrintUsage(argv[0]);
        return 1;
    }

    // Create output directory if it doesn't exist
    std::filesystem::path dir_path(output_dir);
    if (!std::filesystem::exists(dir_path)) {
        if (!std::filesystem::create_directories(dir_path)) {
            spdlog::error("Error: Failed to create output directory: {}",
                          output_dir);
            return 1;
        }
    }

    if (input_mode == InputMode::SINGLE_FILE) {
        bool success = false;
        if (mode == ConversionMode::HDR_TO_UHDR) {
            success = ProcessHDRToUHDR(input_path, output_stem, output_dir,
                                       clip_percentile);
        } else if (mode == ConversionMode::UHDR_TO_HDR) {
            success = ProcessUHDRToHDR(input_path, gainmap_file, metadata_file,
                                       output_stem, output_dir);
        } else if (mode == ConversionMode::COMPARE_HDR_UHDR) {
            utils::Error error;
            std::unique_ptr<imageops::Image> hdr_image =
                imageops::LoadImage(input_path, error);
            if (error.raise) {
                spdlog::error("Failed to load HDR image: {}", error.message);
                return 1;
            }

            std::unique_ptr<imageops::Image> sdr_image =
                imageops::LoadImage(sdr_path, error);
            if (error.raise) {
                spdlog::error("Failed to load SDR image: {}", error.message);
                return 1;
            }

            // Load gainmap data
            std::vector<unsigned long> shape;
            bool fortran_order;
            std::vector<float> gainmap_data;
            try {
                npy::LoadArrayFromNumpy(gainmap_file, shape, fortran_order,
                                        gainmap_data);
            } catch (const std::runtime_error &e) {
                spdlog::error("Failed to load gainmap NPY file: {}", e.what());
                return 1;
            }

            gainmap::CompareHDRToUHDR(hdr_image, sdr_image, gainmap_data,
                                      metadata_file, output_stem, output_dir,
                                      error);
            if (error.raise) {
                spdlog::error("Failed to compare HDR to UHDR: {}",
                              error.message);
                return 1;
            }
            success = true;
        }
        if (!success) {
            return 1;
        }
    } else {
        // Process directory
        spdlog::info("Processing directory: {}", input_path);
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
            if (ext != ".exr" && ext != ".HDR" && ext != ".png" &&
                ext != ".jpg" && ext != ".jpeg") {
                continue;
            }

            bool success = false;
            if (mode == ConversionMode::HDR_TO_UHDR) {
                success = ProcessHDRToUHDR(current_file, current_stem,
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
                    spdlog::error(
                        "Skipping {}: Missing gainmap or metadata files",
                        current_file);
                    failed++;
                    continue;
                }

                success = ProcessUHDRToHDR(current_file, current_gainmap,
                                           current_metadata, current_stem,
                                           output_dir);
            }

            if (success) {
                processed++;
            } else {
                failed++;
            }
        }

        spdlog::info("\nBatch processing completed:\nSuccessfully processed: "
                     "{} files\nFailed: {} files",
                     processed, failed);

        if (processed == 0) {
            spdlog::error("No valid images found in directory");
            return 1;
        }
    }

    spdlog::info("Conversion completed successfully");
    return 0;
}
