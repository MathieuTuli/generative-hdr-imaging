#include "imageio.hpp"
#include <algorithm>
#include <cctype>
#include <filesystem>

namespace imageio {

bool HasExtension(const std::string &filename, const std::string &ext) {
    std::filesystem::path path(filename);
    std::string file_ext = path.extension().string();
    std::transform(file_ext.begin(), file_ext.end(), file_ext.begin(),
                   ::tolower);
    std::string compare_ext = ext[0] == '.' ? ext : "." + ext;
    std::transform(compare_ext.begin(), compare_ext.end(), compare_ext.begin(),
                   ::tolower);
    return file_ext == compare_ext;
}

bool ValidateHeader(const std::string &filename, utils::Error &error) {
    if (!std::filesystem::exists(filename)) {
        error = {true, "File does not exist: " + filename};
        return false;
    }
    // REVISIT: Add specific format validation here
    return true;
}

HDRFormat DetectFormat(const std::string &filename) {
    if (HasExtension(filename, ".avif"))
        return HDRFormat::AVIF;
    if (HasExtension(filename, ".exr"))
        return HDRFormat::EXR;
    if (HasExtension(filename, ".hdr"))
        return HDRFormat::PNG_HDR;
    return HDRFormat::UNKNOWN;
}

HDRImage LoadImage(const std::string &filename, utils::Error &error) {
    HDRFormat format = DetectFormat(filename);
    switch (format) {
    case HDRFormat::AVIF:
        return LoadAVIF(filename, error);
    case HDRFormat::EXR:
        return LoadEXR(filename, error);
    case HDRFormat::PNG_HDR:
        return LoadPNGHDR(filename, error);
    default:
        error = {true, "Unsupported format"};
        return HDRImage{};
    }
}

HDRImage LoadAVIF(const std::string &filename, utils::Error &error) {
    // TODO: Implement AVIF loading using libavif
    error = {true, "AVIF loading not implemented"};
    return HDRImage{};
}

HDRImage LoadEXR(const std::string &filename, utils::Error &error) {
    // TODO: Implement EXR loading using OpenEXR
    error = {true, "EXR loading not implemented"};
    return HDRImage{};
}

HDRImage LoadPNGHDR(const std::string &filename, utils::Error &error) {
    // TODO: Implement HDR PNG loading
    error = {true, "HDR PNG loading not implemented"};
    return HDRImage{};
}

ImageMetadata ReadMetadata(const std::string &filename, HDRFormat format,
                           utils::Error &error) {
    switch (format) {
    case HDRFormat::AVIF:
        return ReadAVIFMetadata(filename, error);
    case HDRFormat::EXR:
        return ReadEXRMetadata(filename, error);
    case HDRFormat::PNG_HDR:
        return ReadPNGHDRMetadata(filename, error);
    default:
        error = {true, "Unsupported format for metadata"};
        return ImageMetadata{};
    }
}

ImageMetadata ReadAVIFMetadata(const std::string &filename, utils::Error &error) {
    // TODO: Implement AVIF metadata reading
    error = {true, "AVIF metadata reading not implemented"};
    return ImageMetadata{};
}

ImageMetadata ReadEXRMetadata(const std::string &filename, utils::Error &error) {
    // TODO: Implement EXR metadata reading
    error = {true, "EXR metadata reading not implemented"};
    return ImageMetadata{};
}

ImageMetadata ReadPNGHDRMetadata(const std::string &filename,
                                 utils::Error &error) {
    // TODO: Implement HDR PNG metadata reading
    error = {true, "HDR PNG metadata reading not implemented"};
    return ImageMetadata{};
}

bool SaveImage(const HDRImage &image, const std::string &filename,
               HDRFormat format, const ImageMetadata &metadata,
               utils::Error &error) {
    switch (format) {
    case HDRFormat::AVIF:
        return SaveAVIF(image, filename, metadata, error);
    case HDRFormat::EXR:
        return SaveEXR(image, filename, metadata, error);
    case HDRFormat::PNG_HDR:
        return SavePNGHDR(image, filename, metadata, error);
    default:
        error = {true, "Unsupported format for saving"};
        return false;
    }
}

bool SaveAVIF(const HDRImage &image, const std::string &filename,
              const ImageMetadata &metadata, utils::Error &error) {
    // TODO: Implement AVIF saving
    error = {true, "AVIF saving not implemented"};
    return false;
}

bool SaveEXR(const HDRImage &image, const std::string &filename,
             const ImageMetadata &metadata, utils::Error &error) {
    // TODO: Implement EXR saving
    error = {true, "EXR saving not implemented"};
    return false;
}

bool SavePNGHDR(const HDRImage &image, const std::string &filename,
                const ImageMetadata &metadata, utils::Error &error) {
    // TODO: Implement HDR PNG saving
    error = {true, "HDR PNG saving not implemented"};
    return false;
}

} // namespace imageio
