#ifndef API_HPP
#define API_HPP
#include "utils.h"
#include <cstdint>
#include <memory>
#include <png.h>
#include <string>
#include <vector>

namespace imageops {

// ----------------------------------------
// DATA CONTAINERS
// ----------------------------------------
enum class ToneMapping {
    BASE,
    REINHARD,
    GAMMA,
    FILMIC,
    ACES,
    UNCHARTED2,
    DRAGO,
    LOTTES,
    HABLE
};

enum class ColorSpace {
    SRGB,
    DISPLAYP3,
    P3PQ,
    REC2020HLG,
    REC2020GAMMA,
};

struct PNGImage {
    size_t width{0};
    size_t height{0};
    uint8_t color_type{0};
    uint8_t bit_depth{0};
    size_t bytes_per_row;
    png_bytep *row_pointers;
};

struct ImageMetadata {
    ColorSpace color_space;
    double clip_low{0.0};
    double clip_high{1.0};
    double offset_hdr{0.015625};
    double offset_sdr{0.015625};
    double min_content_boost{1.0};
    double max_content_boost{4.0};
    double map_gamma{1.0};
};

enum class HDRFormat {
    UNKNOWN = -1,
    HDRPNG = 0,
    AVIF = 1,
};

// ----------------------------------------
// BASIC IO
// ----------------------------------------
bool HasExtension(const std::string &filename, const std::string &ext);
bool ValidateHeader(const std::string &filename, utils::Error &error);
HDRFormat DetectFormat(const std::string &filename);

// ----------------------------------------
// INVOLVED IO
// ----------------------------------------
std::unique_ptr<PNGImage> LoadImage(const std::string &filename,
                                    utils::Error &error);
void LoadAVIF(const std::string &filename, utils::Error &error);
std::unique_ptr<PNGImage> LoadHDRPNG(const std::string &filename,
                                     utils::Error &error);
ImageMetadata ReadAVIFMetadata(const std::string &filename,
                               utils::Error &error);
ImageMetadata ReadHDRPNGMetadata(const std::string &filename,
                                 utils::Error &error);

bool WriteToPNG(const std::unique_ptr<PNGImage> &image,
                const std::string &filename, utils::Error &error);
bool WriteToNumpy(const std::vector<double> &data, int width, int height,
                  int channels, const std::string &dtype_str,
                  const std::string &output_path, utils::Error &error);

// ----------------------------------------
// CONVERSION
// ----------------------------------------
// 1. HLG ->
// 2. LINEAR ->
// 3. CLIP/COMPRESSION/HDR-SDR TONEMAP ->
// 4. COLOR SPACE ->
// 5. sRGB TONEMAP/TRANSFER ->
// 6. QUANT
#define CLIP(x, min, max) ((x) < (min)) ? (min) : ((x) > (max)) ? (max) : (x)

double Rec2020HLGToLinear(double x);
double Rec2020GammaToLinear(double x);
double P3PQToLinear(double x);
double sRGBToLinear(double x);

double LinearToRec2020HLG(double x);
double LinearToRec2020Gamma(double x);
double LinearToP3PQ(double x);
double LinearTosRGB(double x);

std::vector<double> LinearRec2020ToXYZ(const std::vector<double> &rgb);
std::vector<double> LinearP3ToXYZ(const std::vector<double> &rgb);
std::vector<double> LinearsRGBToXYZ(const std::vector<double> &rgb);

std::vector<double> XYZToLinearRec2020(const std::vector<double> &xyz);
std::vector<double> XYZToLinearP3(const std::vector<double> &xyz);
std::vector<double> XYZToLinearsRGB(const std::vector<double> &xyz);
std::vector<double> XYZToRec2020HLG(const std::vector<double> &xyz);
std::vector<double> XYZToRec2020Gamma(const std::vector<double> &xyz);
std::vector<double> XYZToP3PQ(const std::vector<double> &xyz);
std::vector<double> XYZTosRGB(const std::vector<double> &xyz);
std::vector<double> XYZToRec2020YUV(const std::vector<double> &xyz);
std::vector<double> XYZToBt709YUV(const std::vector<double> &xyz);

double ApplyToneMapping(double x, ToneMapping mode, double target_nits,
                        double max_nits);
// DEPRECATE:
void LinearRec2020ToLinearsRGB(double &r, double &g, double &b);

std::unique_ptr<PNGImage> HDRToYUV(const std::unique_ptr<PNGImage> &hdr_image,
                                   double clip_low, double clip_high,
                                   utils::Error &error, ToneMapping mode);
std::unique_ptr<PNGImage> HDRToSDR(const std::unique_ptr<PNGImage> &hdr_image,
                                   double clip_low, double clip_high,
                                   utils::Error &error, ToneMapping mode);

} // namespace imageops
#endif
