#ifndef API_HPP
#define API_HPP
#include "imageio.hpp"
#include "utils.h"
namespace imageops {

// HDR processing functions
struct HDRProcessingParams {
    float exposure{0.0f};
    float saturation{1.0f};
    float contrast{1.0f};
    float gamma{2.2f};
};

struct SDRConversionParams {
    float max_nits{100.0f};
    float target_gamma{2.2f};
    bool preserve_highlights{true};
    float knee_point{0.75f};
};
bool HDRtoRAW(imageio::HDRImage &image, const HDRProcessingParams &params, utils::Error &error);

imageio::HDRImage RAWtoSDR(const imageio::HDRImage &hdr_image,
                           const SDRConversionParams &params, utils::Error &error);

imageio::HDRImage HDRtoSDR(const imageio::HDRImage &hdr_image,
                           const SDRConversionParams &params, utils::Error &error);
} // namespace imageops
#endif
