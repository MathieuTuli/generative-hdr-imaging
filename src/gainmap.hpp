#ifndef GAINMAP_CPP
#define GAINMAP_CPP
#include "imageops.hpp"
namespace gainmap {
std::unique_ptr<imageops::PNGImage>
ComputeGainMap(std::vector<double> hdr_yuv, std::vector<double> sdr_yuv,
               double offset_hdr, double offset_sdr, double min_content_boost,
               double max_content_boost, double map_gamma, utils::Error error);

std::unique_ptr<imageops::PNGImage>
HDRToGainMap(const std::unique_ptr<imageops::PNGImage> &hdr_image,
             double offset_hdr, double offset_sdr, double min_content_boost,
             double max_content_boost, double map_gamma, utils::Error &error);
} // namespace gainmap
#endif
