#ifndef GAINMAP_CPP
#define GAINMAP_CPP
#include "imageops.hpp"
namespace gainmap {
float ComputeGain(float hdr_y_nits, float sdr_y_nits, float hdr_offset,
                  float sdr_offset);
float AffineMapGain(float gainlog2, float min_gainlog2, float max_gainlog2,
                      float gamma);

void HDRToGainMap(const std::unique_ptr<imageops::Image> &hdr_image,
                  float map_gamma, utils::Error &error);
} // namespace gainmap
#endif
