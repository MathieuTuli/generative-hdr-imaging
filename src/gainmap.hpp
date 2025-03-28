#ifndef GAINMAP_CPP
#define GAINMAP_CPP
#include "colorspace.hpp"
#include "imageops.hpp"
namespace gainmap {
float ComputeGain(float hdr_y_nits, float sdr_y_nits, float hdr_offset,
                  float sdr_offset);
float AffineMapGain(float gainlog2, float min_gainlog2, float max_gainlog2,
                    float map_gamma);
colorspace::Color ApplyGain(colorspace::Color e, float gain, float map_gamma,
                            float min_content_boost, float max_content_boost,
                            float hdr_offset, float sdr_offset);

void HDRToGainMap(const std::unique_ptr<imageops::Image> &hdr_image,
                  float clip_percentile, float map_map_gamma,
                  const std::string &file_stem, const std::string &output_dir,
                  utils::Error &error);
void GainmapSDRToHDR(const std::unique_ptr<imageops::Image> &sdr_image,
                     const std::vector<float> gainmap,
                     const std::string &metadata, const std::string &file_stem,
                     const std::string &output_dir, utils::Error &error);
void CompareHDRToUHDR(const std::unique_ptr<imageops::Image> &hdr_image,
                      const std::unique_ptr<imageops::Image> &sdr_image,
                      const std::vector<float> gainmap,
                      const std::string &metadata, const std::string
                      &file_stem, const std::string &output_dir, utils::Error
                      &error);
} // namespace gainmap
#endif
