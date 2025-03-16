#ifndef ULTRAHDR_H
#define ULTRAHDR_H

#include <string>

typedef enum uhdr_img_fmt {
    UHDR_IMG_FMT_UNSPECIFIED = -1, /**< Unspecified */
    UHDR_IMG_FMT_24bppYCbCrP010 =
        0, /**< 10-bit-per component 4:2:0 YCbCr semiplanar format.
       Each chroma and luma component has 16 allocated bits in
       little-endian configuration with 10 MSB of actual data.*/
    UHDR_IMG_FMT_12bppYCbCr420 =
        1, /**< 8-bit-per component 4:2:0 YCbCr planar format */
    UHDR_IMG_FMT_8bppYCbCr400 = 2, /**< 8-bit-per component Monochrome format */
    UHDR_IMG_FMT_32bppRGBA8888 =
        3, /**< 32 bits per pixel RGBA color format, with 8-bit red, green, blue
          and alpha components. Using 32-bit little-endian representation,
          colors stored as Red 7:0, Green 15:8, Blue 23:16, Alpha 31:24. */
    UHDR_IMG_FMT_64bppRGBAHalfFloat =
        4, /**< 64 bits per pixel, 16 bits per channel, half-precision floating
              point RGBA color format. colors stored as Red 15:0, Green 31:16,
              Blue 47:32, Alpha 63:48. In a pixel even though each channel has
              storage space of 16 bits, the nominal range is expected to be
              [0.0..(10000/203)] */
    UHDR_IMG_FMT_32bppRGBA1010102 =
        5, /**< 32 bits per pixel RGBA color format, with 10-bit red,
          green,   blue, and 2-bit alpha components. Using 32-bit
          little-endian   representation, colors stored as Red 9:0, Green
          19:10, Blue   29:20, and Alpha 31:30. */
    UHDR_IMG_FMT_24bppYCbCr444 =
        6, /**< 8-bit-per component 4:4:4 YCbCr planar format */
    UHDR_IMG_FMT_16bppYCbCr422 =
        7, /**< 8-bit-per component 4:2:2 YCbCr planar format */
    UHDR_IMG_FMT_16bppYCbCr440 =
        8, /**< 8-bit-per component 4:4:0 YCbCr planar format */
    UHDR_IMG_FMT_12bppYCbCr411 =
        9, /**< 8-bit-per component 4:1:1 YCbCr planar format */
    UHDR_IMG_FMT_10bppYCbCr410 =
        10, /**< 8-bit-per component 4:1:0 YCbCr planar format */
    UHDR_IMG_FMT_24bppRGB888 =
        11, /**< 8-bit-per component RGB interleaved format */
    UHDR_IMG_FMT_30bppYCbCr444 =
        12,       /**< 10-bit-per component 4:4:4 YCbCr planar format */
} uhdr_img_fmt_t; /**< alias for enum uhdr_img_fmt */

/*!\brief List of supported color ranges */
typedef enum uhdr_color_range {
    UHDR_CR_UNSPECIFIED = -1, /**< Unspecified */
    UHDR_CR_LIMITED_RANGE =
        0, /**< Y {[16..235], UV [16..240]} * pow(2, (bpc - 8)) */
    UHDR_CR_FULL_RANGE = 1, /**< YUV/RGB {[0..255]} * pow(2, (bpc - 8)) */
} uhdr_color_range_t;       /**< alias for enum uhdr_color_range */

typedef enum {
    COLORGAMUT_UNSPECIFIED = -1,
    COLORGAMUT_BT709,
    COLORGAMUT_P3,
    COLORGAMUT_BT2100,
    COLORGAMUT_MAX = COLORGAMUT_BT2100,
} ultrahdr_color_gamut;

/*
 * Holds information for gain map related metadata.
 *
 * Not: all values stored in linear. This differs from the metadata encoding in
 * XMP, where maxContentBoost (aka gainMapMax), minContentBoost (aka
 * gainMapMin), hdrCapacityMin, and hdrCapacityMax are stored in log2 space.
 */
struct ultrahdr_metadata_struct {
    // Ultra HDR format version
    std::string version;
    // Max Content Boost for the map
    float maxContentBoost;
    // Min Content Boost for the map
    float minContentBoost;
    // Gamma of the map data
    float gamma;
    // Offset for SDR data in map calculations
    float offsetSdr;
    // Offset for HDR data in map calculations
    float offsetHdr;
    // HDR capacity to apply the map at all
    float hdrCapacityMin;
    // HDR capacity to apply the map completely
    float hdrCapacityMax;
};

/*
 * Holds information for uncompressed image or gain map.
 */
struct jpegr_uncompressed_struct {
    // Pointer to the data location.
    void *data;
    // Width of the gain map or the luma plane of the image in pixels.
    unsigned int width;
    // Height of the gain map or the luma plane of the image in pixels.
    unsigned int height;
    // Color gamut.
    ultrahdr_color_gamut colorGamut;

    // Values below are optional
    // Pointer to chroma data, if it's NULL, chroma plane is considered to be
    // immediately after the luma plane.
    void *chroma_data = nullptr;
    // Stride of Y plane in number of pixels. 0 indicates the member is
    // uninitialized. If non-zero this value must be larger than or equal to
    // luma width. If stride is uninitialized then it is assumed to be equal to
    // luma width.
    unsigned int luma_stride = 0;
    // Stride of UV plane in number of pixels.
    // 1. If this handle points to P010 image then this value must be larger
    // than
    //    or equal to luma width.
    // 2. If this handle points to 420 image then this value must be larger than
    //    or equal to (luma width / 2).
    // NOTE: if chroma_data is nullptr, chroma_stride is irrelevant. Just as the
    // way, chroma_data is derived from luma ptr, chroma stride is derived from
    // luma stride.
    unsigned int chroma_stride = 0;
    // Pixel format.
    uhdr_img_fmt_t pixelFormat = UHDR_IMG_FMT_UNSPECIFIED;
    // Color range.
    uhdr_color_range_t colorRange = UHDR_CR_UNSPECIFIED;
};

/*
 * Holds information for compressed image or gain map.
 */
struct jpegr_compressed_struct {
    // Pointer to the data location.
    void *data;
    // Used data length in bytes.
    size_t length;
    // Maximum available data length in bytes.
    size_t maxLength;
    // Color gamut.
    ultrahdr_color_gamut colorGamut;
};

/*
 * Holds information for EXIF metadata.
 */
struct jpegr_exif_struct {
    // Pointer to the data location.
    void *data;
    // Data length;
    size_t length;
};

typedef struct jpegr_uncompressed_struct *jr_uncompressed_ptr;
typedef struct jpegr_compressed_struct *jr_compressed_ptr;
typedef struct jpegr_exif_struct *jr_exif_ptr;
typedef struct ultrahdr_metadata_struct *ultrahdr_metadata_ptr;

#endif
