# functions
## `src/imageio.cpp`
- [ ] `LoadAVIF()`: AVIF loading using libavif
- [ ] `ReadAVIFMetadata()`: AVIF metadata reading
- [x] `LoadHDRPNG()`: HDR PNG loading
- [ ] `ReadHDRPNGMetadata()`: HDR PNG metadata reading
- [ ] `SaveRAW()`: raw saving (to 16-bit png?)

## `src/imageops.cpp`
- are the chroma channels in the sdr/hdr the same? they should be...
    - only y is changed
- order
    - linear
    - convert to rec2020 in linear space
    - for gt gainmap
        - hdr image
        - linear space
        - convert to rec2020
            - hdr yuv
        - clip
        - YUV
        - tonemap
        - quantize

## generate gain map from libultrahdr steps
- first, convert both from RGB to YUV (which is YCbCr)
- gets the InvOETF function for HDR image
- gets the luminance function for HDR image
- gets the OOTF function (i.e. if HLG, needs OOTF, identity otherwise)
- gets the peak luminance based on colorspace for HDR image
- if the sdr and hdr images are not the same color gamut
    - get conversion to convert sdr to hdr
    - only use SDR color gamut if it's bt2100 lol
    - otherwise, cnovert to hdr's gamut (note that this won't convert to bt2100 like we might wanna do)
- next it grabs the yuv to rgb function for the sdr image and hdr image
- get the luminance function for sdr now  
- it also gets the pixel sampler, but, for now I will handle that some other way
- then we generate the map
