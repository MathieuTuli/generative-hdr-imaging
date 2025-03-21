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
