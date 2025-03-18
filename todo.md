# functions
## `src/imageio.cpp`
- [ ] `LoadAVIF()`: AVIF loading using libavif
- [ ] `ReadAVIFMetadata()`: AVIF metadata reading
- [x] `LoadHDRPNG()`: HDR PNG loading
- [ ] `ReadHDRPNGMetadata()`: HDR PNG metadata reading
- [ ] `SaveRAW()`: raw saving (to 16-bit png?)

## `src/imageops.cpp`
- [ ] `HDRtoRAW()`:
    - load HDR image
    - linearize
    - convert to uniform colorspace
    - save to raw
- [ ] `HDRtoSDR()`: EXR loading using OpenEXR
    - loads an HDR image
    - converts to raw, linear, uniform colorspace
    - calls RAWtoSDR
- [ ] `RAWtoSDR()`: EXR loading using OpenEXR
    - clips the raw above
    - non-linear correction (i.e. conversion to gamma, srgb, something)
    - quantization
    - save image
