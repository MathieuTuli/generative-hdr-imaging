# functions
## `src/imageio.cpp`
1. `LoadAVIF()`: AVIF loading using libavif
2. `LoadEXR()`: EXR loading using OpenEXR
3. `LoadPNGHDR()`: HDR PNG loading
4. `ReadAVIFMetadata()`: AVIF metadata reading
5. `ReadEXRMetadata()`: EXR metadata reading
6. `ReadPNGHDRMetadata()`: HDR PNG metadata reading
7. `SaveAVIF()`: AVIF saving
8. `SaveEXR()`: EXR saving
9. `SavePNGHDR()`: HDR PNG saving

## `src/imageops.cpp`
1. `HDRtoRAW()`:
    - load HDR image
    - linearize
    - convert to uniform colorspace
    - save to raw
2. `HDRtoSDR()`: EXR loading using OpenEXR
    - loads an HDR image
    - converts to raw, linear, uniform colorspace
    - calls RAWtoSDR
2. `RAWtoSDR()`: EXR loading using OpenEXR
    - clips the raw above
    - non-linear correction (i.e. conversion to gamma, srgb, something)
    - quantization
    - save image
