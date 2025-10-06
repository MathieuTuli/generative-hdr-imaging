# HDR Gainmap Conversion 

A set of python scripts for converting HDR images to gainmaps, and combining gainmaps with SDRs into HDR images.

## installation
```console
pip install loguru fire
pip install torch einops imageio
```

## usage
Run `python api.py` to see the available commands:
```
> python api.py
NAME
    api.py

SYNOPSIS
    api.py - COMMAND

COMMANDS
    COMMAND is one of the following:

     compare_reconstruction

     compare_reconstruction_batched

     hdr_to_gainmap
       - outdir: None | str - if None - will return the data - used for in loop loading

     hdr_to_gainmap_batched

     reconstruct_hdr

     reconstruct_hdr_batched
```
