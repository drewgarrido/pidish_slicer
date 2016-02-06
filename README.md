# pidish slicer
## Overview
Slices a 3d file in the STL file format into PNGs for use on an SLA 3D printer,
like my personal [pidish](http://pidish3dprinter.blogspot.com/) printer.
The script will find the best position for the object on the printer bed,
both by translation and rotation, using a mask file representing areas on the
lift to avoid. Then the object will be sliced.

## Running
This script requires:
* Python 3 (tested on 3.5)
* numpy
* numpy-stl
* pillow

You'll also need:
* A mask file that indicates preferred areas to print in. See below for more.
* The units your STL file (millimeters, inches, meters, etc.)
* Your printer resolution (smallest feature size)

The scale value is the number of pixels per unit in the STL file. I usually
have STL files using millimeters, and my resolution is 0.1 mm,
so my scale value is 10. You can use this formula:

scale_value = (STL units * (unit conversion to match printer)) / printer resolution

On a command line, run:
python slicer.py -i input.stl -o output_dir -m mask_file -s scale_value

This script can take a few minutes to complete on a Core i5, so I wouldn't
put it on an embedded system.

## Mask file
The resolution of this mask should match the projector. Dark areas are less
favorable, so I made any holes black and good areas white. My mask file
is a RGBA .png, but I suspect other formats work as well since I'm using
Pillow's open function. I would NOT use a single channel grayscale image.

## Lessons Demonstrated
* Numpy
* Voxelization
