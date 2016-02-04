# Copyright (C) 2016  Drew Garrido
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.

usageStr = """\
Usage:      python slicer.py input.stl output_dir [mask.png]

For the mask.png, white pixels denote supported areas, and black pixels \
denote holes to avoid. The pixel resolution of the slices are dependent on \
the mask.png resolution. \
"""

import numpy as np
import math
from PIL import Image, ImageDraw, ImageChops
import stl
import math
import traceback

WIDTH_MM = 102.4
HEIGHT_MM = 76.8

Z_RESOLUTION_MM = 0.1

def main(args):
    import re, os.path

    input_path = ""
    mask_path = ""
    output_path = ""

    # Enough arguments?
    if (len(args) in [3,4]):

        if (re.match(".+?\.stl", args[1])):
            input_path = args[1]

        output_path = args[2]

        if (len(args) == 4):
            if (re.match(".+?\.png", args[3])):
                mask_path = args[3]
    else:
        print(usageStr)
        return 0

    # Test for missing arguments and files
    if (input_path == ""):
        print("Please specify an input .stl file.")
        print(usageStr)
        return 0
    elif (not os.path.isfile(input_path)):
        print(input_path + " does not exist!")
        return 0

    if (not os.path.isdir(output_path)):
        print("Please specify a valid output directory.")
        print(usageStr)
        return 0

    if (mask_path == ""):
        if (os.path.isfile('mask.png')):
            mask_path = "mask.png"
        else:
            print("Please specify a mask.png path or place a mask.png file into the script directory.")
            print("White areas are supported areas, black are holes.")
            return 0
    elif (not os.path.isfile(mask_path)):
        print(mask_path + " does not exist!")
        return 0


    slicer(input_path, mask_path, output_path)

    return 0



def slicer(input_path, mask_path, output_path):
    try:
        f = open(input_path, "rb")
        identifier = f.read(6)
        f.seek(0,0)
        if (identifier == b"solid "):
            print('ascii file')
            # TODO: fix stl.read_ascii_file for python 3.5
            solid = stl.read_ascii_file(f)
        else:
            print('binary file')
            solid = stl.read_binary_file(f)
        f.close()
    except:
        print("Unable to parse the .stl file!")
        print(traceback.format_exc())
        return 0

    try:
        mask = Image.open(mask_path)
    except:
        print("Unable to read the mask.png file!")
        return 0


    # Scale the vertices to the actual pixels
    scale = mask.width / WIDTH_MM

    x_list = []
    y_list = []
    z_list = []
    for facet in solid.facets:
        for vertex in facet.vertices:
            x_list.append(vertex.x)
            y_list.append(vertex.y)
            z_list.append(vertex.z)

        facet.normal = compute_normal(facet.vertices)

    object_height = max(z_list) - min(z_list)

    bottom_z = min(z_list) + Z_RESOLUTION_MM / 2

    points = []
    midpoints = []

    # center the bottom image in order to find the best position
    #translate_x = (mask.width / 2) - ((max(x_list) + min(x_list)) / 2) * scale
    #translate_y = (mask.height / 2) - ((max(y_list) + min(y_list)) / 2) * scale

    # translate just enough to get the bottom in the top left
    translate_x = -1 * (min(x_list) * scale)
    translate_y = -1 * (min(y_list) * scale)
    translate_z = -1 * (min(z_list) * scale)

    clip_x = round(max(x_list) * scale + translate_x)
    clip_y = round(max(y_list) * scale + translate_y)

    # Find all line segments and midpoints
    for facet in solid.facets:
        below = []
        above = []
        for vertex in facet.vertices:
            if (vertex.z < bottom_z):
                below.append(vertex)
            else:
                above.append(vertex)

        if (len(below) == 1):
            scale0 = (bottom_z - below[0].z) / (above[0].z - below[0].z)
            scale1 = (bottom_z - below[0].z) / (above[1].z - below[0].z)

            x0 = ((above[0].x - below[0].x) * scale0 + below[0].x) * scale + translate_x
            y0 = ((above[0].y - below[0].y) * scale0 + below[0].y) * scale + translate_y
            x1 = ((above[1].x - below[0].x) * scale1 + below[0].x) * scale + translate_x
            y1 = ((above[1].y - below[0].y) * scale1 + below[0].y) * scale + translate_y

            nx = facet.normal[0]
            ny = facet.normal[1]

            mx = int(round((x0 + x1) / 2 - nx))
            my = int(round((y0 + y1) / 2 - ny))

            points.append(((round(x0),round(y0)),(round(x1),round(y1))))

            if (points[-1][0] != points[-1][1]):
                midpoints.append((mx, my))



        elif (len(above) == 1):
            scale0 = (bottom_z - below[0].z) / (above[0].z - below[0].z)
            scale1 = (bottom_z - below[1].z) / (above[0].z - below[1].z)

            x0 = ((above[0].x - below[0].x) * scale0 + below[0].x) * scale + translate_x
            y0 = ((above[0].y - below[0].y) * scale0 + below[0].y) * scale + translate_y
            x1 = ((above[0].x - below[1].x) * scale1 + below[1].x) * scale + translate_x
            y1 = ((above[0].y - below[1].y) * scale1 + below[1].y) * scale + translate_y

            nx = facet.normal[0]
            ny = facet.normal[1]

            mx = int(round((x0 + x1) / 2 - nx))
            my = int(round((y0 + y1) / 2 - ny))

            points.append(((round(x0),round(y0)),(round(x1),round(y1))))

            if (points[-1][0] != points[-1][1]):
                midpoints.append((mx, my))

    bottom_im = Image.new('RGBA',(mask.width, mask.height), (0,0,0,255))
    bottom_draw = ImageDraw.Draw(bottom_im)

    for couple in points:
        bottom_draw.line(couple, fill=(255,255,255,255))

    for x,y in midpoints:
        floodfill(bottom_im, x, y, (255,255,255,255))


    ###########################################################################
    ##
    ## Find the best position for the bottom
    ##
    ###########################################################################

    print("Finding best location")

    cropped_bottom = bottom_im.crop((0,0,clip_x,clip_y))

    crop_pixels = []
    for x in range(clip_x):
        crop_pixels.append([])
        for y in range(clip_y):
            crop_pixels[x].append(cropped_bottom.getpixel((x,y))[0]/255.0)

    crop_pixels = np.array(crop_pixels).flatten()

    # Normalize by the number of pixels involved
    crop_pixels = crop_pixels / np.sum(crop_pixels)

    mask_pixels = []
    for x in range(mask.width):
        mask_pixels.append([])
        for y in range(mask.height):
            mask_pixels[x].append(mask.getpixel((x,y))[0]/255.0)

    mask_pixels = np.array(mask_pixels)

    best_score = 0
    best_xy = (0,0)

    for x in range(bottom_im.width - clip_x):
        for y in range(bottom_im.height - clip_y):
            accum = mask_pixels[x:x+clip_x, y:y+clip_y].flatten().dot(crop_pixels)

            if (best_score < accum):
                best_xy = (x,y)
                best_score = accum

    print(best_xy)
    print(best_score)

    ###########################################################################
    ##
    #   Print rest of object
    ##
    ###########################################################################

    # Create transformation matrix
    angle = 0   # TODO: Find the best angle the bottom should be placed in
    normal_transform = np.array([[math.cos(angle),  math.sin(angle), 0],
                                 [-math.sin(angle), math.cos(angle), 0],
                                 [0,                0,               1]])

    transform = np.array([[scale * math.cos(angle),  scale * math.sin(angle), 0,     translate_x + best_xy[0]],
                          [scale * -math.sin(angle), scale * math.cos(angle), 0,     translate_y + best_xy[1]],
                          [0,                        0,                       scale, translate_z],
                          [0,                        0,                       0,     0]])

    for facet in solid.facets:
        facet.vertices = np.dot(transform, np.concatenate((np.array(facet.vertices),np.ones((3,1))),axis=1).transpose())[:3, :].transpose()
        facet.normal = np.dot(normal_transform, np.array(facet.normal))


    # Find all line segments and midpoints
    layers = round(object_height * scale)
    for layer in range(layers):

        z_plane = layer + 0.5

        points = []
        midpoints = []

        for facet in solid.facets:
            below = []
            above = []
            for vertex in facet.vertices:
                if (vertex[2] < z_plane):
                    below.append(vertex)
                else:
                    above.append(vertex)

            if (len(below) == 1):
                scale0 = (z_plane - below[0][2]) / (above[0][2] - below[0][2])
                scale1 = (z_plane - below[0][2]) / (above[1][2] - below[0][2])

                x0 = (above[0][0] - below[0][0]) * scale0 + below[0][0]
                y0 = (above[0][1] - below[0][1]) * scale0 + below[0][1]
                x1 = (above[1][0] - below[0][0]) * scale1 + below[0][0]
                y1 = (above[1][1] - below[0][1]) * scale1 + below[0][1]

                mx = int(round((x0 + x1) / 2 - facet.normal[0]))
                my = int(round((y0 + y1) / 2 - facet.normal[1]))

                points.append(((round(x0),round(y0)),(round(x1),round(y1))))

                if (points[-1][0] != points[-1][1]):
                    midpoints.append((mx, my))

            elif (len(above) == 1):
                scale0 = (z_plane - below[0][2]) / (above[0][2] - below[0][2])
                scale1 = (z_plane - below[1][2]) / (above[0][2] - below[1][2])

                x0 = (above[0][0] - below[0][0]) * scale0 + below[0][0]
                y0 = (above[0][1] - below[0][1]) * scale0 + below[0][1]
                x1 = (above[0][0] - below[1][0]) * scale1 + below[1][0]
                y1 = (above[0][1] - below[1][1]) * scale1 + below[1][1]

                mx = int(round((x0 + x1) / 2 - facet.normal[0]))
                my = int(round((y0 + y1) / 2 - facet.normal[1]))

                points.append(((round(x0),round(y0)),(round(x1),round(y1))))

                if (points[-1][0] != points[-1][1]):
                    midpoints.append((mx, my))

        layer_im = Image.new('RGBA',(mask.width, mask.height), (0,0,0,255))
        layer_draw = ImageDraw.Draw(layer_im)

        for couple in points:
            layer_draw.line(couple, fill=(255,255,255,255))

        for x,y in midpoints:
            floodfill(layer_im, x, y, (255,255,255,255))

        layer_im.save('./{!s}/layer{:04d}.png'.format(output_path, layer))


    return 0


def floodfill(image, x, y, color):
    if x < 0 or y < 0 or x >= image.width or y >= image.height or image.getpixel((x, y)) == color:
        return
    edge = [(x, y)]
    image.putpixel((x, y), (255,0,0,255))
    while edge:
        newedge = []
        for (x, y) in edge:
            for (s, t) in ((x+1, y), (x-1, y), (x, y+1), (x, y-1)):
                if s >= 0 and t >= 0 and s < image.width and t < image.height and image.getpixel((s, t)) == (0,0,0,255):
                    image.putpixel((s, t), color)
                    newedge.append((s, t))
        edge = newedge

###############################################################################
##
#   Assumes the order of the vertices follows the right hand rule
#
#   @param vertices     tuple/list of 3D coordinates, also a tuple
##
###############################################################################
def compute_normal(vertices):
    base = vertices[0]
    point_a = vertices[1]
    point_b = vertices[2]

    vect_a = []
    vect_b = []
    for idx in range(3):
        vect_a.append(point_a[idx] - base[idx])
        vect_b.append(point_b[idx] - base[idx])

    normal = np.cross(vect_a, vect_b)

    scale_n = math.sqrt(normal[0]**2 + normal[1]**2)

    if scale_n == 0:
        scale_n = 1

    return [normal[0] / scale_n, normal[1] / scale_n, normal[2] / scale_n]




if __name__ == '__main__':
    import sys
    sys.exit(main(sys.argv))
