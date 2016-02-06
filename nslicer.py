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

import argparse
import numpy as np
import math
from PIL import Image, ImageDraw, ImageChops
from stl import mesh
import traceback
import copy
import re, os.path, sys, shutil

WIDTH_MM = 102.4
HEIGHT_MM = 76.8

Z_RESOLUTION_MM = 0.1

def main():

    parser = argparse.ArgumentParser(
        description='Slices a .stl file into .png images.')
    parser.add_argument('-i', '--input-file', required=True,
        help='Input .stl file')
    parser.add_argument('-o', '--output-dir', default=None,
        help='If unspecified, the name of the input will be used as a new directory')
    parser.add_argument('-m', '--mask-file', default='mask.png',
        help='Mask file with black as areas to avoid. Can be grayscale to specify preferred regions \
        If unspecified, mask.png will be sought in the current working directory.')

    args = parser.parse_args()

    if (not re.match(".+?\.stl", args.input_file)):
        print("Please specify an input .stl file.")
        return 0

    if (not os.path.isfile(args.input_file)):
        print(args.input_file + " does not exist!")
        return 0

    if (not os.path.isfile(args.mask_file)):
        print(args.mask_file + " does not exist!")
        return 0

    # Fix default for output
    if (args.output_dir is None):
        args.output_dir = args.input_file[:-4]

    if (os.path.isdir(args.output_dir)):
        shutil.rmtree(args.output_dir)

    os.mkdir(args.output_dir)

    # Open the input files
    try:
        scene = mesh.Mesh.from_file(args.input_file)
    except:
        print("Unable to read the .stl file!")
        return 0

    try:
        mask = Image.open(args.mask_file)
    except:
        print("Unable to read the mask.png file!")
        return 0

    ###########################################################################
    # Files successfully opened! Now the the 3d math can begin.
    ###########################################################################

    # Note that we also mirror about the x-axis since CAD has lower-left
    # as origin, but graphics put top-left as origin
    scene = mirror_scene(scene,(1,-1,1))

    # Scale the vertices to the actual pixels
    scale = mask.width / WIDTH_MM

    # Find the dimensions of the scene without rotation
    x_min = min(scene.points[:,0:9:3].flatten())
    x_max = max(scene.points[:,0:9:3].flatten())
    y_min = min(scene.points[:,1:9:3].flatten())
    y_max = max(scene.points[:,1:9:3].flatten())
    z_min = min(scene.points[:,2:9:3].flatten())
    z_max = max(scene.points[:,2:9:3].flatten())

    print("Translating object...")
    bottom_transform = transform_scene(scene, scale, 0, (-x_min*scale, -y_min*scale, -z_min*scale))

    clip_x = int(round((x_max - x_min) * scale))
    clip_y = int(round((y_max - y_min) * scale))

    print("Generating bottom image...")
    bottom_image_data = get_slice_image_data(bottom_transform, 0.5, mask.width, mask.height)

    print("Finding best location...")

    # Crop the bottom image to the object size
    cropped_bottom = np.uint32(bottom_image_data[:clip_y,:clip_x].flatten())

    # Get the red channel of the mask as the gray value and normalize
    mask_pixels = np.uint32(np.array(mask)[:,:,0])

    best_score = 0
    best_xy = (0,0)

    for x in range(0, mask.width - clip_x, 4):
        for y in range(0, mask.height - clip_y, 4):
            accum = mask_pixels[y:y+clip_y, x:x+clip_x].flatten().dot(cropped_bottom)

            if (best_score < accum):
                best_xy = (x,y)
                best_score = accum

    print(best_xy, best_score)

    print("Moving object to best position...")

    best_scene = transform_scene(scene, scale, 0, (-x_min*scale + best_xy[0], -y_min*scale + best_xy[1], -z_min*scale))

    print("Generating layers...")
    num_layers = int(round((z_max - z_min) * scale))

    for layer_z in range(num_layers):
        layer_image_data = get_slice_image_data(best_scene, layer_z + 0.5, mask.width, mask.height)
        save_image_data(layer_image_data, '{!s}/layer{:04d}.png'.format(args.output_dir, layer_z))


###############################################################################
##
#   Saves a numpy array to .png image
#
#   @param image_data       2D numpy array consisting of single 0.0 - 1.0
#                           elements (no color)
#
#   @param filename         Name of .png file
##
###############################################################################
def save_image_data(image_data, filename):
    width = len(image_data[0])
    height = len(image_data)
    num_pixels = width * height

    RGB_data = np.dot(image_data.reshape((num_pixels,1)),[[255,255,255]])
    RGBA_data = np.concatenate((RGB_data, np.array([[255]]*num_pixels)), axis=1)

    reshaped_data = np.uint8(RGBA_data.reshape((height,width,4)))

    Image.fromarray(reshaped_data).save(filename)


###############################################################################
##
#   Mirrors the object over a particular axis
#
#   @param scene        stl scene
#   @param mirror       3-tuple of which to mirror over
##
###############################################################################
def mirror_scene(scene, mirror):
    transform = np.array([[mirror[0],  0,           0],
                          [0,          mirror[1],   0],
                          [0,          0,           mirror[2]]])

    num_facets = len(scene.points)
    num_points = num_facets * 3

    tmp_scene = copy.copy(scene)

    tmp_scene.points = np.dot(tmp_scene.points.reshape((num_points, 3)), transform.transpose()).reshape(num_facets, 9)
    tmp_scene.update_normals()

    return tmp_scene


###############################################################################
##
#   Scales, rotates, and translates the object per the parameters
#
#   @param scene        stl scene
#   @param scale        amount to scale the scene
#   @param angle        amount to rotate the object around the z-axis
#   @param translate    3-tuple of (x,y,z) to move (relative to pixels)
##
###############################################################################
def transform_scene(scene, scale, angle, translate):
    transform = np.array([[scale * math.cos(angle),  scale * math.sin(angle),  0,     translate[0]],
                          [scale * -math.sin(angle), scale * math.cos(angle),  0,     translate[1]],
                          [0,                        0,                        scale, translate[2]]])

    num_facets = len(scene.points)
    num_points = num_facets * 3

    tmp_scene = copy.copy(scene)

    tmp_scene.points = np.dot(np.concatenate((tmp_scene.points.reshape((num_points, 3)),np.ones((num_points,1))),axis=1), transform.transpose()).reshape(num_facets, 9)
    tmp_scene.update_normals()

    return tmp_scene

###############################################################################
##
#   Produces a slice image at the z-height
#
#   @param scene        stl scene
#   @param z_height     z height of the layer
#   @param width        image width
#   @param height       image height
##
###############################################################################
def get_slice_image_data(scene, z_height, width, height):
    image = np.array([[0.0]*width]*height)

    # Remove facets that are too low in z
    facet_points = scene.points[scene.points[:,2:9:3].min(axis=1)<=z_height]

    # Remove facets that are too high in z
    facet_points = facet_points[facet_points[:,2:9:3].max(axis=1)>z_height]

    x_list = [[] for _ in range(height)]

    for facet in facet_points:
        above = []
        below = []
        for vertex in [facet[0:3], facet[3:6],facet[6:9]]:
            if (vertex[2] <= z_height):
                below.append(vertex)
            else:
                above.append(vertex)

        if (len(below) == 1):
            scale0 = (z_height - below[0][2]) / (above[0][2] - below[0][2])
            scale1 = (z_height - below[0][2]) / (above[1][2] - below[0][2])

            x0 = (above[0][0] - below[0][0]) * scale0 + below[0][0]
            y0 = (above[0][1] - below[0][1]) * scale0 + below[0][1]
            x1 = (above[1][0] - below[0][0]) * scale1 + below[0][0]
            y1 = (above[1][1] - below[0][1]) * scale1 + below[0][1]

        else:
            scale0 = (z_height - below[0][2]) / (above[0][2] - below[0][2])
            scale1 = (z_height - below[1][2]) / (above[0][2] - below[1][2])

            x0 = (above[0][0] - below[0][0]) * scale0 + below[0][0]
            y0 = (above[0][1] - below[0][1]) * scale0 + below[0][1]
            x1 = (above[0][0] - below[1][0]) * scale1 + below[1][0]
            y1 = (above[0][1] - below[1][1]) * scale1 + below[1][1]

        y_min = int(min(y0, y1))+1
        y_max = int(max(y0, y1))+1
        for row in range(y_min,y_max):
            x = ((row - y0) / (y1 - y0)) * (x1 - x0) + x0
            x_list[row].append(int(round(x)))



    for row in range(height):
        # Pair the list elements [(idx0, idx1), (idx2,idx3)...]
        x_pairs = zip(*[iter(sorted(x_list[row]))]*2)

        # Draw lines between the pairs
        #~ for start, end in x_pairs:
            #~ for idx in range(start, end):
                #~ image[row][idx] = 1.0

        for start, end in x_pairs:
            image[row][start:end] = [1.0]*(end - start)

    return image


if __name__ == '__main__':
    sys.exit(main())
