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

# Python 3 script
# Standard libraries
import argparse, copy, math, os.path, re, shutil, sys, time

# numpy library
import numpy as np

# PIL/Pillow library
from PIL import Image

# numpy-stl library
from stl import mesh

def main():

    parser = argparse.ArgumentParser(
        description='Python 3.5 script that slices an .stl file into .png \
        images. Dependent on numpy, numpy-stl, and pillow')
    parser.add_argument('-i', '--input-file', required=True,
        help='Input .stl file using mm units')
    parser.add_argument('-o', '--output-dir', default=None,
        help='If unspecified, the name of the input will be used as a new \
        directory')
    parser.add_argument('-m', '--mask-file', default='mask.png',
        help='Mask image of the lift platform with black as areas to avoid. Can \
        be RGBA grayscale to specify preferred regions. The resolution of       \
        the mask is used to determine the slice resolution. If                  \
        unspecified, mask.png will be sought in the current working             \
        directory.')

    parser.add_argument('-s', '--scale', default='10',
        help='Pixels per unit in the .stl file. For instance, a value of 10 \
        will provide 0.1 mm resolution if the .stl file uses mm for units. \
        If unspecified, resolution will be set to 10.')

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

    if (not re.match("^([0-9]+|[0-9]+\.[0-9]*)$", args.scale)):
        print(args.scale + " is not a valid float value for scale!")
        return 0

    # Fix default for output
    if (args.output_dir is None):
        args.output_dir = args.input_file[:-4]

    if (os.path.isdir(args.output_dir)):
        shutil.rmtree(args.output_dir)
        time.sleep(1.0)

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
        print("Unable to read the mask file!")
        return 0

    ###########################################################################
    # Files successfully opened! Now the the 3d math can begin.
    ###########################################################################

    # Mirror about the xz-plane since CAD has lower-left
    # as origin, but graphics put top-left as origin
    scene = mirror_scene(scene,(1,-1,1))

    # Scale the vertices to the actual pixels
    scale = float(args.scale)

    print("Finding best location...")

    z_min = min(scene.points[:,2:9:3].flatten())
    z_max = max(scene.points[:,2:9:3].flatten())

    best_score = 0
    best_xy = (0,0)
    best_angle = 0.0
    best_min_xy = (0,0)

    for angle in (0.0, math.radians(45)):

        bottom_transform = transform_scene(scene, 1, angle, (0,0,0))

        # Find the dimensions of the rotated scene
        x_min = min(bottom_transform.points[:,0:9:3].flatten())
        x_max = max(bottom_transform.points[:,0:9:3].flatten())
        y_min = min(bottom_transform.points[:,1:9:3].flatten())
        y_max = max(bottom_transform.points[:,1:9:3].flatten())

        bottom_transform = transform_scene(bottom_transform, scale, 0, (-x_min*scale, -y_min*scale, -z_min*scale))

        clip_x = int(round((x_max - x_min) * scale))
        clip_y = int(round((y_max - y_min) * scale))

        downscale = 4

        # change scene to [polygon][vertex][dimension]
        optimized_scene = bottom_transform.points.reshape((-1,3,3))

        # sort the scene on z-axis per polygon
        optimized_scene = optimized_scene[np.arange(len(optimized_scene))[:,np.newaxis], np.argsort(optimized_scene[:, :, 2])]

        bottom_image_data = get_slice_image_data(optimized_scene, 0.5, mask.width, mask.height)

        # Crop the bottom image to the object size
        cropped_bottom = (bottom_image_data[:clip_y:downscale,:clip_x:downscale] == 1)

        # Get the red channel of the mask as the gray value and normalize
        mask_pixels = np.uint32(np.array(mask)[::downscale,::downscale,0])

        cb_x = cropped_bottom.shape[1]
        cb_y = cropped_bottom.shape[0]

        for x in range(0, mask_pixels.shape[1] - cb_x):
            for y in range(0, mask_pixels.shape[0] - cb_y):
                accum = mask_pixels[y:y+cb_y,x:x+cb_x][cropped_bottom].sum()

                if (best_score < accum):
                    best_xy = (x*downscale,y*downscale)
                    best_score = accum
                    best_angle = angle
                    best_min_xy = (x_min, y_min)

    print(best_xy, math.degrees(best_angle), best_score)

    print("Moving object to best position...")

    best_scene = transform_scene(scene, 1, best_angle, (0,0,0))
    best_scene = transform_scene(best_scene, scale, 0, (-best_min_xy[0]*scale + best_xy[0], -best_min_xy[1]*scale + best_xy[1], -z_min*scale))

    print("Generating layers...")
    num_layers = int(round((z_max - z_min) * scale))

    # change scene to [polygon][vertex][dimension]
    optimized_scene = best_scene.points.reshape((-1,3,3))

    # sort the scene on z-axis per polygon
    optimized_scene = optimized_scene[np.arange(len(optimized_scene))[:,np.newaxis], np.argsort(optimized_scene[:, :, 2])]

    for layer_z in range(num_layers):
        layer_image_data = get_slice_image_data(optimized_scene, layer_z + 0.5, mask.width, mask.height)
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

    R_data = (image_data * 255)

    Image.fromarray(R_data).convert('RGBA').save(filename)


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
#   Scales, rotates, and then translates the object per the parameters
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
#   Produces a slice image at the z-height. Determines the relevant facets
#   for the layer. For every row of pixels, a ray is shot in the x-direction
#   and finds x-coordinates where the ray passes through facets/triangles.
#   After passing through an odd number of facets, the ray is in the solid
#   portion of the object, and outside after passing an even number of facets.
#
#   @param scene        stl scene
#   @param z_height     z height of the layer
#   @param width        image width
#   @param height       image height
##
###############################################################################
def get_slice_image_data(scene, z_height, width, height):
    image = np.zeros((height, width), dtype='uint8')

    new_facet_points = scene

    # filter the matrix
    new_facet_points = new_facet_points[new_facet_points[:,0,2] <= z_height]
    new_facet_points = new_facet_points[new_facet_points[:,2,2] > z_height]

    # split the matrix based on whether the base is above or
    # below the z-height plane
    base_above = new_facet_points[new_facet_points[:,1,2] <= z_height]
    base_below = new_facet_points[new_facet_points[:,1,2] > z_height]

    scale_below = (z_height - base_below[:,0,2]).reshape(-1,1) / (base_below[:,1:3,2] - base_below[:,0,2].reshape(-1,1))

    scale_above = (z_height - base_above[:,0:2,2]) / (base_above[:,2,2].reshape(-1,1) - base_above[:,0:2,2])

    # should be a [polygon][vertex][x,y]
    above_coords = (base_above[:,2,0:2].reshape(-1,1,2) - base_above[:,0:2,0:2]) * scale_above.reshape(-1,2,1) + base_above[:,0:2,0:2]
    below_coords = (base_below[:,1:3,0:2] - base_below[:,0,0:2].reshape(-1,1,2)) * scale_below.reshape(-1,2,1) + base_below[:,0,0:2].reshape(-1,1,2)

    coords = np.concatenate((above_coords, below_coords), axis=0)

    # Filter out points where the y values don't have a difference
    coords = coords[coords[:,0,1].astype(int) != coords[:,1,1].astype(int)]

    # sort the coords so smallest y is listed first in the vertex dimension
    coords = coords[np.arange(len(coords))[:,np.newaxis], np.argsort(coords[:,:,1])].reshape(-1,4)

    #   The line between the coords pairs marks the transition points (outside
    #   -> inside or inside -> outside). Find all the points on that line and
    #   mark them

    x_trans = []

    total_row_list = np.arange(1, height + 1, dtype='float')

    for x0,y0,x1,y1 in coords:
        row_list = total_row_list[int(y0):int(y1)]
        col_list = (row_list - y0) * ((x1 - x0) / (y1 - y0)) + x0
        x_trans.append((col_list,row_list))

    transition_points = np.round(np.hstack(x_trans)).T.astype(int)

    # sort the coords by first by ascending y then by ascending x.
    # Remove redundant y dimension, since these coordinates will have y-pairs
    transition_points = transition_points[np.lexsort((transition_points[:,0], transition_points[:,1]))].reshape(-1,4)[:,0:3]

    char_one = np.uint8(1)
    for x0,y0,x1 in transition_points:
        image[y0,x0:x1] = char_one

    return image


if __name__ == '__main__':
    sys.exit(main())
