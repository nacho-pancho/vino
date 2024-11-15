#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""
Correct the white balance and exposure of a frame using calibration data.
"""

# importing the necessary libraries
import os
import sys
import argparse
import json

import numpy as np
import skimage
from skimage import io as imgio


if __name__ == "__main__":

    ap = argparse.ArgumentParser()
    ap.add_argument("-j","--calib_json", required=True, help="JSON file with calibration data.")
    ap.add_argument("-r","--rotate",type=int,  default=0, help="Rotate white frame if necessary.")
    ap.add_argument("-s","--scale",type=int,  default=1, help="Scale down by this integer factor, if necessary.")
    ap.add_argument("-w","--calib_frame", required=True, help="Calibration white frame")
    ap.add_argument("-i","--input", required=True, help="image, image list or folder.")
    ap.add_argument("-o","--output_prefix",required=True,help="output prefix. May produce one or more images depending on input.")

    args = ap.parse_args()
    calibration_json  = args.calib_json
    calibration_frame = args.calib_frame

    if not os.path.exists(calibration_json):
        print(f"Calibration file {calibration_json} not found.")
        sys.exit(1)
    with open(calibration_json,'r') as fc:
        calibration = json.loads(fc.read())


    if not os.path.exists(calibration_frame):
        print(f"Calibration image file {calibration_frame} not found.")
        sys.exit(1)
    white_frame = skimage.img_as_float(imgio.imread(calibration_frame))

    if not os.path.exists(args.input):
        print("Input file not found.")
        sys.exit(1)
    
    #
    # detect type of input
    #
    input_files = list()
    if os.path.isdir(args.input):
        input_files = [os.path.join(args.input,f) for f in os.listdir(args.input) if f.endswith(".jpg") or f.endswith(".png")]
    else:
        _,ext = os.path.splitext(args.input)
        extl = ext.lower()
        if extl == '.list':
            with open(args.input,'r') as fl:
                input_files = [l.strip() for l in fl.readlines()]
        elif extl == '.jpg' or extl == '.png':
            input_files = [args.input]
    print('Images to process:',len(input_files))

    #
    # prepare calibration for max speed
    #     
    white_balance = calibration["white_balance"]
    white_balance['red'] /= 255
    white_balance['green'] /= 255
    white_balance['blue'] /= 255

    inv_white_frame = 1/np.maximum(white_frame,1e-3)
    if args.rotate == 90:
        inv_white_frame = np.rot90(inv_white_frame)
    elif args.rotate == 180:
        inv_white_frame = np.fliplr(np.flipud(inv_white_frame))
    elif args.rotate == 270:
        inv_white_frame = np.rot90(inv_white_frame)
        inv_white_frame = np.fliplr(np.flipud(inv_white_frame))

    #
    # adjust white frame size to input frame if necessary
    #
    if len(input_files) == 0:
        print("No input files.")
        sys.exit(1)
    
    first_frame = skimage.img_as_float(imgio.imread(input_files[0]))
    wh, ww = inv_white_frame.shape[:2]
    ih, iw = first_frame.shape[:2]
    if wh != ih and ww != iw:
        wratio = ww/wh
        iratio = iw/ih
        if wratio != iratio:
            print(f"Input frames have different aspect ratio ({iratio}) than calibration frame ({wratio}).")
            sys.exit(1)
        else:
            if args.scale > 1:
                inv_white_frame = skimage.transform.rescale(inv_white_frame,1/args.scale)
            wh, ww = inv_white_frame.shape[:2]
            if wh != ih and ww != iw:
                print(f"Wrong scaling factor: input frames have different shape ({wh}x{ww}) than calibration frame ({ih}x{iw}).")
                print(f"Wrong scaling factor.")
                sys.exit(1)

    norm_frame = np.empty((*inv_white_frame.shape,3))
    norm_frame[:,:,0] = inv_white_frame*(1/white_balance["red"])
    norm_frame[:,:,1] = inv_white_frame*(1/white_balance["green"])
    norm_frame[:,:,2] = inv_white_frame*(1/white_balance["blue"])

    output_dir = os.path.dirname(args.output_prefix)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    if output_dir == '':
        output_dir = '.'
    print(f'Output directory: {output_dir}')
    for i,input_file in enumerate(input_files):
        print(f'{i+1}/{len(input_files)}: {input_file}')
        input_frame = skimage.img_as_float(imgio.imread(input_file))
        output_frame = np.maximum(0,np.minimum(1,input_frame * norm_frame))
        output_file = os.path.join(args.output_prefix,os.path.basename(input_file))
        imgio.imsave(output_file,skimage.img_as_ubyte(output_frame))


