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
from skimage import transform as trans

def correct(input_frame,white_frame,white_balance):
    #
    # apply rectification
    #
    input_frame[:,:,0] = (input_frame[:,:,0]/white_frame)*(1/white_balance["red"]) # both white balance and white frame are 0-255
    input_frame[:,:,1] = (input_frame[:,:,1]/white_frame)*(1/white_balance["green"])
    input_frame[:,:,2] = (input_frame[:,:,2]/white_frame)*(1/white_balance["blue"])            
    input_frame = np.maximum(0,np.minimum(1,input_frame))
    return input_frame


if __name__ == "__main__":

    ap = argparse.ArgumentParser()
    ap.add_argument("calib_params", help="JSON file with calibration data.")
    ap.add_argument("calib_white", help="Calibration white frame")
    ap.add_argument("input", help="input image")
    ap.add_argument("-o","--output",default="output.png",help="output image.")

    args = ap.parse_args()
    calibration_file = args.calib_params
    calibration_img  = args.calib_white

    if not os.path.exists(calibration_file):
        print(f"Calibration file {calibration_file} not found.")
        sys.exit(1)

    if not os.path.exists(calibration_img):
        print(f"Calibration image file {calibration_img} not found.")
        sys.exit(1)

    if not os.path.exists(args.input):
        print("Input file not found.")
        sys.exit(1)
    
    with open(calibration_file,'r') as fc:
        calibration = json.loads(fc.read())

    white_balance = calibration["white_balance"]
    white_balance['red'] /= 255
    white_balance['green'] /= 255
    white_balance['blue'] /= 255
    #white_frame = np.load(os.path.join(args.calibration_dir,white_frame_file))
    white_frame = imgio.imread(calibration_img)
    input_frame = skimage.img_as_float(imgio.imread(args.input))
    print('input shape',input_frame.shape)
    white_frame = skimage.img_as_float(imgio.imread(calibration_img))
    print('white shape',white_frame.shape)
    output_frame = correct(input_frame,white_frame,white_balance)
    imgio.imsave(args.output,skimage.img_as_ubyte(output_frame))


