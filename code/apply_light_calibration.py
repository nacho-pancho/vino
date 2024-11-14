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
from skimage import io as imgio
from skimage import transform as trans

def correct(input_frame,white_frame,white_balance):
    #
    # apply rectification
    #
    input_frame[:,:,0] = (input_frame[:,:,0]/white_frame)*(255/white_balance["red"]) # both white balance and white frame are 0-255
    input_frame[:,:,1] = (input_frame[:,:,1]/white_frame)*(255/white_balance["green"])
    input_frame[:,:,2] = (input_frame[:,:,2]/white_frame)*(255/white_balance["blue"])            
    input_frame = np.maximum(0,np.minimum(255,input_frame)).astype(np.uint8)
    return input_frame


if __name__ == "__main__":

    ap = argparse.ArgumentParser()
    ap.add_argument("calibration_dir", help="directory containing calibration information")
    ap.add_argument("input", help="input image")
    ap.add_argument("-o","--output",default="output.png",help="output image.")

    args = ap.parse_args()
    calibration_file = os.path.join(args.calibration_dir,"calibration.json")
    if not os.path.exists(calibration_file):
        print("Calibration file not found.")
        sys.exit(1)
    
    if not os.path.exists(args.input):
        print("Input file not found.")
        sys.exit(1)
    
    with open(calibration_file,'r') as fc:
        calibration = json.loads(fc.read())

    input_frame = imgio.imread(args.input)    
    white_balance = calibration["white_balance"]
    white_balance['red'] /= 255
    white_balance['green'] /= 255
    white_balance['blue'] /= 255
    white_frame_file = calibration["white_frame"]
    white_frame = np.load(os.path.join(args.calibration_dir,white_frame_file))
    output_frame = correct(input_frame,white_frame,white_balance)
    imgio.imsave(args.output,output_frame)


