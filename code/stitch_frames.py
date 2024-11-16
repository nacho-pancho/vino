#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""
Correct the white balance and exposure of a frame using calibration data.
"""

# importing the necessary libraries
import os
import sys
import argparse

import numpy as np
from numpy import fft
import skimage
from skimage import io as imgio
from skimage.filters import gaussian
import matplotlib.pyplot as plt


if __name__ == "__main__":

    ap = argparse.ArgumentParser()
    ap.add_argument("-i","--input", required=True, help="image list or folder.")
    ap.add_argument("-i","--movement", required=True, help="movement from frame to frame.")
    ap.add_argument("-d","--base_dir", required=True, help="base dir for files.")
    ap.add_argument("-w","--filter_radius", type=float, default=8)

    args = ap.parse_args()
    if not os.path.exists(args.input):
        print("Input file not found.")
        sys.exit(1)
    
    if not os.path.exists(args.movement):
        print("Movements file not found.")
        sys.exit(1)
    
    #
    # detect type of input
    #
    input_files = list()
    if os.path.isdir(args.input):
        input_files = [os.path.join(args.base_dir,f) for f in os.listdir(args.input) if f.endswith(".jpg") or f.endswith(".png")]
    else:
        _,ext = os.path.splitext(args.input)
        extl = ext.lower()
        if extl == '.list':
            with open(args.input,'r') as fl:
                input_files = [l.strip() for l in fl.readlines()]
        else:
            print("Input must be a list or a folder.")

    movs = np.loadtxt(args.movement)
    dx = movs[0,:]
    dy = movs[1,:]
    print('Images to process:',len(input_files))

    #
    # adjust white frame size to input frame if necessary
    #
    if len(input_files) == 0:
        print("No input files.")
        sys.exit(1)
    h = None
    w = None
    for i,input_rel_path in enumerate(input_files):
        input_abs_path = os.path.join(args.base_dir,input_rel_path)
        input_frame = np.squeeze(skimage.img_as_float(imgio.imread(input_abs_path))[:,:,1])
        if i == 0:
            h,w = input_frame.shape
            sh = h + dy[-1]
            sw = w + dx[-1]
            print(f'New frame size: {sh}x{sw}')


