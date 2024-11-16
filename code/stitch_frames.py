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
    ap.add_argument("-m","--movement", required=True, help="movement from frame to frame.")
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
    dx = movs[0,:].astype(int)
    dy = movs[1,:].astype(int)
    print(dx)
    print(dy)
    print('Images to process:',len(input_files))
    minx,maxx = np.min(dx),np.max(dx)
    miny,maxy = np.min(dy),np.max(dy)
    offx = -minx
    offy = -miny
    #
    # adjust white frame size to input frame if necessary
    #
    if len(input_files) == 0:
        print("No input files.")
        sys.exit(1)
    h = None
    w = None
    output_image = None
    norm_image = None
    for i,input_rel_path in enumerate(input_files):
        input_abs_path = os.path.join(args.base_dir,input_rel_path)
        input_frame = imgio.imread(input_abs_path)
        if i == 0:
            h,w,_ = input_frame.shape
            sh = h + maxy - miny
            sw = w + maxx - minx
            print(f'New frame size: {sh}x{sw}')
            output_image = np.zeros((sh,sw,3),dtype=int)
            norm_image = np.zeros((sh,sw,3),dtype=int)
            output_image[0:h,0:w,:] = input_frame
            norm_image[0:h,0:w,:] = 1
        else:
            x0 = dx[i-1] + offx
            y0 = dy[i-1] + offy
            output_image[y0:y0+h,x0:x0+w,:] += input_frame
            norm_image[y0:y0+h,x0:x0+w,:] += 1
    norm_image = np.maximum(norm_image,1)
    output_image = output_image // norm_image
    print(np.max(output_image))
    plt.imsave('output_image.png',skimage.img_as_ubyte(output_image))
    

