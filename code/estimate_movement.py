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
    ap.add_argument("-d","--base_dir", required=True, help="base dir for files.")
    ap.add_argument("-w","--filter_radius", type=float, default=8)

    args = ap.parse_args()
    if not os.path.exists(args.input):
        print("Input file not found.")
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

    print('Images to process:',len(input_files))

    #
    # adjust white frame size to input frame if necessary
    #
    if len(input_files) == 0:
        print("No input files.")
        sys.exit(1)
    
    prev_frame = None
    prev_frame_fft = None
    xi = 0
    yi = 0
    x = list()
    y = list()
    for i,input_rel_path in enumerate(input_files):
        x.append(xi)
        y.append(yi)
        input_abs_path = os.path.join(args.base_dir,input_rel_path)
        #print(f'{i+1}/{len(input_files)}: {input_abs_path}')
        input_frame = np.squeeze(skimage.img_as_float(imgio.imread(input_abs_path))[:,:,1])
        h,w = input_frame.shape
        #input_frame = skimage.transform.resize(input_frame,(h,w))
        h,w = input_frame.shape
        #input_frame = input_frame[h//2-512:h//2+512,:]
        input_frame = input_frame[h//2-512:h//2+512,:]
        h,w = input_frame.shape
        curr_frame_fft = fft.fft2(input_frame)
        if i > 0:
            # estimate movement
            cross_power_spectrum_fft = curr_frame_fft * np.conj(prev_frame_fft)
            cross_power_spectrum_fft = cross_power_spectrum_fft / np.abs(cross_power_spectrum_fft)
            cross_power_spectrum = np.abs(fft.ifft2(cross_power_spectrum_fft))
            cross_power_spectrum = gaussian(cross_power_spectrum,sigma=args.filter_radius,mode='wrap')
            cross_power_spectrum[0,0] = 0
            flat_max = np.argmax(cross_power_spectrum)
            imax = flat_max // w
            jmax = flat_max % w
            if imax > h//2:
                imax = imax - h
            if jmax > w//2:
                jmax = jmax - w
            #imax *= 8
            #jmax *= 8
            yi -= imax
            xi -= jmax            
            print(f'from {i} to {i-1}: {imax:5d} {jmax:5d}')
            if False:
                plt.figure()
                plt.imshow((cross_power_spectrum))
                plt.colorbar()
                plt.show()
        prev_frame_fft = curr_frame_fft
    plt.figure()
    plt.scatter(x,y)
    plt.show()
    np.savetxt('movement.txt',np.array([x,y]),fmt='%8.2f')


