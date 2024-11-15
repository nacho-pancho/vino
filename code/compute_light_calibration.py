#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""
This program takes frames directly from a video file and produces two main outputs:
* a white frame to be used for correcting for non-uniform illumination; this is a grayscale map computed as 0.5G+0.25R+0.25B
* a correction factor to the R G and B factors so that a white balance is defined independently from the camera settings
The user must also provide:
* the starting frame in the video
* the final frame in the video
* a cropping region
"""

# importing the necessary libraries
import argparse
import sys
import time
import numpy as np
import cv2
import matplotlib.pyplot as plt
from skimage import io as imgio
from skimage import transform as trans
import skimage
import json
import os
from vutils import *


def do_white(input_fname, ini_frame, end_frame, output_prefix):
    calibration = dict()
    res_fac = 8 # args["rescale_factor"]
    input_fname = input_fname
    n_frames = end_frame - ini_frame
    cap = cv2.VideoCapture(input_fname)
    t_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    fps = cap.get(cv2.CAP_PROP_FPS) 
    print("\tframes per second: ",fps)
    max_frame = None
    t0 = time.time()
    mean_red = 0
    mean_green = 0
    mean_blue = 0
    num_valid = 0
    n = 0
    calibration = dict()
    while (cap.isOpened()) and n < n_frames: # ----- loop over frames
        # Capture frame-by-frame
        ret, input_frame = cap.read()
        if not ret:
            print("Error reading frame!")
            break
        h,w,_ = input_frame.shape
        small_frame = cv2.resize(input_frame,(w//res_fac,h//res_fac))
        hr,wr = small_frame.shape[:2]
        # BGR -> RGB
        small_frame = np.flip(np.array(small_frame),axis=2)
        if max_frame is None:
            gray_frame = np.zeros((hr,wr),dtype=np.uint8)
            valid_pixels = np.zeros_like(gray_frame,dtype=bool)
            max_frame = np.zeros_like(gray_frame,dtype=np.uint8)

        gray_frame[:] = np.squeeze((np.sum(small_frame,axis=2))//3) # R + G + B
        valid_pixels[:] = gray_frame < 255
        num_valid  += np.sum(valid_pixels)
        mean_red   += np.sum(small_frame[:,:,0]*valid_pixels)
        mean_green += np.sum(small_frame[:,:,1]*valid_pixels)
        mean_blue  += np.sum(small_frame[:,:,2]*valid_pixels)
        max_frame[:] = np.maximum(max_frame,gray_frame)

        if not n % 5:
            _fps = n/(time.time()-t0)
            imgio.imsave(f'{output_prefix}_frame_{n+ini_frame:05d}.jpg',small_frame)
            print(f'frame {n+ini_frame:05d}  fps {_fps:7.1f}')

        n += 1
        # release the video capture object
    cap.release()
    i0r = 0
    j0r = 0
    i1r = max_frame.shape[0]
    j1r = max_frame.shape[1]
    hr = i1r - i0r
    wr = j1r - j0r
    #
    # upsample
    #
    # save calibration white frame
    #
    max_frame_up = cv2.resize(max_frame,(w,h))
    wf_avg_preview = f'{output_prefix}_average_white_frame.png'
    imgio.imsave(wf_avg_preview,skimage.img_as_ubyte(max_frame_up))
    means = (mean_red/num_valid,mean_green/num_valid,mean_blue/num_valid)
    #
    # save calibration params
    #
    calibration["white_balance"] = {"red":means[0],"green":means[1],"blue":means[2]}
    print(f"mean white frame color:",means)
    #
    # compute illumination curve as a second order curve from the non-saturated pixels
    #
    # we build a regression problem of the form (1,r,c,r^2,r*c,c^2) -> L
    # using as r and c the _unscaled_ and _uncropped_ row and column indexes
    # 
    ri = np.arange(i0r,i1r)/hr
    ci = np.arange(j0r,j1r)/wr
    Ri,Ci = np.meshgrid(ri,ci,indexing='ij')
    Ri = Ri.ravel()
    Ci = Ci.ravel()
    L = max_frame.ravel()
    print(len(L))
    VP = np.flatnonzero(L < 255)
    L = L[VP]
    Ri = Ri[VP]
    Ci = Ci[VP]
    N = len(VP)
    TN = np.prod(max_frame.shape)
    print('Total pixels:',TN,' Valid pixels:',N)
    X = np.ones((N,6))
    X[:,1] = Ri
    X[:,2] = Ci
    X[:,3] = Ri**2
    X[:,4] = Ri*Ci
    X[:,5] = Ci**2
    a,rss,rank,sval = np.linalg.lstsq(X,L,rcond=None)
    calibration["white_frame_parameters"] = a.tolist()

    ri = np.arange(h)/h
    ci = np.arange(w)/w
    Ri,Ci = np.meshgrid(ri,ci,indexing='ij')
    white_frame = a[0] + a[1]*Ri + a[2]*Ci + a[3]*(Ri**2) + a[4]*(Ri*Ci) + a[5]*(Ci**2)
    wf_matrix_fname = f'{output_prefix}_white_frame_par.npy'
    np.save(wf_matrix_fname,white_frame)
    # very likely, the computed parametric white frame falls out of the valid range if there were many saturated
    # pixels, so we downscale the output image. We will use the saved matrix (npy), not this, for normalization
    white_frame = (white_frame*(255/np.max(white_frame))).astype(np.uint8)
    wf_image_fname = f'{output_prefix}_white_frame_par.png'
    imgio.imsave(wf_image_fname,white_frame)
    return calibration



if __name__ == "__main__":

    ap = argparse.ArgumentParser()
    #
    # mmetadata
    #
    ap.add_argument("input_video", type=str)
    ap.add_argument("ini_frame", type=int)
    ap.add_argument("end_frame", type=int)
    ap.add_argument('-R',"--rescale-factor", type=int, default=8,
                    help="Reduce resolution this many times (defaults to 8 -- brutal). ")
    ap.add_argument('-o',"--output_prefix", type=str, default="output_",
                    help="Output prefix for data produced by this function.")
    ap.add_argument('-m',"--method", type=str, default="max",
                    help="Method for computing the white frame. May be average,max,or an integer for the percentile (much slower).")
    args = ap.parse_args()
    calibration = do_white(args.input_video, args.ini_frame, args.end_frame, args.output_prefix)
    txt = json.dumps(calibration,indent=4)
    print("CALIBRATION:",txt)    
    with open(f'{args.output_prefix}_white.json',"w") as f:
        f.write(txt)

