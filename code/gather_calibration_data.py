#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""
This program takes frames directly from a video file and produces two main outputs:
* a white frame to be used for correcting for non-uniform illumination
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

if __name__ == "__main__":

    def type_cropbox(s):
        cs = [int(x.strip()) for x in s.split(',')]
        return cs
        
    ap = argparse.ArgumentParser()
    #
    # mmetadata
    #
    ap.add_argument('-s',"--start", type=int, required=True,
                    help="First calibration frame.")
    ap.add_argument('-f',"--finish", type=int, required=True,
                    help="Last calibration frame.")
    ap.add_argument('-c',"--cropbox", type=type_cropbox, required="True",help="cropping box: top,bottom,left,right")
    ap.add_argument('-i',"--input", type=str, required=True,
                    help="input video")
    ap.add_argument('-o',"--output", type=str, required=True,
                    help="output prefix. Three files are produced: one is an image with name prefix_wf.png and the other is a txt with three numbers called prefix_wb.txt, and another with the cropbox: prefix_cropbox.txt")
    ap.add_argument('-m',"--method", type=str, default="max",
                    help="Method for computing the white frame. May be average,max,or an integer for the percentile (much slower).")

    args = vars(ap.parse_args())

    cap = cv2.VideoCapture(args["input"])
    n0  = args["start"]
    n1  = args["finish"]
    cropbox = args["cropbox"]
    prefix  = args["output"]

    # Loop until the end of the video
    u = None
    frame = None
    n = 0
    t0 = time.time()
    while (cap.isOpened()):
        # Capture frame-by-frame
        if frame is None:
            ret, frame = cap.read()
        else:
            ret, frame = cap.read(frame)
        if not ret:
            break

        x = np.array(frame)
        if u is None:
            u = np.zeros(x.shape)
        
        if n >= n0:
            u = np.maximum(u,x)
        n += 1

        if not n % 120:
            fps = n/(time.time()-t0)
            print(f'frame {n:05d}  fps {fps:7.1f}')
        
        if n > n1:
            break

    # release the video capture object
    cap.release()
    u = np.round(u).astype(np.uint8)
    u = np.flip(u,axis=2) # BGR -> RGB
    u = u[cropbox[0]:cropbox[1],cropbox[2]:cropbox[3]]
    imgio.imsave(f'{prefix}_white_frame.png',u)
    means = np.mean(np.mean(u,axis=0),axis=0)
    np.savetxt(f'{prefix}_white_balance.txt',means,fmt='%8.4f')
    np.savetxt(f'{prefix}_cropbox.txt',cropbox,fmt='%5d')
    print(means)


