#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""
Takes a video and produces a sequence of frames, skipping if requested.
Inputs:
* the input video (mandatory)
* initial frame (optional)
* last frame (optional)
* cropbox (optional)
* skip (optional)
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
    ap.add_argument('-s',"--start", type=int, default=0,
                    help="First calibration frame.")
    ap.add_argument('-f',"--finish", type=int, default=10000000,
                    help="Last calibration frame.")
    ap.add_argument('-c',"--cropbox", type=type_cropbox, default=[],help="cropping box: top,bottom,left,right")
    ap.add_argument('-i',"--input", type=str, required=True,
                    help="input video")
    ap.add_argument('-o',"--output", type=str, required=True,
                    help="output prefix. Three files are produced: one is an image with name prefix_wf.png and the other is a txt with three numbers called prefix_wb.txt, and another with the cropbox: prefix_cropbox.txt")
    ap.add_argument('-r',"--skip", type=int, default=1,
                    help="Save a frame every <skip> frames. Defaults to 1 (no skip).")

    args = vars(ap.parse_args())

    cap = cv2.VideoCapture(args["input"])
    cropbox = args["cropbox"]
    prefix  = args["output"]
    n0  = args["start"]
    n1  = args["finish"]
    skip    = args["skip"]

    # Loop until the end of the video
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
        n += 1

        if (n < n0) or (n % skip):
            continue

        if n >= n1:
            break

        x = np.array(frame)
        x = np.flip(x,axis=2)
        if not n % skip:
            imgio.imsave(f'{prefix}_{n:05d}.jpg',x,quality=90)
        
        if n > n1:
            break



