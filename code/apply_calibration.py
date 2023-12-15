#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""
This program takes a white frame and a set of average R, G and B values and adjusts the global white balance and the illumination of a video.
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

    ap = argparse.ArgumentParser()
    #
    # mmetadata
    #
    ap.add_argument('-w',"--wframe", type=int, required=True,
                    help="White calibration frame.")
    ap.add_argument('-f',"--wbalance", type=int, required=True,
                    help="Calibration vector (this is a txt file with 3 numbers).")
    ap.add_argument('-i',"--input", type=str, required=True,
                    help="input video")
    ap.add_argument('-o',"--output", type=str, required=True,
                    help="output prefix. ")
    ap.add_argument('-e',"--exposure", type=float, default=0.5,
                    help="Global exposure, between 0 and 1. Defaults to 0.5. ")

    args = vars(ap.parse_args())

    cap = cv2.VideoCapture(args["input"])
    wffile  = args["wframe"]
    wbfile  = args["wbalance"]
    input   = args["input"]
    prefix  = args["output"]
    expo    = args["exposure"]

    frame = None
    n = 0
    t0 = time.time()
    wframe = imgio.imread(wffile)
    wbal   = np.loadtxt(wbfile)
    #
    # compute normalization matrix from wframe and wbal and exposure
    #
    wmax = np.max(wbal)
    wbal *= 128/wmax # maximum value = half
    W = expo/wframe
    W[:,:,0] *= wbal[0]
    W[:,:,1] *= wbal[1]
    W[:,:,2] *= wbal[2]

    while (cap.isOpened()):
        if frame is None:
            ret, frame = cap.read()
        else:
            ret, frame = cap.read(frame)
        if not ret:
            break

        x = np.flip(np.array(frame),axis=2) # BGR -> RGB
        # adjust global exposure

