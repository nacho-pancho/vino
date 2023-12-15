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
    ap.add_argument('-i',"--input", type=str, required=True,
                    help="input video")
    ap.add_argument('-s',"--start", type=int, default=0,
                    help="First calibration frame.")
    ap.add_argument('-f',"--finish", type=int, default=10000000,
                    help="Last calibration frame.")
    ap.add_argument('-r',"--skip", type=int, default=6,
                    help="Save a frame every <skip> frames. Defaults to 1 (no skip).")

    ap.add_argument('-w',"--wcurve", type=str, required=True,
                    help="White calibration curve as grayscale image.")
    ap.add_argument('-b',"--wbalance", type=str, required=True,
                    help="R,G,B calibration vector (this is a txt file with 3 numbers).")
    ap.add_argument('-o',"--output", type=str, required=True,
                    help="output prefix. ")
    ap.add_argument('-e',"--exposure", type=float, default=0.25,
                    help="Global exposure, between 0 and 1. Defaults to 0.25. ")

    args = vars(ap.parse_args())

    input   = args["input"]
    n0      = args["start"]
    n1      = args["finish"]
    skip    = args["skip"]
    wffile  = args["wcurve"]
    wbfile  = args["wbalance"]
    prefix  = args["output"]
    expo    = args["exposure"]


    cap = cv2.VideoCapture(input)
    frame = None
    n = 0
    t0 = time.time()
    wframe = imgio.imread(wffile)
    wbal   = np.loadtxt(wbfile)
    print('expo',expo)
    print('wbal',wbal)
    #
    # compute normalization matrix from wframe and wbal and exposure
    #
    W = None
    while (cap.isOpened()):
        if frame is None:
            ret, frame = cap.read()
            x = np.array(frame)
            wmax = np.max(wbal)
            #print('wmax',wmax)
            W = np.zeros(x.shape)
            W[:,:,0] = 255*expo/wbal[0]
            W[:,:,1] = 255*expo/wbal[1]
            W[:,:,2] = 255*expo/wbal[2]
            #print('Wmax',np.max(W))
        else:
            ret, frame = cap.read(frame)
        if not ret:
            break
        
        n += 1
        
        if n < n0 or n % skip:
            continue
        
        if n >= n1:
            break

        x = np.flip(np.array(frame),axis=2)
        x_adj = np.minimum(255,np.maximum(0,(x * W)))
        x_adj = x_adj.astype(np.uint8)
        fps = (n-n0)/(time.time()-t0)
        print(f'frame {n:05d}  fps {fps:7.1f}')
        #plt.figure()
        #plt.subplot(2,1,1)
        #plt.imshow(x)
        #plt.subplot(2,1,2)
        #plt.imshow(x_adj)
        #plt.show()
        imgio.imsave(f'{prefix}_frame_{n:05d}_orig.png',x)
        imgio.imsave(f'{prefix}_frame_{n:05d}_adj.png',x_adj)
    cap.release()


