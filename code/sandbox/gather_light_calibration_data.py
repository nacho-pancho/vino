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

if __name__ == "__main__":

    def type_cropbox(s):
        cs = [int(x.strip()) for x in s.split(',')]
        return cs
        
    ap = argparse.ArgumentParser()
    #
    # mmetadata
    #
    ap.add_argument('-s',"--start", type=int, required=True,
                    help="First calibration frame in seconds.")
    ap.add_argument('-f',"--finish", type=int, required=True,
                    help="Last calibration frame in seconds.")
    ap.add_argument('-c',"--cropbox", type=type_cropbox, default=None,help="cropping box: top,bottom,left,right")
    ap.add_argument('-i',"--input", type=str, required=True,
                    help="input video")
    ap.add_argument('-o',"--output", type=str, required=True,
                    help="output prefix. Three files are produced: one is an image with name prefix_wf.png and the other is a txt with three numbers called prefix_wb.txt, and another with the cropbox: prefix_cropbox.txt")
    ap.add_argument('-m',"--method", type=str, default="max",
                    help="Method for computing the white frame. May be average,max,or an integer for the percentile (much slower).")
    ap.add_argument('-R','--rotation',type=int,default=0,help="Optionally rotate the frames this many degrees clockwise (Defualts to 0).")

    args = vars(ap.parse_args())

    cap = cv2.VideoCapture(args["input"])
    fps = cap.get(cv2.CAP_PROP_FPS) 
    print("frames per second: ",fps)
    n0  = (np.round(args["start"]*fps))
    n1  = int(np.round(args["finish"]*fps))
    cropbox = args["cropbox"]
    prefix  = args["output"]
    rot     = args["rotation"]
    # Loop until the end of the video
    max_frame = None
    mean_frame = None
    frame = None
    n = 0
    t0 = time.time()
    print(f"going from frame {n0} to frame {n1}")
    while (cap.isOpened()):
        # Capture frame-by-frame
        if frame is None:
            ret, frame = cap.read()
        else:
            ret, frame = cap.read(frame)
        if not ret:
            break
        
        n += 1
        if n < n0:
            continue

        color_frame = np.flip(np.array(frame),axis=2)
        h,w,c = color_frame.shape
        if n == n0:
            print(f"input frame dimensions: height={h} width={w} channels={c}")        
            print(f"rotating {rot} degrees.")
        if rot:
            color_frame = 255*trans.rotate(color_frame,-rot,resize=True) # rotation scales colors to 0-1!!

        gray_frame = 0.25*color_frame[:,:,0] + 0.5*color_frame[:,:,1] + 0.25*color_frame[:,:,2]
        if n == n0:
            h,w,c = color_frame.shape
            print(f"output frame dimensions: height={h} width={w} channels={c}")        
            mean_frame = np.zeros(color_frame.shape)
            max_frame = np.zeros(gray_frame.shape)
        

        max_frame = np.maximum(max_frame,gray_frame)
        mean_frame += color_frame

        if not n % 10:
            fps = (n-n0)/(time.time()-t0)
            imgio.imsave(f'{prefix}_input_white_{n:05d}.jpg',np.round(color_frame).astype(np.uint8))
            print(f'frame {n:05d}  fps {fps:7.1f}')
        
        if n > n1:
            break

    # release the video capture object
    cap.release()
    max_frame = np.round(max_frame).astype(np.uint8)
    if cropbox is not None:
        max_frame = max_frame[cropbox[0]:cropbox[1],cropbox[2]:cropbox[3]]
        np.savetxt(f'{prefix}_cropbox.txt',cropbox,fmt='%5d')
    imgio.imsave(f'{prefix}_white_frame.png',max_frame)

    mean_frame /= (n-n0)
    means = np.mean(np.mean(mean_frame,axis=0),axis=0)/255
    np.savetxt(f'{prefix}_white_balance.txt',means,fmt='%8.4f')
    print(f"mean white frame color:",means)


