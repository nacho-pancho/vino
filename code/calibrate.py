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
import json
import os

def compute_offsets(annotations):
    sync_1 = annotations["sync_1_frame"]
    sync_2 = annotations["sync_2_frame"]

    if sync_1 < 0:
        sync_1 = 0
        print("WARNING: Assuming sync frame 1 is 0 (does not seem right but...)")

    if sync_2 < 0:
        sync_2 = 0
        print("WARNING: Assuming sync frame 2 is 0 (does not seem right but...)")

    if sync_1 == 0 and sync_2 == 0:
        print("WARNING: both sync frames are 0. Did you really annotate this?")
    
    if sync_1 < sync_2:
        # marker appeared in an earlier frame in camera 1 => it started AFTER camera 2
        # so we discard sync_1 - sync_2 frames from camera 2 to put them in sync
        offset = [0,sync_2 - sync_1]
    else:
        # vice versa
        offset = [sync_1 - sync_2,0]
    print(f"Frame offsets: input 1 {offset[0]} input 2 {offset[1]}")
    return offset



def do_white(annotations,args):

    input_fname = [None,None]
    input_fname[0] = os.path.join(basedir,annotations["input_one"])
    if annotations["input_two"]:
        input_fname[1] = os.path.join(basedir,annotations["input_two"])
        ncam = 2
    else:
        ncam = 1
    rot = [ annotations["rot1"], annotations["rot2"]]

    offset = compute_offsets(annotations)
    cropbox = annotations["crop_box"]
    prefix  = os.path.join(args["basedir"],args["output"])
    # white frame
    ini_white = annotations["ini_white_frame"]
    end_white = annotations["fin_white_frame"]    
    n_white = end_white - ini_white
    fps = [None,None]
    cap = [None,None]
    for c in range(ncam):
        print(f"camera {c}:")
        cap[c] = cv2.VideoCapture(input_fname[c])
        cap[c].set(cv2.CAP_PROP_POS_FRAMES, offset[c]+ini_white)
        fps[c] = cap[c].get(cv2.CAP_PROP_FPS) 
        print("\tframes per second: ",fps[c])
        print("\trotation:",rot[c])

        # Loop until the end of the video
        max_frame = None
        mean_frame = None
        frame = None
        n = 0
        t0 = time.time()
        while (cap[c].isOpened()) and n < n_white:
            # Capture frame-by-frame
            if frame is None:
                ret, frame = cap[c].read()
            else:
                ret, frame = cap[c].read(frame)
            if not ret:
                break
            
            color_frame = np.flip(np.array(frame),axis=2)
            h,w,_ = color_frame.shape
            if rot[c]:
                color_frame = 255*trans.rotate(color_frame,-rot,resize=True) # rotation scales colors to 0-1!!

            gray_frame = 0.25*color_frame[:,:,0] + 0.5*color_frame[:,:,1] + 0.25*color_frame[:,:,2]
            if n == 0:
                h,w,_ = color_frame.shape
                mean_frame = np.zeros(color_frame.shape)
                max_frame = np.zeros(gray_frame.shape)
            

            max_frame = np.maximum(max_frame,gray_frame)
            mean_frame += color_frame

            if not n % 10:
                fps = n/(time.time()-t0)
                imgio.imsave(os.path.join(prefix,f'input_white_{n:05d}.jpg',np.round(color_frame).astype(np.uint8)))
                print(f'frame {n:05d}  fps {fps:7.1f}')

            n += 1

        # release the video capture object
        cap[c].release()
        max_frame = np.round(max_frame).astype(np.uint8)
        if cropbox is not None:
            max_frame = max_frame[cropbox[0]:cropbox[1],cropbox[2]:cropbox[3]]
            np.savetxt(f'{prefix}_cropbox.txt',cropbox,fmt='%5d')
        imgio.imsave(f'{prefix}_white_frame.png',max_frame)

        mean_frame /= n_white
        means = np.mean(np.mean(mean_frame,axis=0),axis=0)/255
        np.savetxt(f'{prefix}_white_balance.txt',means,fmt='%8.4f')
        print(f"mean white frame color:",means)


def do_calib(cap,annotations,args):
    #calib_info = [None,None]
    #for c in range(ncam):
    #    calib_info[c] = calibrate_single_camera(cap[c],annotations,args)
    pass


if __name__ == "__main__":

    ap = argparse.ArgumentParser()
    #
    # mmetadata
    #
    ap.add_argument('-D',"--basedir", type=str, default=".",
                    help="Base directory. Everything else is relative to this one. ")
    ap.add_argument('-a',"--annotation", type=str, required=True,
                    help="Calibration JSON file produced by annotate. ")
    ap.add_argument('-o',"--output", type=str, required=True,
                    help="Output prefix for data produced by this function. This is appended to basedir.")
    ap.add_argument('-m',"--method", type=str, default="max",
                    help="Method for computing the white frame. May be average,max,or an integer for the percentile (much slower).")
    args = vars(ap.parse_args())

    json_fname = args["annotation"]
    basedir = args["basedir"]

    with open(json_fname,"r") as f:
        annotations = json.loads(f.read())
        print(json.dumps(annotations,indent="    "))
    
        prefix  = args["output"]
        # white frame
        ini_white = annotations["ini_white_frame"]
        end_white = annotations["fin_white_frame"]    
        if ini_white * end_white >= 0:
            print(f"Computing white frame using frames from {ini_white} to {end_white}")
            do_white(annotations,args)
        else:
            print("No white frame will be computed.")

        ini_calib = annotations["ini_calib_frame"]
        end_calib = annotations["fin_calib_frame"]
        if ini_calib * end_calib >= 0:
            print(f"Computing 3D calibration  using frames from {ini_calib} to {end_calib}")
            do_calib(cap,annotations,args)
        else:
            print("No white frame will be computed.")

