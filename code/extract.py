#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""
This program takes one or two video inputs and already computed calibration information
and extracts pairs of frames at given intervals, correcting the illumination and white balance
using the calibration information.
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


def fast_rot(img,rot):
    if rot < 0:
        rot += 360
    if rot == 0:
        return img
    elif rot == 90:
        return np.transpose(np.flip(img,axis=0),(1,0,2))
    elif rot == 270:
        return np.flip(np.transpose(img,(1,0,2)),axis=0)
    elif rot == 180:
        return np.flip(np.flip(img,axis=0),axis=1)
    else:
        return (255*trans.rotate(img,rot,resize=True)).astype(np.uint8) # rotation scales colors to 0-1!!


def extract(annotations,calibration,args):

    input_fname = [None,None]
    input_a = annotations["input_a"]
    take = annotations["take"]
    res_fac = args["rescale_factor"]
    input_fname[0] = os.path.join(basedir,f'{input_a}/{input_a}_toma{take}_parte1.mp4')
    prefix = args["output"]
    if annotations["input_b"]:
        input_b = annotations["input_b"]
        input_fname[1] = os.path.join(basedir,f'{input_b}/{input_b}_toma{take}_parte1.mp4')
        if prefix is None:
            prefix  = os.path.join(args["basedir"],f'{input_a}+{input_b}.calib')
        ncam = 2
    else:
        if prefix is None:
            prefix  = os.path.join(args["basedir"],f'{input_a}.calib')
        ncam = 1
    rot = [ annotations["rot1"], annotations["rot2"]]
    if not os.path.exists(prefix):
        os.makedirs(prefix,exist_ok=True)

    offset = compute_offsets(annotations)
    cropbox = annotations["crop_box"]
    # white frame
    ini_white = annotations["ini_white_frame"]
    end_white = annotations["fin_white_frame"]
    n_white = end_white - ini_white
    #n_white = 20 # DEBUG
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
        print(ini_white)    
        mean_red = 0
        mean_green = 0
        mean_blue = 0
        num_valid = 0
        while (cap[c].isOpened()) and n < n_white: # ----- loop over frames
            # Capture frame-by-frame
            if frame is None:
                ret, frame = cap[c].read()
            else:
                ret, frame = cap[c].read(frame)
            if not ret:
                break
            h,w,ch = frame.shape
            #color_frame = cv2.resize(frame,(w//res_fac,h//res_fac))
            color_frame = np.flip(np.array(color_frame),axis=2)
            h,w,ch = color_frame.shape
            color_frame = fast_rot(color_frame,-rot[c])

            #if not n % 30:
            #    _fps = n/(time.time()-t0)
            #    imgio.imsave(os.path.join(prefix,f'input_{c}_white_{n+ini_white:05d}.jpg'),color_frame)
            #    print(f'frame {n+ini_white:05d}  fps {_fps:7.1f}')

            n += 1
            # ------------------- end loop over frames

        # release the video capture object
        cap[c].release()


if __name__ == "__main__":

    ap = argparse.ArgumentParser()
    #
    # mmetadata
    #
    ap.add_argument('-r',"--rescale-factor", type=int, default=8,
                    help="Reduce resolution this many times (defaults to 8 -- brutal). ")
    ap.add_argument('-D',"--basedir", type=str, default=".",
                    help="Base directory. Everything else is relative to this one. ")
    ap.add_argument('-a',"--annotation", type=str, required=True,
                    help="Calibration JSON file produced by annotate. ")
    ap.add_argument('-o',"--output", type=str, default=None,
                    help="Output prefix for data produced by this function. This is appended to basedir.")
    ap.add_argument('-m',"--method", type=str, default="max",
                    help="Method for computing the white frame. May be average,max,or an integer for the percentile (much slower).")
    args = vars(ap.parse_args())

    json_fname = args["annotation"]
    basedir = args["basedir"]

    with open(json_fname,"r") as f:
        annotations = json.loads(f.read())
        print(json.dumps(annotations,indent="    "))
        calibration = None
        extract(annotations,calibration,args)

