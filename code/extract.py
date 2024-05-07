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
    camera_a = annotations["camera_a"]
    take = annotations["take"]
    res_fac = args["rescale_factor"]
    skip = args["skip"]
    basedir = os.path.join(args["datadir"],args["adqdir"])
    input_fname[0] = os.path.join(basedir,f'{camera_a}/{camera_a}_toma{take}_parte1.mp4')
    output_dir = args["output"]
    annotation_file = args["annotation"]
    calibration_dir,_ = os.path.splitext(annotation_file) 
    calibration_dir = calibration_dir + ".calib"
    if annotations["camera_b"]:
        camera_b = annotations["camera_b"]
        input_fname[1] = os.path.join(basedir,f'{camera_b}/{camera_b}_toma{take}_parte1.mp4')
        if output_dir is None:
            output_dir  = os.path.join(args["basedir"],f'{camera_a}+{camera_b}.output')
        ncam = 2
    else:
        if output_dir is None:
            output_dir  = os.path.join(args["basedir"],f'{camera_a}.output')
        ncam = 1
    rot = [ annotations["rot1"], annotations["rot2"]]
    if not os.path.exists(output_dir):
        os.makedirs(output_dir,exist_ok=True)

    print("extracting frames to output directory:",output_dir)
    print("skipping every ",skip,"frames","reduced by a factor of ",res_fac)
    offset = compute_offsets(annotations)
    cropbox = annotations["crop_box"]
    # white frame
    ini_data = args["ini_data_frame"]
    if ini_data < 0:
        ini_data = 0
    end_data = args["fin_data_frame"]
    if end_data < 0:
        end_data = skip*100
    n_data = end_data - ini_data
    fps = [None,None]
    cap = [None,None]
    for c in range(ncam):
        camera = f"camera{c+1}"
        print(f"camera {c}:")
        white_balance = calibration[camera]["white_balance"]
        cap[c] = cv2.VideoCapture(input_fname[c])
        cap[c].set(cv2.CAP_PROP_POS_FRAMES, offset[c]+ini_data)
        fps[c] = cap[c].get(cv2.CAP_PROP_FPS) 
        print("\tframes per second: ",fps[c])
        print("\trotation:",rot[c])

        # Loop until the end of the video
        frame = None
        white_frame = None
        n = 0
        t0 = time.time()
        while (cap[c].isOpened()) and n < n_data: # ----- loop over frames
            # Capture frame-by-frame
            if frame is None:
                ret, frame = cap[c].read()
            else:
                ret, frame = cap[c].read(frame)
            if not ret:
                break
            n += 1
            frame_name = f'camera{c+1}_frame_{ini_data+n:05d}'
            if n % skip:
                continue
            h,w,ch = frame.shape
            color_frame = cv2.resize(frame,(w//res_fac,h//res_fac))

            color_frame = np.flip(np.array(color_frame),axis=2)            
            color_frame = fast_rot(color_frame,rot[c])
            #imgio.imsave(os.path.join(output_dir,frame_name+'_before.jpg'),color_frame)
            color_frame = color_frame.astype(float)
            if white_frame is None:
                white_frame = np.load(os.path.join(calibration_dir,calibration[camera]["white_frame_matrix"]))
                hw,ww = white_frame.shape
                white_frame = trans.resize(white_frame,(hw//res_fac,ww//res_fac))*(1/255)
            h,w,ch = color_frame.shape
            color_frame[:,:,0] = (color_frame[:,:,0]/white_frame)*(255/white_balance["red"]) # both white balance and white frame are 0-255
            color_frame[:,:,1] = (color_frame[:,:,1]/white_frame)*(255/white_balance["green"])
            color_frame[:,:,2] = (color_frame[:,:,2]/white_frame)*(255/white_balance["blue"])            
            color_frame = np.maximum(0,np.minimum(255,color_frame)).astype(np.uint8)
            imgio.imsave(os.path.join(output_dir,frame_name+'.jpg'),color_frame)
            _fps = (n+1)/(time.time()-t0)
            print(f'frame {n+ini_data:05d}  fps {_fps:7.1f}')
            # ------------------- end loop over frames

        # release the video capture object
        cap[c].release()


if __name__ == "__main__":

    ap = argparse.ArgumentParser()
    #
    # mmetadata
    #
    ap.add_argument("-D","--datadir",type=str,required=True,help="directorio donde se encuentran todos los datos.")
    ap.add_argument("-A","--adqdir", type=str, required=True,
                    help="nombre de directorio de la instancia de adquisicion, por ej: 2024-01-03-vino_fino SIN terminadores (barras)")

    ap.add_argument('-r',"--rescale-factor", type=int, default=4,
                    help="Reduce output this many times (defaults to 4). ")
    ap.add_argument('-i',"--ini-data-frame", type=int, required=True,
                    help="First data frame to extract. ")
    ap.add_argument('-f',"--fin-data-frame", type=int, required=True,
                    help="First data frame to extract. ")
    ap.add_argument('-s',"--skip", type=int, default=5,
                    help="Output every this number of frames (defaults to 5). ")
    ap.add_argument('-a',"--annotation", type=str, required=True,
                    help="Calibration info JSON file produced by annotate. ")
    ap.add_argument('-c',"--calibration", type=str, default=None,
                    help="Directory where calibration results were stored. Defaults to annotations file name with .calib suffix. ")
    ap.add_argument('-o',"--output", type=str, default=None,
                    help="Output directory for data produced by this function. This is appended to basedir. Default is the name of the annotations file with a .output prefix")
    args = vars(ap.parse_args())

    annotations_json_fname = args["annotation"]    
    if args["calibration"] is None:
        annotations_base_fname,_ = os.path.splitext(annotations_json_fname)
        calibration_basedir = os.path.join(annotations_base_fname+".calib")
        calibration_json_fname = os.path.join(calibration_basedir,"calibration.json")
    with open(annotations_json_fname,"r") as fa:
        with open(calibration_json_fname,'r') as fc:
            annotations = json.loads(fa.read())
            calibration = json.loads(fc.read())
            print("annotations:",json.dumps(annotations,indent="  "))
            print("calibration:",json.dumps(annotations,indent="    "))
            extract(annotations,calibration,args)

