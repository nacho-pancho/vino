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
import csv
from vutils import *





def extract(input_dir, annotations, args, calibration_dir):

    input_fname = [None,None]
    camera_a = args["camera_a"]
    camera_b = args["camera_b"]
    cameras = (camera_a,camera_b)
    if camera_b is None:
        ncam = 1
    else:
        ncam = 2
    res_fac = args["rescale_factor"]
    print("number of cameras",ncam)
    if not os.path.exists(calibration_dir):
        os.makedirs(calibration_dir,exist_ok=True)
    #
    # we store the QR info in a CSV table along with the files
    #
    offset = compute_offsets(annotations)
    # white frame
    for c in range(ncam):
        camera = cameras[c]
        # must reset for each camera because ini_frame gets modified
        ini_frame = annotations["ini_calib_frame"]
        if ini_frame < 0:
            ini_frame = 0
        final_frame = annotations["fin_calib_frame"]
        if final_frame <= 0:
            final_frame = 1000000000
        frame = None
        n = 0
        t0 = time.time()
        print("="*80)
        print(f"camara {camera}:")
        ini_frame += offset[c]
        input_fname = os.path.join(input_dir,f'{camera}.mp4')
        print("input video file:",input_fname)
        if not os.path.exists(input_fname):
            print(f'ERROR: no se encuentra video {input_fname}.')
            break

        cap = cv2.VideoCapture(input_fname)
        nframes = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        print("number of frames:",nframes)
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        print("input frame height",h)
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        print("input frame width",w)
        frame = np.empty((h,w,3))
        #
        # ningun metodo de posicionamiento parece andar 100% bien así que 
        # voy a hacerlo a lo bestia y leer todos los frames iniciales hasta llegar al objetivo
        #
        print(f'grabbing frames {ini_frame} to {final_frame}')
        frame_index = 0
        print(f'skipping initial frames')
        while cap.isOpened() and frame_index < ini_frame:
            res = cap.read(frame) # do not decode
            if not res:
                print(f'Fin de stream en frame {frame_index}')
                break
            if not frame_index % 100:
                print(f'{frame_index:07d}...')
            frame_index += 1

        # Loop until the end of the video
        rot = int(annotations[f"rot{c+1}"])
        while cap.isOpened() and frame_index < min(final_frame,nframes): # ----- loop over frames
            ret, frame = cap.read(frame)
            if not ret:
                print('End of stream reached.')
                break
            color_frame = cv2.resize(frame,(w//res_fac,h//res_fac))
            color_frame = np.flip(np.array(color_frame),axis=2)            
            color_frame = fast_rot(color_frame,rot)
            #color_frame = color_frame.astype(float)
            
            frame_name = f'{camera}_frame_{frame_index:07d}'
            imgio.imsave(os.path.join(calibration_dir,frame_name+'.png'),color_frame,quality=4)
            frame_index += 1
        print(f"finished with camera {camera}")    
        cap.release()
        # -------- end for: we have read all parts from this camera
        # end for: we have processed all cameras
    # end function    


if __name__ == "__main__":

    ap = argparse.ArgumentParser()
    #
    # mmetadata
    #
    ap.add_argument("-D","--datadir",type=str,default=".",help="directorio donde se encuentran todos los datos.")
    ap.add_argument("camera_a", type=str,
                    help="primera cámara")
    ap.add_argument("camera_b", type=str,
                    help="segunda cámara")
    ap.add_argument('-r',"--rescale-factor", type=int, default=2,
                    help="Reduce output this many times (defaults to 2). ")
    args = vars(ap.parse_args())
    camera_a = args["camera_a"]
    camera_b = args["camera_b"]
    input_dir = os.path.join(args["datadir"])
    annotations_file = os.path.join(input_dir,generate_annotations_filename(camera_a,camera_b))
    calibration_dir,_ = os.path.splitext(annotations_file) 
    calibration_dir = calibration_dir + ".calib"
    annotations_base_fname,_ = os.path.splitext(annotations_file)
    print("="*80)
    print("Parameters:")
    print("\tinput dir:",input_dir)
    print("\tsource annotation file:",annotations_file)
    print("\toutput calibration directory:",calibration_dir)
    
    with open(annotations_file,"r") as fa:
        annotations = json.loads(fa.read())
        print("annotations:",json.dumps(annotations,indent="  "))
        extract(input_dir, annotations, args, calibration_dir)

