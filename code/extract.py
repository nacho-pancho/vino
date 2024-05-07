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





def extract(annotations,calibration,args):

    input_fname = [None,None]
    camera_a = annotations["camera_a"]
    take = annotations["take"]
    res_fac = args["rescale_factor"]
    skip = args["skip"]
    basedir = os.path.join(args["datadir"],args["adqdir"])
    output_dir = args["output"]
    annotation_file = args["annotation"]
    calibration_dir,_ = os.path.splitext(annotation_file) 
    calibration_dir = calibration_dir + ".calib"
    if annotations["camera_b"]:
        camera_b = annotations["camera_b"]
        if output_dir is None:
            output_dir  = os.path.join(basedir,f'{camera_a}+{camera_b}_toma{annotations["take"]}.output')
        else:
            output_dir = args["output"]
        ncam = 2
    else:
        if output_dir is None:
            output_dir  = os.path.join(basedir,f'{camera_a}_toma{args["take"]}.output')
        else:
            output_dir = args["output"]
        ncam = 1
    rot = [ annotations["rot1"], annotations["rot2"]]
    if not os.path.exists(output_dir):
        os.makedirs(output_dir,exist_ok=True)
    ini_frame = args["ini_data_frame"]
    if ini_frame < 0:
        ini_frame = 0
    final_frame = args["fin_data_frame"]
    if final_frame <= 0:
        final_frame = 0
    n_data = final_frame - ini_frame
    print("="*80)
    print("Parameters:")
    print("\toutput dir:",output_dir)
    print("\tsource annotation file:",annotation_file)
    print("\tsource calibration directory:",calibration_dir)
    print("\tbase directory:",basedir)
    print("\tinitial frame:",ini_frame)
    print("\tfinal frame",final_frame)
    print("\twrite every this many frames: ",skip)
    #
    # we store the QR info in a CSV table along with the files
    #
    qr_csv_path = os.path.join(output_dir,'qr.csv')
    qr_csv_file = open(qr_csv_path,'w')
    print('Creating csv file ',qr_csv_path)
    csv_writer = csv.writer(qr_csv_file,delimiter=',')
    csv_writer.writerow(('camera','frame','data','x1', 'y1', 'x2','y2','x3','y3','x4','y4'))
    print("extracting frames to output directory:",output_dir)
    print("skipping every ",skip,"frames","reduced by a factor of ",res_fac)
    offset = compute_offsets(annotations)
    # white frame
    fps = [None,None]
    cap = [None,None]
    input_fname = [None,None]
    for c in range(ncam):

        frame = None
        white_frame = None
        n = 0
        t0 = time.time()
        nframes = [list(),list()]
        camera = f"camera{c+1}"
        print("="*80)
        print(f"camera {c+1}:")
        white_balance = calibration[camera]["white_balance"]
        for p in range(1,10):
            input_fname[c] = os.path.join(basedir,f'{camera_a}/{camera_a}_toma{take}_parte{p}.mp4')
            if not os.path.exists(input_fname[c]):
                break
            print("-"*80)
            print("part ",p)
            cap[c] = cv2.VideoCapture(input_fname[c])
            cap[c].set(cv2.CAP_PROP_POS_FRAMES, offset[c]+ini_frame)
            fps[c] = cap[c].get(cv2.CAP_PROP_FPS) 
            nframes[c].append(int(cap[c].get(cv2.CAP_PROP_FRAME_COUNT)))
            if final_frame <= 0:
                final_frame = nframes[c][-1]+1 # read until the very bitter end in all parts
            print("frames per second: ",fps[c])
            print("rotation:",rot[c])
            print("number of frames in this part:",nframes[c][-1])
            #
            # create QR code detector instance
            #
            qr_detector = cv2.QRCodeDetector()

            # Loop until the end of the video
            while (cap[c].isOpened()) and n < n_data: # ----- loop over frames
                # Capture frame-by-frame
                if frame is None:
                    ret, frame = cap[c].read()
                else:
                    ret, frame = cap[c].read(frame)
                if not ret:
                    break
                n += 1
                if n % skip:
                    continue
                h,w,_ = frame.shape
                color_frame = cv2.resize(frame,(w//res_fac,h//res_fac))
                color_frame = np.flip(np.array(color_frame),axis=2)            
                color_frame = fast_rot(color_frame,rot[c])
                hr,wr,_ = color_frame.shape
                color_frame = color_frame.astype(float)
                if white_frame is None:
                    top_crop = hr*args["top_crop"]//100
                    bottom_crop = hr*(100-args["bottom_crop"])//100
                    print("uncropped frame size: h=",hr,"w=",wr)
                    print("crop: top",top_crop, "bottom",bottom_crop)
                    white_frame = np.load(os.path.join(calibration_dir,calibration[camera]["white_frame_matrix"]))
                    hw,ww = white_frame.shape
                    white_frame = trans.resize(white_frame,(hw//res_fac,ww//res_fac))*(1/255)
                #
                # apply rectification
                #
                color_frame[:,:,0] = (color_frame[:,:,0]/white_frame)*(255/white_balance["red"]) # both white balance and white frame are 0-255
                color_frame[:,:,1] = (color_frame[:,:,1]/white_frame)*(255/white_balance["green"])
                color_frame[:,:,2] = (color_frame[:,:,2]/white_frame)*(255/white_balance["blue"])            
                color_frame = np.maximum(0,np.minimum(255,color_frame)).astype(np.uint8)
                frame_index = ini_frame + n
                #
                # detect QR code, if any
                #
                try:
                    qr_info, qr_points, qr_data = qr_detector.detectAndDecode(color_frame[:,:,1])
                except:
                    qr_info = ""
                if qr_info is not None and len(qr_info):
                    qr_info = int(qr_info)
                    print(f'{frame_index:06d}: QR detected: {qr_info:03d}')
                    qr_points = np.squeeze(np.round(qr_points).astype(int))
                    csv_row = [c+1,frame_index]
                    csv_row.extend(qr_points.ravel().tolist())
                    csv_writer.writerow(csv_row)
                    #qr_data = {'data':qr_info,'points':qr_points.tolist()}
                #
                # crop and save frame
                #
                if top_crop >0 or bottom_crop < hr:
                    color_frame = color_frame[top_crop:bottom_crop,:,:]
                frame_name = f'camera{c+1}_frame_{frame_index:07d}'
                imgio.imsave(os.path.join(output_dir,frame_name+'.jpg'),color_frame,quality=90)

                #_fps = (n+1)/(time.time()-t0)
                #print(f'frame {n+ini_data:05d}  fps {_fps:7.1f}')
                # ------- end while : we have read all frames from a given part and camera
                
            # release the video capture object 
            cap[c].release()
            # -------- end for: we have read all parts from this camera
        # end for: we have processed all cameras
    qr_csv_file.close()
    # end function    


if __name__ == "__main__":

    ap = argparse.ArgumentParser()
    #
    # mmetadata
    #
    ap.add_argument("-D","--datadir",type=str,required=True,help="directorio donde se encuentran todos los datos.")
    ap.add_argument("-A","--adqdir", type=str, required=True,
                    help="nombre de directorio de la instancia de adquisicion, por ej: 2024-01-03-vino_fino SIN terminadores (barras).")

    ap.add_argument('-r',"--rescale-factor", type=int, default=4,
                    help="Reduce output this many times (defaults to 4). ")
    ap.add_argument('-i',"--ini-data-frame", type=int, required=True,
                    help="First data frame to extract. ")
    ap.add_argument('-f',"--fin-data-frame", type=int, default=0,
                    help="First data frame to extract. ")
    ap.add_argument('-s',"--skip", type=int, default=5,
                    help="Output every this number of frames (defaults to 5). ")
    ap.add_argument('-a',"--annotation", type=str, required=True,
                    help="Calibration info JSON file produced by annotate. ")
    ap.add_argument('-c',"--calibration", type=str, default=None,
                    help="Directory where calibration results were stored. Defaults to annotations file name with .calib suffix. ")
    ap.add_argument('-o',"--output", type=str, default=None,
                    help="Output directory for data produced by this function. This is appended to basedir. Default is the name of the annotations file with a .output prefix")
    ap.add_argument('-T',"--top-crop", type=int,  default=0, help="Crop the top T%% pixels from the output image. This does not affect QR detection as it is done after that stage.")
    ap.add_argument('-B',"--bottom-crop", type=int,  default=0, help="Crop the top T%% pixels from the output image. This does not affect QR detection as it is done after that stage.")
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

