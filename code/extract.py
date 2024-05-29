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





def extract(input_dir, annotations, calibration, args, output_dir):

    input_fname = [None,None]
    camera =[ calibration["camera1"]["name"], calibration["camera2"]["name"]]
    take = annotations["take"]
    part = args["part"]
    res_fac = args["rescale_factor"]
    skip = args["skip"]
    camera_b = annotations["camera_b"]
    if camera_b is None:
        ncam = 1
    else:
        ncam = 2
    print("number of cameras",ncam)
    rot = [ annotations["rot1"], annotations["rot2"]]
    if not os.path.exists(output_dir):
        os.makedirs(output_dir,exist_ok=True)
    frames_in_seconds = args["seconds"]
    #
    # we store the QR info in a CSV table along with the files
    #
    if args["create_csv"]:
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
        # must reset for each camera because ini_frame gets modified
        ini_frame = args["ini_data_frame"]
        if ini_frame < 0:
            ini_frame = 0
        final_frame = args["fin_data_frame"]
        if final_frame <= 0:
            final_frame = 1000000000
        frame = None
        white_frame = None
        n = 0
        t0 = time.time()
        nframes = [list(),list()]
        print("="*80)
        print(f"camara {camera[c]}:")
        print(f"parte {part} ")
        white_balance = calibration[f"camera{c+1}"]["white_balance"]
        fps[c] = calibration[f'camera{c+1}']["fps"]
        if frames_in_seconds:
            ini_frame = int(ini_frame*fps[c])
            final_frame = int(final_frame*fps[c])
        ini_frame += offset[c]
        input_fname[c] = os.path.join(input_dir,f'{camera[c]}/{camera[c]}_toma{take}_parte{part}.mp4')
        print(input_fname[c])
        if not os.path.exists(input_fname[c]):
            print(f'ERROR: no se encuentra video {input_fname[c]}.')
            break

        cap[c] = cv2.VideoCapture(input_fname[c])
        nframes_c_p = int(cap[c].get(cv2.CAP_PROP_FRAME_COUNT))
        nframes[c].append(nframes_c_p)
        print("number of frames:",nframes_c_p)
        h = int(cap[c].get(cv2.CAP_PROP_FRAME_HEIGHT))
        print("input frame height",h)
        w = int(cap[c].get(cv2.CAP_PROP_FRAME_WIDTH))
        print("input frame width",w)
        frame = np.empty((h,w,3))
        # initial frame is beyond this part
        
        if ini_frame < 0:
            ini_frame = 0
            
        #
        # ningun metodo de posicionamiento parece andar 100% bien así que 
        # voy a hacerlo a lo bestia y leer todos los frames iniciales hasta llegar al objetivo
        #
        print(f'grabbing frames {ini_frame} to {final_frame}')
        frame_index = 0
        print(f'skipping initial frames')
        while cap[c].isOpened() and frame_index < ini_frame:
            #res, frame = cap[c].read(frame)
            res = cap[c].read(frame) # do not decode
            if not res:
                print(f'Fin de stream en frame {frame_index}')
                break
            print('.',end='')
            if not frame_index % 50:
                print()
            frame_index += 1
        #
        # create QR code detector instance
        #
        qr_detector = cv2.QRCodeDetector()
        # Loop until the end of the video
        while cap[c].isOpened() and frame_index < min(final_frame,nframes_c_p): # ----- loop over frames
            ret, frame = cap[c].read(frame)
            if not ret:
                print('End of stream reached.')
                break

            n = frame_index - ini_frame
            if n > 0 and n % skip:
                frame_index += 1
                print('x',end='')
                continue
            else:
                print('o',end='')
            if not n % 10:
                print()
            h,w,_ = frame.shape
            color_frame = cv2.resize(frame,(w//res_fac,h//res_fac))
            color_frame = np.flip(np.array(color_frame),axis=2)            
            color_frame = fast_rot(color_frame,rot[c])
            hr,wr,_ = color_frame.shape
            color_frame = color_frame.astype(float)
            
            if white_frame is None:
                print('Loading white frame')
                top_crop = hr*args["top_crop"]//100
                bottom_crop = hr*(100-args["bottom_crop"])//100
                print("uncropped frame size: h=",hr,"w=",wr)
                print("crop: top",top_crop, "bottom",bottom_crop)
                white_frame = np.load(os.path.join(calibration_dir,calibration[f'camera{c+1}']["white_frame_matrix"]))
                hw,ww = white_frame.shape
                white_frame = trans.resize(white_frame,(hw//res_fac,ww//res_fac))*(1/255)
            #
            # apply rectification
            #
            color_frame[:,:,0] = (color_frame[:,:,0]/white_frame)*(255/white_balance["red"]) # both white balance and white frame are 0-255
            color_frame[:,:,1] = (color_frame[:,:,1]/white_frame)*(255/white_balance["green"])
            color_frame[:,:,2] = (color_frame[:,:,2]/white_frame)*(255/white_balance["blue"])            
            color_frame = np.maximum(0,np.minimum(255,color_frame)).astype(np.uint8)
            #
            # detect QR code, if any
            #
            try:
                qr_info, qr_points, qr_data = qr_detector.detectAndDecode(color_frame[:,:,1])
            except:
                qr_info = ""
            if qr_info is not None and len(qr_info):
                qr_info = int(qr_info)
                frame_time_s = frame_index / fps[c]
                frame_time_min = int(np.floor(frame_time_s / 60))
                frame_time_s -= frame_time_min*60
                print(f'frame {frame_index:06d} (time {frame_time_min:02d}:{frame_time_s:5.2f}s: QR detected: {qr_info:03d}')
                qr_points = np.squeeze(np.round(qr_points).astype(int))
                if args["create_csv"]:
                    csv_row = [c+1,frame_index]
                    csv_row.extend(qr_points.ravel().tolist())
                    csv_writer.writerow(csv_row)
            #
            # crop and save frame
            #
            if top_crop >0 or bottom_crop < hr:
                color_frame = color_frame[top_crop:bottom_crop,:,:]
            frame_name = f'camera{c+1}_frame_{frame_index:07d}'
            imgio.imsave(os.path.join(output_dir,frame_name+'.jpg'),color_frame,quality=90)
            if args["crude"]:
                crude_frame = cv2.resize(frame,(w//res_fac,h//res_fac))
                crude_frame = np.flip(np.array(crude_frame),axis=2)            
                crude_frame = fast_rot(crude_frame,rot[c])
                imgio.imsave(os.path.join(output_dir,frame_name+'_crude.jpg'),crude_frame,quality=90)
                imgio.imsave(os.path.join(output_dir,frame_name+'_refined.jpg'),color_frame,quality=90)
            else:
                imgio.imsave(os.path.join(output_dir,frame_name+'.jpg'),color_frame,quality=90)

            frame_index += 1
            #_fps = (n+1)/(time.time()-t0)
            #print(f'frame {n+ini_data:05d}  fps {_fps:7.1f}')
            # ------- end while : we have read all frames from a given part and camera
        print("finished with camera ",c+1)    
        cap[c].release()
        # -------- end for: we have read all parts from this camera
        # end for: we have processed all cameras
    if args["create_csv"]:
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
    ap.add_argument("-a","--camera-a", type=str, required=True,
                    help="primera cámara (siempre tiene que estar)")
    ap.add_argument("-b","--camera-b", type=str, default=None,
                    help="segunda cámara (si es un par)")
    ap.add_argument("-t","--take", type=int, default=1,
                    help="número de toma")
    ap.add_argument("-p","--part", type=int, default=1,
                    help="número de parte (en gral. para calibrar usamos siempre la 1)")
    ap.add_argument('-r',"--rescale-factor", type=int, default=1,
                    help="Reduce output this many times (defaults to 1). ")
    ap.add_argument('-i',"--ini-data-frame", type=int, default=-1,
                    help="First data frame to extract. ")
    ap.add_argument('-f',"--fin-data-frame", type=int, default=-1,
                    help="First data frame to extract. ")
    ap.add_argument('-s',"--skip", type=int, default=6,
                    help="Output every this number of frames (defaults to 5). ")
    ap.add_argument('-S',"--seconds", action="store_true",
                    help="Initial and final frames are given in seconds, not number of frame. ")    
    ap.add_argument('-o',"--output-dir", type=str, default=None,
                    help="Output directory for data produced by this function. This is appended to basedir. Default is the name of the annotations file with a .output prefix")
    ap.add_argument('-T',"--top-crop", type=int,  default=0, help="Crop the top T%% pixels from the output image. This does not affect QR detection as it is done after that stage.")
    ap.add_argument('-B',"--bottom-crop", type=int,  default=0, help="Crop the top T%% pixels from the output image. This does not affect QR detection as it is done after that stage.")
    ap.add_argument('-v','--create-csv',action='store_true')
    ap.add_argument('-x','--crude',action="store_true",help='Save non-rectified frames as well. For comparison.')
    args = vars(ap.parse_args())
    camera_a = args["camera_a"]
    camera_b = args["camera_b"]
    take     = args["take"]
    input_dir = os.path.join(args["datadir"],args["adqdir"])
    annotations_file = os.path.join(input_dir,generate_annotations_filename(camera_a,camera_b,take))
    calibration_dir,_ = os.path.splitext(annotations_file) 
    calibration_dir = calibration_dir + ".calib"
    calibration_file = os.path.join(calibration_dir,"calibration.json")
    annotations_base_fname,_ = os.path.splitext(annotations_file)
    ini_frame = args["ini_data_frame"]
    final_frame = args["fin_data_frame"]
    if args["output_dir"] is None:
        output_dir = os.path.join(input_dir,generate_output_dir(camera_a,camera_b,take,ini_frame,final_frame))
    else:
        output_dir = args["output_dir"]
    print("="*80)
    print("Parameters:")
    print("\tinput dir:",input_dir)
    print("\toutput dir:",output_dir)
    print("\tsource annotation file:",annotations_file)
    print("\tsource calibration directory:",calibration_dir)
    print("\tbase directory:",input_dir)
    print("\tinitial frame:",args["ini_data_frame"])
    print("\tfinal frame",args["fin_data_frame"])
    print("\twrite every this many frames: ",args["skip"])
    
    with open(annotations_file,"r") as fa:
        with open(calibration_file,'r') as fc:
            annotations = json.loads(fa.read())
            calibration = json.loads(fc.read())
            print("annotations:",json.dumps(annotations,indent="  "))
            print("calibration:",json.dumps(calibration,indent="  "))
            extract(input_dir, annotations, calibration, args, output_dir)

