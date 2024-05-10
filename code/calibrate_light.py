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
from vutils import *


def do_white(annotations,args,output_dir):
    calibration = dict()
    input_fname = [None,None]
    camera_a = annotations["camera_a"]
    camera_b = annotations["camera_b"]
    take = annotations["take"]
    res_fac = args["rescale_factor"]
    basedir = os.path.join(args["datadir"],args["adqdir"])
    input_fname[0] = os.path.join(basedir,f'{camera_a}/{camera_a}_toma{take}_parte1.mp4')
    rot = annotations["rot1"]
    camera = [camera_a,camera_b]
    if annotations["camera_b"]:
        camera_b = annotations["camera_b"]
        input_fname[1] = os.path.join(basedir,f'{camera_b}/{camera_b}_toma{take}_parte1.mp4')
        ncam = 2
        calibration["ncam"] = 2
        rot = [ annotations["rot1"], annotations["rot2"]]
        offset = compute_offsets(annotations)
    else:
        ncam = 1
        calibration["ncam"] = 1
        offset = [0,0]
    if not os.path.exists(output_dir):
        os.makedirs(output_dir,exist_ok=True)
        
    cropbox = annotations["crop_box"]
    i0,j0,i1,j1 = cropbox
    i0r = i0//res_fac
    j0r = j0//res_fac
    i1r = i1//res_fac
    j1r = j1//res_fac
    calibration["cropbox_rescaled"] = {"top":i0r,"left":j0r,"bottom":i1r,"right":j1r}
    calibration["cropbox_orig"] = cropbox
    # white frame
    ini_white = annotations["ini_white_frame"]
    end_white = annotations["fin_white_frame"]
    n_white = end_white - ini_white
    n_white = 20 # DEBUG
    fps = [None,None]
    cap = [None,None]
    nframes = [None,None]
    for c in range(ncam):
        print(f"camera {c}:")

        cap[c] = cv2.VideoCapture(input_fname[c])
        nframes[c] = cap[c].get(cv2.CAP_PROP_FRAME_COUNT)
        cap[c].set(cv2.CAP_PROP_POS_AVI_RATIO, (offset[c]+ini_white)/nframes[c])
        fps[c] = cap[c].get(cv2.CAP_PROP_FPS) 
        camera_c_key = f"camera{c+1}"
        calibration_c = dict() 
        calibration_c["input_fname"] = input_fname[c]
        calibration_c["offset"]=offset[c]
        calibration_c["fps"] = fps[c]
        calibration_c["rotation"] = rot[c]
        calibration_c["name"] = camera[c]
        print("\tframes per second: ",fps[c])
        print("\trotation:",rot[c])

        # Loop until the end of the video
        max_frame = None
        mean_frame = None
        frame = None
        n = 0
        t0 = time.time()
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
            color_frame = fast_rot(frame,rot[c])
            h0,w0,_ = color_frame.shape
            color_frame = cv2.resize(color_frame,(w0//res_fac,h0//res_fac))
            color_frame = np.flip(np.array(color_frame),axis=2)
            # brutal resizing
            hr,wr,_ = color_frame.shape
            #print('Saturados:',100*np.sum(gray_frame == 255)/np.prod(gray_frame.shape),'%')
            if n == 0:
                gray_frame = np.zeros((hr,wr),dtype=np.uint8)
                valid_pixels = np.zeros((hr,wr),dtype=bool)
                max_frame = np.zeros(gray_frame.shape,dtype=np.uint8)
            
            gray_frame[:] = np.squeeze((np.sum(color_frame,axis=2))//3) # R + G + B
            valid_pixels[:] = gray_frame < 255
            num_valid  += np.sum(valid_pixels)
            mean_red   += np.sum(color_frame[:,:,0]*valid_pixels)
            mean_green += np.sum(color_frame[:,:,1]*valid_pixels)
            mean_blue  += np.sum(color_frame[:,:,2]*valid_pixels)
            max_frame[:] = np.maximum(max_frame,gray_frame)

            if not n % 5:
                _fps = n/(time.time()-t0)
                imgio.imsave(os.path.join(output_dir,f'camera{c+1}_white_{n+ini_white:05d}.jpg'),color_frame)
                print(f'frame {n+ini_white:05d}  fps {_fps:7.1f}')

            n += 1
            # ------------------- end loop over frames

        # release the video capture object
        cap[c].release()
        if cropbox is not None:
            # cropbox is top left bottom right
            max_frame = max_frame[i0r:i1r,j0r:j1r]
        else:
            calibration_c["cropbox_rescaled"] = ""
        wf_avg_preview = os.path.join(output_dir,f'camera{c+1}_average_cropped_scaled_white_frame.png')
        calibration_c["white_frame_average"] = f'camera{c+1}_average_cropped_scaled_white_frame.png'
        imgio.imsave(wf_avg_preview,max_frame.astype(np.uint8))
        means = (mean_red/num_valid,mean_green/num_valid,mean_blue/num_valid)
        calibration_c["white_balance"] = {"red":means[0],"green":means[1],"blue":means[2]}
        print(f"mean white frame color:",means)
        #
        # compute illumination curve as a second order curve from the non-saturated pixels
        #
        # we build a regression problem of the form (1,r,c,r^2,r*c,c^2) -> L
        # using as r and c the _unscaled_ and _uncropped_ row and column indexes
        # 
        ri = np.arange(i0r,i1r)/hr
        ci = np.arange(j0r,j1r)/wr
        Ri,Ci = np.meshgrid(ri,ci,indexing='ij')
        Ri = Ri.ravel()
        Ci = Ci.ravel()
        L = max_frame.ravel()
        VP = np.flatnonzero(L < 255)
        L = L[VP]
        Ri = Ri[VP]
        Ci = Ci[VP]
        N = len(VP)
        TN = np.prod(max_frame.shape)
        print('Total pixels:',TN,' Valid pixels:',N)
        X = np.ones((N,6))
        X[:,1] = Ri
        X[:,2] = Ci
        X[:,3] = Ri**2
        X[:,4] = Ri*Ci
        X[:,5] = Ci**2
        a,rss,rank,sval = np.linalg.lstsq(X,L,rcond=None)
        calibration_c["white_frame_parameters"] = a.tolist()
        #
        # compute approximated white frame 
        #
        # DEBUG
        #ri = np.arange(i0r,i1r)/hr
        #ci = np.arange(j0r,j1r)/wr
        #Ri,Ci = np.meshgrid(ri,ci,indexing='ij')
        #white_frame = a[0] + a[1]*Ri + a[2]*Ci + a[3]*(Ri**2) + a[4]*(Ri*Ci) + a[5]*(Ci**2)
        #white_frame = np.maximum(0,np.minimum(255,white_frame)).astype(np.uint8)
        #wf_image_fname = os.path.join(output_dir,f'camera{c+1}_white_frame_par_test.png')
        #imgio.imsave(wf_image_fname,white_frame)
        #plt.figure()
        #plt.imshow(white_frame)
        #plt.show()
        # end debug

        ri = np.arange(h0)/h0
        ci = np.arange(w0)/w0
        Ri,Ci = np.meshgrid(ri,ci,indexing='ij')
        white_frame = a[0] + a[1]*Ri + a[2]*Ci + a[3]*(Ri**2) + a[4]*(Ri*Ci) + a[5]*(Ci**2)
        wf_matrix_fname = os.path.join(output_dir,f'camera{c+1}_white_frame_par.npy')
        np.save(wf_matrix_fname,white_frame)
        calibration_c["white_frame_matrix"] = f'camera{c+1}_white_frame_par.npy'
        # very likely, the computed parametric white frame falls out of the valid range if there were many saturated
        # pixels, so we downscale the output image. We will use the saved matrix (npy), not this, for normalization
        white_frame = (white_frame*(255/np.max(white_frame))).astype(np.uint8)
        wf_image_fname = os.path.join(output_dir,f'camera{c+1}_white_frame_par.png')
        imgio.imsave(wf_image_fname,white_frame)
        calibration_c["white_frame_parametric_image"] = f'camera{c+1}_white_frame_par.png'
        calibration[camera_c_key] = calibration_c
    return calibration

def do_calib(annotations,args,calibration):
    #calib_info = [None,None]
    #for c in range(ncam):
    #    calib_info[c] = calibrate_single_camera(cap[c],annotations,args)
    return calibration


if __name__ == "__main__":

    ap = argparse.ArgumentParser()
    #
    # mmetadata
    #
    ap.add_argument("-a","--camera-a", type=str, required=True,
                    help="primera cámara (siempre tiene que estar)")
    ap.add_argument("-b","--camera-b", type=str, default=None,
                    help="segunda cámara (si es un par)")
    ap.add_argument("-t","--take", type=int, default=1,
                    help="número de toma")
    ap.add_argument("-D","--datadir",type=str,required=True,help="directorio donde se encuentran todos los datos.")
    ap.add_argument("-A","--adqdir", type=str, required=True,
                    help="nombre de directorio de la instancia de adquisicion, por ej: 2024-01-03-vino_fino SIN terminadores (barras)")

    ap.add_argument('-R',"--rescale-factor", type=int, default=8,
                    help="Reduce resolution this many times (defaults to 8 -- brutal). ")
    ap.add_argument('-o',"--output", type=str, default=None,
                    help="Output prefix for data produced by this function. This is appended to basedir.")
    ap.add_argument('-m',"--method", type=str, default="max",
                    help="Method for computing the white frame. May be average,max,or an integer for the percentile (much slower).")
    args = vars(ap.parse_args())

    rel_json_fname = generate_annotations_filename(args["camera_a"],args["camera_b"],args["take"])
    json_fname = os.path.join(args["datadir"],args["adqdir"],rel_json_fname)
    print("JSON file: ",json_fname)
    with open(json_fname,"r") as f:
        annotations = json.loads(f.read())
        print(json.dumps(annotations,indent="    "))
        output_info = dict()
        output_info        
        output_dir  = args["output"]
        if output_dir is None:
            base_fname,_ = os.path.splitext(json_fname)
            output_dir = base_fname+".calib"
        print("output_dir",output_dir)
        # white frame
        ini_white = annotations["ini_white_frame"]
        end_white = annotations["fin_white_frame"]    
        if ini_white * end_white >= 0:
            print(f"Computing white frame using frames from {ini_white} to {end_white}")
            calibration = do_white(annotations,args, output_dir)
        else:
            print("No white frame will be computed.")

        ini_calib = annotations["ini_calib_frame"]
        end_calib = annotations["fin_calib_frame"]
        if ini_calib * end_calib >= 0:
            print(f"Computing 3D calibration  using frames from {ini_calib} to {end_calib}")
            calibration = do_calib(annotations,args, calibration)
        else:
            print("No calibration  will be computed.")

        
        calibration_file = os.path.join(output_dir,"calibration.json")
        txt = json.dumps(calibration,indent=4)
        print("CALIBRATION:",txt)
        with open(calibration_file,"w") as f:
            f.write(txt)

