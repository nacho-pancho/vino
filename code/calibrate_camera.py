#!/usr/bin/env python3
import numpy as np
import cv2
import os
import argparse 
import json
import skimage.io as imgio

from vutils import *

def gather_calibration_data(annotations,args):
    input_fname = [None,None]
    camera = [None,None]
    fps = [None,None]
    cap = [None,None]
    input_fname = [None,None]
    obj_points = [None,None]
    img_points =[None,None]

    M = args["nrows"]
    N = args["ncols"]
    window_size = 11 # for finding cheessboard points
    objp = np.zeros((M*N,3), np.float32)
    objp[:,:2] = np.mgrid[0:M,0:N].T.reshape(-1,2) * args["pattern_size"]

    ini_frame = annotations["ini_calib_frame"]
    if ini_frame <= 0:
        print("ERROR: initial calibration frame not annotated.")
        exit(1)
    final_frame = annotations["fin_calib_frame"]
    if final_frame <= 0:
        print("ERROR: final calibration frame not annotated.")
        exit(1)

    camera[0] = annotations["camera_a"]
    input_dir=os.path.join(args["datadir"],args["adqdir"])
    output_dir=args["outdir"]
    if output_dir is None:
        annotations_rel_fname = generate_annotations_filename(args["camera_a"],args["camera_b"],args[ "take"])
        input_dir = os.path.join(args["datadir"],args["adqdir"])
        annotations_fname = os.path.join(input_dir,annotations_rel_fname)
        print("annotations file:",annotations_fname)
        annotations_base,_ = os.path.splitext(annotations_fname)
        output_dir = os.path.join(annotations_base+".calib")
    res_fac=args["rescale_factor"]
    if annotations["camera_b"]:
        camera[1] = annotations["camera_b"]
        ncam = 2
    else:
        ncam = 1
    rot = [ annotations["rot1"], annotations["rot2"]]
    print("output dir",output_dir)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir,exist_ok=True)


    offset = compute_offsets(annotations)
    # white frame
    for c in range(ncam):
        print("-"*80)
        input_fname[c] = os.path.join(input_dir,f'{camera[c]}/{camera[c]}_toma{take}_parte1.mp4')
        if not os.path.exists(input_fname[c]):
            print("ERROR: file ",input_fname[c]," not found.")
            exit(1)
        cap[c] = cv2.VideoCapture(input_fname[c])
        cap[c].set(cv2.CAP_PROP_POS_FRAMES, offset[c]+ini_frame)
        fps[c] = cap[c].get(cv2.CAP_PROP_FPS) 
        # Arrays to store object points and image points from all the images.
        objpoints_c = [] # 3d point in real world space
        imgpoints_c = [] # 2d points in image plane.
        frame = None
        frame_index = ini_frame
        good_frames = 0
        while (cap[c].isOpened()) and frame_index <= final_frame: # ----- loop over frames
            # Capture frame-by-frame
            if frame is None:
                ret, frame = cap[c].read()
            else:
                ret, frame = cap[c].read(frame)
            if not ret:
                break
            h,w,_ = frame.shape
            color_frame = cv2.resize(frame,(w//res_fac,h//res_fac))
            color_frame = np.flip(np.array(color_frame),axis=2)            
            color_frame = fast_rot(color_frame,rot[c])
            color_frame = color_frame.astype(float)
            gray_frame = color_frame[:,:,1]
            gray = np.array(gray_frame)
            gray = gray - np.min(gray)
            gray = gray*(255/np.max(gray))
            gray = gray.astype(np.uint8)
            frame_name = f'camera{c+1}_calib_{frame_index:07d}.jpg'
            frame_full_path = os.path.join(output_dir,frame_name)
            imgio.imsave(frame_full_path,gray,quality=90)

            # Find the chess board corners
            ret, centers = cv2.findCirclesGrid(gray, (M,N), flags=cv2.CALIB_CB_ASYMMETRIC_GRID)
            # If found, add object points, image points (after refining them)
            print(ret,len(centers))
            if ret == True:
                objpoints_c.append(objp)
                #criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, args["maxiter"], args["epsilon"])
                #corners2 = cv2.cornerSubPix(gray,centers, (window_size,window_size), (-1,-1), criteria)
                centers2 = centers
                imgpoints_c.append(centers2)
                good_frames += 1
                print('OK!')
            else:
                print('Pattern not found in frame ',frame_index)
                objpoints_c.append(None)
                imgpoints_c.append(None)
            for ce in centers:
                ce = np.squeeze(ce).astype(int)
                gray[ce[1]-2:ce[1]+2,ce[0]-2:ce[0]+2] = 255
            debug_frame_name = f'camera{c+1}_debug_{frame_index:07d}.jpg'
            debug_frame_full_path = os.path.join(output_dir,debug_frame_name)
            imgio.imsave(debug_frame_full_path,gray,quality=90)
            frame_index += 1
            print(f"got {good_frames} out of {final_frame-ini_frame} frames.")
            # end while: go over calibration frames for current camera
        obj_points[c] = objpoints_c
        img_points[c] = imgpoints_c
        # end for: end gathering data for current camera
    return np.flip(gray_frame.shape), obj_points, img_points

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("-a","--camera-a", type=str, required=True,
                    help="primera cámara (siempre tiene que estar)")
    ap.add_argument("-b","--camera-b", type=str, default=None,
                    help="segunda cámara (si es un par)")
    ap.add_argument('-D',"--datadir", type=str, required=True,
                    help="Base directory for all gathered data.")
    ap.add_argument('-R',"--rescale-factor", type=int, default=8,
                    help="Reduce resolution this many times (defaults to 8 -- brutal). ")
    ap.add_argument('-l',"--alpha", type=float, default=1,
                    help="Parameter for OpenCV getOptimalNewCameraMatrix.")
    ap.add_argument('-E',"--epsilon", type=float, default=1e-3,
                    help="Tolerance for OpenCV corner subpixel estimation.")
    ap.add_argument('-I',"--maxiter", type=int, default=100,help="Parameter for OpenCV corner subpixel estimation.")
    ap.add_argument('-o',"--outdir", type=str, default=None,
                    help="output directory. Defaults to the same name as the annotation file with .calib instead of .json as suffix.")
    ap.add_argument('-p',"--pattern", type=str, default="circles",
                    help="Type of calibration pattern. May be chessboard or circles.")
    ap.add_argument('-M',"--nrows", type=int, default=5,
                    help="Number of rows in pattern.")
    ap.add_argument('-N',"--ncols", type=int, default=4,
                    help="Number of columns in pattern.")
    ap.add_argument('-r',"--rotate", default=90,
                    help="Save debugging info (frames).")
    ap.add_argument('-L',"--pattern-size", type=float, default=1,
                    help="Size of patterns in real world (in your units of preference, doesn't matter).")

    args = vars(ap.parse_args())
    annotations_rel_fname = generate_annotations_filename(args["camera_a"],args["camera_b"],args[ "take"])
    input_dir = os.path.join(args["datadir"],args["adqdir"])
    annotations_fname = os.path.join(input_dir,annotations_rel_fname)
    print("annotations file:",annotations_fname)
    with open(annotations_fname,"r") as fa:
        annotations = json.loads(fa.read())
        print("annotations:",json.dumps(annotations,indent="  "))
        #
        # grab and store calibration frames
        #
        frame_size, obj_points, img_points = gather_calibration_data(annotations,args)
        #
        # calibrating
        #
        print("Initial calibration")
        ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(obj_points, img_points, frame_size, None, None)
        print('RMSE',ret)
        rvecs = np.squeeze(np.array(rvecs))
        tvecs = np.squeeze(np.array(tvecs))
        dist = np.squeeze(dist)
        #
        # refining model
        #
        alpha = args["alpha"]
        print('Refining calibration')
        newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, frame_size, alpha, frame_size)

        print('Saving results')
        input_dir=os.path.join(args["datadir"],args["adqdir"])
        output_dir=args["outdir"]
        if output_dir is None:
            annotations_base,_ = os.path.splitext(annotations_fname)
            output_dir = os.path.join(input_dir,annotations_base+".calib")
        np.savetxt(os.path.join(output_dir,'calibration_matrix.txt'),mtx,fmt='%10.6f')
        np.savetxt(os.path.join(output_dir,'calibration_distortion_coeffs.txt'),dist,fmt='%10.6f')
        np.savetxt(os.path.join(output_dir,'calibration_rotation_vectors.txt'),rvecs,fmt='%10.6f')
        np.savetxt(os.path.join(output_dir,'calibration_translation_vectors.txt'),tvecs,fmt='%10.6f')
        np.savetxt(os.path.join(output_dir,'calibration_optimal_matrix.txt'),newcameramtx,fmt='%10.6f')
        cv2.destroyAllWindows()
