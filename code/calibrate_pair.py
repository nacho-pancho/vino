#!/usr/bin/env python3
import numpy as np
import cv2 as cv
import os
import argparse 

def gather_calibration_points(img_dir,img_list,args):
    """
    Calibrates a single camera.
    Takes the args produced by argparse as input
    """
    M = args["nrows"]
    N = args["ncols"]
    downscale=args["downscale"]
    window_size = 11 # for finding cheessboard points

    objp = np.zeros((M*N,3), np.float32)
    objp[:,:2] = np.mgrid[0:M,0:N].T.reshape(-1,2) * args["pattern_size"]
    # Arrays to store object points and image points from all the images.
    objpoints = [] # 3d point in real world space
    imgpoints = [] # 2d points in image plane.

    n = 0
    with open(img_list) as flist:
        for relfname in flist:
            fname = os.path.join(img_dir,relfname.strip())
            print('Finding calibration points in ',fname,end=' ... ')
            color_frame = cv.imread(fname)
            h,w,c = color_frame.shape
            color_frame = cv.resize(color_frame,(w//downscale,h//downscale))
            gray_frame = cv.cvtColor(color_frame, cv.COLOR_BGR2GRAY)
            gray = np.array(gray_frame)
            gray = gray - np.min(gray)
            gray = gray*(255/np.max(gray))
            gray = gray.astype(np.uint8)
            # Find the chess board corners
            ret, corners = cv.findChessboardCorners(gray, (M,N), None)
            # If found, add object points, image points (after refining them)
            if ret == True:
                objpoints.append(objp)
                criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, args["maxiter"], args["epsilon"])
                corners2 = cv.cornerSubPix(gray,corners, (window_size,window_size), (-1,-1), criteria)
                imgpoints.append(corners2)
                n += 1
                print('OK!')
            else:
                print('Pattern not found!')
                objpoints.append(None)
                imgpoints.append(None)

            if n >= args["nframes"]: 
                break
    print(f"processed {n} valid frames")
    return np.flip(gray_frame.shape), objpoints,imgpoints


def calibrate_single_camera(args,img_dir,img_list):
    """
    Calibrates a single camera.
    Takes the args produced by argparse as input
    """
    M = args["nrows"]
    N = args["ncols"]

    img_size, objpoints, imgpoints = gather_calibration_points(img_dir,img_list)
    #
    # calibrating
    #
    print(f"Initial calibration using {n} valid frames")
    ret, camera_matrix, dist, rvecs, tvecs = cv.calibrateCamera(objpoints, imgpoints, img_size, None, None)
    print('RMSE',ret)
    rvecs = np.squeeze(np.array(rvecs))
    tvecs = np.squeeze(np.array(tvecs))
    camera_distortion_coeffs = np.squeeze(dist)

    #
    # refining model
    #
    print('Refining calibration')
    new_camera_matrix, roi = cv.getOptimalNewCameraMatrix(camera_matrix, dist, img_size, args["alpha"], img_size)
    return new_camera_matrix,camera_distortion_coeffs


def calibrate_single(objpoints,imgpoints,img_size,args):
    #
    # calibrating
    #
    print("Initial calibration")
    ret, mtx, dist, rvecs, tvecs = cv.calibrateCamera(objpoints, imgpoints, img_size, None, None)
    print('RMSE',ret)
    rvecs = np.squeeze(np.array(rvecs))
    tvecs = np.squeeze(np.array(tvecs))
    dist = np.squeeze(dist)

    #
    # refining model
    #
    alpha = args["alpha"]
    print('Refining calibration')
    newcameramtx, roi = cv.getOptimalNewCameraMatrix(mtx, dist, img_size, alpha, img_size)
    return newcameramtx, dist


def calibrate_pair(img_dir, img_list1,img_list2,args):
    """
    Calibrates a pair of cameras using previously computed matrices and distortion coeffs for both cameras
    and a list of synchronized frames for each 
    """
    print("Gathering calibration points")
    img_size, objpoints1,imgpoints1 = gather_calibration_points(img_dir,img_list1,args)
    img_size, objpoints2,imgpoints2 = gather_calibration_points(img_dir,img_list2,args)
    valid_objpoints = [o1 for o1,o2 in zip(objpoints1,objpoints2) if o1 is not None and o2 is not None]
    valid_imgpoints_1 = [o1 for o1,o2 in zip(imgpoints1,imgpoints2) if o1 is not None and o2 is not None]
    valid_imgpoints_2 = [o2 for o1,o2 in zip(imgpoints1,imgpoints2) if o1 is not None and o2 is not None]

    mtx1,dist1 = calibrate_single(valid_objpoints,valid_imgpoints_1, img_size, args)
    mtx2,dist2 = calibrate_single(valid_objpoints,valid_imgpoints_2, img_size, args)

    print('Calibrating pair')
    stopping_criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, args["maxiter"], args["epsilon"])
    stereo_calibration_flags = cv.CALIB_FIX_INTRINSIC
    ret, CM1, dist1, CM2, dist2, R, T, E, F = cv.stereoCalibrate(valid_objpoints, 
        valid_imgpoints_1, 
        valid_imgpoints_2, 
        mtx1, dist1,
        mtx2, dist2, img_size, 
        criteria = stopping_criteria, 
        flags = stereo_calibration_flags)
    
    print('RMSE:',ret)
    return CM1, dist1, CM2, dist2, R, T, E, F
 

if __name__ == "__main__":
    ap = argparse.ArgumentParser()

    ap.add_argument('-a',"--alpha", type=float, default=1,
                    help="Parameter for OpenCV getOptimalNewCameraMatrix.")
    ap.add_argument('-E',"--epsilon", type=float, default=1e-4,
                    help="Tolerance for OpenCV corner subpixel estimation.")
    ap.add_argument('-I',"--maxiter", type=int, default=100,help="Parameter for OpenCV corner subpixel estimation.")
    ap.add_argument('-l',"--list-left", type=str, required=True,
                    help="input file list for left camera, where the files are relative to indir")
    ap.add_argument('-r',"--list-right", type=str, required=True,
                    help="input file list for right camera, where the files are relative to indir")
    ap.add_argument('-i',"--indir", type=str, required=True,
                    help="input directory.")
    ap.add_argument('-o',"--outdir", type=str, required=True,
                    help="output directory.")
    ap.add_argument('-n',"--nframes", type=int, default=100,
                    help="Use only this number of frames.")
    ap.add_argument('-d',"--downscale", type=int, default=2,
                    help="Downscale input images by this amount.")
    ap.add_argument('-p',"--pattern", type=str, default="chessboard",
                    help="Type of calibration pattern. May be chessboard or circles.")
    ap.add_argument('-M',"--nrows", type=int, default=9,
                    help="Number of rows in pattern.")
    ap.add_argument('-N',"--ncols", type=int, default=6,
                    help="Number of columns in pattern.")
    ap.add_argument('-D',"--debug", action="store_true",
                    help="Save debugging info (frames).")
    ap.add_argument('-L',"--pattern-size", type=float, default=1,
                    help="Size of patterns in real world (in your units of preference, doesn't matter).")

    args = vars(ap.parse_args())
    CM1, dist1, CM2, dist2, R, T, E, F = calibrate_pair(args["indir"],args["list_left"],args["list_right"],args)
    outdir = args["outdir"]

    print('Saving results')
    np.savetxt(os.path.join(outdir,'calibration_matrix_1.txt'),CM1,fmt='%10.6f')
    np.savetxt(os.path.join(outdir,'calibration_distortion_coeffs_1.txt'),dist1,fmt='%10.6f')
    np.savetxt(os.path.join(outdir,'calibration_matrix_2.txt'),CM1,fmt='%10.6f')
    np.savetxt(os.path.join(outdir,'calibration_distortion_coeffs_2.txt'),dist2,fmt='%10.6f')

    np.savetxt(os.path.join(outdir,'calibration_pair_rotation_matrix.txt'),R,fmt='%10.6f')
    np.savetxt(os.path.join(outdir,'calibration_pair_translation_matrix.txt'),R,fmt='%10.6f')
    cv.destroyAllWindows()
