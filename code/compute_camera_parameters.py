#!/usr/bin/env python3
import numpy as np
import cv2 as cv
import os
import argparse 

def gather_camera_points(img_dir,img_list,args):
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
    return np.flip(gray_frame.shape), objpoints,imgpoints


if __name__ == "__main__":
    ap = argparse.ArgumentParser()

    ap.add_argument('-a',"--alpha", type=float, default=1,
                    help="Parameter for OpenCV getOptimalNewCameraMatrix.")
    ap.add_argument('-E',"--epsilon", type=float, default=1e-3,
                    help="Tolerance for OpenCV corner subpixel estimation.")
    ap.add_argument('-I',"--maxiter", type=int, default=100,help="Parameter for OpenCV corner subpixel estimation.")
    ap.add_argument('-l',"--list", type=str, required=True,
                    help="input file list, where the files are relative to indir")
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
    ap.add_argument('-L',"--pattern-size", type=float,
                    help="Size of patterns in real world (in your units of preference, doesn't matter).")

    args = vars(ap.parse_args())

    M = args["nrows"]
    N = args["ncols"]
    indir=args["indir"]
    outdir=args["outdir"]
    downscale=args["downscale"]
    window_size = 11 # for finding cheessboard points

    objp = np.zeros((M*N,3), np.float32)
    objp[:,:2] = np.mgrid[0:M,0:N].T.reshape(-1,2) * args["pattern_size"]
    # Arrays to store object points and image points from all the images.
    objpoints = [] # 3d point in real world space
    imgpoints = [] # 2d points in image plane.

    n = 0
    with open(args["list"]) as flist:
        for relfname in flist:
            fname = os.path.join(indir,relfname.strip())
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
                # Draw and display the corners
                if args["debug"]:
                    cv.drawChessboardCorners(gray, (M,N), corners2, ret)
                    #cv.imshow('img', gray)
                    cv.imwrite(os.path.join(outdir,f'calibration_frame_{n:05d}.jpg'),gray)
                    #cv.waitKey(100)
                n += 1
                print('OK!')
            else:
                print('Pattern not found!')
            if n >= args["nframes"]: 
                break

    #img_size, obj_points, img_points = gather_camera_points(args["indir"],args["list"],args)

    #
    # calibrating
    #
    print("Initial calibration")
    ret, mtx, dist, rvecs, tvecs = cv.calibrateCamera(objpoints, imgpoints, (w,h), None, None)
    print('RMSE',ret)
    rvecs = np.squeeze(np.array(rvecs))
    tvecs = np.squeeze(np.array(tvecs))
    dist = np.squeeze(dist)

    #
    # refining model
    #
    alpha = args["alpha"]
    print('Refining calibration')
    newcameramtx, roi = cv.getOptimalNewCameraMatrix(mtx, dist, (w,h), alpha, (w,h))
    #print('newcameramtx')
    #print('roi',roi)
    #np.savetxt(os.path.join(outdir,'calibration_rmse.txt'),ret,fmt='10.6f')
    print('Saving results')
    np.savetxt(os.path.join(outdir,'calibration_matrix.txt'),mtx,fmt='%10.6f')
    np.savetxt(os.path.join(outdir,'calibration_distortion_coeffs.txt'),dist,fmt='%10.6f')
    np.savetxt(os.path.join(outdir,'calibration_rotation_vectors.txt'),rvecs,fmt='%10.6f')
    np.savetxt(os.path.join(outdir,'calibration_translation_vectors.txt'),tvecs,fmt='%10.6f')
    np.savetxt(os.path.join(outdir,'calibration_optimal_matrix.txt'),newcameramtx,fmt='%10.6f')
    cv.destroyAllWindows()
