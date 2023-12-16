#!/usr/bin/env python3
import numpy as np
import cv2 as cv
import os

# termination criteria
criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)
# prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
M = 9
N = 6
objp = np.zeros((M*N,3), np.float32)
objp[:,:2] = np.mgrid[0:M,0:N].T.reshape(-1,2)
# Arrays to store object points and image points from all the images.
objpoints = [] # 3d point in real world space
imgpoints = [] # 2d points in image plane.
indir='../../results'
n = 0
with open(f'{indir}/calib3d_gopro1.list') as flist:
    for fname in flist:
        print('Processing ',fname.strip())
        color_frame = cv.imread(os.path.join(indir,fname.strip()))        
        h,w,c = color_frame.shape
        color_frame = cv.resize(color_frame,(w//4,h//4))
        gray_frame = cv.cvtColor(color_frame, cv.COLOR_BGR2GRAY)
        gray_frame = cv.resize(gray_frame,(w//4,h//4))
        gray = np.array(gray_frame)
        gray = gray - np.min(gray)
        gray = gray*(255/np.max(gray))
        gray = gray.astype(np.uint8)
        # Find the chess board corners
        ret, corners = cv.findChessboardCorners(gray, (M,N), None)
        # If found, add object points, image points (after refining them)
        if ret == True:
            objpoints.append(objp)
            corners2 = cv.cornerSubPix(gray,corners, (11,11), (-1,-1), criteria)
            imgpoints.append(corners2)
            # Draw and display the corners
            cv.drawChessboardCorners(gray, (M,N), corners2, ret)
            cv.imshow('img', gray)
            cv.waitKey(100)
            n += 1
        else:
            print('something went wrong')            
        if n >= 30: # enough
            break

h,w = gray_frame.shape
#
# calibrating
#
print("calibrating")
ret, mtx, dist, rvecs, tvecs = cv.calibrateCamera(objpoints, imgpoints, (w,h), None, None)
#print(ret,mtx,dist,rvecs,tvecs)
alpha = 1
#
# refining model
#
newcameramtx, roi = cv.getOptimalNewCameraMatrix(mtx, dist, (w,h), alpha, (w,h))
print(newcameramtx,roi)
#
# sample undistort
#
# undistort
n = 0
mapx, mapy = cv.initUndistortRectifyMap(mtx, dist, None, newcameramtx, (w,h), 5)
with open(f'{indir}/calib3d_gopro1.list') as flist:
    for fname in flist:
        print('Undistorting ',fname.strip())
        color_frame = cv.imread(os.path.join(indir,fname.strip()))        
        h,w,c = color_frame.shape
        color_frame = cv.resize(color_frame,(w//4,h//4))
        dst = cv.remap(color_frame, mapx, mapy, cv.INTER_LINEAR)
        # crop the image
        x, y, w, h = roi
        dst = dst[y:y+h, x:x+w]
        cv.imshow('undistorted',dst)
        cv.waitKey(100)
        #cv.imwrite('calibresult.png', dst)    
        n += 1
        if n >= 30:
            break
cv.destroyAllWindows()
