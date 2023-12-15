#!/usr/bin/env python3

import argparse
import sys
import time
import numpy as np

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d
from skimage import io as imgio
from skimage import filters as imgfilt
from skimage import morphology as morph
import numpy.linalg as la



if __name__ == "__main__":

    def type_cropbox(s):
        cs = [int(x.strip()) for x in s.split(',')]
        return cs

    ap = argparse.ArgumentParser()

    ap.add_argument('-c',"--cropbox", type=type_cropbox, default=None,help="cropping box: top,bottom,left,right")
    ap.add_argument('-i',"--input", type=str, required=True,
                    help="input, a white frame")
    ap.add_argument('-o',"--output", type=str, required=True,
                    help="output prefix. ")
    ap.add_argument('-p',"--order", type=int, default=3,
                    help="Order of approximation polynomial.")
    ap.add_argument('-d',"--decimation", type=int, default=5,
                    help="Decimate input by this factor (after median filter).")

    args = vars(ap.parse_args())

    cropbox = args["cropbox"]
    prefix  = args["output"]
    order   = args["order"]
    decim   = args["decimation"]
    input   = args["input"]

    #
    # index=0: original size and offset
    #
    W0 = imgio.imread(input)
    W0 = imgfilt.median(W0,morph.disk(8))
    h0,w0 = W0.shape
    x0 = np.arange(w0)/w0
    y0 = np.arange(h0)/h0
    X0,Y0 = np.meshgrid(x0,y0)
    # 
    # index=1: downsize by a factor of 5
    #
    W1 = W0[::decim,::decim]

    h1,w1 = W1.shape
    x1 = np.arange(w1)/w1
    y1 = np.arange(h1)/h1
    X1,Y1 = np.meshgrid(x1,y1)

    fig = plt.figure()
    plt.imshow(W1)

    fig = plt.figure(figsize=(10,10))
    ax = fig.add_subplot(projection='3d')
    ax.plot_surface(X1, Y1, W1, cmap=plt.cm.coolwarm)
    plt.title('Input')
    plt.savefig(f'{prefix}_input_curve.png')

    if cropbox is not None:
        mtop,mbottom,mleft,mright = cropbox
        W2 = W1[mtop:mbottom,mleft:mright]
        X2 = X1[mtop:mbottom,mleft:mright]
        Y2 = Y1[mtop:mbottom,mleft:mright]
    else:
        W2 = W1
        X2 = X1
        Y2 = Y1

    x2 = X2.ravel()
    y2 = Y2.ravel()
    w2 = W2.ravel()
    n2 = len(x2)


    fig = plt.figure(figsize=(10,10))
    ax = fig.add_subplot(projection='3d')
    ax.plot_surface(X2, Y2, W2, cmap=plt.cm.coolwarm)
    plt.title('Ground truth (cropped)')
    #
    # 2D polynomial in (x,y)
    # w = a0 + a1*x + a2*y + a3*x*y + a4*x^2 + a5*y^2
    # 2D polynomial in (x,y)
    # w = a0 + a1*x + a2*y + a3*x*y + a4*x^2 + a5*y^2
    if order == 2:
        v2 = np.array((np.ones(n2),x2,y2,x2*y2,x2**2,y2**2))
    else:
        v2 = np.array((np.ones(n2),x2,y2,x2*y2,x2**2,y2**2,(x2**2)*y2,x2*(y2**2),x2**3,y2**3))
    ahat,err,rnk,sval = la.lstsq(v2.T,w2)

    w2hat = np.dot(ahat,v2)
    W2hat = np.reshape(w2hat,W2.shape)

    fig = plt.figure(figsize=(10,10))
    ax = fig.add_subplot(projection='3d')
    ax.plot_surface(X2, Y2, W2hat, cmap=plt.cm.coolwarm)
    plt.title('Approximation as 3rd order polynomial')
    plt.savefig(f'{prefix}_approximated_curve.png')

    fig = plt.figure(figsize=(10,10))
    ax = fig.add_subplot(projection='3d')
    E = W2hat-W2
    maxE = np.max(np.abs(E))
    ax.plot_surface(X2, Y2, E, cmap=plt.cm.coolwarm,vmin=-maxE,vmax=maxE)
    plt.title('Approximation error')
    plt.savefig(f'{prefix}_approximation_error.png')

    x0 = X0.ravel()
    y0 = Y0.ravel()
    n0 = len(x0)
    if order == 2:
        v0 = np.array((np.ones(n0),x0,y0,x0*y0,x0**2,y0**2))
    else:
        v0 = np.array((np.ones(n0),x0,y0,x0*y0,x0**2,y0**2,(x0**2)*y0,x0*(y0**2),x0**3,y0**3))
    w0hat = np.minimum(255,np.maximum(0,np.dot(ahat,v0))).astype(np.uint8)
    W0hat = np.reshape(w0hat,W0.shape)
    imgio.imsave(f'{prefix}_curve.png',W0hat)
    plt.show()

