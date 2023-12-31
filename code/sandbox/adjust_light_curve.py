#!/usr/bin/env python3

import numpy as np
import sys
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d
from skimage import io as imgio
from skimage import filters as imgfilt
from skimage import morphology as morph
import numpy.linalg as la

#
# index=0: original size and offset
#
if len(sys.argv) < 3:
    print(f'usage: {sys.argv[0]} input_image output_image')

W0 = imgio.imread(sys.argv[1])
#W0 = W0*(1/255)
W0 = imgfilt.median(W0,morph.disk(8))
h0,w0 = W0.shape
x0 = np.arange(w0)/w0
y0 = np.arange(h0)/h0
X0,Y0 = np.meshgrid(x0,y0)
# 
# index=1: downsize by a factor of 5
#
W1 = W0[::5,::5]

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
plt.savefig('wf_ground.png')
#
# discard borders that may appear outside the white frame
# this is because we couldn't fit the scene with the white board :)
maxW = np.max(W1)
print(W1.shape)
W1v = np.mean(W1,axis=1)
W1h = np.mean(W1,axis=0)

fig = plt.figure()
plt.title('Vertical profile')
plt.plot(W1v)
plt.ylim(0,255)
plt.grid(True)

fig = plt.figure()
plt.title('Horizontal profile')
plt.plot(W1h)
plt.ylim(0,255)
plt.grid(True)

aux = np.flatnonzero(W1h>0.5*np.max(W1h))
mleft,mright = aux[0],aux[-1]

aux = np.flatnonzero(W1v>0.5*np.max(W1v))
mtop,mbottom = 0,-1
print(mleft,mright,mtop,mbottom)

W2 = W1[mtop:mbottom,mleft:mright]
X2 = X1[mtop:mbottom,mleft:mright]
Y2 = Y1[mtop:mbottom,mleft:mright]
print(W2.shape)

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

#v = np.array((np.ones(n),x,y,x*y,x**2,y**2))
v2 = np.array((np.ones(n2),x2,y2,x2*y2,x2**2,y2**2,(x2**2)*y2,x2*(y2**2),x2**3,y2**3))
ahat,err,rnk,sval = la.lstsq(v2.T,w2)
print(ahat)
w2hat = np.dot(ahat,v2)
W2hat = np.reshape(w2hat,W2.shape)

fig = plt.figure(figsize=(10,10))
ax = fig.add_subplot(projection='3d')
ax.plot_surface(X2, Y2, W2hat, cmap=plt.cm.coolwarm)
plt.title('Approximation as 3rd order polynomial')
plt.savefig('wf_approx.png')


fig = plt.figure(figsize=(10,10))
ax = fig.add_subplot(projection='3d')
E = W2hat-W2
maxE = np.max(np.abs(E))
ax.plot_surface(X2, Y2, E, cmap=plt.cm.coolwarm,vmin=-maxE,vmax=maxE)
plt.title('Approximation error')
plt.savefig('wf_error.png')


x0 = X0.ravel()
y0 = Y0.ravel()
n0 = len(x0)
v0 = np.array((np.ones(n0),x0,y0,x0*y0,x0**2,y0**2,(x0**2)*y0,x0*(y0**2),x0**3,y0**3))
w0hat = np.minimum(255,np.maximum(0,np.dot(ahat,v0))).astype(np.uint8)
W0hat = np.reshape(w0hat,W0.shape)
imgio.imsave(sys.argv[2],W0hat)
plt.show()

