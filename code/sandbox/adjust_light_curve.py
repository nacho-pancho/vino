#!/usr/bin/env python3

import numpy as np
import sys
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d

W = plt.imread('whiteframe.pgm')
print(W.shape)
h,w = W.shape
x = np.arange(w)
y = np.arange(h)
X,Y = np.meshgrid(x,y)


fig = plt.figure()
ax = fig.add_subplot(projection='3d')
# Plot a basic wireframe.
ax.plot_wireframe(X, Y, W, rstride=10, cstride=10)
plt.show()