#!/usr/bin/env python3

import json
import os.path as osp
import labelme
import imgviz 
import cv2
import numpy as np
import sys
# this is mixed up now; just a test
import patch_mapping
import dictionary
import numpy.random as rand 

def create_mask(filename):
    assert osp.exists(filename)

    data = json.load(open(filename))

    assert "imagePath" in data
    imageData = data.get("imageData", None)
    if imageData is None:
        parent_dir = osp.dirname(filename)
        img_file = osp.join(parent_dir, data["imagePath"])
        assert osp.exists(img_file)
        img = imgviz.io.imread(img_file)
    else:
        img = labelme.utils.img_b64_to_arr(imageData)
    mask = np.zeros(img.shape[:2],dtype=np.uint8)

    H, W = img.shape[:2]
    assert H == data["imageHeight"]
    assert W == data["imageWidth"]

    assert "shapes" in data
    N = 0
    for shape in data["shapes"]:
        assert "label" in shape
        assert "points" in shape
        p = list()
        for x, y in shape["points"]:
            x = int(x)
            y = int(y)
            assert 0 <= x <= W
            assert 0 <= y <= H
            p.append((x,y))
            print(x,y)
        N += 1
        print('shape',N, 'points',len(p))
        cv2.fillPoly(mask,pts=[np.array(p)],color=255)
    imgviz.io.imsave('mask.png',mask)
    # this is for debugging; should go somewhere else; definitely not here
    w = 16
    m = w*w
    s = 2
    Y = patch_mapping.extract_with_mask(np.mean(img,axis=2),w,s,mask)
    m,n = Y.shape
    K   = 4*m
    rng = rand.default_rng(42)
    A = rng.normal(size=(m,K))
    dictionary.normalize_dict(A)
    dictionary.train_dict(A,Y,0.1,1e-3)
    I = dictionary.show_bw(A,2,0)
    imgviz.io.imsave('dict.png',(255*I).astype(np.uint8))


if __name__ == '__main__':
    create_mask(sys.argv[1])