# -*- coding: utf-8 -*-
"""
Created on Wed Mar  7 04:37:04 2018

@author: HOSSAM ABDELHAMID
"""
import numpy as np
from scipy.ndimage.filters import gaussian_filter
from scipy.ndimage.interpolation import map_coordinates
import cv2

def flip(x):
    factor = np.random.uniform(-1, 1)
    process=x
    if factor > 0:
        process=[]
        for data in x:
            process.append(np.fliplr(data))
    return np.asarray(process)
    
#  This function is copied from 
# http://tensorlayer.readthedocs.io/en/latest/_modules/tensorlayer/prepro.html#elastic_transform       
def elastic_transform(x, alpha, sigma, mode="constant", cval=0):
    state = np.random.RandomState(None)
    shape = x[0].shape
    if len(shape) == 3:
        shape = (shape[0], shape[1])
    new_shape = state.rand(*shape)

    results = []
    for data in x:
        is_3d = False
        if len(data.shape) == 3 and data.shape[-1] == 1:
            data = data[:, :, 0]
            is_3d = True
        elif len(data.shape) == 3 and data.shape[-1] != 1:
            raise Exception("Only support greyscale image")
        assert len(data.shape) == 2, "input should be grey-scale image"

        dx = gaussian_filter((new_shape * 2 - 1), sigma, mode=mode, cval=cval) * alpha
        dy = gaussian_filter((new_shape * 2 - 1), sigma, mode=mode, cval=cval) * alpha

        x_, y_ = np.meshgrid(np.arange(shape[0]), np.arange(shape[1]), indexing='ij')
        indices = np.reshape(x_ + dx, (-1, 1)), np.reshape(y_ + dy, (-1, 1))
        # logging.info(data.shape)
        if is_3d:
            results.append(map_coordinates(data, indices, order=1).reshape((shape[0], shape[1], 1)))
        else:
            results.append(map_coordinates(data, indices, order=1).reshape(shape))
    return np.asarray(results)
    
        
def rotation(x):
    theta = int(np.random.uniform(0, 360))
    num_rows, num_cols = x[0].shape[:2]
    rotation_matrix = cv2.getRotationMatrix2D((num_cols/2, num_rows/2), theta, 1)
    results=[]
    for data in x:
        results.append(cv2.warpAffine(data, rotation_matrix, (num_cols, num_rows)))
    return np.asarray(results)


    
def shift(x, wrg=0.1, hrg=0.1): 
    h,w=x[0].shape
    tx = int(np.random.uniform(-hrg, hrg) * h)
    ty = int(np.random.uniform(-wrg, wrg) * w)
    M = np.float32([[1,0,tx],[0,1,ty]])
    results=[]
    for data in x:
        results.append(cv2.warpAffine(data,M,(w,h)))
    return np.asarray(results)


def shear(x,intensity=0.1):
    # Shear
    pts1 = np.float32([[[4,7],[19,5],[6,22]]])
    pt1 = 5+intensity*np.random.uniform()-intensity/2
    pt2 = 20+intensity*np.random.uniform()-intensity/2
    pts2 = np.float32([[pt1,5],[pt2,pt1],[5,pt2]])
    M = cv2.getAffineTransform(pts1,pts2)
    h,w=x[0].shape
    results=[]
    for data in x:
        results.append(cv2.warpAffine(data,M,(w,h)))
    return np.asarray(results)


def zoom(x,zoom_range=(1, 1.2)):
    zx, zy = np.random.uniform(zoom_range[0], zoom_range[1], 2)
    process=[]
    for data in x:
        n=cv2.resize(data,None,fx=zx, fy=zy, interpolation = cv2.INTER_LINEAR)
        row,col= n.shape
        print
        row=int(row/2)
        col=int(col/2)
        x1=row-120
        x2=row+120
        y1=col-120
        y2=col+120
        process.append(n[x1:x2,y1:y2])
    return np.asarray(process)

    







def data_aug(flair, t1, t1ce, t2, seg):
    """ data augumentation """
    flair, t1, t1ce, t2, seg = flip([flair, t1, t1ce, t2, seg]) # left right
    flair, t1, t1ce, t2, seg = elastic_transform([flair, t1, t1ce, t2, seg],alpha=720, sigma=24)
    flair, t1, t1ce, t2, seg = rotation([flair, t1, t1ce, t2, seg]) # nearest, constant
    flair, t1, t1ce, t2, seg = shift([flair, t1, t1ce, t2, seg], wrg=0.10,hrg=0.10)
    flair, t1, t1ce, t2, seg = shear([flair, t1, t1ce, t2, seg], 0.05)
    flair, t1, t1ce, t2, seg = zoom([flair, t1, t1ce, t2, seg], zoom_range=[1, 1.2])
    return flair, t1, t1ce, t2, seg