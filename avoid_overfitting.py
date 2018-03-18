# -*- coding: utf-8 -*-
"""
Created on Tue Mar  6 21:33:57 2018

@author: HOSSAM ABDELHAMID
"""
"""This file is to truncate the whole black images that can cause overfitting"""
import os
import scipy.misc
import numpy as np
dst_dir_HGG= "data/processed_data/HGG1/seg"
dst_dir_LGG= "data/processed_data/LGG1/seg"


arr = os.listdir(dst_dir_HGG)
target_list=[]
for i in range(1, len(arr)):
    image = scipy.misc.imread(dst_dir_HGG+"/"+str(i)+".png")
    if (np.amax(image)!=0):
        target_list.append(i)  
target = np.asarray(target_list)
scipy.io.savemat(os.getcwd()+'\\'+'HGG.mat', mdict={'arr': target})

arr = os.listdir(dst_dir_LGG)
target_list=[]
for i in range(1, len(arr)):
    image = scipy.misc.imread(dst_dir_LGG+"/"+str(i)+".png")
    if (np.amax(image)!=0):
        target_list.append(i)  
target = np.asarray(target_list)
scipy.io.savemat(os.getcwd()+'\\'+'LGG.mat', mdict={'arr': target})