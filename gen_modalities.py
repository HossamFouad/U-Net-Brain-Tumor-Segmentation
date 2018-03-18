# -*- coding: utf-8 -*-
"""
Created on Mon Mar  5 23:09:46 2018

@author: HOSSAM ABDELHAMID
"""
import numpy as np
import os, cv2
import nibabel as nib

# Dataset Paths 
data_path_HGG="data/MICCAI_BraTS17_Data_Training/HGG"
data_path_LGG="data/MICCAI_BraTS17_Data_Training/LGG"
# Destination Paths
dst_dir_HGG= "data/processed_data/HGG1"
dst_dir_LGG= "data/processed_data/LGG1"
# list Folders
HGG_dir_list=os.listdir(data_path_HGG)
LGG_dir_list=os.listdir(data_path_LGG)
# modalities 
data_types = ['flair', 't1', 't1ce', 't2']

print('HGG modalities processing')
count0=1
for i in HGG_dir_list:
    print(i)
    for j in data_types:
        # image path
        img_path = os.path.join(data_path_HGG, i, i + '_' + j + '.nii.gz')
        # read image modality 
        img = nib.load(img_path).get_data()
        img=img[:,:,31:104]
        # scale image to [0,255]
        max_val=np.amax(img)
        img=img*255.0/max_val
        img=img.astype(int)
        count1=0
        for k in range(img.shape[2]):
            num=count0+count1
            cv2.imwrite(dst_dir_HGG +"/" + j +"/" +str(num)+".png", img[:,:,k])
            count1+=1
    count0+=img.shape[2]

print('HGG segmentation processing')    
count=1
for i in HGG_dir_list:
    print(i)
    # image path
    img_path = os.path.join(data_path_HGG, i, i + '_' + 'seg' + '.nii.gz')
    # read image modality 
    target = nib.load(img_path).get_data()
    target=target[:,:,31:104]
    target= (target > 0).astype(int)
    target= target*255 
    target=target.astype(int)
    for k in range(target.shape[2]):
        cv2.imwrite(dst_dir_HGG +"/" + 'seg' +"/" +str(count)+".png", target[:,:,k])
        count+=1 

print('LGG modalities processing')
count0=1
for i in LGG_dir_list:
    print(i)
    for j in data_types:
        # image path
        img_path = os.path.join(data_path_LGG, i, i + '_' + j + '.nii.gz')
        # read image modality 
        img = nib.load(img_path).get_data()
        img=img[:,:,31:104]
        # scale image to [0,255]
        max_val=np.amax(img)
        img=img*255.0/max_val
        img=img.astype(int)
        count1=0
        for k in range(img.shape[2]):
            num=count0+count1
            cv2.imwrite(dst_dir_LGG +"/" + j +"/" +str(num)+".png", img[:,:,k])
            count1+=1
    count0+=img.shape[2]

print('LGG segmentation processing')     
count=1
for i in LGG_dir_list:
    print(i)
    # image path
    img_path = os.path.join(data_path_LGG, i, i + '_' + 'seg' + '.nii.gz')
    # read target modality 
    target = nib.load(img_path).get_data()
    target=target[:,:,31:104]
    target= (target > 0).astype(int)
    target= target*255 
    target=target.astype(int)
    for k in range(target.shape[2]):
        cv2.imwrite(dst_dir_LGG +"/" + 'seg' +"/" +str(count)+".png", target[:,:,k])
        count+=1 
