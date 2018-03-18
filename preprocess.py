# -*- coding: utf-8 -*-
"""
Created on Mon Mar  5 23:12:13 2018

@author: HOSSAM ABDELHAMID
"""

import scipy.misc
import random
import csv
import numpy as np
from data_aug import data_aug
# modalities 
dst="data/processed_data/"
data_types = ["flair/", "t1/", "t1ce/", "t2/", "seg/"]
SIZE=240
class DataReader(object):
    def __init__(self):
        self.load()

    def load(self):
        grade_train = []     #input data
        image_train = []
        self.train_batch_pointer = 0 #pointer for taking mini batch one after another
        self.test_batch_pointer = 0
        self.num_images = 0  # Number of training samples
        # CVS file that has all images names, Class and quality factor
        with open('TRAIN.csv') as f:
            reader = csv.DictReader(f)
            print("Fetching Data...")
            for row in reader:
                grade_train.append(row['grade']+"/")
                image_train.append(row['image']+'.png')
                self.num_images += 1       
        print('Total training data: ' + str(self.num_images))
        
        c = list(zip(grade_train, image_train))
        random.shuffle(c)
        # Random Data xs->images , ys->ouptut labels one hot encoded matrix 
        self.grade_train, self.image_train = zip(*c)
        grade_test = []     #input data
        image_test = []
        self.total_test = 0
        # CVS file that has all images names, Class and quality factor
        with open('TEST.csv') as f:
            reader = csv.DictReader(f)
            print("Fetching Testing Data ...")
            for row in reader:
                grade_test.append(row['grade']+"/")
                image_test.append(row['image']+'.png')
                self.total_test += 1
        print('Total test data: ' + str(self.total_test))
        c = list(zip(grade_test, image_test))
        # Random Data xs->images , ys->ouptut labels one hot encoded matrix 
        random.shuffle(c)
        self.grade_test, self.image_test = zip(*c)

        # Get Random mini batch of size batch_size
    def load_train_batch(self, batch_size):
        x_out = np.zeros((batch_size,SIZE,SIZE,4))
        y_out = np.zeros((batch_size,SIZE,SIZE,1))
        t1 = np.zeros((SIZE,SIZE,1))
        t1ce = np.zeros((SIZE,SIZE,1))
        t2 = np.zeros((SIZE,SIZE,1))
        flair = np.zeros((SIZE,SIZE,1))
        target = np.zeros((SIZE,SIZE,1))
        for i in range(0, batch_size):
            t1 = scipy.misc.imread(dst+self.grade_train[(self.train_batch_pointer + i) % self.num_images]+data_types[0]+self.image_train[(self.train_batch_pointer + i) % self.num_images])
            t1ce = scipy.misc.imread(dst+self.grade_train[(self.train_batch_pointer + i) % self.num_images]+data_types[1]+self.image_train[(self.train_batch_pointer + i) % self.num_images])
            t2 = scipy.misc.imread(dst+self.grade_train[(self.train_batch_pointer + i) % self.num_images]+data_types[2]+self.image_train[(self.train_batch_pointer + i) % self.num_images])
            flair = scipy.misc.imread(dst+self.grade_train[(self.train_batch_pointer + i) % self.num_images]+data_types[3]+self.image_train[(self.train_batch_pointer + i) % self.num_images])
            target=scipy.misc.imread(dst+self.grade_train[(self.train_batch_pointer + i) % self.num_images]+data_types[4]+self.image_train[(self.train_batch_pointer + i) % self.num_images])            
            flair, t1, t1ce, t2, target=data_aug(flair, t1, t1ce, t2, target)
            combined_array = np.stack((t1,t1ce, t2, flair), axis=2)
            combined_array=combined_array.reshape(SIZE,SIZE,4)
            x_out[i,:,:,:]=combined_array / 255.0
            target=target.reshape(SIZE,SIZE,1)
            target=target/255
            y_out[i,:,:,:]=target
        y_out=y_out.astype(int)
        self.train_batch_pointer += batch_size
        return x_out, y_out
    
    def load_test_data(self, test_size):
        x_out = np.zeros((test_size,SIZE,SIZE,4))
        y_out = np.zeros((test_size,SIZE,SIZE,1))
        t1 = np.zeros((SIZE,SIZE,1))
        t1ce = np.zeros((SIZE,SIZE,1))
        t2 = np.zeros((SIZE,SIZE,1))
        flair = np.zeros((SIZE,SIZE,1))
        target = np.zeros((SIZE,SIZE,1))
        for i in range(0, test_size):
            t1 = scipy.misc.imread(dst+self.grade_test[(self.test_batch_pointer + i) % self.total_test]+data_types[0]+self.image_test[(self.test_batch_pointer + i) % self.total_test])
            t1ce = scipy.misc.imread(dst+self.grade_test[(self.test_batch_pointer + i) % self.total_test]+data_types[1]+self.image_test[(self.test_batch_pointer + i) % self.total_test])
            t2 = scipy.misc.imread(dst+self.grade_test[(self.test_batch_pointer + i) % self.total_test]+data_types[2]+self.image_test[(self.test_batch_pointer + i) % self.total_test])
            flair = scipy.misc.imread(dst+self.grade_test[(self.test_batch_pointer + i) % self.total_test]+data_types[3]+self.image_test[(self.test_batch_pointer + i) % self.total_test])
            target=scipy.misc.imread(dst+self.grade_test[(self.test_batch_pointer + i) % self.total_test]+data_types[4]+self.image_test[(self.test_batch_pointer + i) % self.total_test])
            flair, t1, t1ce, t2, target=data_aug(flair, t1, t1ce, t2, target)
            combined_array = np.stack((t1,t1ce, t2, flair), axis=2)
            combined_array=combined_array.reshape(SIZE,SIZE,4)
            target=target.reshape(SIZE,SIZE,1)
            x_out[i,:,:,:]=combined_array / 255.0
            target=target/255
            y_out[i,:,:,:]=target
        y_out=y_out.astype(int)
        self.test_batch_pointer += test_size
        return x_out, y_out
''''
x=DataReader()
d,c=x.load_test_data(1)
import matplotlib.pyplot as plt
print(d[0,:,:,:].shape)
imgplot = plt.imshow(d[0,:,:,0],cmap='gray')
plt.show()
'''