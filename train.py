# -*- coding: utf-8 -*-
"""
Created on Wed Mar  7 04:17:50 2018

@author: HOSSAM ABDELHAMID
"""


import os
import tensorflow as tf
import time
from model import unet 
from preprocess import DataReader  
import numpy as np
import matplotlib.pyplot as plt

def Hard_Dice(out,target):
    output = tf.cast(out > 0.5, dtype=tf.float32)
    target = tf.cast(target > 0.5, dtype=tf.float32)
    inse = tf.reduce_sum(tf.multiply(out, target), axis=[0,1,2,3])
    l = tf.reduce_sum(output, axis=[0,1,2,3])
    r = tf.reduce_sum(target, axis=[0,1,2,3])
    ## old axis=[0,1,2,3]
    # hard_dice = 2 * (inse) / (l + r)
    # epsilon = 1e-5
    # hard_dice = tf.clip_by_value(hard_dice, 0, 1.0-epsilon)
    ## new haodong
    hard_dice = (2. * inse + 1e-5) / (l + r + 1e-5)
    ##
    hard_dice = tf.reduce_mean(hard_dice)
    return hard_dice
def Dice(out,target):
    ins = tf.reduce_sum(out * target,axis=[0,1,2,3])
    l = tf.reduce_sum(out, axis=[0,1,2,3])
    r = tf.reduce_sum(target,  axis=[0,1,2,3])
    dice = (2. * ins + 1e-5) / (l + r + 1e-5)
    dice = tf.reduce_mean(dice)
    return dice

CHECKPOINT = 10
LOGDIR = 'vol'
RESTORE_FROM='model-step-229-val-0.296382.ckpt'
# Hyperparameters
BATCH_SIZE = 32
L_R = 0.0001 
BETA = 0.9
NUM_EPOCH = 100



def main():
        tf.reset_default_graph()
        sess = tf.InteractiveSession()
        ####Model###
        unet_model = unet()

        ####Compute loss###
        loss=1-Dice(unet_model.out_seg,unet_model.ground_truth_seg)
        HardDice=Hard_Dice(unet_model.out_seg,unet_model.ground_truth_seg)
        #Dice_hard=Hard_Dice(unet_model.out_seg,unet_model.ground_truth_seg)
        train_op = tf.train.AdamOptimizer(L_R).minimize(loss)
        ### Read Data###
        data_reader = DataReader()
        NUM_BATCHES_PER_EPOCH=int(data_reader.num_images/BATCH_SIZE)
        print('Num of batches per epoch :',NUM_BATCHES_PER_EPOCH)
        NUM_TEST_DATA=int(data_reader.total_test/BATCH_SIZE)
        NUM_STEPS=NUM_BATCHES_PER_EPOCH*NUM_EPOCH
        print("Total No. of iterations :",NUM_STEPS)
 
        steps=0
        test_error=0
        dice_over_epoch=[]
        ### Save ###
        saver = tf.train.Saver()
        ### Init Variables###
        init = tf.global_variables_initializer()
        sess.run(init)
          #restoring the model
        if RESTORE_FROM is not None:
            saver.restore(sess, os.getcwd()+'\\'+LOGDIR+'\\'+RESTORE_FROM)
            print('Model restored from ' + os.getcwd()+'\\'+LOGDIR+'\\'+RESTORE_FROM)
        for epoch in range(NUM_EPOCH):
            Hard_Dice_avg = 0.0
            for i in range(NUM_BATCHES_PER_EPOCH):
                start= time.time()
                steps+=1
                #get minibatch
                in_batch, out_batch = data_reader.load_train_batch(BATCH_SIZE)
                # run optimizer and loss
                _ , error = sess.run([train_op,loss], feed_dict={unet_model.in_img_modal:in_batch, unet_model.ground_truth_seg: out_batch })
                #evauate train error
                Dice_score = HardDice.eval(feed_dict={unet_model.in_img_modal:in_batch,  unet_model.ground_truth_seg: out_batch})
                #evaluate dice average loss per epoch
                Hard_Dice_avg += Dice_score/ NUM_BATCHES_PER_EPOCH
                end = time.time()
                elapsed = end - start 
                print("Step%d [Hard Dice=%g, Loss= %g , elapse= %g min]"  % (steps,Dice_score,error,elapsed*(NUM_STEPS-steps)/60))

        #test every 100 iteration
                if steps% 100 == 0 or steps==NUM_BATCHES_PER_EPOCH*NUM_EPOCH-1:
                    loss_test_avg = 0.0
                    test_hard_dice=00
                    for j in range(NUM_TEST_DATA):
                        in_batch, out_batch = data_reader.load_test_data(BATCH_SIZE)
                        test_error = loss.eval(feed_dict={unet_model.in_img_modal:in_batch,  unet_model.ground_truth_seg: out_batch})
                        loss_test_avg +=  test_error/ NUM_TEST_DATA
                        hard_dice = HardDice.eval(feed_dict={unet_model.in_img_modal:in_batch,  unet_model.ground_truth_seg: out_batch})
                        test_hard_dice +=  hard_dice/ NUM_TEST_DATA                        
                    print("Testing... Hard Dice=%g, loss= %g " % (test_hard_dice, loss_test_avg))
            #saving every 10 iteration
                if steps > 0 and steps % CHECKPOINT == 0:
                    if not os.path.exists(LOGDIR):
                        os.makedirs(LOGDIR)
                    checkpoint_path = os.path.join(LOGDIR, "model-step-%d-val-%g.ckpt" % (i, test_error))
                    filename = saver.save(sess, checkpoint_path)
                    print("Model saved in file: %s" % filename)
            dice_over_epoch.append(Hard_Dice_avg)
        
        checkpoint_path = os.path.join(LOGDIR, "model-step-final.ckpt")
        filename = saver.save(sess, checkpoint_path)
        print("Model saved in file: %s" % filename)
        # plot the cost
        plt.plot(np.squeeze(dice_over_epoch))
        plt.ylabel('cost')
        plt.xlabel('iterations (per tens)')
        plt.show()  
    
if __name__ == '__main__':
    main()