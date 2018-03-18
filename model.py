import tensorflow as tf


SIZE=240
W,H,Z=(240,240,4)

def lrelu(x, alpha=0.2):
  return tf.nn.relu(x) - alpha * tf.nn.relu(-x)

class unet():
    def __init__(self,pad='SAME'):
        self.in_img_modal = tf.placeholder('float32', [None,W , H, Z])
        self.ground_truth_seg = tf.placeholder('float32', [None,W , H, 1])
        self.conv1 = tf.layers.conv2d(inputs=self.in_img_modal,filters=64,kernel_size=[4, 4],strides=(2, 2),padding=pad)
        
        conv2 = tf.layers.conv2d(inputs=self.conv1,filters=128,kernel_size=[4, 4],strides=(2, 2),padding=pad)
        conv2 = tf.layers.batch_normalization(conv2)
        self.conv2=lrelu(conv2)
    
        conv3 = tf.layers.conv2d(inputs=self.conv2,filters=256,kernel_size=[4, 4],strides=(2, 2),padding=pad)
        conv3 = tf.layers.batch_normalization(conv3)
        self.conv3=lrelu(conv3)
    
        conv4 = tf.layers.conv2d(inputs=self.conv3,filters=512,kernel_size=[4, 4],strides=(2, 2),padding=pad)
        conv4 = tf.layers.batch_normalization(conv4)
        self.conv4=lrelu(conv4)
    
        conv5 = tf.layers.conv2d(inputs=self.conv4,filters=512,kernel_size=[4, 4],strides=(2, 2),padding=pad)
        conv5 = tf.layers.batch_normalization(conv5)
        self.conv5=lrelu(conv5)
    
        conv6 = tf.layers.conv2d(inputs=self.conv5,filters=512,kernel_size=[4, 4],strides=(2, 2),padding=pad)
        conv6 = tf.layers.batch_normalization(conv6)
        self.conv6=lrelu(conv6)
        
        conv7 = tf.layers.conv2d(inputs=self.conv6,filters=512,kernel_size=[4, 4],strides=(2, 2),padding=pad)
        conv7 = tf.layers.batch_normalization(conv7)
        self.conv7=lrelu(conv7)
    
        conv8 = tf.layers.conv2d(inputs=self.conv7,filters=512,kernel_size=[4, 4],strides=(2, 2),padding=pad)
        self.conv8=lrelu(conv8)      
        ###########################################################
        
        upconv7=tf.layers.conv2d_transpose(self.conv8,filters=512,kernel_size=[4, 4], strides=(2, 2),padding=pad)
        upconv7 = tf.layers.batch_normalization(upconv7)
        self.upconv7=tf.nn.relu(upconv7)
        
        upconv6=tf.concat([self.upconv7, self.conv7], axis=3)
        upconv6=tf.layers.conv2d_transpose(upconv6,filters=1024, kernel_size=[4, 4], strides=(2, 2),padding=pad)
        upconv6 = tf.layers.batch_normalization(upconv6)
        self.upconv6=tf.nn.relu(upconv6)
   
        upconv5=tf.concat([self.upconv6, self.conv6], axis=3)
        upconv5=tf.layers.conv2d_transpose(upconv5,filters=1024, kernel_size=[4, 4], strides=(2, 2),padding=pad)
        upconv5 = tf.layers.batch_normalization(upconv5)
        self.upconv5=tf.nn.relu(upconv5)
    
        upconv4=tf.concat([self.upconv5, self.conv5], axis=3)
        upconv4=tf.layers.conv2d_transpose(upconv4,filters=1024, kernel_size=[4, 4], strides=(2, 2),padding=pad)
        upconv4 = tf.layers.batch_normalization(upconv4)
        self.upconv4=tf.nn.relu(upconv4)

        self.upconv4=tf.slice(self.upconv4, [0, 1,1, 0], [-1,15,15,1024])

        upconv3=tf.concat([self.upconv4, self.conv4], axis=3)
        upconv3=tf.layers.conv2d_transpose(upconv3,filters=256, kernel_size=[4, 4], strides=(2, 2),padding=pad)
        upconv3 = tf.layers.batch_normalization(upconv3)
        self.upconv3=tf.nn.relu(upconv3)
    
        upconv2=tf.concat([self.upconv3, self.conv3], axis=3)
        upconv2=tf.layers.conv2d_transpose(upconv2,filters=128, kernel_size=[4, 4], strides=(2, 2),padding=pad)
        upconv2 = tf.layers.batch_normalization(upconv2)
        self.upconv2=tf.nn.relu(upconv2)
   
        upconv1=tf.concat([self.upconv2, self.conv2], axis=3)
        upconv1=tf.layers.conv2d_transpose(upconv1,filters=64, kernel_size=[4, 4], strides=(2, 2),padding=pad)
        upconv1 = tf.layers.batch_normalization(upconv1)
        self.upconv1=tf.nn.relu(upconv1)
 
        upconv0=tf.concat([self.upconv1, self.conv1], axis=3)
        upconv0=tf.layers.conv2d_transpose(upconv0,filters=64, kernel_size=[4, 4], strides=(2, 2),padding=pad)
        upconv0 = tf.layers.batch_normalization(upconv0)
        self.upconv0=tf.nn.relu(upconv0)
   ###################################################
        self.out_seg = tf.layers.conv2d(inputs=self.upconv0,filters=1,kernel_size=[1, 1],activation=tf.nn.sigmoid)
    
