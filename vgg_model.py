#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May  7 15:03:04 2018

@author: dayea
"""


#input image_size = [n, 224, 224, 3]

import tensorflow as tf
import numpy as np
import os

class vgg16:
  def __init__(self, transfer_para=None, trained_para=None, class_num=None, transfer_flag=True, continue_flag=True):
    
    self.class_num = class_num
    self.para_file = trained_para
    
    if continue_flag:
      if os.path.exists(trained_para):
        print('Loading trained parameters')
        self.weights = np.load(open(trained_para, "rb")).item()
        if transfer_flag:
          print('Restoring trained transfer parameters in continue_transfer mode')
          self.init_para_transfer(trained_flag=True)
        else:
          print('Restoring trained no_transfer parameters in continue_no_transfer mode')
          self.init_para_no_transfer(trained_flag=True)
      else:
        if transfer_flag:
          if os.path.exists(transfer_para):
            print('Loading transfer parameters in cintinue_transfer mode')
            self.weights = np.load(transfer_para)
            print('Restoring transfer parameters in cintinue_transfer mode')
            self.init_para_transfer(trained_flag=False)
          else:
            print('error: Cannot find offical parameters file')
        else:
          print('Initializing new parameters in cintinue_no_transfer mode')
          self.init_para_no_transfer(trained_flag=False)
    else:
      if transfer_flag:
        if os.path.exists(transfer_para):
          print('Loading transfer parameters')
          self.weights = np.load(transfer_para)
          print('Restoring transfer parameters in once_transfer mode')
          self.init_para_transfer(trained_flag=False)
        else:
          print('error: Cannot find offical parameters file')
      else:
        print('Initializing new parameters in once_no_transfer mode')
        self.init_para_no_transfer(trained_flag=False)
    
    
  def init_para_transfer(self, trained_flag=False):
    self.init_para_transfer_variable(trained_flag=trained_flag)
    
    
  def init_para_transfer_constant(self, trained_flag=False):

    # conv1_1
    self.conv1_1_kernel = tf.constant(self.weights['conv1_1_W'], dtype=tf.float32, shape=[3, 3, 3, 64], name='weights')
    self.conv1_1_biases = tf.constant(self.weights['conv1_1_b'], dtype=tf.float32, shape=[64], name='biases')
    
    # conv1_2
    self.conv1_2_kernel = tf.constant(self.weights['conv1_2_W'], dtype=tf.float32, shape=[3, 3, 64, 64], name='weights')
    self.conv1_2_biases = tf.constant(self.weights['conv1_2_b'],  dtype=tf.float32, shape=[64], name='biases')
    
    # conv2_1
    self.conv2_1_kernel = tf.constant(self.weights['conv2_1_W'], dtype=tf.float32, shape=[3, 3, 64, 128], name='weights')
    self.conv2_1_biases = tf.constant(self.weights['conv2_1_b'], dtype=tf.float32, shape=[128], name='biases')
    # conv2_2
    self.conv2_2_kernel = tf.constant(self.weights['conv2_2_W'], dtype=tf.float32, shape=[3, 3, 128, 128], name='weights')
    self.conv2_2_biases = tf.constant(self.weights['conv2_2_b'], dtype=tf.float32, shape=[128], name='biases')
    
    self.conv3_1_kernel = tf.constant(self.weights['conv3_1_W'], dtype=tf.float32, shape=[3, 3, 128, 256], name='weights')
    self.conv3_1_biases = tf.constant(self.weights['conv3_1_b'], dtype=tf.float32, shape=[256], name='biases')
    
    # conv3_2
    self.conv3_2_kernel = tf.constant(self.weights['conv3_2_W'], dtype=tf.float32, shape=[3, 3, 256, 256], name='weights')
    self.conv3_2_biases = tf.constant(self.weights['conv3_2_b'], dtype=tf.float32, shape=[256], name='biases')
    
    # conv3_3
    self.conv3_3_kernel = tf.constant(self.weights['conv3_3_W'], dtype=tf.float32, shape=[3, 3, 256, 256], name='weights')
    self.conv3_3_biases = tf.constant(self.weights['conv3_3_b'], dtype=tf.float32, shape=[256], name='biases')
    
    # conv4_1
    self.conv4_1_kernel = tf.constant(self.weights['conv4_1_W'], dtype=tf.float32, shape=[3, 3, 256, 512], name='weights')
    self.conv4_1_biases = tf.constant(self.weights['conv4_1_b'], dtype=tf.float32, shape=[512], name='biases')
    
    # conv4_2
    self.conv4_2_kernel = tf.constant(self.weights['conv4_2_W'], dtype=tf.float32, shape=[3, 3, 512, 512], name='weights')
    self.conv4_2_biases = tf.constant(self.weights['conv4_2_b'], dtype=tf.float32, shape=[512], name='biases')
    
    # conv4_3
    self.conv4_3_kernel = tf.constant(self.weights['conv4_3_W'], dtype=tf.float32, shape=[3, 3, 512, 512], name='weights')
    self.conv4_3_biases = tf.constant(self.weights['conv4_3_b'], dtype=tf.float32, shape=[512], name='biases')
    
    
    # conv5_1
    self.conv5_1_kernel = tf.constant(self.weights['conv5_1_W'], dtype=tf.float32, shape=[3, 3, 512, 512], name='weights')
    self.conv5_1_biases = tf.constant(self.weights['conv5_1_b'], dtype=tf.float32, shape=[512], name='biases')
    
    # conv5_2
    self.conv5_2_kernel = tf.constant(self.weights['conv5_2_W'], dtype=tf.float32, shape=[3, 3, 512, 512], name='weights')
    self.conv5_2_biases = tf.constant(self.weights['conv5_2_b'], dtype=tf.float32, shape=[512], name='biases')
    
    # conv5_3
    self.conv5_3_kernel = tf.constant(self.weights['conv5_3_W'], dtype=tf.float32, shape=[3, 3, 512, 512], name='weights')
    self.conv5_3_biases = tf.constant(self.weights['conv5_3_b'], dtype=tf.float32, shape=[512], name='biases')
    
    
#    self.fc1w = tf.constant(self.weights['fc6_W'], dtype=tf.float32, shape=[25088, 4096], name='weights')
#    self.fc1b = tf.constant(self.weights['fc6_b'], dtype=tf.float32, shape=[4096], name='biases')
#    
    self.fc1w = tf.Variable(self.weights['fc6_W'], dtype=tf.float32, name='weights')
    self.fc1b = tf.Variable(self.weights['fc6_b'], dtype=tf.float32, name='biases')
    
    # fc2
    self.fc2w = tf.Variable(self.weights['fc7_W'], dtype=tf.float32, name='weights')
    self.fc2b = tf.Variable(self.weights['fc7_b'], dtype=tf.float32, name='biases')
    
    if trained_flag:
      self.fc3w = tf.Variable(self.weights['fc8_W'], dtype=tf.float32, name='weights')
      self.fc3b = tf.Variable(self.weights['fc8_b'], dtype=tf.float32, name='biases')
    else:
      self.fc3w = tf.Variable(tf.random_normal(shape=[4096, self.class_num], stddev=1, dtype=tf.float32), name='weights')
      self.fc3b = tf.Variable(tf.random_normal(shape=[self.class_num], stddev=1, dtype=tf.float32), name='biases')
          
  def init_para_transfer_variable(self, trained_flag=False):

    # conv1_1
    self.conv1_1_kernel = tf.Variable(self.weights['conv1_1_W'], dtype=tf.float32, name='weights')
    self.conv1_1_biases = tf.Variable(self.weights['conv1_1_b'], dtype=tf.float32, name='biases')
    
    # conv1_2
    self.conv1_2_kernel = tf.Variable(self.weights['conv1_2_W'], dtype=tf.float32, name='weights')
    self.conv1_2_biases = tf.Variable(self.weights['conv1_2_b'], dtype=tf.float32, name='biases')
    
    # conv2_1
    self.conv2_1_kernel = tf.Variable(self.weights['conv2_1_W'], dtype=tf.float32, name='weights')
    self.conv2_1_biases = tf.Variable(self.weights['conv2_1_b'], dtype=tf.float32, name='biases')
    # conv2_2
    self.conv2_2_kernel = tf.Variable(self.weights['conv2_2_W'], dtype=tf.float32, name='weights')
    self.conv2_2_biases = tf.Variable(self.weights['conv2_2_b'], dtype=tf.float32, name='biases')
    
    self.conv3_1_kernel = tf.Variable(self.weights['conv3_1_W'], dtype=tf.float32, name='weights')
    self.conv3_1_biases = tf.Variable(self.weights['conv3_1_b'], dtype=tf.float32, name='biases')
    
    # conv3_2
    self.conv3_2_kernel = tf.Variable(self.weights['conv3_2_W'], dtype=tf.float32, name='weights')
    self.conv3_2_biases = tf.Variable(self.weights['conv3_2_b'], dtype=tf.float32, name='biases')
    
    # conv3_3
    self.conv3_3_kernel = tf.Variable(self.weights['conv3_3_W'], dtype=tf.float32, name='weights')
    self.conv3_3_biases = tf.Variable(self.weights['conv3_3_b'], dtype=tf.float32, name='biases')
    
    # conv4_1
    self.conv4_1_kernel = tf.Variable(self.weights['conv4_1_W'], dtype=tf.float32, name='weights')
    self.conv4_1_biases = tf.Variable(self.weights['conv4_1_b'], dtype=tf.float32, name='biases')
    
    # conv4_2
    self.conv4_2_kernel = tf.Variable(self.weights['conv4_2_W'], dtype=tf.float32, name='weights')
    self.conv4_2_biases = tf.Variable(self.weights['conv4_2_b'], dtype=tf.float32, name='biases')
    
    # conv4_3
    self.conv4_3_kernel = tf.Variable(self.weights['conv4_3_W'], dtype=tf.float32, name='weights')
    self.conv4_3_biases = tf.Variable(self.weights['conv4_3_b'], dtype=tf.float32, name='biases')
    
    
    # conv5_1
    self.conv5_1_kernel = tf.Variable(self.weights['conv5_1_W'], dtype=tf.float32, name='weights')
    self.conv5_1_biases = tf.Variable(self.weights['conv5_1_b'], dtype=tf.float32, name='biases')
    
    # conv5_2
    self.conv5_2_kernel = tf.Variable(self.weights['conv5_2_W'], dtype=tf.float32, name='weights')
    self.conv5_2_biases = tf.Variable(self.weights['conv5_2_b'], dtype=tf.float32, name='biases')
    
    # conv5_3
    self.conv5_3_kernel = tf.Variable(self.weights['conv5_3_W'], dtype=tf.float32, name='weights')
    self.conv5_3_biases = tf.Variable(self.weights['conv5_3_b'], dtype=tf.float32, name='biases')
    

#    
    self.fc1w = tf.Variable(self.weights['fc6_W'], dtype=tf.float32, name='weights')
    self.fc1b = tf.Variable(self.weights['fc6_b'], dtype=tf.float32, name='biases')
    
    # fc2
    self.fc2w = tf.Variable(self.weights['fc7_W'], dtype=tf.float32, name='weights')
    self.fc2b = tf.Variable(self.weights['fc7_b'], dtype=tf.float32, name='biases')
    
    if trained_flag:
      self.fc3w = tf.Variable(self.weights['fc8_W'], dtype=tf.float32, name='weights')
      self.fc3b = tf.Variable(self.weights['fc8_b'], dtype=tf.float32, name='biases')
    else:
      self.fc3w = tf.Variable(tf.random_normal(shape=[4096, self.class_num], stddev=1, dtype=tf.float32), name='weights')
      self.fc3b = tf.Variable(tf.random_normal(shape=[self.class_num], stddev=1, dtype=tf.float32), name='biases')


      

  def init_para_no_transfer(self, trained_flag=False):
    
    if trained_flag:
      # conv1_1
      self.conv1_1_kernel = tf.Variable(self.weights['conv1_1_W'], dtype=tf.float32,  name='weights')
      self.conv1_1_biases = tf.Variable(self.weights['conv1_1_b'], dtype=tf.float32, name='biases')
      
      # conv1_2
      self.conv1_2_kernel = tf.Variable(self.weights['conv1_2_W'], dtype=tf.float32, name='weights')
      self.conv1_2_biases = tf.Variable(self.weights['conv1_2_b'],  dtype=tf.float32, name='biases')
      
      # pool1
      
      # conv2_1
      self.conv2_1_kernel = tf.Variable(self.weights['conv2_1_W'], dtype=tf.float32, name='weights')
      self.conv2_1_biases = tf.Variable(self.weights['conv2_1_b'], dtype=tf.float32, name='biases')
      # conv2_2
      self.conv2_2_kernel = tf.Variable(self.weights['conv2_2_W'], dtype=tf.float32, name='weights')
      self.conv2_2_biases = tf.Variable(self.weights['conv2_2_b'], dtype=tf.float32, name='biases')
      
      self.conv3_1_kernel = tf.Variable(self.weights['conv3_1_W'], dtype=tf.float32, name='weights')
      self.conv3_1_biases = tf.Variable(self.weights['conv3_1_b'], dtype=tf.float32, name='biases')
      
      # conv3_2
      self.conv3_2_kernel = tf.Variable(self.weights['conv3_2_W'], dtype=tf.float32, name='weights')
      self.conv3_2_biases = tf.Variable(self.weights['conv3_2_b'], dtype=tf.float32, name='biases')
      
      # conv3_3
      self.conv3_3_kernel = tf.Variable(self.weights['conv3_3_W'], dtype=tf.float32, name='weights')
      self.conv3_3_biases = tf.Variable(self.weights['conv3_3_b'], dtype=tf.float32, name='biases')
      
      # conv4_1
      self.conv4_1_kernel = tf.Variable(self.weights['conv4_1_W'], dtype=tf.float32, name='weights')
      self.conv4_1_biases = tf.Variable(self.weights['conv4_1_b'], dtype=tf.float32, name='biases')
      
      # conv4_2
      self.conv4_2_kernel = tf.Variable(self.weights['conv4_2_W'], dtype=tf.float32, name='weights')
      self.conv4_2_biases = tf.Variable(self.weights['conv4_2_b'], dtype=tf.float32, name='biases')
      
      # conv4_3
      self.conv4_3_kernel = tf.Variable(self.weights['conv4_3_W'], dtype=tf.float32, name='weights')
      self.conv4_3_biases = tf.Variable(self.weights['conv4_3_b'], dtype=tf.float32, name='biases')
      
      
      # conv5_1
      self.conv5_1_kernel = tf.Variable(self.weights['conv5_1_W'], dtype=tf.float32, name='weights')
      self.conv5_1_biases = tf.Variable(self.weights['conv5_1_b'], dtype=tf.float32, name='biases')
      
      # conv5_2
      self.conv5_2_kernel = tf.Variable(self.weights['conv5_2_W'], dtype=tf.float32, name='weights')
      self.conv5_2_biases = tf.Variable(self.weights['conv5_2_b'], dtype=tf.float32, name='biases')
      
      # conv5_3
      self.conv5_3_kernel = tf.Variable(self.weights['conv5_3_W'], dtype=tf.float32, name='weights')
      self.conv5_3_biases = tf.Variable(self.weights['conv5_3_b'], dtype=tf.float32, name='biases')
      
      
  #    self.fc1w = tf.constant(self.weights['fc6_W'], dtype=tf.float32, shape=[25088, 4096], name='weights')
  #    self.fc1b = tf.constant(self.weights['fc6_b'], dtype=tf.float32, shape=[4096], name='biases')
  #    
      self.fc1w = tf.Variable(self.weights['fc6_W'], dtype=tf.float32, name='weights')
      self.fc1b = tf.Variable(self.weights['fc6_b'], dtype=tf.float32, name='biases')
      
      # fc2
      self.fc2w = tf.Variable(self.weights['fc7_W'], dtype=tf.float32, name='weights')
      self.fc2b = tf.Variable(self.weights['fc7_b'], dtype=tf.float32, name='biases')
      
      self.fc3w = tf.Variable(self.weights['fc8_W'], name='weights')
      self.fc3b = tf.Variable(self.weights['fc8_b'], name='biases')
      
    else:
    
      # conv1_1  
      self.conv1_1_kernel = tf.Variable(tf.random_normal(shape=[3, 3, 3, 64], stddev=1, dtype=tf.float32), name='conv1_1_W')
      self.conv1_1_biases = tf.Variable(tf.random_normal(shape=[64], stddev=1, dtype=tf.float32), name='conv1_1_b')
      
      # conv1_2
      self.conv1_2_kernel = tf.Variable(tf.random_normal(shape=[3, 3, 64, 64], stddev=1, dtype=tf.float32), name='conv1_2_W')
      self.conv1_2_biases = tf.Variable(tf.random_normal(shape=[64], stddev=1, dtype=tf.float32), name='conv1_2_b')
      
      # conv2_1
      self.conv2_1_kernel = tf.Variable(tf.random_normal(shape=[3, 3, 64, 128], stddev=1, dtype=tf.float32), name='conv2_1_W')
      self.conv2_1_biases = tf.Variable(tf.random_normal(shape=[128], stddev=1, dtype=tf.float32), name='conv2_1_b')
      
      # conv2_2
      self.conv2_2_kernel = tf.Variable(tf.random_normal(shape=[3, 3, 128, 128], stddev=1, dtype=tf.float32), name='conv2_2_W')
      self.conv2_2_biases = tf.Variable(tf.random_normal(shape=[128], stddev=1, dtype=tf.float32), name='conv2_2_b')
      
      self.conv3_1_kernel = tf.Variable(tf.random_normal(shape=[3, 3, 128, 256], stddev=1, dtype=tf.float32), name='conv3_1_W')
      self.conv3_1_biases = tf.Variable(tf.random_normal(shape=[256], stddev=1, dtype=tf.float32), name='conv3_1_b')
      
      # conv3_2
      self.conv3_2_kernel = tf.Variable(tf.random_normal(shape=[3, 3, 256, 256], stddev=1, dtype=tf.float32), name='conv3_2_W')
      self.conv3_2_biases = tf.Variable(tf.random_normal(shape=[256], stddev=1, dtype=tf.float32), name='conv3_2_b')
      
      # conv3_3
      self.conv3_3_kernel = tf.Variable(tf.random_normal(shape=[3, 3, 256, 256], stddev=1, dtype=tf.float32), name='conv3_3_W')
      self.conv3_3_biases = tf.Variable(tf.random_normal(shape=[256], stddev=1, dtype=tf.float32), name='conv3_3_b')
      
      # conv4_1
      self.conv4_1_kernel = tf.Variable(tf.random_normal(shape=[3, 3, 256, 512], stddev=1, dtype=tf.float32), name='conv4_1_W')
      self.conv4_1_biases = tf.Variable(tf.random_normal(shape=[512], stddev=1, dtype=tf.float32), name='conv4_1_b')
      
      # conv4_2
      self.conv4_2_kernel = tf.Variable(tf.random_normal(shape=[3, 3, 512, 512], stddev=1, dtype=tf.float32), name='conv4_2_W')
      self.conv4_2_biases = tf.Variable(tf.random_normal(shape=[512], stddev=1, dtype=tf.float32), name='conv4_2_b')
      
      # conv4_3
      self.conv4_3_kernel = tf.Variable(tf.random_normal(shape=[3, 3, 512, 512], stddev=1, dtype=tf.float32), name='conv4_3_W')
      self.conv4_3_biases = tf.Variable(tf.random_normal(shape=[512], stddev=1, dtype=tf.float32), name='conv4_3_b')
      
      
      # conv5_1
      self.conv5_1_kernel = tf.Variable(tf.random_normal(shape=[3, 3, 512, 512], stddev=1, dtype=tf.float32), name='conv5_1_W')
      self.conv5_1_biases = tf.Variable(tf.random_normal(shape=[512], stddev=1, dtype=tf.float32), name='conv5_1_b')
      
      # conv5_2
      self.conv5_2_kernel = tf.Variable(tf.random_normal(shape=[3, 3, 512, 512], stddev=1, dtype=tf.float32), name='conv5_2_W')
      self.conv5_2_biases = tf.Variable(tf.random_normal(shape=[512], stddev=1, dtype=tf.float32), name='conv5_2_b')
      
      # conv5_3
      self.conv5_3_kernel = tf.Variable(tf.random_normal(shape=[3, 3, 512, 512], stddev=1, dtype=tf.float32), name='conv5_3_W')
      self.conv5_3_biases = tf.Variable(tf.random_normal(shape=[512], stddev=1, dtype=tf.float32), name='conv5_3_b')
      
      
      self.fc1w = tf.Variable(tf.random_normal(shape=[25088, 4096], stddev=1, dtype=tf.float32), name='weights')
      self.fc1b = tf.Variable(tf.random_normal(shape=[4096], stddev=1, dtype=tf.float32), name='biases')
      
      # fc2
      self.fc2w = tf.Variable(tf.random_normal(shape=[4096, 4096], stddev=1, dtype=tf.float32), name='weights')
      self.fc2b = tf.Variable(tf.random_normal(shape=[4096], stddev=1, dtype=tf.float32), name='biases')
      
      self.fc3w = tf.Variable(tf.random_normal(shape=[4096, self.class_num], stddev=1, dtype=tf.float32), name='weights')
      self.fc3b = tf.Variable(tf.random_normal(shape=[self.class_num], stddev=1, dtype=tf.float32), name='biases')
      

  def model(self, images):
    
#        # zero-mean input
#        with tf.name_scope('preprocess') as scope:
#            mean = tf.constant([123.68, 116.779, 103.939], dtype=tf.float32, shape=[1, 1, 1, 3], name='img_mean')
#            

    # conv1_1
    conv = tf.nn.conv2d(images, self.conv1_1_kernel, [1, 1, 1, 1], padding='SAME')
    out = tf.nn.bias_add(conv, self.conv1_1_biases)
    conv1_1 = tf.nn.relu(out)
    
    # conv1_2
    conv = tf.nn.conv2d(conv1_1, self.conv1_2_kernel, [1, 1, 1, 1], padding='SAME')
    out = tf.nn.bias_add(conv, self.conv1_2_biases)
    conv1_2 = tf.nn.relu(out)
    
    # pool1
    pool1 = tf.nn.max_pool(conv1_2,
                       ksize=[1, 2, 2, 1],
                       strides=[1, 2, 2, 1],
                       padding='SAME',
                       name='pool1')
    
    # conv2_1
    conv = tf.nn.conv2d(pool1, self.conv2_1_kernel, [1, 1, 1, 1], padding='SAME')
    out = tf.nn.bias_add(conv, self.conv2_1_biases)
    conv2_1 = tf.nn.relu(out)
    
    # conv2_2
    conv = tf.nn.conv2d(conv2_1, self.conv2_2_kernel, [1, 1, 1, 1], padding='SAME')
    out = tf.nn.bias_add(conv, self.conv2_2_biases)
    conv2_2 = tf.nn.relu(out)
    
    # pool2
    pool2 = tf.nn.max_pool(conv2_2,
                       ksize=[1, 2, 2, 1],
                       strides=[1, 2, 2, 1],
                       padding='SAME',
                       name='pool2')
    
    # conv3_1
    conv = tf.nn.conv2d(pool2, self.conv3_1_kernel, [1, 1, 1, 1], padding='SAME')
    out = tf.nn.bias_add(conv, self.conv3_1_biases)
    conv3_1 = tf.nn.relu(out)
    
    # conv3_2
    conv = tf.nn.conv2d(conv3_1, self.conv3_2_kernel, [1, 1, 1, 1], padding='SAME')
    out = tf.nn.bias_add(conv, self.conv3_2_biases)
    conv3_2 = tf.nn.relu(out)
    
    # conv3_3
    conv = tf.nn.conv2d(conv3_2, self.conv3_3_kernel, [1, 1, 1, 1], padding='SAME')
    out = tf.nn.bias_add(conv, self.conv3_3_biases)
    conv3_3 = tf.nn.relu(out)
    
    # pool3
    pool3 = tf.nn.max_pool(conv3_3,
                       ksize=[1, 2, 2, 1],
                       strides=[1, 2, 2, 1],
                       padding='SAME',
                       name='pool3')
    
    # conv4_1
    conv = tf.nn.conv2d(pool3, self.conv4_1_kernel, [1, 1, 1, 1], padding='SAME')
    out = tf.nn.bias_add(conv, self.conv4_1_biases)
    conv4_1 = tf.nn.relu(out)
    
    # conv4_2
    conv = tf.nn.conv2d(conv4_1, self.conv4_2_kernel, [1, 1, 1, 1], padding='SAME')
    out = tf.nn.bias_add(conv, self.conv4_2_biases)
    conv4_2 = tf.nn.relu(out)
    
    # conv4_3
    conv = tf.nn.conv2d(conv4_2, self.conv4_3_kernel, [1, 1, 1, 1], padding='SAME')
    out = tf.nn.bias_add(conv, self.conv4_3_biases)
    conv4_3 = tf.nn.relu(out)
    
    # pool4
    pool4 = tf.nn.max_pool(conv4_3,
                       ksize=[1, 2, 2, 1],
                       strides=[1, 2, 2, 1],
                       padding='SAME',
                       name='pool4')
    
    # conv5_1
    conv = tf.nn.conv2d(pool4, self.conv5_1_kernel, [1, 1, 1, 1], padding='SAME')
    out = tf.nn.bias_add(conv, self.conv5_1_biases)
    conv5_1 = tf.nn.relu(out)
    
    # conv5_2
    conv = tf.nn.conv2d(conv5_1, self.conv5_2_kernel, [1, 1, 1, 1], padding='SAME')
    out = tf.nn.bias_add(conv, self.conv5_2_biases)
    conv5_2 = tf.nn.relu(out)
    
    # conv5_3
    conv = tf.nn.conv2d(conv5_2, self.conv5_3_kernel, [1, 1, 1, 1], padding='SAME')
    out = tf.nn.bias_add(conv, self.conv5_3_biases)
    conv5_3 = tf.nn.relu(out)
    
    # pool5
    pool5 = tf.nn.max_pool(conv5_3,
                       ksize=[1, 2, 2, 1],
                       strides=[1, 2, 2, 1],
                       padding='SAME',
                       name='pool4')
    
    
    shape = int(np.prod(pool5.get_shape()[1:]))
    pool5_flat = tf.reshape(pool5, [-1, shape])
    fc1l = tf.nn.bias_add(tf.matmul(pool5_flat, self.fc1w), self.fc1b)
    fc1 = tf.nn.relu(fc1l)
    
    # fc2
    
     
    fc2l = tf.nn.bias_add(tf.matmul(fc1, self.fc2w), self.fc2b)
    fc2 = tf.nn.relu(fc2l)
    
    
    
    fc3l = tf.nn.bias_add(tf.matmul(fc2, self.fc3w), self.fc3b)
    
    return fc3l, pool5 #[512]
  
  
  def save_model(self, sess):
    print('Saving model...')
#    para_name
    
    # conv1_1
    conv1_1_kernel, conv1_1_biases = sess.run([self.conv1_1_kernel,self.conv1_1_biases])
    
    # conv1_2
    conv1_2_kernel,conv1_2_biases = sess.run([self.conv1_2_kernel,self.conv1_2_biases])
    
    # conv2_1
    conv2_1_kernel,conv2_1_biases = sess.run([self.conv2_1_kernel,self.conv2_1_biases])
    # conv2_2
    conv2_2_kernel,conv2_2_biases = sess.run([self.conv2_2_kernel,self.conv2_2_biases])
    
    conv3_1_kernel,conv3_1_biases = sess.run([self.conv3_1_kernel,self.conv3_1_biases])
    
    # conv3_2
    conv3_2_kernel,conv3_2_biases = sess.run([self.conv3_2_kernel,self.conv3_2_biases])
    
    # conv3_3
    conv3_3_kernel,conv3_3_biases = sess.run([self.conv3_3_kernel,self.conv3_3_biases])
    
    # conv4_1
    conv4_1_kernel,conv4_1_biases = sess.run([self.conv4_1_kernel,self.conv4_1_biases])
    
    # conv4_2
    conv4_2_kernel,conv4_2_biases = sess.run([self.conv4_2_kernel,self.conv4_2_biases])
    
    # conv4_3
    conv4_3_kernel,conv4_3_biases = sess.run([self.conv4_3_kernel,self.conv4_3_biases])
    
    # conv5_1
    conv5_1_kernel,conv5_1_biases = sess.run([self.conv5_1_kernel,self.conv5_1_biases])
    
    # conv5_2
    conv5_2_kernel,conv5_2_biases = sess.run([self.conv5_2_kernel,self.conv5_2_biases])
    
    # conv5_3
    conv5_3_kernel,conv5_3_biases = sess.run([self.conv5_3_kernel,self.conv5_3_biases])
    
    
    fc1w,fc1b = sess.run([self.fc1w,self.fc1b])
    
    fc2w,fc2b = sess.run([self.fc2w,self.fc2b])
    
    fc3w,fc3b  = sess.run([self.fc3w,self.fc3b])
    
    para = {
      'conv1_1_W': conv1_1_kernel,
      'conv1_1_b': conv1_1_biases,
      'conv1_2_W': conv1_2_kernel,
      'conv1_2_b': conv1_2_biases,
      'conv2_1_W': conv2_1_kernel,
      'conv2_1_b': conv2_1_biases,
      'conv2_2_W': conv2_2_kernel,
      'conv2_2_b': conv2_2_biases,
      'conv3_1_W': conv3_1_kernel,
      'conv3_1_b': conv3_1_biases,
      'conv3_2_W': conv3_2_kernel,
      'conv3_2_b': conv3_2_biases,
      'conv3_3_W': conv3_3_kernel,
      'conv3_3_b': conv3_3_biases,
      'conv4_1_W': conv4_1_kernel,
      'conv4_1_b': conv4_1_biases,
      'conv4_2_W': conv4_2_kernel,
      'conv4_2_b': conv4_2_biases,
      'conv4_3_W': conv4_3_kernel,
      'conv4_3_b': conv4_3_biases,
      'conv5_1_W': conv5_1_kernel,
      'conv5_1_b': conv5_1_biases,
      'conv5_2_W': conv5_2_kernel,
      'conv5_2_b': conv5_2_biases,
      'conv5_3_W': conv5_3_kernel,
      'conv5_3_b': conv5_3_biases,
      
      'fc6_W': fc1w,
      'fc6_b': fc1b,
      'fc7_W': fc2w,
      'fc7_b': fc2b,
      'fc8_W': fc3w,
      'fc8_b': fc3b
      }
    new_para = self.para_file[:-4]+'_new'+'.npy'
    np.save(new_para, para)
    print('Model saved.')