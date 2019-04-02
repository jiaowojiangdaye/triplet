#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 23 19:40:15 2018

@author: dayea
"""


#from pylab import *
import os
import tensorflow as tf
import numpy as np
#from matplotlib import pyplot as plt
#from scipy.ndimage import filters
#import Queue
#from PIL import Image  #


#%%
def draw_point(plant):
  
  cod = np.random.randint(2,6,[2])
  plant[cod[0],cod[1]] = 1
    
  return plant

def draw_line(plant):
  
  cod = np.random.randint(2,6,[2])
  for j in range(cod[1]-2, cod[1]+2+1):
    plant[cod[0],j] = 1
    
  return plant

def draw_square(plant):
  
  cod = np.random.randint(2,6,[2])
  for i in range(cod[0]-2, cod[0]+2+1):
    for j in range(cod[1]-2, cod[1]+2+1):
      plant[i,j] = 1
  
  
  return plant

def draw_triangle(plant):
  
  
  cod = np.random.randint(2,6,[2])
  for i in range(cod[0]-2, cod[0]+2+1):
    for j in range(cod[1]-2, cod[1]+2+1):
      plant[i,j] = 1

  plant[cod[0]-2,cod[1]-2] = 0
  plant[cod[0]-2,cod[1]-1] = 0
  plant[cod[0]-1,cod[1]-2] = 0
  plant[cod[0]-1,cod[1]-1] = 0
  plant[cod[0],  cod[1]-2] = 0
  plant[cod[0]+1,cod[1]-2] = 0
  

  plant[cod[0]-2,cod[1]+2] = 0
  plant[cod[0]-2,cod[1]+1] = 0
  plant[cod[0]-1,cod[1]+2] = 0
  plant[cod[0]-1,cod[1]+1] = 0
  plant[cod[0],  cod[1]+2] = 0
  plant[cod[0]+1,cod[1]+2] = 0
  
  return plant
  

def generate_one_touple_triangle():
  data = np.zeros([3,8,8])
  
  ancher = np.random.randint(2)
 
  if ancher == 0:
    data[0] = draw_square(data[0])
    data[1] = draw_square(data[1])
    data[2] = draw_triangle(data[2])
  else:
    data[0] = draw_triangle(data[0])
    data[1] = draw_triangle(data[1])
    data[2] = draw_square(data[2])
      
  return data

 
def generate_one_touple_line():
  
  data = np.zeros([3,8,8])
  label = np.zeros([3])
  ancher = np.random.randint(2)
 
  if ancher == 0:
    data[0] = draw_square(data[0])
    data[1] = draw_square(data[1])
    data[2] = draw_line(data[2])
    label[0] = 0
    label[1] = 0
    label[2] = 1
  else:
    data[0] = draw_line(data[0])
    data[1] = draw_line(data[1])
    data[2] = draw_square(data[2])
    label[0] = 1
    label[1] = 1
    label[2] = 0
      
  return data, label

def generate_one_touple_point():
  
  data = np.zeros([3,8,8])
  label = np.zeros([3])
  ancher = np.random.randint(2)
 
  if ancher == 0:
    data[0] = draw_square(data[0])
    data[1] = draw_square(data[1])
    data[2] = draw_point(data[2])
    label[0] = 0
    label[1] = 0
    label[2] = 1
  else:
    data[0] = draw_point(data[0])
    data[1] = draw_point(data[1])
    data[2] = draw_square(data[2])
    label[0] = 1
    label[1] = 1
    label[2] = 0
      
  return data, label

def generate_group_touple(batch_size=1):
  
  image = np.zeros([3,batch_size,8,8], dtype=np.float)
  label = np.zeros([3,batch_size], dtype=np.int64)
  for i in range(batch_size):
    image[:,i,:,:], label[:,i] = generate_one_touple_point()
    
  image = image.reshape(3,batch_size,8,8,1)
  label = label.reshape(3,batch_size)
  return image, label


def generate_train_classifier_touple(batch_size=100):
  
  data = np.zeros([batch_size,8,8])
  label = np.zeros([batch_size], dtype=np.int64)
  
  for i in range(batch_size):
    
    class_type = np.random.randint(2)
    data[i,:,:] = draw_square(data[i,:,:]) if class_type == 0 else draw_point(data[i,:,:])
    label[i] = class_type

  data = data.reshape(batch_size,8,8,1)
  label = label.reshape(batch_size)
  return data, label

def generate_test_touple(batch_size=100):
  
  data = np.zeros([batch_size,8,8])
  label = np.zeros([batch_size], dtype=np.int64)
  class1_num = batch_size // 2
  
  for i in range(class1_num):
    data[i,:,:] = draw_square(data[i,:,:])  # maybe the return parameter is unnessisary, bacause the 
    label[i] = 0
  for i in range(class1_num, batch_size):
    data[i,:,:] = draw_point(data[i,:,:])  # maybe the return parameter is unnessisary, bacause the 
    label[i] = 1
  data = data.reshape(batch_size,8,8,1)
  
 
  return data, label


#%%
  
def save_model(para_file, net, sess):
  print('Saving model...')
  conv1W_np, conv1b_np = sess.run([net.conv1W, net.conv1b])
  conv2W_np, conv2b_np = sess.run([net.conv2W, net.conv2b])
  fc1W_np, fc1b_np = sess.run([net.fc1W, net.fc1b])


  para = {
    'conv1W': conv1W_np,
    'conv1b': conv1b_np,
    'conv2W': conv2W_np,
    'conv2b': conv2b_np,
    'fc1W': fc1W_np,
    'fc1b': fc1b_np
    
    }
  np.save(para_file, para)
  print('Model saved.')

def save_test(test_file, data):
  np.save(test_file, data)


class test_show():
  
  def __init__(self, file):
    
    ori = np.load(file, "r")
    [num, w, h, _] = np.shape(ori)
    
    self.data = ori.reshape([num, w, h])
    self.diff = np.zeros([num, w, h])
    self.get_diff()
    
  
  
  def show_img(self, img):
    plt.figure()
    for i in range(0, 50):
      plt.imshow(self.img[i,:,:])
      
    for i in range(50, 100):
      plt.imshow(img[i,:,:])
    
    
    
  def get_diff(self):
    
  
    for i in range(50):
      self.diff[i] = self.data[i]-self.data[i+50]
  
    for i in range(25):
      self.diff[i+50] = self.data[i]-self.data[i+25]
      
    for i in range(25):
      self.diff[i+75] = self.data[i+50]-self.data[i+75]
    

    for i in range(0, 100):
      plt.figure()
      plt.imshow(self.diff[i,:,:])
    
    
  def show_ori(self):
    self.show_img(self, self.data)
    
  def show_diff(self):
    self.show_img(self, self.diff)
  
  

    
    

    


  #%%

class triplet_net():
  
  def __init__(self, parameter_file):
    if os.path.isfile(parameter_file):
      print('Restoring model...')
      self.para = np.load(open(parameter_file, "rb")).item()
      
      self.transfer_flag = True
    else:
      self.transfer_flag = False
    self.init_Wb()
    
  def init_Wb(self):
    self.x0 = tf.placeholder(shape=[None,8,8,1], dtype=tf.float32)
    self.x1 = tf.placeholder(shape=[None,8,8,1], dtype=tf.float32)
    self.x2 = tf.placeholder(shape=[None,8,8,1], dtype=tf.float32)
    self.lab0 = tf.placeholder(shape=[None,], dtype=tf.int64)
    self.lab1 = tf.placeholder(shape=[None,], dtype=tf.int64)
    self.lab2 = tf.placeholder(shape=[None,], dtype=tf.int64)
    
    self.train_x = tf.placeholder(shape=[None,8,8,1], dtype=tf.float32)
    self.train_lab = tf.placeholder(shape=[None,], dtype=tf.int64)
    
    if self.transfer_flag:
      self.conv1W = tf.Variable(self.para['conv1W'], dtype=tf.float32)
      self.conv1b = tf.Variable(self.para['conv1b'], dtype=tf.float32)
      self.conv2W = tf.Variable(self.para['conv2W'], dtype=tf.float32)
      self.conv2b = tf.Variable(self.para['conv2b'], dtype=tf.float32)
      self.fc1W = tf.Variable(self.para['fc1W'], dtype=tf.float32)
      self.fc1b = tf.Variable(self.para['fc1b'], dtype=tf.float32)
    else:
      self.conv1W = tf.Variable(tf.random_normal([3,3,1,10], dtype=tf.float32))
      self.conv1b = tf.Variable(tf.random_normal([10,], dtype=tf.float32))
      self.conv2W = tf.Variable(tf.random_normal([3,3,10,1], dtype=tf.float32))
      self.conv2b = tf.Variable(tf.random_normal([1,], dtype=tf.float32))
      self.fc1W = tf.Variable(tf.random_normal([16,2], dtype=tf.float32))
      self.fc1b = tf.Variable(tf.random_normal([2], dtype=tf.float32))
      
    
  def conv(self, input, kernel, biases, k_h, k_w, c_o, s_h, s_w,  padding='VALID', group=1):
    '''From https://github.com/ethereon/caffe-tensorflow
    '''
    c_i = input.get_shape()[-1]
    assert c_i%group==0
    assert c_o%group==0
    convolve = lambda i, k: tf.nn.conv2d(i, k, [1, s_h, s_w, 1], padding=padding)
    
    if group==1:
        conv = convolve(input, kernel)
    else:
        input_groups =  tf.split(input, group, 3)   #tf.split(3, group, input)
        kernel_groups = tf.split(kernel, group, 3)  #tf.split(3, group, kernel) 
        output_groups = [convolve(i, k) for i,k in zip(input_groups, kernel_groups)]
        conv = tf.concat(output_groups, 3)          #tf.concat(3, output_groups)
        
    return  tf.reshape(tf.nn.bias_add(conv, biases), [-1]+conv.get_shape().as_list()[1:])
    
  def feed_train(self, iamge, label):
    feed_dict = {self.x0: iamge[0,:,:,:,:], 
                 self.x1: iamge[1,:,:,:,:], 
                 self.x2: iamge[2,:,:,:,:],
                 self.lab0: label[0,:],
                 self.lab1: label[1,:],
                 self.lab2: label[2,:]}
    
    return feed_dict
  
  def feed_train_classifier(self, iamge, label):
    feed_dict = {self.train_x: iamge, 
                 self.train_lab: label}
    
    return feed_dict
  
  def net_model(self, xx):
    k_h = 3; k_w = 3; c_o = 1; s_h = 1; s_w = 1
    conv1 = self.conv(xx, self.conv1W, self.conv1b, k_h, k_w, c_o, s_h, s_w, padding="SAME", group=1)
    conv1 = tf.nn.relu(conv1)
#    y2 = tf.nn.max_pool(x2_conv1, ksize=[1, 4, 4, 1], strides=[1, 4, 4, 1], padding='SAME')
    pool1 = tf.nn.max_pool(conv1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
    
    conv2 = self.conv(pool1, self.conv2W, self.conv2b, k_h, k_w, c_o, s_h, s_w, padding="SAME", group=1)
    conv2 = tf.nn.relu(conv2)
    pool2 = tf.nn.max_pool(conv2, ksize=[1, 2, 2, 1], strides=[1, 1, 1, 1], padding='SAME')
    
    shape = int(np.prod(pool2.get_shape()[1:]))
    pool5_flat = tf.reshape(pool2, [-1, shape])
    fc1l = tf.nn.bias_add(tf.matmul(pool5_flat, self.fc1W), self.fc1b)
    fc1 = tf.nn.relu(fc1l)
    
    
    return pool2, fc1


  def anchor_model(self):
    y, out = self.net_model(self.x0)
    return y, out, self.lab0

  def positive_model(self):
    y, out = self.net_model(self.x1)
    return y, out, self.lab1
  
  def negative_model(self):
    y, out = self.net_model(self.x2)
    return y, out, self.lab2
  
  def classifier_model(self):
    y, out = self.net_model(self.train_x)
    return y, out, self.train_lab
    

  
  def feed_test(self, data):
    feed_dict = {self.test_x: data}
  
    return feed_dict
