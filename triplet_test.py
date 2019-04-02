#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 23 15:02:15 2018

@author: dayea
"""



#from pylab import *
import numpy as np
#from matplotlib import pyplot as plt
#from scipy.ndimage import filters
#import Queue
#from PIL import Image  #
import tensorflow as tf
import triplet_utility as trip
from triplet_utility import triplet_net





#%%
MODE = 'test'
#MODE = 'training'
#MODE = 'training_classifier'
#MODE = 'generate_database'
para_file = 'saved_file/para.npy'
test_file = 'saved_file/test_result.npy'
lamda = 0.1
LEARN_RATE = 0.00001
alpha = 100
_WEIGHT_DECAY = 0.0002
#%%

def build_train(net):
  
  y0, out0, lab0 = net.anchor_model()
  y1, out1, lab1 = net.positive_model()
  y2, out2, lab2 = net.negative_model()


#triplet loss
  
  pos_dist = tf.reduce_sum(tf.square(tf.subtract(y0, y1)), [1,2])
  neg_dist = tf.reduce_sum(tf.square(tf.subtract(y0, y2)), [1,2])
  neg_dist1 = tf.multiply(neg_dist, lamda)
#  neg_dist = tf.minimum(neg_dist, -10000)
  basic_loss = tf.add(tf.subtract(pos_dist, neg_dist1), alpha)
  
  basic_loss = tf.maximum(basic_loss, 0.0)
  
#classify loss
  
  out = tf.concat([out0, out1, out2], 0)
  lab = tf.concat([lab0, lab1, lab2], 0)
  
  class_loss = tf.nn.softmax_cross_entropy_with_logits(logits=out, labels=tf.one_hot(lab, 2))
  
  #compute softmax softmax_cross_entropy_with_logits
  probs = tf.nn.softmax(out)
  prediction = tf.argmax(probs, 1)
  correct_prediction = tf.equal(prediction, lab)
  acc = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
  
  
  
  loss =  class_loss + 0.01*basic_loss + _WEIGHT_DECAY * tf.add_n([tf.nn.l2_loss(v) for v in tf.trainable_variables()])
#  loss = basic_loss
  opt = tf.train.GradientDescentOptimizer(LEARN_RATE).minimize(loss) 
  
  return opt, loss, y0, y1, y2, pos_dist, neg_dist, basic_loss, class_loss, acc


def build_classifier_train(net):
  y, out, lab = net.classifier_model()

  classifier_loss = tf.nn.softmax_cross_entropy_with_logits(logits=out, labels=tf.one_hot(lab, 2))
  loss = classifier_loss + _WEIGHT_DECAY * tf.add_n([tf.nn.l2_loss(v) for v in tf.trainable_variables()])
  opt = tf.train.GradientDescentOptimizer(LEARN_RATE).minimize(loss)
  
  #compute softmax softmax_cross_entropy_with_logits
  probs = tf.nn.softmax(out)
  prediction = tf.argmax(probs, 1)
  correct_prediction = tf.equal(prediction, lab)
  acc = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
  
  return opt, loss, acc, y



#%%
if __name__=="__main__":
  
  net = triplet_net(para_file)

  
  
  if MODE == 'training':
    opt, loss, y0, y1, y2, pos_dist, neg_dist, basic_loss, class_loss, train_acc= build_train(net)
    init = tf.global_variables_initializer()
    loss_array = []
    with tf.Session() as sess:
      sess.run(init)
      
      total = 10000
      for i in range(total):
        
        image, label= trip.generate_group_touple(batch_size=20)
        feed_dict = net.feed_train(image, label)
        
#        x0_np, lab0_np= sess.run([net.x0, net.lab0], feed_dict = feed_dict)
        
        
        
        
        _, loss_np, _, _, _, pos_dist_np, neg_dist_np, basic_loss_np, class_loss_np, train_acc_np = sess.run([opt, loss, y0, y1, y2, pos_dist, neg_dist, basic_loss, class_loss, train_acc], feed_dict = feed_dict) 
        loss_dis = np.mean(np.mean(loss_np))
        bas_loss_dis = np.mean(np.mean(basic_loss_np))
#        if i == np.floor(total*0.7):
#          loss_dis = np.mean(loss_np)
#          print('step: %d loss: %.3f     pos_loss: %.3f     neg_loss: %.3f      '%(i,
#                                                                                  loss_dis, 
#                                                                                  np.mean(pos_dist_np),
#                                                                                  np.mean(neg_dist_np)
#                                                                                  ))
          
        if i % (total / 100) == 0:
          loss_dis = np.mean(loss_np)
          loss_array.append(loss_dis)
          print('step: %d  loss: %.5f  bas_loss: %.5f  pos_loss: %.5f   neg_loss: %.5f   class_loss: %.5f  acc: %.5f  '%(i,
                                                                                  loss_dis, 
                                                                                  bas_loss_dis,
                                                                                  np.mean(pos_dist_np),
                                                                                  np.mean(neg_dist_np),
                                                                                  np.mean(class_loss_np),
                                                                                  np.mean(train_acc_np)
                                                                                  ))
          
    #      if i == 0 or i ==total-1:
    #        print(y0_np, y1_np, y2_np)
      
      
      
      trip.save_model(para_file, net, sess)
      
    sess.close()

  elif MODE == 'training_classifier' :
    opt, classifier_loss, acc, _ = build_classifier_train(net)
    init = tf.global_variables_initializer()
    loss_array = []
    
    with tf.Session() as sess:
      sess.run(init)
      
      total = 10000
      for i in range(total):
        
        data, lab = trip.generate_train_classifier_touple(batch_size=100)
        feed_dict = net.feed_train_classifier(data, lab)
        
        _, classifier_loss_np, acc_np = sess.run([opt, classifier_loss, acc], feed_dict = feed_dict) 
        if i % (total / 100) == 0:
          print('step: %d  loss: %.5f  acc: %.5f  '%(i,
                                                   np.mean(classifier_loss_np), 
                                                   np.mean(acc_np)
                                                                                  ))
        
      trip.save_model(para_file, net, sess)
    sess.close()
#    trip.save_test(test_file, classifier_loss_np)
#    show_test = trip.test_show(test_file)
#    show_test.show_diff()

  elif MODE == 'test' :
    _, classifier_loss, acc, y = build_classifier_train(net)
    init = tf.global_variables_initializer()
    loss_array = []
    
    with tf.Session() as sess:
      sess.run(init)
      
      total = 1
      for i in range(total):
        
        data, lab = trip.generate_test_touple(batch_size=100)
        feed_dict = net.feed_train_classifier(data, lab)
        
        classifier_loss_np, acc_np, y_np = sess.run([classifier_loss, acc, y], feed_dict = feed_dict) 
        print('step: %d  loss: %.5f  acc: %.5f  '%(i,
                                                   np.mean(classifier_loss_np), 
                                                   np.mean(acc_np)
                                                                                  ))
        trip.save_test(test_file, y_np)
      
    sess.close()
    
    show_test = trip.test_show(test_file)
    show_test.show_ori()
#    show_test.show_diff()









