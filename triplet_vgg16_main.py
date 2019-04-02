#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 23 15:02:15 2018

@author: dayea
"""

        

#from pylab import *
import numpy as np
import os
from vgg_model import vgg16
import triplet_vgg16_net as trinet
from matplotlib import pyplot as plt
#from scipy.ndimage import filters
#import Queue
#from PIL import Image  #
import tensorflow as tf




#%%
def task_train_classifier(tf_config, args, net, iamges, labels, total_step):
  
  opt, classifier_loss, loss, acc, probs, correct, pool5 = trinet.interface_train_classifier(args, net, iamges, labels)
  init = tf.global_variables_initializer()
  with tf.Session(config=tf_config) as sess:
    with tf.device("/gpu:1"):
      sess.run(init)
      coord = tf.train.Coordinator()
      threads = tf.train.start_queue_runners(coord=coord, sess=sess)
      
      
      for i in range(total_step):
        
        
        data_queue.feed_data_to_queue(sess)
  
  #        images_np, labels_np = sess.run([iamges, labels])
  #        print(str(i), labels_np)
  #        
  #        
  #        figure = plt.figure()
  #        img = images_np[0,:,:,:].reshape(224,224,3)
  #        img = np.uint8(img)
  #        plt.imshow(img)
  #        plt.show()
        
        
        _, classifier_loss_np, loss_np, acc_np, correct_np, probs_np, labels_np, pool5_np = sess.run([opt, classifier_loss, loss, acc, correct, probs, labels, pool5]) 
        
        if i%(total_step/100) == 0:
          print('step: %d   classifier_loss: %.5f   loss: %.5f  acc: %.5f  '%(i, classifier_loss_np, loss_np, acc_np))
          
      net.save_model(sess=sess)
      coord.request_stop()#queue need be turned off, otherwise it will report errors
      coord.join(threads)
  sess.close()
  
  
  
  
def task_train_triplet(tf_config, args, net, iamges, labels, total_step):
  
  opt, pos_dist, neg_dist, loss, feature, embeddings, neg_dist1, triplet_loss = trinet.interface_train_triplet(args, net, iamges, labels)
  init = tf.global_variables_initializer()
  with tf.Session(config=tf_config) as sess:
    sess.run(init)
    coord = tf.train.Coordinator()
#    enqueue_threads = qr.create_threads(sess, coord=coord, start=True)  
    threads = tf.train.start_queue_runners(coord=coord, sess=sess)
    
    
    for i in range(total_step):
      
      data_queue.feed_data_to_queue(sess)

#        images_np, labels_np = sess.run([iamges, labels])
#        labels_np1 = [[labels_np[i:i+3]] for i in np.arange(0,len(labels_np)-2, 3)]
#        print(str(i), labels_np)
      
      _, pos_dist_np, neg_dist_np, loss_np, feature_np, embeddings_np, neg_dist1_np, triplet_loss_np = sess.run([opt, pos_dist, neg_dist, loss, feature, embeddings, neg_dist1, triplet_loss]) 
#      if i%(total_step/100) == 0:
      print('step: %d   pos_dist: %.5f  neg_dist: %.5f  loss: %.5f  '%(i, np.mean(pos_dist_np), np.mean(neg_dist_np), np.mean(loss_np)))
    
    acc_arr.append(loss_np)
    net.save_model(sess=sess)
    
    coord.request_stop()#queue need be turned off, otherwise it will report errors
    coord.join(threads)
  sess.close()
  
  
  
def task_train_classifier_triplet(tf_config, args, net, iamges, labels, total_step):
  acc_arr = []
  opt, pos_dist, neg_dist, classifier_loss, loss, acc, probs= trinet.interface_train_triplet_classifier(args, net, iamges, labels)
  init = tf.global_variables_initializer()
  with tf.Session(config=tf_config) as sess:
    sess.run(init)
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(coord=coord, sess=sess)

    for i in range(total_step):
      
      
      data_queue.feed_data_to_queue(sess)

#        images_np, labels_np = sess.run([iamges, labels])
#        print(str(i), labels_np)
      
      _, pos_dist_np, neg_dist_np, classifier_loss_np, loss_np, acc_np = sess.run([opt, pos_dist, neg_dist, classifier_loss, loss, acc]) 
      
      
#      if i%(total_step/1) == 0:
      print('step: %d   pos_dist: %.5f  neg_dist: %.5f classifier_loss: %.5f  loss:%.5f  acc: %.5f  '%(i, 
                                                                                            np.mean(pos_dist_np), 
                                                                                            np.mean(neg_dist_np), 
                                                                                            classifier_loss_np, 
                                                                                            loss_np, 
                                                                                            acc_np))
      acc_arr.append(acc_np)
      
    print('Finish triplet classifier training! accuracy=', np.mean(acc_arr))
    net.save_model(sess=sess)
    coord.request_stop()#queue need be turned off, otherwise it will report errors
    coord.join(threads)
  sess.close()
  
  
  
def task_eval_classifier(tf_config, args, net, iamges, labels, total_step_eval):
  acc_arr = []
  
  loss, acc, probs, pos_dist, neg_dist= trinet.interface_eval_classifier(args, net, iamges, labels)
  init = tf.global_variables_initializer()
  with tf.Session(config=tf_config) as sess:
    sess.run(init)
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(coord=coord, sess=sess)
    
    for i in range(total_step_eval):
      
      
      data_queue.feed_data_to_queue(sess)

#        images_np, labels_np = sess.run([iamges, labels])
#        labels_np1 = [[labels_np[i:i+3]] for i in np.arange(0,len(labels_np)-2, 3)]
#        print(str(i), labels_np)
      
      loss_np, acc_np, probs_np, pos_dist_np, neg_dist_np = sess.run([loss, acc, probs,  pos_dist, neg_dist]) 
      acc_arr.append(acc_np)
      print('step: %d  loss: %.5f  acc: %.5f pos_dist: %.5f neg: %.5f '%(i, 
                                                                         loss_np, 
                                                                         acc_np, 
                                                                         np.mean(pos_dist_np), 
                                                                         np.mean(neg_dist_np)))
    
    print('Finish evaluating! accuracy=', np.mean(acc_arr))
    coord.request_stop()#queue need be turned off, otherwise it will report errors
    coord.join(threads)
  sess.close()


def build_retrival_dataset(tf_config, args, net, iamges, labels, total_step_eval):
  
  acc_arr = []
  
  loss, acc, probs, pos_dist, neg_dist= trinet.interface_eval_classifier(args, net, iamges, labels)
  init = tf.global_variables_initializer()
  with tf.Session(config=tf_config) as sess:
    sess.run(init)
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(coord=coord, sess=sess)
    
    for i in range(total_step_eval):
      
      
      data_queue.feed_data_to_queue(sess)

#        images_np, labels_np = sess.run([iamges, labels])
#        labels_np1 = [[labels_np[i:i+3]] for i in np.arange(0,len(labels_np)-2, 3)]
#        print(str(i), labels_np)
      
      loss_np, acc_np, probs_np, pos_dist_np, neg_dist_np = sess.run([loss, acc, probs,  pos_dist, neg_dist]) 
      acc_arr.append(acc_np)
      print('step: %d  loss: %.5f  acc: %.5f pos_dist: %.5f neg: %.5f '%(i, 
                                                                         loss_np, 
                                                                         acc_np, 
                                                                         np.mean(pos_dist_np), 
                                                                         np.mean(neg_dist_np)))
    
    print('Finish evaluating! accuracy=', np.mean(acc_arr))
    coord.request_stop()#queue need be turned off, otherwise it will report errors
    coord.join(threads)
  sess.close()


def task_generate_dataset(tf_config, args, net, total_step_eval):
    
  data_queue = trinet.data_queue_v2(args, mode='train')
  images = data_queue.create_data_queue()
  
  features = trinet.interface_generate_feat(args, net, images)
  init = tf.global_variables_initializer()
  with tf.Session(config=tf_config) as sess:
    sess.run(init)
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(coord=coord, sess=sess)
    i = 0
    path_list = []
    feature_list = []
    for i in range(total_step_eval):
      
      
      image_paths, finish = data_queue.feed_data_to_queue(sess)
      if finish:
        break
#        images_np, labels_np = sess.run([iamges, labels])
#        labels_np1 = [[labels_np[i:i+3]] for i in np.arange(0,len(labels_np)-2, 3)]
#        print(str(i), labels_np)
      
      features_np = sess.run([features])
      path_list = path_list + image_paths.tolist()
      feature_list = feature_list + [one.tolist() for one in features_np[0]]

#      path_list.
#      figure = plt.figure()
#      img = images_np[0,:,:,:].reshape(224,224,3)
#      img = np.uint8(img)
#      plt.imshow(img)
#      plt.show()
      
      print(i-1)
#      acc_arr.append(acc_np)
#      print('step: %d  loss: %.5f  acc: %.5f pos_dist: %.5f neg: %.5f '%(i, 
#                                                                         loss_np, 
#                                                                         acc_np, 
#                                                                         np.mean(pos_dist_np), 
#                                                                         np.mean(neg_dist_np)))
    np.save('path_list.npy', np.array(path_list))
    np.save('feature_list.npy', np.array(feature_list))
    print('Finish generating dataset! batches=', i)
    coord.request_stop()#queue need be turned off, otherwise it will report errors
    coord.join(threads)
  sess.close()



def task_retrival(tf_config, args, net, total_step_eval):
  
#  image = tf.placeholder()
  data_queue = trinet.data_queue_v2(args, mode='train')
  images = data_queue.create_data_queue()
  retrivaler = trinet.retrivaler('retrival_results', 'path_list.npy', 'feature_list.npy')
  
  features = trinet.interface_generate_feat(args, net, images)
  init = tf.global_variables_initializer()
  with tf.Session(config=tf_config) as sess:
    sess.run(init)
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(coord=coord, sess=sess)
    i = 0
    results = []
    for i in range(total_step_eval):
      
      
      image_paths, finish = data_queue.feed_data_to_queue(sess)
      if finish:
        break
#        images_np, labels_np = sess.run([iamges, labels])
#        labels_np1 = [[labels_np[i:i+3]] for i in np.arange(0,len(labels_np)-2, 3)]
#        print(str(i), labels_np)
      
      
      features_np = sess.run([features])
      if i % 10==0:
        result = retrivaler.retrival_once(str(i), features_np[0][0], image_paths.tolist()[0])
        results.append(result)
      print(i-1)
      

    trinet.mergeReport2('all_result.png', results)
    print('Finish retrival! batches=', i)
    coord.request_stop()#queue need be turned off, otherwise it will report errors
    coord.join(threads)
  sess.close()








#%%main
if __name__=="__main__":
  
  
#  MODE = 'test'
#  MODE = 'eval'
#  MODE = 'classifier'
  MODE = 'triplet_classifier'
#  MODE = 'triplet'
#  MODE = 'generate_database'
  
#  MODE = 'retrival'
  
#  os.environ["CUDA_VISIBLE_DEVICES"] = "2,3"
  tf_config = tf.ConfigProto()
#  tf_config.gpu_options.per_process_gpu_memory_fraction = 0.5 # 分配50%  
  
  
  args = trinet.get_args()
  total_step = 30000
  total_step_eval = 500
  args.LEARN_RATE = 0.00001
  args.WEIGHT_DECAY = 0.00005
  args.lamda = 0.1
  args.alpha = 10
  args.beta = 5
  transfer_flag=False
  continue_flag=True
  if MODE == 'generate_database' or MODE == 'retrival':
      continue_flag=True

  net = vgg16(transfer_para=args.offical_vgg_para, 
              trained_para=args.trained_para_file, 
              class_num=args.nrof_classes, 
              transfer_flag=transfer_flag, 
              continue_flag=continue_flag)
  
  if MODE != 'generate_database':
    data_queue = trinet.data_queue(args, mode='train' if MODE != 'eval' else 'eval')
    iamges, labels, qr = data_queue.create_data_queue()
  
  acc_arr = []
  if MODE == 'triplet_classifier':
    task_train_classifier_triplet(tf_config, args, net, iamges, labels, total_step)

  elif MODE == 'classifier' :
    task_train_classifier(tf_config, args, net, iamges, labels, total_step)

  elif MODE == 'triplet' :
    task_train_triplet(tf_config, args, net, iamges, labels, total_step)
    
  elif MODE == 'eval' :
    task_eval_classifier(tf_config, args, net, iamges, labels, total_step_eval)
  elif MODE == 'generate_database':
    task_generate_dataset(tf_config, args, net, total_step_eval)
    
  elif MODE == 'retrival':
    task_retrival(tf_config, args, net, total_step_eval)
    
    








