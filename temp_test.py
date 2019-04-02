#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 30 21:54:46 2018

@author: dayea
"""

import tensorflow as tf
import numpy as np
import argparse
from matplotlib import pyplot as plt
#def record_dataset(filenames):
#  """Returns an input pipeline Dataset from `filenames`."""
#  record_bytes = 227 * 227 * 3 + 1
#  return tf.data.FixedLengthRecordDataset(filenames, record_bytes)

#%%
# =============================================================================
# 
# def functiona(a):
#     print("functiona  a = ", a)
#     
#     def functionb(b):
#         print("functiona  a*b = ", a*b)
#         return a*b
# 
#     return functionb
#         
#         
#         
# 
# c = functiona(a = 12)
# print c(3)
# 
# 
# =============================================================================

#%%
#dataset = record_dataset(['data/tianchi_data/TFrTrainPart1_collar_design_labels.tfrecords'])
#
#
## When choosing shuffle buffer sizes, larger sizes result in better
## randomness, while smaller sizes have better performance. Because CIFAR-10
## is a relatively small dataset, we choose to shuffle the full epoch.
#dataset = dataset.shuffle(buffer_size=_NUM_IMAGES['train'])
#
#dataset = dataset.map(parse_record)
#dataset = dataset.map(
#        lambda image, label: (preprocess_image(image, is_training), label))
#
#dataset = dataset.prefetch(2 * batch_size)
#
#  # We call repeat after shuffling, rather than before, to prevent separate
#  # epochs from blending together.
#dataset = dataset.repeat(num_epochs)
#
#  # Batch results by up to batch_size, and then fetch the tuple from the
#  # iterator.
#dataset = dataset.batch(batch_size)
#iterator = dataset.make_one_shot_iterator()
#images, labels = iterator.get_next()

#%%
#dataset = tf.data.Dataset.from_tensor_slices(np.array([1.0, 2.0, 3.0, 4.0,5.0]))
#dataset = tf.data.Dataset.from_tensor_slices(
#        {
#                "a": np.array([1.0, 2.0, 3.0, 4.0,5.0]),
#                "b": np.random.uniform(size=(5,2))
#        }
#        )
#
#
#iterator = dataset.make_one_shot_iterator()
#one_element = iterator.get_next()
#
#with tf.Session() as sess:
#    for i in range(5):
#        print(sess.run(one_element))
  
#%%

#
#net_data = load(open("bvlc_alexnet.npy", "rb")).item()
#
#dict_a = {
#    'a': np.zeros([1,4]),
#    'b': np.ones([2,2])
#    }
#
# 
#f = open('temp_file/temp.txt','w')  
#f.write(str(dict_a))  
#f.close()  
#
#
#
#
##读取  
#f = open('temp_file/temp.txt','r')  
#a = f.read()  
#dict_name = eval(a)  
#f.close()



#%%
#parser = argparse.ArgumentParser()
#
#parser.add_argument(
#    '--data_dir', type=dict, default={'a': 'dsds', 
#                                      'b': 'hfjdskj'},
#    help='The directory where the ImageNet input data is stored.')
#

#%%
#x = tf.placeholder(tf.float32, shape=[None, 1])  
#y = 4 * x + 4  
#  
#w = tf.Variable(tf.random_normal([1], -1, 1))  
#b = tf.Variable(tf.zeros([1]))  
#y_predict = w * x + b  
#  
#loss = tf.reduce_mean(tf.square(y - y_predict))  
#optimizer = tf.train.GradientDescentOptimizer(0.5)  
#train = optimizer.minimize(loss)  
#  
#isTrain = False  
#train_steps = 100  
#
#checkpoint_steps = 50  
#checkpoint_dir = 'temp_file/'  
#
#saver = tf.train.Saver()  # defaults to saving all variables - in this case w and b  
#x_data = np.reshape(np.random.rand(10).astype(np.float32), (10, 1))  
#  
#with tf.Session() as sess:  
#    sess.run(tf.initialize_all_variables())  
#    if isTrain:  
#        for i in range(train_steps):  
#            sess.run(train, feed_dict={x: x_data})  
#            if (i + 1) % checkpoint_steps == 0:  
#                saver.save(sess, checkpoint_dir + 'model.ckpt', global_step=i+1)  
#    else:  
#        ckpt = tf.train.get_checkpoint_state(checkpoint_dir)  
#        if ckpt and ckpt.model_checkpoint_path:  
#            saver.restore(sess, ckpt.model_checkpoint_path)  
#        else:  
#            pass  
#        print(sess.run(w))  
#        print(sess.run(b))  

#%%

#a = np.load('temp_file/conv1W.npy')
#
#b = np.load('temp_file/conv1W_27.npy')


#weights = np.load('vgg16_weights.npz')
#keys = sorted(weights.keys())
#for i, k in enumerate(keys):
#  print(i, k, np.shape(weights[k]))
#file=open('a.txt','w') 
#file.write(str(a)) 
#file.write(str(c)) 
#file.close()
#
#file=open('a.txt','r') 
#b = file.read()
#file.close()



#a = np.ones([2,3,3])
#print(a.shape())
#print(a.reshape([2,2]))

#%%
#dict_a = {
#    'a': np.zeros([1,4]),
#    'b': np.ones([2,2])
#    }
#
#np.save('temp_file/temp.npy', dict_a)
#
#b = np.load(open("temp_file/temp.npy", "rb")).item()


#%%
#b = np.load(open("saved_file/test_result.npy", "rb")).item()
#b = np.load("saved_file/test_result.npy", "r")
#
#c = b.reshape([100,4,4])
#
#for i in range(50, 60):
#  plt.figure()
#  plt.imshow(c[i,:,:])
#  
#for i in range(0, 10):
#  plt.figure()
#  plt.imshow(c[i,:,:])

#
#def f1(a):
#  a[0] = 1
#  
#b = [2,3]
#f1(b)
#
#print(b)
#
#a = 1 if False else 3
#
#


#%%

#x = tf.placeholder(dtype=tf.float32, shape=[20, 20,3,  5])
#
#
#
#y = tf.unstack(tf.reshape(x, [-1,1,5]), 1, 1)
#
#
#
#init = tf.global_variables_initializer()
#
#
#
#with tf.Session() as sess:
#
#  sess.run(init)
#  
#  feed_dict = {x: y}
#
#  xx = sess.run([x], feed_dict = feed_dict)


#%%
#x = tf.placeholder(dtype=tf.float32, shape=[1, 2, 2, 3])
#
#z = tf.reduce_max(x, reduction_indices=[1,2])
#
#y = [[[[1, 2, 3], [4, 3, 2]], [[5, 6, 7], [8, 7, 6]]]]
#
#
#
#init = tf.global_variables_initializer()
#
#
#
#with tf.Session() as sess:
#
#  sess.run(init)
#  
#  feed_dict = {x: y}
#
#  xx = sess.run([z], feed_dict = feed_dict)

#%%
#
#a = np.arange(10)
#
#b= np.arange(10,20)
#c = []
#for i in range(10):
#  c.append((a[i], b[i]))
#  
#  
#np.random.shuffle(c)
#a = [triplets2[i][0][:] for i in range(6)]

#%%

from tensorflow.python.ops import data_flow_ops

image_paths_placeholder = tf.placeholder(tf.string, shape=(None,3), name='image_paths')
labels_placeholder = tf.placeholder(tf.int64, shape=(None,3), name='labels')
    
x =image_paths_placeholder 

input_queue = data_flow_ops.FIFOQueue(capacity=100000,
                            dtypes=[tf.string, tf.int64],
                            shapes=[(3,), (3,)],
                            shared_name=None, name=None)
enqueue_op = input_queue.enqueue_many([image_paths_placeholder, labels_placeholder])





#filenames, label = input_queue.dequeue()

nrof_preprocess_threads = 4
images_and_labels = []
for _ in range(nrof_preprocess_threads):
    filenames, label = input_queue.dequeue()
    images = []
    
    for filename in tf.unstack(filenames):
        file_contents = tf.read_file(filename)
        image = tf.image.decode_png(file_contents)
        
    
        image = tf.image.resize_image_with_crop_or_pad(image, 224, 224)
    
        #pylint: disable=no-member
        image.set_shape((224, 224, 3))
        images.append(tf.image.per_image_standardization(image))
    images_and_labels.append([images])

image_batch = tf.train.batch_join(
    images_and_labels, 
    batch_size=6,
    shapes=[(224, 224, 3)],
    enqueue_many=True,
    capacity=4 * 6,
    allow_smaller_final_batch=True)

#nrof_preprocess_threads = 4
#images_and_labels = []
#for _ in range(nrof_preprocess_threads):
#    filenames, label = input_queue.dequeue()
#    images = []
#    
#    for filename in tf.unstack(filenames):
#        file_contents = tf.read_file(filename)
#        image = tf.image.decode_png(file_contents)
#        
#    
#        image = tf.image.resize_image_with_crop_or_pad(image, 224, 224)
#    
#        #pylint: disable=no-member
#        image.set_shape((224, 224, 3))
#        images.append(tf.image.per_image_standardization(image))
#    images_and_labels.append([images, label])
#
#image_batch, labels_batch = tf.train.batch_join(
#    images_and_labels, 
#    batch_size=6,
#    shapes=[(224, 224, 3), ()],
#    enqueue_many=True,
#    capacity=4 * 6,
#    allow_smaller_final_batch=True)




#
#""" Select the triplets for training
#"""
#nrof_images_per_class=[3,3]
#base_path = '/home/dayea/Documents/MyWs/triplet_ws/test_data/'
#class_name = ['cat', 'dog', 'mice']
#
#
#image_path1=[]
#image_path2=[]
#for i in range(1,4):
#  image_path1.append(base_path+class_name[0]+'/'+str(i)+'.jpeg')
#  
#for i in range(1,4):
#  image_path2.append(base_path+class_name[1]+'/'+str(i)+'.jpeg')
#
#image_paths = [image_path1[0], image_path1[1], image_path1[2], image_path2[0], image_path2[1], image_path2[2]]
#
#labels = ['1','1','1','2','2','2']
#
#emb_start_idx = 0
#triplets = []
## VGG Face: Choosing good triplets is crucial and should strike a balance between
##  selecting informative (i.e. challenging) examples and swamping training with examples that
##  are too hard. This is achieve by extending each pair (a, p) to a triplet (a, p, n) by sampling
##  the image n at random, but only between the ones that violate the triplet loss margin. The
##  latter is a form of hard-negative mining, but it is not as aggressive (and much cheaper) than
##  choosing the maximally violating example, as often done in structured output learning.
#
#for i in range(2):
#  nrof_images = int(nrof_images_per_class[i])  # image number of every class
#  for j in range(1,nrof_images):
#      a_idx = emb_start_idx + j - 1
#      for pair in range(j, nrof_images): # For every possible positive pair.
#          p_idx = emb_start_idx + pair
#          pre_range = np.arange(emb_start_idx)
#          pos_range = np.arange(emb_start_idx+nrof_images,len(image_paths))
#          if len(pos_range)!=0:
#              total_range  = np.append(pre_range,pos_range)
#          nrof_range = len(total_range)
#          rnd_idx = np.random.randint(nrof_range)
#          n_idx = total_range[rnd_idx]
#          
#          triplets.append(((image_paths[a_idx], image_paths[p_idx], image_paths[n_idx]), (labels[a_idx], labels[p_idx], labels[n_idx])))
##          triplets2.append([[image_paths[a_idx], image_paths[p_idx], image_paths[n_idx]], [labels[a_idx], labels[p_idx], labels[n_idx]]])
#  emb_start_idx += nrof_images
#
##np.random.shuffle(triplets)
#
#triplets = np.array(triplets)
#triplets_paths = triplets[:, 0, :]
#triplets_labels = triplets[:, 1, :]
#
##yyyy = triplets_labels.reshape([-1]).tolist()
##[int(item) for item in yyyy]
#
#
##fdsfs=[triplets2[i][0][:] for i in range(int(len(image_paths)/3))]
#
##triplets_paths = [('a','a','a'),('a','a','a')]
##triplets_labels= np.array([[1,2,3],[4,5,6]], dtype=int)
#
#
#init = tf.global_variables_initializer()
#with tf.Session() as sess:
#  
#  
#  
#  sess.run(init)
#  
#  coord = tf.train.Coordinator()
#  threads = tf.train.start_queue_runners(coord=coord, sess=sess)
#  
#  feed_dict = {image_paths_placeholder: triplets_paths,
#               labels_placeholder: triplets_labels
#               }
#  
#  for i in range(20):
#    sess.run([enqueue_op], feed_dict = feed_dict) 
#  
#  filenames_np, label_np = sess.run([filenames, label])
#  
#  images_np = sess.run([image_batch])
#  
#  coord.request_stop()#queue need be turned off, otherwise it will report errors
#  coord.join(threads)
#  


#
#a = [2,3,1,4]
#b = np.array(a)
#
#c = np.argsort(b)
#
#a = np.load('path_list.npy')
#b = np.load('feature_list.npy')

path = '/raid/data/for_retrival/images'

  