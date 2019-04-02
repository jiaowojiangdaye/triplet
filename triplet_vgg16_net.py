#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr 21 23:46:00 2018

@author: mbzhao
"""

import tensorflow as tf
import numpy as np
from tensorflow.python.ops import data_flow_ops
from matplotlib import pyplot as plt
from PIL import Image  
import matplotlib.image as mpimg # mpimg
import os
import argparse
import csv
import random

#%%
def get_args():
  
  parser = argparse.ArgumentParser()
  
  parser.add_argument(
      '--image_base_dir',
      type=str,
      default='test_data',
#      default='/raid/Guests/DaYea/AmazonData/AmazonFigureEasy',
#      default='/raid/Guests/DaYea/ILSVRC2012_20',
#      default='/raid/data/for_retrival/images',
      help='The directory where the TFrecord will be loaded.')
  
  parser.add_argument(
      '--class_list_file',
      type=str,
      default='test_data/class_label_list.csv',
#      default='/raid/Guests/DaYea/AmazonData/Easy_class_names.txt',
#      default='/raid/Guests/DaYea/ILSVRC2012_20/class_list.txt',
#      default='/raid/data/for_retrival/class_list.csv'  ,
      help='The class name list file (.txt .csv one name per line). If you don\'t have it, just name a name for it '
            'and we will generate a new one for you according to the data in image_base_dir.')
  
  
#    parser.add_argument(
#      '--train_proportion',
#      type=float,
#      default=0.7,
#      help='The directory where the TFrecord will be loaded.')
  
  
  parser.add_argument(
      '--nrof_classes',
      type=int,
      default=4
      )
  
  parser.add_argument(
      '--offical_vgg_para',
      type=str,
      default='vgg16_weights.npz'
      )
  
  parser.add_argument(
      '--trained_para_file',
      type=str,
#      default='trained_para_for_vgg_triplet/vgg16_para.npy'
      default='saved_file/vgg16_para.npy'
      )
  
  parser.add_argument(
      '--image_size', type=int,
      default= 224,
      help='T')
  
  parser.add_argument(
      '--batch_size', type=str,
      default=20,
      help='T')
  
  parser.add_argument(
      '--classes_per_batch', type=str,
      default=4,
      help='T')

  parser.add_argument(
      '--items_per_class', type=str,
      default=5,
      help='T')

  parser.add_argument(
      '--num_epochs', type=int,
      default= 1,
      help='T')
  
  parser.add_argument(
      '--LEARN_RATE', type=str,
      default=0.01,
      help='T')

  parser.add_argument(
      '--WEIGHT_DECAY', type=str,
      default=0.0002,
      help='T')

  parser.add_argument(
      '--random_crop', type=bool,
      default=False,
      help='T')

  parser.add_argument(
      '--random_flip', type=bool,
      default=False,
      help='T')
  
  
  parser.add_argument(
      '--lamda', type=int,
      default=10,
      help='T')

  parser.add_argument(
      '--alpha', type=int,
      default=100,
      help='T')
  
  parser.add_argument(
      '--beta', type=int,
      default=0.1,
      help='T')
  
  

  
  
  
  arg, unparsed = parser.parse_known_args()
  
  return arg




#%%
def read_class_label_csv(path):
    
  class_label_list = []
  csvfile = open(path, 'r')
  print('读取的csv文件：')
  csvreader = csv.reader(csvfile)
  
  for line in csvreader:
    class_label_list.append(line[1])
          
      
  csvfile.close()
  
  return class_label_list

def read_class_label_txt(path):
  
  class_label_list = []
  
  file = open(path, 'r')
  line = file.readline()
  
  while line:
    line = line.split(' ')[0]
    class_label_list.append(line)
    line = file.readline()
  
  
  return class_label_list

class ImageClass():
    "Stores the paths to images for a given class"
    def __init__(self, name, image_paths):
        self.name = name
        self.image_paths = image_paths
        self.num = len(self.image_paths)
    def __str__(self):
        return self.name + ', ' + str(len(self.image_paths)) + ' images'
  
    def __len__(self):
        return len(self.image_paths)



def generate_class_list_csv(image_path, class_list_file):

  class_name_list = os.listdir(image_path)
  class_name_list.sort()
        
  
  print('generating class name list csv文件：')
  print(class_list_file)
  csvfile = open(class_list_file, 'w', newline='')
  csvwriter = csv.writer(csvfile)
  for class_name in class_name_list:
    csvwriter.writerow([class_name])
  csvfile.close() 
  
  return class_name_list
  




def get_dataset(image_path, class_list_file):

  dataset = []
  if os.path.exists(class_list_file):
    if class_list_file[-4:] == '.txt':
      class_name_list = read_class_label_txt(class_list_file)
    elif class_list_file[-4:] == '.csv':
      class_name_list = read_class_label_csv(class_list_file)
    else:
      print('invailded class list file formate!!!we will generate a new csv file for use.')
  else:
    print('No class list file, we will generate a new csv file for use.')
    class_name_list = generate_class_list_csv(image_path, class_list_file)
      
  
  for class_name in class_name_list:
      
    class_path = os.path.join(image_path, class_name.strip())
    image_of_class = os.listdir(class_path)
    
    image_paths = [os.path.join(class_path, image) for image in image_of_class]
    dataset.append(ImageClass(class_name, image_paths))

  return dataset, class_name_list




#%%

class data_queue:
  def __init__(self, args, mode='train'):
    self.classes_per_batch = args.classes_per_batch
    self.items_per_class = args.items_per_class
    self.args = args
    self.data_dir = os.path.join(args.image_base_dir, mode)
    
    
    self.prepare_batch()
    
    

  def create_data_queue(self):
    
    self.image_paths_placeholder = tf.placeholder(tf.string, shape=(None,3), name='image_paths')
    self.labels_placeholder = tf.placeholder(tf.int64, shape=(None,3), name='labels')
    
    
    input_queue = data_flow_ops.FIFOQueue(capacity=100000,
                                dtypes=[tf.string, tf.int64],
                                shapes=[(3,), (3,)],
                                shared_name=None, name='Dayea_queue')
    self.enqueue_op = input_queue.enqueue_many([self.image_paths_placeholder, self.labels_placeholder])
    qr = tf.train.QueueRunner(input_queue, enqueue_ops=[self.enqueue_op] *4)
    
#    tf.train.add_queue_runner(qr)

    nrof_preprocess_threads = 4
    images_and_labels = []
    for _ in range(nrof_preprocess_threads):
        filenames, label = input_queue.dequeue()
        images = []
        for filename in tf.unstack(filenames):
            file_contents = tf.read_file(filename)
            image = tf.image.decode_png(file_contents)
            
            if self.args.random_crop:
                image = tf.random_crop(image, [self.args.image_size, self.args.image_size, 3])
            else:
#                image = tf.image.resize_image_with_crop_or_pad(image, self.args.image_size, self.args.image_size)
                image = tf.image.resize_images(image, (self.args.image_size, self.args.image_size))
            if self.args.random_flip:
                image = tf.image.random_flip_left_right(image)

            #pylint: disable=no-member
            image.set_shape((self.args.image_size, self.args.image_size, 3))
            images.append(image)
#            images.append(tf.image.per_image_standardization(image))
        images_and_labels.append([images, label])

    image_batch, labels_batch = tf.train.batch_join(
        images_and_labels, batch_size=self.args.batch_size*3, #%%%%%%%%%%%%%%%%%%?????????????????????
        shapes=[(self.args.image_size, self.args.image_size, 3), ()], enqueue_many=True,
        capacity=4 * nrof_preprocess_threads * self.args.batch_size,
        allow_smaller_final_batch=True, name='queue2')
        
    
    
    return image_batch, labels_batch, qr


  def sample_people(self):

    """ input
        dataset: train dataset
        people_per_batch: number of people in a batch
        images_per_person: image number of every person
            
        return
        image_paths: image path in dataset(list)
    """
  
    nrof_images = self.classes_per_batch * self.items_per_class
  
    # Sample classes from the dataset
    nrof_classes = len(self.dataset)
    class_indices = np.arange(nrof_classes)
    np.random.shuffle(class_indices)
    
    i = 0
    image_paths = []
    num_per_class = []
    sampled_class_indices = []
    # Sample images from these classes until we have enough
    while len(image_paths)<nrof_images:
        class_index = class_indices[i]
        nrof_images_in_class = len(self.dataset[class_index])  # totle number of  images in the class
        image_indices = np.arange(nrof_images_in_class)
        np.random.shuffle(image_indices)
        nrof_images_from_class = min(nrof_images_in_class, self.items_per_class, nrof_images-len(image_paths))  
        idx = image_indices[0:nrof_images_from_class]
        image_paths_for_class = [self.dataset[class_index].image_paths[j] for j in idx]
        sampled_class_indices += [class_index]*nrof_images_from_class
        image_paths += image_paths_for_class
        num_per_class.append(nrof_images_from_class)
        i+=1
  
    return image_paths, sampled_class_indices, num_per_class
  
  def select_triplets(self, nrof_images_per_class, image_paths, labels):
      
    """ Select the triplets for training
    """
    batch_size = self.args.batch_size
    triplets = []
    class_indincs = [np.random.randint(self.classes_per_batch) for i in range(batch_size)]
    class_indincs.sort()
    
    for class_idx in class_indincs:
      class_start_idx = sum(nrof_images_per_class[0:class_idx])
      nrof_images = int(nrof_images_per_class[class_idx])
      pos_range = range(class_start_idx, class_start_idx + nrof_images)
      a_idx, p_idx = random.sample(pos_range, 2)
      pre_range = list(range(class_start_idx))
      res_range = list(range(class_start_idx+nrof_images,len(image_paths)))
      neg_range  = pre_range+ res_range
      n_idx = random.sample(neg_range, 1)[0]
      triplets.append(((image_paths[a_idx], image_paths[p_idx], image_paths[n_idx]), 
                       (labels[a_idx], labels[p_idx], labels[n_idx])))
    

    np.random.shuffle(triplets)
    
    triplets = np.array(triplets)
    triplets_paths = triplets[:, 0, :]
    triplets_labels = triplets[:, 1, :]
    
    
    return triplets_paths, triplets_labels, len(triplets)


#  def select_triplets(self, nrof_images_per_class, image_paths, labels):
#      
#    """ Select the triplets for training
#    """
#    
#    emb_start_idx = 0
#    triplets = []
#    # VGG Face: Choosing good triplets is crucial and should strike a balance between
#    #  selecting informative (i.e. challenging) examples and swamping training with examples that
#    #  are too hard. This is achieve by extending each pair (a, p) to a triplet (a, p, n) by sampling
#    #  the image n at random, but only between the ones that violate the triplet loss margin. The
#    #  latter is a form of hard-negative mining, but it is not as aggressive (and much cheaper) than
#    #  choosing the maximally violating example, as often done in structured output learning.
#
#    for i in range(self.classes_per_batch):
#      nrof_images = int(nrof_images_per_class[i])  # image number of every class
#      for j in range(1,nrof_images):
#          a_idx = emb_start_idx + j
#          for pair in range(j, nrof_images): # For every possible positive pair.
#              p_idx = emb_start_idx + pair
#              pre_range = np.arange(emb_start_idx)
#              pos_range = np.arange(emb_start_idx+nrof_images,len(image_paths))
#                  total_range  = np.append(pre_range,pos_range)
#              nrof_range = len(total_range)
#              rnd_idx = np.random.randint(nrof_range)
#              n_idx = total_range[rnd_idx]
#              
#              triplets.append(((image_paths[a_idx], image_paths[p_idx], image_paths[n_idx]), (labels[a_idx], labels[p_idx], labels[n_idx])))
#      emb_start_idx += nrof_images
#
#    np.random.shuffle(triplets)
#    
#    triplets = np.array(triplets)
#    triplets_paths = triplets[:, 0, :]
#    triplets_labels = triplets[:, 1, :]
#    
#    
#    return triplets_paths, triplets_labels, len(triplets)


  def prepare_batch(self):
    
    self.dataset, self.class_name_list = get_dataset(self.data_dir, self.args.class_list_file)
    
    
  def feed_data_to_queue(self, sess):
    
    image_paths, labels, nrof_images_per_class = self.sample_people()
    triplets_paths, triplets_labels, nrof_triplets = self.select_triplets(
        nrof_images_per_class, 
        image_paths,
        labels)
    
    
    
    
    feed_dict = {self.image_paths_placeholder: triplets_paths,
                 self.labels_placeholder: triplets_labels
                 }

    
    
    
    sess.run(self.enqueue_op, feed_dict = feed_dict)
    
    




class data_queue_v2:
  def __init__(self, args, mode='train'):
    self.classes_per_batch = args.classes_per_batch
    self.args = args
    self.data_dir = os.path.join(args.image_base_dir, mode)
    self.batch_step = 0
    self.prepare_batch()
    
    
  def create_data_queue(self):
    
    self.image_paths_placeholder = tf.placeholder(tf.string, shape=(None,), name='image_paths')
    
    
    input_queue = data_flow_ops.FIFOQueue(capacity=100000,
                                dtypes=[tf.string],
                                shapes=[()],
                                shared_name=None, name='Dayea_queue')
    self.enqueue_op = input_queue.enqueue_many([self.image_paths_placeholder])
#    qr = tf.train.QueueRunner(input_queue, enqueue_ops=[self.enqueue_op] *4)
    
#    tf.train.add_queue_runner(qr)

#    nrof_preprocess_threads = 2
#    images_and_labels = []
#    for _ in range(nrof_preprocess_threads):
#        filenames = input_queue.dequeue()
#        images_and_labels.append([filenames])
#
#          
#    image_batch = tf.train.batch_join(
#        images_and_labels, batch_size=self.args.batch_size, #%%%%%%%%%%%%%%%%%%?????????????????????
#        shapes=[()], enqueue_many=True,
#        capacity=4 * nrof_preprocess_threads * self.args.batch_size,
#        allow_smaller_final_batch=True, name='queue2')

    nrof_preprocess_threads = 2
    images_and_labels = []
    for _ in range(nrof_preprocess_threads):
      filename = input_queue.dequeue()
      file_contents = tf.read_file(filename)
      image = tf.image.decode_png(file_contents)
      
      if self.args.random_crop:
          image = tf.random_crop(image, [self.args.image_size, self.args.image_size, 3])
      else:
#                image = tf.image.resize_image_with_crop_or_pad(image, self.args.image_size, self.args.image_size)
          image = tf.image.resize_images(image, (self.args.image_size, self.args.image_size))
      if self.args.random_flip:
          image = tf.image.random_flip_left_right(image)

      #pylint: disable=no-member
      image.set_shape((self.args.image_size, self.args.image_size, 3))
      images_and_labels.append([[image]])

          

    image_batch = tf.train.batch_join(
        images_and_labels, batch_size=self.args.batch_size, #%%%%%%%%%%%%%%%%%%?????????????????????
        shapes=[(self.args.image_size, self.args.image_size, 3)], enqueue_many=True,
        capacity=4 * nrof_preprocess_threads * self.args.batch_size,
        allow_smaller_final_batch=True, name='queue2')
        
    

    return image_batch
  
  
  def get_dataset(self, image_path, class_list_file):

    dataset = []
    
    if class_list_file[-3:] == 'txt':
        class_name_list = read_class_label_txt(class_list_file)
    elif class_list_file[-3:] == 'csv':
        class_name_list = read_class_label_csv(class_list_file)
    
    for class_name in class_name_list:
      class_path = os.path.join(image_path, class_name.strip())
      image_of_class = os.listdir(class_path)
      
      dataset = dataset + [os.path.join(class_path, image) for image in image_of_class]
  
    return dataset, class_name_list

  def get_batch(self):

    finish = False
    image_paths = []
    now_img_idx = self.batch_step * self.args.batch_size
    if self.batch_step >= self.total_steps:
      print('not enough images!')
      self.batch_step = 0
      finish = True
    if not finish:
      image_paths = self.dataset[now_img_idx: now_img_idx + self.args.batch_size]
      image_paths = np.array(image_paths)
      self.batch_step = self.batch_step + 1
    return image_paths, finish



  def prepare_batch(self):
    
    self.dataset, self.class_name_list = self.get_dataset(self.data_dir, self.args.class_list_file)
    self.total_steps = len(self.dataset) // self.args.batch_size
    
    
  def feed_data_to_queue(self, sess):
    
    image_paths, finish = self.get_batch()
    feed_dict = {self.image_paths_placeholder: image_paths}

    
    if not finish:
      sess.run(self.enqueue_op, feed_dict = feed_dict)
    
    return image_paths, finish
  
  

#%%
    
  
  
  
  
def mergeReport2(name, files):

  baseimg=Image.open(files[0])
  sz = baseimg.size
  basemat=np.atleast_2d(baseimg)
  for file in files[1:]:
      im=Image.open(file)
  #resize to same width
      sz2 = im.size
      if sz2!=sz:
          im=im.resize((sz[0],round(sz2[0] / sz[0] * sz2[1])),Image.ANTIALIAS)
      mat=np.atleast_2d(im)
      basemat=np.append(basemat,mat,axis=0)
  report_img=Image.fromarray(basemat)
  report_img.save(name)
  
def mergeReport(name, files, size=(224,224),axis=0):

  baseimg=Image.open(files[0])
  baseimg=baseimg.resize(size,Image.ANTIALIAS)
  basemat=np.atleast_2d(baseimg)
  for file in files[1:]:
      im=Image.open(file)
  #resize to same width

      im=im.resize(size,Image.ANTIALIAS)
      mat=np.atleast_2d(im)
      basemat=np.append(basemat,mat,axis=axis)
  report_img=Image.fromarray(basemat)
  report_img.save(name)
  
class retrivaler:
  def __init__(self, save_path, path_file, feature_file):
    self.save_path = save_path
    self.image_paths = np.load(path_file)
    self.features = np.load(feature_file)
    
    
    
  def calEuclideanDistance(self,vec1,vec2):  
    dist = np.sqrt(np.sum(np.square(vec1 - vec2)))  
    return dist
    
  def retrival_once(self, name, key, key_path, num = 10):
    
    dists = []
    for i, element in enumerate(self.features):
      dists.append(self.calEuclideanDistance(element, key))
      
    
    
    dists_idx = np.argsort(np.array(dists))[:10]
    
    retrival_paths = []
    for idx in dists_idx:
      retrival_paths.append(self.image_paths[idx])
    
    img_name = self.save_path+"/re"+name + '.png'
    mergeReport(img_name, 
                [key_path]+retrival_paths, 
                size=(224,224),
                axis=1)
    
    return img_name
    #show results
#    im = Image.open(key_path)  
#    im.save(self.save_path + "/re"+name+"_" + str(0) + ".jpg")
    
    
    
#    for index, element in enumerate(retrival_paths):
#      
#      im = Image.open(element)
#      im.save(self.save_path + "/re"+name+"_" + str(index+1) + ".jpg")

    
#    return retrival_paths
  
  



#%%
def feature_map_2_vector(feature):
  

#  channel = feature.get_shape()[3]
  shape2 = int(np.prod(feature.get_shape()[1]))
#  max_per_channel = tf.reduce_max(feature, reduction_indices=[1,2])
  
  pool = tf.nn.max_pool(feature,
                   ksize=[1, shape2, shape2, 1],
                   strides=[1, shape2, shape2, 1],
                   padding='VALID',
                   name='pool4')
  shape = int(np.prod(pool.get_shape()[1:]))
  pool_flat = tf.reshape(pool, [-1, shape])
  
  return pool_flat
    
  
  
def interface_train_classifier(args, net, images, labels):
  
  LEARN_RATE = args.LEARN_RATE
  WEIGHT_DECAY=args.WEIGHT_DECAY
  
  out, pool5 = net.model(images)
  one_hot_labels = tf.one_hot(labels, args.nrof_classes)
  classifier_loss = tf.nn.softmax_cross_entropy_with_logits(logits=out, labels=one_hot_labels)
  classifier_loss = tf.reduce_mean(classifier_loss, 0)
  
  #l2 loss
  l2_loss = WEIGHT_DECAY * tf.add_n([tf.nn.l2_loss(v) for v in tf.trainable_variables()])
 
  loss = classifier_loss + l2_loss
  
  opt = tf.train.GradientDescentOptimizer(LEARN_RATE).minimize(loss) 
  
  probs = tf.argmax(tf.nn.softmax(out), 1)
  correct_prediction = tf.equal(probs, labels)
  acc = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
  
  return opt, classifier_loss, loss, acc, probs, correct_prediction, pool5
  


def interface_train_triplet(args, net, images, labels):
  
  
  LEARN_RATE = args.LEARN_RATE
  WEIGHT_DECAY=args.WEIGHT_DECAY
  
  lamda = args.lamda
  alpha = args.alpha
  
  _, feature = net.model(images)   #[512]
  
  embeddings = feature_map_2_vector(feature)   #[512]
#  embeddings = feature   #[512]
  
  vector_length = int(embeddings.get_shape()[1])
  #triplet loss
  embeddings = tf.nn.l2_normalize(x=embeddings, axis=1, name='embeddings')
  # Split embeddings into anchor, positive and negative and calculate triplet loss
  anchor, positive, negative = tf.unstack(tf.reshape(embeddings, [-1,3,vector_length]), 3, 1)
  
  pos_dist = tf.reduce_sum(tf.square(tf.subtract(anchor, positive)), 1)
  neg_dist = tf.reduce_sum(tf.square(tf.subtract(anchor, negative)), 1)
  neg_dist1 = tf.multiply(neg_dist, lamda)
  
  triplet_loss = tf.add(tf.subtract(pos_dist, neg_dist1), alpha)
  triplet_loss = tf.reduce_mean(tf.maximum(triplet_loss, 0.0), 0)

  #l2 loss
  l2_loss = WEIGHT_DECAY * tf.add_n([tf.nn.l2_loss(v) for v in tf.trainable_variables()])
  
  #total loss  tf.reduce_mean(-neg_dist, 0)
  loss = triplet_loss + l2_loss

  #compute softmax softmax_cross_entropy_with_logits
  opt = tf.train.GradientDescentOptimizer(LEARN_RATE).minimize(loss) 
  
  
  return opt, pos_dist, neg_dist, loss, feature, embeddings, neg_dist1, triplet_loss



def interface_train_triplet_classifier(args, net, images, labels):
  
  LEARN_RATE = args.LEARN_RATE
  WEIGHT_DECAY=args.WEIGHT_DECAY
  
  lamda = args.lamda
  alpha = args.alpha
  beta = args.beta
  
  out,   feature = net.model(images)   #[512]
  
  vector = feature_map_2_vector(feature)   #[512]
  
  vector_length = int(vector.get_shape()[1])
  #triplet loss
  embeddings = tf.nn.l2_normalize(x=vector, axis=1, name='embeddings')
  # Split embeddings into anchor, positive and negative and calculate triplet loss
  anchor, positive, negative = tf.unstack(tf.reshape(embeddings, [-1,3,vector_length]), 3, 1)
  
  pos_dist = tf.reduce_sum(tf.square(tf.subtract(anchor, positive)), 1)
  neg_dist = tf.reduce_sum(tf.square(tf.subtract(anchor, negative)), 1)
  neg_dist1 = tf.multiply(neg_dist, lamda)
  
  basic_loss = tf.add(tf.subtract(pos_dist, neg_dist1), alpha)
  triplet_loss = tf.reduce_mean(tf.maximum(basic_loss, 0.0), 0)
  
  #classifier loss
  one_hot_lab = tf.one_hot(labels, args.nrof_classes)
  
  classifier_loss = tf.nn.softmax_cross_entropy_with_logits(logits=out, labels=one_hot_lab)
  classifier_loss = tf.reduce_mean(classifier_loss, 0)
  
  
  #l2 loss
  l2_loss = WEIGHT_DECAY * tf.add_n([tf.nn.l2_loss(v) for v in tf.trainable_variables()])
  
  #total loss
  loss = classifier_loss + beta*triplet_loss + l2_loss

  #compute softmax softmax_cross_entropy_with_logits
  opt = tf.train.GradientDescentOptimizer(LEARN_RATE).minimize(loss) 
  
  
  #about classifier performance
  probs = tf.nn.softmax(out)
  prediction = tf.argmax(probs, 1)
  correct_prediction = tf.equal(prediction, labels)
  acc = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
  
  
  return opt, pos_dist, neg_dist, classifier_loss, loss, acc, probs
  

  
def interface_eval_classifier(args, net, images, labels):
  
  lamda = args.lamda
  alpha = args.alpha
  out,   feature = net.model(images)   #[512]
  
  vector = feature_map_2_vector(feature)   #[512]
  
  vector_length = int(vector.get_shape()[1])
  #triplet loss
  embeddings = tf.nn.l2_normalize(x=vector, axis=1, name='embeddings')
  # Split embeddings into anchor, positive and negative and calculate triplet loss
  anchor, positive, negative = tf.unstack(tf.reshape(embeddings, [-1,3,vector_length]), 3, 1)
  
  pos_dist = tf.reduce_sum(tf.square(tf.subtract(anchor, positive)), 1)
  neg_dist = tf.reduce_sum(tf.square(tf.subtract(anchor, negative)), 1)
  neg_dist1 = tf.multiply(neg_dist, lamda)
  
  basic_loss = tf.add(tf.subtract(pos_dist, neg_dist1), alpha)
  triplet_loss = tf.reduce_mean(tf.maximum(basic_loss, 0.0), 0)
  
 
  one_hot_labels = tf.one_hot(labels, args.nrof_classes)
  classifier_loss = tf.nn.softmax_cross_entropy_with_logits(logits=out, labels=one_hot_labels)
  classifier_loss = tf.reduce_mean(classifier_loss, 0)
  
  #l2 loss
#  l2_loss = WEIGHT_DECAY * tf.add_n([tf.nn.l2_loss(v) for v in tf.trainable_variables()])
 
  loss = classifier_loss + triplet_loss#+ l2_loss
  
  probs = tf.argmax(tf.nn.softmax(out), 1)
  correct_prediction = tf.equal(probs, labels)
  acc = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
  
  return loss, acc, probs, pos_dist, neg_dist



def interface_generate_feat(args, net, images):
  

  out,   feature = net.model(images)   #[512]
  vector = feature_map_2_vector(feature)   #[512]
  embeddings = tf.nn.l2_normalize(x=vector, axis=1, name='embeddings')
  
  return embeddings
  