#"""
#Created on Thu Apr 12 03:15:44 2018
#
#@author: dayea
#"""
"""Runs a ResNet model on the ImageNet dataset."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os
import sys

import tensorflow as tf
#%% Add by DY
  


def extra_input_processing(images, labels):
  imgs = tf.image.resize_images(images, [224, 224])
  mean = tf.constant([123.68, 116.779, 103.939], dtype=tf.float32, shape=[1, 1, 1, 3], name='img_mean')
  images = imgs-mean
  
  
  
  return images, labels


def get_tf_filenames(is_training, data_dir):
    
  """Returns a list of filenames."""

  if is_training:
    return [os.path.join(data_dir, 'TFrTrainPart1_collar_design_labels.tfrecords')]
  else:
    return [os.path.join(data_dir, 'TFrTrainPart2_collar_design_labels.tfrecords')]

def get_tf_filenames_pred(data_dir):
    
  """Returns a list of filenames."""
  return [os.path.join(data_dir, 'TFrQuest_collar_design_labels.tfrecords')]


def record_parser(value, is_training, image_shape, label_length):
  """Parse an ImageNet record from `value`."""
  
  keys_to_features = {
      'img_raw' : 
          tf.FixedLenFeature([], tf.string, default_value=''),
      'label': 
          tf.FixedLenFeature([label_length], tf.float32),
  }

  parsed = tf.parse_single_example(value, keys_to_features)

#  image = tf.decode_raw(tf.reshape(parsed['img_raw'], shape=[]), 3)
  image = tf.decode_raw(parsed['img_raw'],out_type=tf.uint8)
  image = tf.image.convert_image_dtype(image, dtype=tf.float32)

  image = tf.reshape(image, [image_shape[0], image_shape[1], image_shape[2]])
#  image = image * (1. / 255) - 0.5
  
  label = tf.cast(
      tf.reshape(parsed['label'], shape=[label_length]),
      dtype=tf.float32)
  image, label = extra_input_processing(image, label)

#  label = tf.arg_max(label, _LABEL_CLASSES)

  return image, label


def record_parser_pred(value, image_shape):
  """Parse an ImageNet record from `value`."""
  
  keys_to_features = {
      'img_raw' : 
          tf.FixedLenFeature([], tf.string, default_value='')
  }

  parsed = tf.parse_single_example(value, keys_to_features)

#  image = tf.decode_raw(tf.reshape(parsed['img_raw'], shape=[]), 3)
  image = tf.decode_raw(parsed['img_raw'],out_type=tf.uint8)
  image = tf.image.convert_image_dtype(image, dtype=tf.float32)

  image = tf.reshape(image, [image_shape[0], image_shape[1], image_shape[2]])
#  image = image * (1. / 255) - 0.5


  return image


def input_fn(is_training, data_dir, batch_size, num_epochs=1):
  """Input_fn using the tf.data input pipeline for CIFAR-10 dataset.

  Args:
    is_training: A boolean denoting whether the input is for training.
    data_dir: The directory containing the input data.
    batch_size: The number of samples per batch.
    num_epochs: The number of epochs to repeat the dataset.

  Returns:
    A tuple of images and labels.
  """
  
  """Input function which provides batches for train or eval."""
  
  _FILE_SHUFFLE_BUFFER = 1
  _SHUFFLE_BUFFER = 10000
  
  dataset = tf.data.Dataset.from_tensor_slices(get_tf_filenames(is_training, data_dir))

  if is_training:
    dataset = dataset.shuffle(buffer_size=_FILE_SHUFFLE_BUFFER)

  dataset = dataset.flat_map(tf.data.TFRecordDataset)
  dataset = dataset.map(lambda value: record_parser(value, is_training),
                        num_parallel_calls=5)
  dataset = dataset.prefetch(batch_size)

  if is_training:
    # When choosing shuffle buffer sizes, larger sizes result in better
    # randomness, while smaller sizes have better performance.
    dataset = dataset.shuffle(buffer_size=_SHUFFLE_BUFFER)

  # We call repeat after shuffling, rather than before, to prevent separate
  # epochs from blending together.
  dataset = dataset.repeat(num_epochs)
  dataset = dataset.batch(batch_size)

  iterator = dataset.make_one_shot_iterator()
  images, labels = iterator.get_next()
  return images, labels


def input_fn_pred(data_dir, batch_size):
  """Input_fn using the tf.data input pipeline for CIFAR-10 dataset.

  Args:
    is_training: A boolean denoting whether the input is for training.
    data_dir: The directory containing the input data.
    batch_size: The number of samples per batch.
    num_epochs: The number of epochs to repeat the dataset.

  Returns:
    A tuple of images and labels.
  """
  
  """Input function which provides batches for train or eval."""
  dataset = tf.data.Dataset.from_tensor_slices(get_tf_filenames_pred(data_dir))

#  if is_training:
#    dataset = dataset.shuffle(buffer_size=_FILE_SHUFFLE_BUFFER)

  dataset = dataset.flat_map(tf.data.TFRecordDataset)
  dataset = dataset.map(lambda value: record_parser_pred(value),
                        num_parallel_calls=5)
  dataset = dataset.prefetch(batch_size)


  # We call repeat after shuffling, rather than before, to prevent separate
  # epochs from blending together.
  dataset = dataset.repeat(1)
  dataset = dataset.batch(batch_size)

  iterator = dataset.make_one_shot_iterator()
  images = iterator.get_next()
  return images
