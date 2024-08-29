from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import numpy as np

import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import tf_slim as slim

_R_MEAN = 123.68
_G_MEAN = 116.78
_B_MEAN = 103.94

_RESIZE_SIDE_MIN = 256
_RESIZE_SIDE_MAX = 512

def _mean_image_subtraction(image, means):

  if image.get_shape().ndims != 3:
    raise ValueError('Input must be of size [height, width, C>0]')
  num_channels = image.get_shape().as_list()[-1]
  if len(means) != num_channels:
    raise ValueError('len(means) must match the number of channels')

  channels = tf.split(axis=2, num_or_size_splits=num_channels, value=image)
#  for i in range(num_channels):
#    channels[i] -= means[i]
#  return tf.concat(axis=2, values=channels)
  red = channels[0]
  green = channels[1]
  blue = channels[2]

  channels[0] = blue - means[2]
  channels[1] = green - means[1]
  channels[2] = red - means[0]
  input_bgr = tf.concat(axis=2, values=channels)
  return input_bgr

def preprocess_for_train(image,
                         output_height,
                         output_width,
                         resize_side_min=_RESIZE_SIDE_MIN,
                         resize_side_max=_RESIZE_SIDE_MAX):

  image.set_shape([output_height, output_width, 3])
  image = tf.to_float(image)
  #image = image / 255.0
  return _mean_image_subtraction(image, [_R_MEAN, _G_MEAN, _B_MEAN])

def preprocess_for_eval(image, output_height, output_width, resize_side):
  #image = tf.convert_to_tensor(image)
  image.set_shape([output_height, output_width, 3])
  #image = np.reshape(image, [output_height, output_width, 3])
  image = tf.to_float(image)
  #image = image / 255.0
  return _mean_image_subtraction(image, [_R_MEAN, _G_MEAN, _B_MEAN])
  #return image

def preprocess_image(image, output_height, output_width, is_training=False,
                     resize_side_min=_RESIZE_SIDE_MIN,
                     resize_side_max=_RESIZE_SIDE_MAX):

  if is_training:
    return preprocess_for_train(image, output_height, output_width,
                                resize_side_min, resize_side_max)
  else:
    return preprocess_for_eval(image, output_height, output_width,
                               resize_side_min)