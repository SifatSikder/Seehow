from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import os
import sys
sys.path.append('lib/')
from scipy import misc
from feature_extractor.feature_extractor import FeatureExtractor
from preprocessing import preprocessing_factory
import feature_extractor.utils as utils
import random
import tensorflow as tf
from tqdm import tqdm
from tensorflow.python import pywrap_tensorflow 

path = '/home/cheer/Project/video_test/action_data/java_4_1'
label_name = 'label.txt'
ck_path = '/home/cheer/Project/ActionNet/models/2s/jp/model.ckpt-10000'

nums = 19
batch_size = 8
net_name = 'action_vgg_l'
input_mode = 0
output_mode = 0


'''
T0 T1: 299*299, RGB, diff
T2 T3: 224*224, RGB, diff, black BG   jp_black
T4 T5: 224*224, graysacle, diff       jp_gray
T6 T7: 224*224, RGB, full             jp_full
T8 T9: 224*224, grayscale, full       jp_full_gray
T10 T11: 224*224, RGB, diff           jp
T12: 224*224, absolute, full          jp_a_o
T13: 224*224, optical flow, full      jp_a_o
T6 T13: 224*224, RGB, optical flow    jp_2s
'''

#parts = ['T0', 'T1']
#parts = ['T2', 'T3']
#parts = ['T4', 'T5']
#parts = ['T6', 'T7']
#parts = ['T8', 'T9']
#parts = ['T10', 'T11']
#parts = ['T12', 'T13']
#parts = ['T6', 'T13']
parts = ['T10', 'T14']

def classification_placeholder_input(feature_extractor, transitions, image_path1, image_path2, logits_name, batch_size, num_classes):

  batch_image1 = np.zeros([batch_size, feature_extractor.image_size, feature_extractor.image_size, 3], dtype=np.float32)
  batch_image2 = np.zeros([batch_size, feature_extractor.image_size, feature_extractor.image_size, 3], dtype=np.float32)

  for i in range(batch_size):
    image_a = misc.imread(image_path1[i])
    image_a = misc.imresize(image_a, (feature_extractor.image_size, feature_extractor.image_size))
    if len(image_a.shape) == 2:
      b = image_a - 103.94
      g = image_a - 116.78
      r = image_a - 123.68
      image_a = np.stack([b, g, r], axis=2)
    else:
      channels = np.split(image_a, 3, axis=2)
      red = channels[0]
      green = channels[1]
      blue = channels[2]
      channels[0] = blue - 103.94
      channels[1] = green - 116.78
      channels[2] = red - 123.68
      image_a = np.concatenate((channels[0],channels[1],channels[2]),axis=2)
    batch_image1[i] = image_a

  for i in range(batch_size):
    image_b = misc.imread(image_path2[i])
    image_b = misc.imresize(image_b, (feature_extractor.image_size, feature_extractor.image_size))
    if len(image_b.shape) == 2:
      b = image_b - 103.94
      g = image_b - 116.78
      r = image_b - 123.68
      image_b = np.stack([b, g, r], axis=2)
    else:
      channels = np.split(image_b, 3, axis=2)
      red = channels[0]
      green = channels[1]
      blue = channels[2]
      channels[0] = blue - 103.94
      channels[1] = green - 116.78
      channels[2] = red - 123.68
      image_b = np.concatenate((channels[0],channels[1],channels[2]),axis=2)

    batch_image2[i] = image_b

  outputs = feature_extractor.feed_forward_batch([logits_name], batch_image1, batch_image2, fetch_images=True)

  predictions = outputs[logits_name]  

  if output_mode == 2:
    viterbi_sequence, viterbi_score = tf.contrib.crf.viterbi_decode(predictions, transitions)
    predictions = viterbi_sequence
  else:
    predictions = np.argmax(predictions, axis=1)
  #print(predictions)
  return predictions

def main(path):
  count = [0 for _ in range(nums)]
  label_count = [0 for _ in range(nums)]
  per_accuracy = [0 for _ in range(nums)]
  TP = [0 for _ in range(nums)]
  TN = [0 for _ in range(nums)]
  FP = [0 for _ in range(nums)]
  FN = [0 for _ in range(nums)]
  recall = [0 for _ in range(nums)]
  precision = [0 for _ in range(nums)]
  f1 = [0 for _ in range(nums)]
  accuracy = [0 for _ in range(nums)]
  total = 0
  confusion = np.zeros((nums, nums), dtype = int)
  TN_matrix = np.zeros((nums, nums), dtype = int)
  with open(os.path.join(path, label_name)) as label_file:
    labels = label_file.readlines()

  transitions = 0
  if output_mode == 2:
    reader=pywrap_tensorflow.NewCheckpointReader(ck_path) 
    var_to_shape_map=reader.get_variable_to_shape_map() 
    transitions = reader.get_tensor('transitions') 

  #labels = random.sample(labels, 1000)

  feature_extractor = FeatureExtractor(
    network_name=net_name,
    input_mode = input_mode,
    output_mode = output_mode,
    checkpoint_path=ck_path,
    batch_size=batch_size,
    num_classes=nums,
    preproc_func_name=net_name)
  feature_extractor.print_network_summary()

  for i in tqdm(range(int(len(labels)/batch_size))):
    image1 = []
    image2 = []
    label_num = []
    for j in range(batch_size):
      file_name = labels[i*batch_size + j].strip().split()[0]
      label_num.append(int(labels[i*batch_size +j].strip().split()[1]))
      label_count[label_num[j]] += 1
      image1.append(os.path.join(path, parts[0], file_name))
      image2.append(os.path.join(path, parts[1], file_name))
    clip_class = classification_placeholder_input(feature_extractor, transitions, image1, image2, net_name + '/fc8',batch_size, 19)
    for j in range(batch_size):
      confusion[label_num[j]][clip_class[j]] += 1
      if clip_class[j] == label_num[j]:
        count[clip_class[j]] += 1
        total += 1
  for i in range(len(label_count)):
    if label_count[i]:
      per_accuracy[i] = count[i]/label_count[i]
  for i in range(nums):
    TP[i] = confusion[i][i]
    FP[i] = np.sum(confusion[:,i]) - TP[i]
    FN[i] = np.sum(confusion[i]) - TP[i]
    TN[i] = len(labels) - TP[i] - FP[i] - FN[i]
    recall[i] = TP[i]/(TP[i]+FN[i])
    precision[i] = TP[i]/(TP[i]+FP[i])
    f1[i] = 2*precision[i]*recall[i]/(precision[i]+recall[i])
    accuracy[i] = (TP[i]+TN[i])/(TP[i]+TN[i]+FP[i]+FN[i]) 
  print (label_count)
  print (count)
  print (per_accuracy)
  print (total)
  print (total/len(labels))
  print (confusion)
  print ('TP:',TP)
  print ('FP:',FP)
  print ('FN:',FN)
  print ('TN:',TN)
  print ('precision:',precision)
  print ('recall:',recall)
  print ('f1:',f1)
  print ('accuracy:',accuracy)

if __name__ == '__main__':
  main(path)
