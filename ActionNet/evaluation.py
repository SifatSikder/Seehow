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
from tqdm import tqdm

path = '/home/cheer/Project/video_test/action_data/java_4_1'
label_name = 'label.txt'
ck_path = '/home/cheer/Project/ActionNet/models/s/jp/model.ckpt-60000'
num = 19
nums = 19

'''
T0 T1: 299*299, RGB, diff
T2 T3: 224*224, RGB, diff, black BG   jp_black
T4 T5: 224*224, graysacle, diff       jp_gray
T6 T7: 224*224, RGB, full             jp_full
T8 T9: 224*224, grayscale, full       jp_full_gray
T10 T11: 224*224, RGB, diff           jp
T12: 224*224, absolute, full          jp_a_o
T13: 224*224, optical flow, full      jp_a_o
'''

#parts = ['T0', 'T1']
#parts = ['T2', 'T3']
#parts = ['T4', 'T5']
#parts = ['T6', 'T7']
#parts = ['T8', 'T9']
parts = ['T10', 'T11']
#parts = ['T12', 'T13']

label_map = [[0,1,22,23,24], [2,5,16,19], [3,4,17,20], [6,8,9,13,21,28], [7,15], [10,29], [11], [26], [14,27], [12,18,25]]

def classification_placeholder_input(feature_extractor, image_path1, image_path2, logits_name, batch_size, num_classes):
  image_file1 = image_path1
  image_file2 = image_path2
  batch_image1 = np.zeros([batch_size, feature_extractor.image_size, feature_extractor.image_size, 3], dtype=np.float32)
  batch_image2 = np.zeros([batch_size, feature_extractor.image_size, feature_extractor.image_size, 3], dtype=np.float32)

  #image_preprocessing_fn = preprocessing_factory.get_preprocessing('action_vgg_s')

  for i in range(batch_size):
    image_a = misc.imread(image_file1)
    image_a = misc.imresize(image_a, (feature_extractor.image_size, feature_extractor.image_size))
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
    image_b = misc.imread(image_file2)
    image_b = misc.imresize(image_b, (feature_extractor.image_size, feature_extractor.image_size))
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
  predictions = np.squeeze(outputs[logits_name])
  predictions = np.argmax(predictions)
  return predictions

def convert_label(label_num):
  for i in range(len(label_map)):
    if label_num in label_map[i]:
      return i
      break
  return 0

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

  labels = random.sample(labels, 1000)

  feature_extractor = FeatureExtractor(
    network_name='action_vgg_s',
    checkpoint_path=ck_path,
    batch_size=1,
    num_classes=num,
    preproc_func_name='action_vgg_s')
  feature_extractor.print_network_summary()

  for label in tqdm(labels):
    file_name = label.strip().split()[0]
    label_num = label.strip().split()[1]
    label_num = int(label_num)
    #label_num = convert_label(label_num)
    label_count[label_num] += 1
    image1 = os.path.join(path, parts[0], file_name)
    image2 = os.path.join(path, parts[1], file_name)
    clip_class = classification_placeholder_input(feature_extractor, image1, image2, 'action_vgg_s/fc8',1, 19)
    #clip_class = convert_label(clip_class)
    confusion[label_num][clip_class] += 1
    if clip_class == label_num:
      count[clip_class] += 1
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
  #start(path)
  main(path)
