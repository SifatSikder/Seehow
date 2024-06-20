import os
import sys
import cv2
import shutil
import numpy as np
from tqdm import tqdm
dataset_path = '/home/cheer/Project/video_test/action_data'
output_path = '/home/cheer/Project/video_test/action_merge'
dir_list = ['java_0_0', 'java_0_1', 'java_0_2', 'java_0_3', 'java_0_4','java_1_0', 'java_1_1', 'java_1_2', 'java_1_3', 'java_1_4','java_2_0', 'java_2_1', 'java_2_2', 'java_2_3', 'java_2_4','java_3_0', 'java_3_1', 'java_3_2', 'java_3_3', 'java_3_4','java_4_0', 'java_4_1', 'java_4_2', 'java_4_3', 'java_4_4']

'''
'java_0_0', 'java_0_1', 'java_0_2', 'java_0_3', 'java_0_4','java_1_0', 'java_1_1', 'java_1_2', 'java_1_3', 'java_1_4','java_2_0', 'java_2_1', 'java_2_2', 'java_2_3', 'java_2_4','java_3_0', 'java_3_1', 'java_3_2', 'java_3_3', 'java_3_4','java_4_0', 'java_4_1', 'java_4_2', 'java_4_3', 'java_4_4'

'python_0_0', 'python_0_1', 'python_0_2', 'python_0_3', 'python_0_4','python_1_0', 'python_1_1', 'python_1_2', 'python_1_3', 'python_1_4','python_2_0', 'python_2_1', 'python_2_2', 'python_2_3', 'python_2_4','python_3_0', 'python_3_1', 'python_3_2', 'python_3_3', 'python_3_4','python_4_0', 'python_4_1', 'python_4_2', 'python_4_3', 'python_4_4'
'''

merge_name = 'java_all'
parts = ['T0', 'T1']
#parts = ['T2', 'T3']
#parts = ['T4', 'T5']
label_file = 'label.txt'

def create_folder(output_path):
  for part in parts:
    if not os.path.exists(os.path.join(output_path, merge_name, part)):
      print 'create folder {}'.format(part)
      os.makedirs(os.path.join(output_path, merge_name, part))
    else:
      print 'folder {} exist'.format(part)

def read_label(label_path):
  if os.path.isfile(os.path.join(label_path, label_file)):
    with open(os.path.join(label_path, label_file), 'r') as label:
      lines = label.readlines()
  else:
    lines = []
  return lines

def merge_label(output_path, merge_name, labels):
  if len(labels) > 0:
    with open(os.path.join(output_path, merge_name, label_file), 'a') as label:
      label.writelines(labels)

def main(dataset_path):
  #dir_list = os.listdir(dataset_path)
  count = 0
  for folder in dir_list:
    if os.path.exists(os.path.join(dataset_path, folder, parts[0])):
      image_list = os.listdir(os.path.join(dataset_path, folder, parts[0]))
    else:
      image_list = []
    print 'merge folder', folder
    for image in tqdm(image_list):
      shutil.copyfile(os.path.join(dataset_path, folder, parts[0], image), os.path.join(output_path, merge_name, parts[0], image))
      shutil.copyfile(os.path.join(dataset_path, folder, parts[1], image), os.path.join(output_path, merge_name, parts[1], image))
    labels = read_label(os.path.join(dataset_path, folder))
    merge_label(output_path, merge_name, labels)
    count += len(image_list)
  print 'totally {} pairs'.format(count)

if __name__ == '__main__':
  create_folder(output_path)
  main(dataset_path)
