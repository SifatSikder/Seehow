import os
import sys
import cv2
import shutil
import numpy as np
from tqdm import tqdm
dataset_path = '/home/cheer/Project/video_test/action_data'
parts1 = ['t0', 't1']
parts2 = ['t2', 't3']
parts3 = ['Annotation']
label_name = 'label.txt'
nums = 19

def check_file():
  dir_list = os.listdir(dataset_path)
  for folder in dir_list:
    lsdir0 = os.listdir(os.path.join(dataset_path, folder, parts1[0]))
    lsdir1 = os.listdir(os.path.join(dataset_path, folder, parts1[1]))
    lsdir2 = os.listdir(os.path.join(dataset_path, folder, parts2[0]))
    lsdir3 = os.listdir(os.path.join(dataset_path, folder, parts2[1]))
    lsdir4 = os.listdir(os.path.join(dataset_path, folder, parts3[0]))
    with open(os.path.join(dataset_path, folder, label_name)) as label_file:
      lines = label_file.readlines()
    if not len(lsdir0)==len(lines) and len(lsdir1)==len(lines) and len(lsdir2)==len(lines) and len(lsdir3)==len(lines) and len(lsdir0)==len(lsdir4):
      print 'error in folder', folder    

def check_label():
  dir_list = os.listdir(dataset_path)
  for folder in dir_list:
    label_count = [0 for _ in range(nums)]
    with open(os.path.join(dataset_path, folder, label_name)) as label_file:
      lines = label_file.readlines()
    for line in lines:
      label = int(line.strip().split()[1])
      label_count[label] += 1
    if label_count[16] != 0:
      print folder
      print label_count
      


if __name__ == '__main__':
  #check_file()
  check_label()
