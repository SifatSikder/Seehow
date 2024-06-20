import os
import sys
import cv2
import glob
from tqdm import tqdm

dataset_path = '/home/cheer/Project/video_test/action_data'
parts = ['T0', 'T1']

def check():
  dir_list = os.listdir(dataset_path)
  for folder in glob.glob(os.path.join(dataset_path, 'java*')):
    image_list = os.listdir(os.path.join(dataset_path, folder, parts[0]))
    print folder
    for image_file in tqdm(image_list):
      image = cv2.imread(os.path.join(dataset_path, folder, parts[0], image_file))
      image = cv2.imread(os.path.join(dataset_path, folder, parts[1], image_file))
      with open(os.path.join(dataset_path, folder, 'label.txt')) as label_file:
        labels = label_file.readlines()

if __name__ == '__main__':
  check()
