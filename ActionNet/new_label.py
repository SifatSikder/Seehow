import os
import sys
import cv2
import shutil
import numpy as np
import xml.etree.ElementTree as ET
from tqdm import tqdm
dataset_path = '/home/cheer/Project/video_test/action_data'
folder_name = 'python_0_4'
label_file = 'label.txt'
parts = ['T0', 'T1']

def create_folder(dataset_path):
  print folder_name
  for part in parts:
    if not os.path.exists(os.path.join(dataset_path, folder_name, part)):
      print 'create folder {}'.format(part)
      os.makedirs(os.path.join(dataset_path, folder_name, part))
    else:
      print 'folder {} exist'.format(part)

def read_i(dataset_path):
  if os.path.isfile(os.path.join(dataset_path, folder_name, label_file)):
    with open(os.path.join(dataset_path, folder_name, label_file), 'r') as label:
      lines = label.readlines()
      return len(lines)
  else:
    return 0

def read_label():
  if os.path.isfile(os.path.join(dataset_path, folder_name, label_file)):
    with open(os.path.join(dataset_path, folder_name, label_file), 'r') as label:
      lines = label.readlines()
      if len(lines) > 0:
        num = int(lines[-1].strip().split()[1])
        return num
      else:
        return 0
  else:
    return 0

def delete_file(i):
  with open(os.path.join(dataset_path, folder_name, label_file), 'r') as label:
    lines = label.readlines()
  del_line = lines.pop(-1)
  with open(os.path.join(dataset_path, folder_name, label_file), 'w') as label:
    label.writelines(lines)
  os.remove(os.path.join(dataset_path, folder_name, parts[0], del_line.split()[0]))
  os.remove(os.path.join(dataset_path, folder_name, parts[1], del_line.split()[0]))
  
def main(dataset_path):
  cv2.namedWindow('image', cv2.WINDOW_NORMAL)
  lsdir = os.listdir(os.path.join(dataset_path, folder_name, 't0'))
  lsdir.sort()
  i = read_i(dataset_path)
  class_num = 0
  num_buffer = []
  while i < (len(lsdir)):
    sys.stdout.write('reading image ' + str(lsdir[i]) + ' ' + str(i) + '/' + str(len(lsdir)) + '\r')
    sys.stdout.flush()
    image1 = cv2.imread(os.path.join(dataset_path, folder_name, 't0', lsdir[i]))
    image2 = cv2.imread(os.path.join(dataset_path, folder_name, 't1', lsdir[i]))
    image3 = cv2.imread(os.path.join(dataset_path, folder_name, 't2', lsdir[i]))
    image4 = cv2.imread(os.path.join(dataset_path, folder_name, 't3', lsdir[i]))

    annofile = open(os.path.join(dataset_path, folder_name, 'Annotation', os.path.splitext(lsdir[i])[0] + '.xml'))
    tree = ET.parse(annofile)
    root = tree.getroot()
    for box in root.iter('bndbox'):
      bnd = [int(box.find('xmin').text),int(box.find('xmax').text),int(box.find('ymin').text),int(box.find('ymax').text)]
      cv2.rectangle(image3, (bnd[0], bnd[2]), (bnd[1], bnd[3]), (0, 255, 0), 3)
      cv2.rectangle(image4, (bnd[0], bnd[2]), (bnd[1], bnd[3]), (0, 255, 0), 3)

    for max_box in root.iter('max_box'):
      max_bnd = [int(max_box.find('xmin').text),int(max_box.find('xmax').text),int(max_box.find('ymin').text),int(max_box.find('ymax').text)]
      cv2.rectangle(image3, (max_bnd[0], max_bnd[2]), (max_bnd[1], max_bnd[3]), (255, 0, 0), 3)
      cv2.rectangle(image4, (max_bnd[0], max_bnd[2]), (max_bnd[1], max_bnd[3]), (255, 0, 0), 3)

    for max_region in root.iter('max_region'):
      max_reg = [int(max_region.find('xmin').text),int(max_region.find('xmax').text),int(max_region.find('ymin').text),int(max_region.find('ymax').text)]
      cv2.rectangle(image3, (max_reg[0], max_reg[2]), (max_reg[1], max_reg[3]), (0, 0, 255), 3)
      cv2.rectangle(image4, (max_reg[0], max_reg[2]), (max_reg[1], max_reg[3]), (0, 0, 255), 3)

    image3 = cv2.resize(image3, (299, 299))
    image4 = cv2.resize(image4, (299, 299))
    image0 = np.zeros((603, 603, 3), np.uint8)
    image0[0:299, 0:299] = image1
    image0[0:299, 304:603] = image2
    image0[304:603, 0:299] = image3
    image0[304:603, 304:603] = image4

    k = cv2.waitKey(1)
    if k == ord('q'):
      break
    elif k == ord('n'):
      with open(os.path.join(dataset_path, folder_name, label_file), 'a') as label:
        label.write(folder_name + '_{} {}\n'.format(lsdir[i], class_num))
      cv2.imwrite(os.path.join(dataset_path, folder_name, parts[0], folder_name + '_{}'.format(lsdir[i])), image1)
      cv2.imwrite(os.path.join(dataset_path, folder_name, parts[1], folder_name + '_{}'.format(lsdir[i])), image2)
      num_buffer = []
      class_num = 0
      i += 1
    elif k == ord('b'):
      i -= 1
      class_num = read_label()
      delete_file(i)
    elif k >= ord('0') and k <= ord('9'):
      num_buffer.append(k-48)
      class_num = 0
      reverse_list = num_buffer[::-1]
      for j in range(len(reverse_list)):
        class_num += 10**j*reverse_list[j]
    elif k == ord('d'):
      num_buffer= []
      class_num = 0
    cv2.putText(image0, str(class_num), (50,50), cv2.FONT_HERSHEY_SIMPLEX, 2, (255,0,255), 3)
    cv2.imshow('image', image0)  
      
if __name__ == '__main__':
  create_folder(dataset_path)
  main(dataset_path)
