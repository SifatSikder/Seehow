import os
import glob
import cv2
import numpy as np
from tqdm import tqdm
import xml.etree.ElementTree as ET

dataset_path = '/home/cheer/Project/video_test/action_data'
input_parts = ['t2', 't3']
parts = ['T2', 'T3', 'T4', 'T5', 'T6', 'T7', 'T8', 'T9', 'T10', 'T11', 'T12', 'T13', 'T14']
size = 224

'''
T0 T1: 299*299, RGB, diff
T2 T3: 224*224, RGB, diff, black BG
T4 T5: 224*224, graysacle, diff
T6 T7: 224*224, RGB, full
T8 T9: 224*224, grayscale, full
T10 T11: 224*224, RGB, diff
T12: 224*224, absolute, full
T13: 224*224, optical flow, full
T14: 224*224, diff optical flow
'''


def create_folder(output_path):
  for part in parts:
    if not os.path.exists(os.path.join(output_path, part)):
      print 'create folder {}'.format(part)
      os.makedirs(os.path.join(output_path, part))
    else:
      print 'folder {} exist'.format(part)

def optical_flow(image_a, image_b):
  hsv = np.zeros_like(image_a)
  hsv[...,1] = 255
  flow = cv2.calcOpticalFlowFarneback(cv2.cvtColor(image_a,cv2.COLOR_BGR2GRAY), cv2.cvtColor(image_b,cv2.COLOR_BGR2GRAY), None, 0.5, 3, 15, 3, 5, 1.2, 0)
  mag, ang = cv2.cartToPolar(flow[...,0], flow[...,1])
  hsv[...,0] = ang*180/np.pi/2
  hsv[...,2] = cv2.normalize(mag,None,0,255,cv2.NORM_MINMAX)
  bgr = cv2.cvtColor(hsv,cv2.COLOR_HSV2BGR)
  image_flow = cv2.resize(bgr, (size, size))
  return image_flow

def absolute(image_a, image_b):
  image_absolute = np.absolute(np.array(image_a) - np.array(image_b))
  image_absolute = cv2.resize(image_absolute, (size, size))
  return image_absolute

def resize(image_a, image_b):
  image_a_resize = cv2.resize(image_a, (size, size))
  image_b_resize = cv2.resize(image_b, (size, size))
  return image_a_resize, image_b_resize

def grayscale(image_a, image_b):
  image_a_grayscale = cv2.cvtColor(image_a, cv2.COLOR_BGR2GRAY)
  image_b_grayscale = cv2.cvtColor(image_b, cv2.COLOR_BGR2GRAY)
  image_a_grayscale, image_b_grayscale = resize(image_a_grayscale, image_b_grayscale)
  return image_a_grayscale, image_b_grayscale

def extract_diff(name, image, image_a, image_b):
  annofile = open(os.path.join(name, 'Annotation', os.path.splitext(image)[0] + '.xml'))
  tree = ET.parse(annofile)
  root = tree.getroot()
  for max_region in root.iter('max_region'):
    max_reg = [int(max_region.find('xmin').text),int(max_region.find('xmax').text),int(max_region.find('ymin').text),int(max_region.find('ymax').text)]
  image_a_max = image_a[max_reg[2]:max_reg[3], max_reg[0]:max_reg[1]]
  image_b_max = image_b[max_reg[2]:max_reg[3], max_reg[0]:max_reg[1]]
  image_a_max, image_b_max = resize(image_a_max, image_b_max)
  return image_a_max, image_b_max

def diff_flow(name, image, image_a, image_b):
  hsv = np.zeros_like(image_a)
  hsv[...,1] = 255
  flow = cv2.calcOpticalFlowFarneback(cv2.cvtColor(image_a,cv2.COLOR_BGR2GRAY), cv2.cvtColor(image_b,cv2.COLOR_BGR2GRAY), None, 0.5, 3, 15, 3, 5, 1.2, 0)
  mag, ang = cv2.cartToPolar(flow[...,0], flow[...,1])
  hsv[...,0] = ang*180/np.pi/2
  hsv[...,2] = cv2.normalize(mag,None,0,255,cv2.NORM_MINMAX)
  bgr = cv2.cvtColor(hsv,cv2.COLOR_HSV2BGR)
  annofile = open(os.path.join(name, 'Annotation', os.path.splitext(image)[0] + '.xml'))
  tree = ET.parse(annofile)
  root = tree.getroot()
  for max_region in root.iter('max_region'):
    max_reg = [int(max_region.find('xmin').text),int(max_region.find('xmax').text),int(max_region.find('ymin').text),int(max_region.find('ymax').text)]
  diff_flow = bgr[max_reg[2]:max_reg[3], max_reg[0]:max_reg[1]]
  diff_flow = cv2.resize(diff_flow, (size, size))
  return diff_flow


def run():
  for name in glob.glob(os.path.join(dataset_path, '*')):
    create_folder(name)
    image_list = os.listdir(os.path.join(name, input_parts[0]))
    print os.path.split(name)[1]
    for image in tqdm(image_list):
      image_a = cv2.imread(os.path.join(name, input_parts[0], image))
      image_b = cv2.imread(os.path.join(name, input_parts[1], image))

      diff_image_flow = diff_flow(name, image, image_a, image_b)
      cv2.imwrite(os.path.join(name, parts[12], os.path.split(name)[1] + '_{}'.format(image)), diff_image_flow)

'''      
      # extract max diff region
      image_a_diff, image_b_diff = extract_diff(name, image, image_a, image_b)
      cv2.imwrite(os.path.join(name, parts[8], os.path.split(name)[1] + '_{}'.format(image)), image_a_diff)
      cv2.imwrite(os.path.join(name, parts[9], os.path.split(name)[1] + '_{}'.format(image)), image_b_diff)

      #
      image_a_diff_gray, image_b_diff_gray = grayscale(image_a_diff, image_b_diff)
      cv2.imwrite(os.path.join(name, parts[2], os.path.split(name)[1] + '_{}'.format(image)), image_a_diff_gray)
      cv2.imwrite(os.path.join(name, parts[3], os.path.split(name)[1] + '_{}'.format(image)), image_b_diff_gray)

      image_absolute = absolute(image_a, image_b)
      cv2.imwrite(os.path.join(name, parts[10], os.path.split(name)[1] + '_{}'.format(image)), image_absolute)
  
      image_flow = optical_flow(image_a, image_b)
      cv2.imwrite(os.path.join(name, parts[11], os.path.split(name)[1] + '_{}'.format(image)), image_flow)
'''
      #cv2.imwrite(os.path.join(name, parts[4], os.path.split(name)[1] + '_{}'.format(image)), image_a_resize)
      #cv2.imwrite(os.path.join(name, parts[5], os.path.split(name)[1] + '_{}'.format(image)), image_b_resize)
      #cv2.imwrite(os.path.join(name, parts[6], os.path.split(name)[1] + '_{}'.format(image)), image_a_grayscale)
      #cv2.imwrite(os.path.join(name, parts[7], os.path.split(name)[1] + '_{}'.format(image)), image_b_grayscale)




if __name__ == '__main__':
  run()
