from skimage.measure import compare_ssim
import imutils
import cv2
import numpy as np
import math
import os
import time
import multiprocessing
import gc
from scipy import misc
from xml.dom import minidom

video_path = '/home/cheer/Project/video_test/videos'
dataset_path = '/home/cheer/Project/video_test/action_data'
parts = ['t0', 't1', 't2', 't3', 'Annotation']
train_size = 299
error_list = ['java_4_0.mkv', 'python_3_0.mp4', 'python_4_2.mkv', 'python_4_3.mp4', 'python_1_1.webm', 'java_0_2.webm', 'java_4_1.mkv', 'java_4_3.mkv']

def nothing(emp):
  pass

def make_dirs(video_name):
  folder_name = os.path.splitext(video_name)[0]
  for part in parts:
    if not os.path.exists(os.path.join(dataset_path, folder_name, part)):
      print 'create folder {}'.format(folder_name)
      os.makedirs(os.path.join(dataset_path, folder_name, part))
    else:
      print 'folder {} exist'.format(folder_name)

def find_max(boxes_nms):
  if len(boxes_nms) == 0:
    return []
  boxes = []
  for box_nms in boxes_nms:
    box_nms = np.append(box_nms, (box_nms[2]-box_nms[0])*(box_nms[3]-box_nms[1]))
    boxes.append(box_nms)
  boxes = np.array(boxes)
  idx = np.argsort(boxes[:,4])
  x_center = boxes[idx[-1]][0] + (boxes[idx[-1]][2] - boxes[idx[-1]][0]) / 2
  y_center = boxes[idx[-1]][1] + (boxes[idx[-1]][3] - boxes[idx[-1]][1]) / 2
  box_max = np.append(boxes[idx[-1]], [x_center, y_center])
  box_max = np.array(box_max, dtype = np.int32)
  return box_max

def non_max_suppression(boxes, overlapThresh):
  if len(boxes) == 0:
    return []
  if boxes.dtype.kind == "i":
    boxes = boxes.astype("float")

  pick = []
  x1 = boxes[:,0]
  y1 = boxes[:,1]
  x2 = boxes[:,2]
  y2 = boxes[:,3]
  area = (x2 - x1 + 1) * (y2 - y1 + 1)
  idxs = np.argsort(y2)

  while len(idxs) > 0:
    last = len(idxs) - 1
    i = idxs[last]
    pick.append(i)
    xx1 = np.maximum(x1[i], x1[idxs[:last]])
    yy1 = np.maximum(y1[i], y1[idxs[:last]])
    xx2 = np.minimum(x2[i], x2[idxs[:last]])
    yy2 = np.minimum(y2[i], y2[idxs[:last]])
    w = np.maximum(0, xx2 - xx1 + 1)
    h = np.maximum(0, yy2 - yy1 + 1)
    overlap = (w * h) / area[idxs[:last]] 
    idxs = np.delete(idxs, np.concatenate(([last], np.where(overlap > overlapThresh)[0])))

  return boxes[pick].astype("int")

def compare_frame(frameA, frameB):
  grayA = cv2.cvtColor(frameA, cv2.COLOR_BGR2GRAY)
  grayB = cv2.cvtColor(frameB, cv2.COLOR_BGR2GRAY)

  score, diff = compare_ssim(grayA, grayB, full=True)
  diff = (diff * 255).astype("uint8")
  print("SSIM: {}".format(score))

  thresh = cv2.threshold(diff, 180, 255, cv2.THRESH_BINARY_INV)[1]
  cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
  cnts = cnts[0] if imutils.is_cv2() else cnts[1]

  return diff, thresh, cnts, score

def convert_box(cnts):
  box = []
  for c in cnts:
    x, y, w, h = cv2.boundingRect(c)
    if w > 6 and h > 15: 
      box.append([x, y, x+w, y+h])
  box = np.array(box)
  return box

def find_max(boxes_nms):
  if len(boxes_nms) == 0:
    return []
  boxes = []
  for box_nms in boxes_nms:
    box_nms = np.append(box_nms, (box_nms[2]-box_nms[0])*(box_nms[3]-box_nms[1]))
    boxes.append(box_nms)
  boxes = np.array(boxes)
  idx = np.argsort(boxes[:,4])
  x_center = boxes[idx[-1]][0] + (boxes[idx[-1]][2] - boxes[idx[-1]][0]) / 2
  y_center = boxes[idx[-1]][1] + (boxes[idx[-1]][3] - boxes[idx[-1]][1]) / 2
  box_max = np.append(boxes[idx[-1]], [x_center, y_center])
  box_max = np.array(box_max, dtype = np.int32)
  return box_max

def find_max_region(boxes_nms):
  if len(boxes_nms) == 0:
    return []
  x1 = boxes_nms[:,0]
  y1 = boxes_nms[:,1]
  x2 = boxes_nms[:,2]
  y2 = boxes_nms[:,3]  
  xx1 = min(x1)
  yy1 = min(y1)
  xx2 = max(x2)
  yy2 = max(y2)
  max_region = np.array([xx1, yy1, xx2, yy2])
  return max_region

def to_canvas(region):
  canvas = np.zeros((train_size, train_size, 3), np.uint8)
  if region.shape[0] > region.shape[1]:
    if region.shape[1] % 2:
      canvas[:, train_size / 2 - (region.shape[1] / 2):train_size / 2 + (region.shape[1] / 2) + 1] = region
    else:
      canvas[:, train_size / 2 - (region.shape[1] / 2):train_size / 2 + (region.shape[1] / 2)] = region
  else:
    if region.shape[0] % 2:
      canvas[train_size / 2 - (region.shape[0] / 2):train_size / 2 + (region.shape[0] / 2) + 1, :] = region
    else:
      canvas[train_size / 2 - (region.shape[0] / 2):train_size / 2 + (region.shape[0] / 2), :] = region
  return canvas

def add_box(doc, new_annotation):
    annotation = doc.documentElement
    bndbox = doc.createElement(new_annotation['name'])
    xmin = doc.createElement('xmin')
    xmin.appendChild(doc.createTextNode(new_annotation['xmin']))
    ymin = doc.createElement('ymin')
    ymin.appendChild(doc.createTextNode(new_annotation['ymin']))
    xmax = doc.createElement('xmax')
    xmax.appendChild(doc.createTextNode(new_annotation['xmax']))
    ymax = doc.createElement('ymax')
    ymax.appendChild(doc.createTextNode(new_annotation['ymax']))
    bndbox.appendChild(xmin)
    bndbox.appendChild(ymin)
    bndbox.appendChild(xmax)
    bndbox.appendChild(ymax)
    annotation.appendChild(bndbox)    
    return doc

def add_image_information(doc, new_information):
    annotation = doc.createElement("annotation")
    doc.appendChild(annotation)
    filename = doc.createElement('videoname')
    filename.appendChild(doc.createTextNode(new_information['videoname']))   
    annotation.appendChild(filename)
    box = doc.createElement("box")
    pos = doc.createElement('pos')
    pos.appendChild(doc.createTextNode(new_information['pos'])) 
    box.appendChild(pos)  
    annotation.appendChild(box)
    return doc

def start():
  video_name = q.get()
  make_dirs(video_name)
  print 'processing video', video_name
  video = os.path.join(video_path, video_name)
  folder_name = os.path.splitext(video_name)[0]
  box_o = [0,0,0,0]
  cv2.namedWindow(video_name, cv2.WINDOW_NORMAL)
  cap = cv2.VideoCapture(video)
  width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
  height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
  frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
  cv2.createTrackbar('time', video_name, 0, frames, nothing)
  loop_flag = 0
  pos = 0
  if cap.isOpened():
    ret, frameA = cap.read()
  while(cap.isOpened()):
    if loop_flag == pos:
      loop_flag = loop_flag + 1
      cv2.setTrackbarPos('time', video_name, loop_flag)
    else:
      pos = cv2.getTrackbarPos('time', video_name)
      loop_flag = pos
      cap.set(cv2.CAP_PROP_POS_FRAMES, pos)
    ret, frameB = cap.read()
    try:
      diff, thresh, cnts, score= compare_frame(frameA, frameB)
    except:
      break
    frameC = frameB.copy()
    frameD = frameB.copy()
    mask_region = np.zeros((frameB.shape[0], frameB.shape[1]), np.uint8)
    boxes = convert_box(cnts)
    boxes_nms = non_max_suppression(boxes, 0.3)
    max_box = find_max(boxes_nms)
    max_region_box = find_max_region(boxes_nms)

    for box in boxes_nms:
      cv2.rectangle(mask_region, (box[0], box[1]), (box[2], box[3]), (255, 255, 255), -1)

    if len(boxes_nms):
      cv2.rectangle(frameC, (max_region_box[0], max_region_box[1]), (max_region_box[2], max_region_box[3]), (0, 255, 0), 2)
      max_region_A = frameA[max_region_box[1]:max_region_box[3], max_region_box[0]:max_region_box[2]].copy()
      mask_region = mask_region[max_region_box[1]:max_region_box[3], max_region_box[0]:max_region_box[2]].copy()
      max_region_D = frameD[max_region_box[1]:max_region_box[3], max_region_box[0]:max_region_box[2]].copy()
      region_A = cv2.bitwise_and(max_region_A, max_region_A, mask = mask_region)
      region_D = cv2.bitwise_and(max_region_D, max_region_D, mask = mask_region) 
      f_rate = train_size * 1.0 / max(region_A.shape[0], region_A.shape[1])
      region_A = cv2.resize(region_A, (0,0), fx = f_rate, fy = f_rate) 
      region_D = cv2.resize(region_D, (0,0), fx = f_rate, fy = f_rate) 
      img_A = to_canvas(region_A)
      img_D = to_canvas(region_D)
      #cv2.imwrite(os.path.join(dataset_path, folder_name, parts[0], '{:05}'.format(pos) + '.jpg'), img_A)  
      #cv2.imwrite(os.path.join(dataset_path, folder_name, parts[1], '{:05}'.format(pos) + '.jpg'), img_D)
      #cv2.imwrite(os.path.join(dataset_path, folder_name, parts[2], '{:05}'.format(pos) + '.jpg'), frameA)  
      #cv2.imwrite(os.path.join(dataset_path, folder_name, parts[3], '{:05}'.format(pos) + '.jpg'), frameD)
      doc = minidom.Document()
      doc = add_image_information(doc, {'videoname':folder_name, 'pos':'{:05}'.format(pos)})
      for bbox in boxes_nms:
        doc = add_box(doc, {'name':'bndbox','xmin':str(bbox[0]),'ymin':str(bbox[1]),'xmax':str(bbox[2]),'ymax':str(bbox[3])})
      doc = add_box(doc, {'name':'max_box','xmin':str(max_box[0]),'ymin':str(max_box[1]),'xmax':str(max_box[2]),'ymax':str(max_box[3])})
      doc = add_box(doc, {'name':'max_region','xmin':str(max_region_box[0]),'ymin':str(max_region_box[1]),'xmax':str(max_region_box[2]),'ymax':str(max_region_box[3])})
      with open(os.path.join(dataset_path, folder_name, 'Annotation', '{:05}'.format(pos) + '.xml'), 'w') as f_annotation:
        f_annotation.write(doc.toprettyxml(indent = "\t", newl = "\n", encoding = "utf-8"))
    cv2.imshow(video_name, frameC)
    frameA = frameB.copy()
    gc.collect()
    if cv2.waitKey(1) & 0xFF == ord('q'):
      break

  cap.release()
  cv2.destroyAllWindows()

if __name__ == '__main__':
  #video_list = os.listdir(video_path)
  video_list = []
  for e_name in error_list:
    video_list.append(os.path.join(video_path, e_name))
  q = multiprocessing.Queue()
  pool = multiprocessing.Pool(processes = 3)
  #processes_number = 5
  for video_name in video_list:
    q.put(video_name)
  qsize = q.qsize()
  for _ in range(qsize):
    pool.apply_async(start, ())
  pool.close()
  pool.join()
  print 'start processing...'
    
  #while q.qsize():
  #  processes = []
  #  for _ in range(processes_number):
  #    p = multiprocessing.Process(target = start, args = (q,))
  #    processes.append(p)
  #    p.start()
  #  for p in processes:
  #    p.join()
  cv2.destroyAllWindows()
