import sys
sys.path.append('lib')
import torch
import difflib
import os
import re
import json
import webvtt
import cv2
import bbox
import skimage
import numpy as np
from skimage.metrics import structural_similarity as compare_ssim
from lib.feature_extractor.feature_extractor import FeatureExtractor
from tqdm import tqdm
from paddleocr import PaddleOCR
from transformers import BertTokenizer, BertForNextSentencePrediction

def rectify(vtt_file_path):
  vtt_file_dir = os.path.dirname(vtt_file_path)
  with open(vtt_file_path, 'r') as vtt_file:
    subtitles = vtt_file.readlines()
  i = 0
  for _ in subtitles:
    if re.search('position:0%', subtitles[i]) and len(subtitles[i+1].strip()) == 0 and len(subtitles[i+2].strip()) > 0:
      del subtitles[i+1]
    i += 1
    if i-2 > len(subtitles):
      break
  with open(os.path.join(vtt_file_dir, 'tmp.en.vtt'), 'w') as vtt_file:
    vtt_file.writelines(subtitles)

def parse_vtt(vtt_file_path,caption_path):
  vtt_file_dir = os.path.dirname(vtt_file_path)
  file_name = os.path.join(vtt_file_dir, 'tmp.en.vtt')
  caption_list = []
  merge_list = []
  for caption in webvtt.read(file_name):
    start = int(caption.start.split(':')[0])*3600 + int(caption.start.split(':')[1])*60 + int(float(caption.start.split(':')[2]))
    end = int(caption.end.split(':')[0])*3600 + int(caption.end.split(':')[1])*60 + int(float(caption.end.split(':')[2]))
    if len(caption.text.split('\n')) > 1:
      text = caption.text.split('\n')[1]
    else:
      text = caption.text.split('\n')[0]
    if end > start:
      caption_list.append(f'{start} {end} {text}\n')
  i = 1
  for caption in caption_list:
    start = caption.split()[0]
    end = caption.split()[1]
    text = ' '.join(caption.split()[2:])
    if i % 2 == 0:
      text_two = text_two + ' ' + text
      merge_list.append(f'{start_old} {end} {text_two}\n')
    else:
      start_old = start
      text_two = text
    i += 1       
  make_dir(caption_path)
  with open(caption_path + 'captions.txt', 'w') as caption_file:
    caption_file.writelines(merge_list)

def make_dir(path):
  if not os.path.exists(path):
    print('Creating folder {}'.format(path))
    os.makedirs(path)
  else:
    print('Folder {} already exists'.format(path))
    
def extract_images(video_path, images_path):
    make_dir(images_path)
    cmd = f'ffmpeg -i "{video_path}" -q:v 1 -r 1 "{images_path}/%05d.jpg"'
    os.system(cmd)

def extract_caption(vtt_file_path,caption_path):
    rectify(vtt_file_path)
    parse_vtt(vtt_file_path,caption_path)

def compare_frame(frameA, frameB):
  grayA = cv2.cvtColor(frameA, cv2.COLOR_BGR2GRAY)
  grayB = cv2.cvtColor(frameB, cv2.COLOR_BGR2GRAY)
  score, diff = compare_ssim(grayA, grayB, full=True)
  diff = (diff * 255).astype("uint8")
  thresh = cv2.threshold(diff, 180, 255, cv2.THRESH_BINARY_INV)[1]
  cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
  cnts = cnts[0]
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
    box_max = np.zeros((7), dtype = np.int32)
    return box_max
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

def extract_annotations(images_path,annotations_path):
    box_list = []
    image_list = os.listdir(os.path.join(images_path))
    image_old = cv2.imread(os.path.join(images_path,image_list[0]))
    for image_file in image_list:
        image = cv2.imread(os.path.join(images_path, image_file))
        diff, thresh, cnts, score= compare_frame(image_old, image)
        boxes = convert_box(cnts)
        boxes_nms = non_max_suppression(boxes, 0.3)
        max_box = find_max(boxes_nms)
        image_old = image.copy()
        box_list.append(f'{image_file.split('.')[0]} {max_box[0]} {max_box[1]} {max_box[2]} {max_box[3]} {max_box[5]} {max_box[6]}\n')
    make_dir(annotations_path)
    with open(os.path.join(annotations_path + 'annotations.txt'), 'w') as annotation_file:
        annotation_file.writelines(box_list)

def classification_placeholder_input(feature_extractor, image_a, image_b, logits_name, batch_size):

  batch_image1 = np.zeros([batch_size, feature_extractor.image_size, feature_extractor.image_size, 3], dtype=np.float32)
  batch_image2 = np.zeros([batch_size, feature_extractor.image_size, feature_extractor.image_size, 3], dtype=np.float32)

  for i in range(batch_size):
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

  predictions = np.argmax(predictions, axis=1)
  return predictions

def extract_action(images_path,annotations_path,model_path,actions_path):
  nums = 19
  batch_size = 1
  net_name = 'action_vgg_e'
  input_mode = 0
  output_mode = 0
  feature_extractor = FeatureExtractor(
    network_name=net_name,
    input_mode = input_mode,
    output_mode = output_mode,
    checkpoint_path=model_path,
    batch_size=batch_size,
    num_classes=nums,
    preproc_func_name=net_name)
  feature_extractor.print_network_summary()

  annotations = os.listdir(annotations_path)
  for annotation in annotations:
    with open(os.path.join(annotations_path, annotation), 'r') as annotation_file:
      annotation_list = annotation_file.readlines()
    image1 = skimage.io.imread(os.path.join(images_path, annotation_list[0].split()[0] + '.jpg'))
    action_list = []
    for line in tqdm(annotation_list):
      image_name = line.split()[0] + '.jpg'
      box = [int(x) for x in line.split()[1:5]]
      image2 = skimage.io.imread(os.path.join(images_path, image_name))
      if sum(box):
        diff = (box[3] - box[1]) * (box[2] - box[0]) / (image2.shape[0] * image2.shape[1])
        image1_clip = image1[box[1]:box[3], box[0]:box[2], :]
        image2_clip = image2[box[1]:box[3], box[0]:box[2], :]
        image1_clip = skimage.transform.resize(image1_clip, (224, 224))
        image2_clip = skimage.transform.resize(image2_clip, (224, 224))
        clip_class = classification_placeholder_input(feature_extractor, skimage.img_as_ubyte(image1_clip), skimage.img_as_ubyte(image2_clip), net_name + '/fc8',batch_size)
        clip_class = clip_class[0] if clip_class[0] in range(6, 15) else 0
      else:
        clip_class = 0
      action_list.append(line.split()[0] + ' ' + str(clip_class) + '\n')
      image1 = image2.copy()
    make_dir(actions_path)
    with open(os.path.join(actions_path, 'actions.txt'), 'w') as action_file:
      action_file.writelines(action_list)

def convert_bbox_to_vertice(bbox):
  x_min = bbox[0][0]
  y_min = bbox[0][1]
  x_max = bbox[2][0]
  y_max = bbox[2][1]
  return {
      "x_min": x_min,
      "y_min": y_min,
      "x_max": x_max,
      "y_max": y_max
  }

def create_json_file(bboxes, texts,image_name):
  lines = []
  for i, (bbox, text) in enumerate(zip(bboxes, texts)):
    lines.append({"id": i,"text": text,"vertice": convert_bbox_to_vertice(bbox)})
  result= {"lines": lines}
  with open(f'{image_name}.json', 'w') as f:
    json.dump(result, f, indent=2)
  print(f'{image_name}.json is created...')

def extract_text(images_path,ocr_path):
  ocr = PaddleOCR(use_angle_cls=True, lang='en')
  bounding_boxes =[]
  texts =[]
  image_files = sorted([f for f in os.listdir(images_path) if f.endswith(('.png', '.jpg', '.jpeg'))])
  make_dir(ocr_path)
  for image_name in image_files:
    image_path = os.path.join(images_path, image_name)
    image = cv2.imread(image_path)
    result = ocr.ocr(image)
    for line in result:
      if line is not None:
        for word in line:
          text = word[1][0]
          bbox = word[0]
          bounding_boxes.append(bbox)
          texts.append(text)
    create_json_file(bounding_boxes, texts,f'{ocr_path}/{image_name[:5]}')

def compute_action_overlap(y1, y2, y3, y4):
  if y3 > y2 or y1 > y4:
    return 0
  else:
    y_list = np.array([y1, y2, y3, y4])
    y_list = np.sort(y_list)
    return (y_list[2] - y_list[1]) / (y_list[3] - y_list[0])

def find_clip(action_overlaps):
  clips = []
  if len(action_overlaps):
    for i in range(len(action_overlaps)):
      if action_overlaps[i]['overlap'] < 0.1:
        start = action_overlaps[i]['id']
        end = action_overlaps[i]['id']
      if i < len(action_overlaps) - 1 and action_overlaps[i+1]['overlap'] < 0.1:
        end = action_overlaps[i]['id']
        action = action_overlaps[i]['action']
        code = action_overlaps[i]['code']
        clips.append({'clip': [start, end], 'action': action, 'code': code})
    end = action_overlaps[i]['id']
    action = action_overlaps[i]['action']
    code = action_overlaps[i]['code']
    clips.append({'clip': [start, end], 'action': action, 'code': code})
  return clips  

def rectify_workflow_captions(i, captions):
  if i > int(captions[-1].split()[0]):
    return int(captions[-1].split()[0]), int(captions[-1].split()[1])
  elif i < int(captions[0].split()[1]):
    return int(captions[0].split()[0]), int(captions[0].split()[1])
  for caption in captions:
    if i in range(int(caption.split()[0]), int(caption.split()[1])):
      return int(caption.split()[0]), int(caption.split()[1])

def merge_two(clip1, clip2):
  clip = {}
  clip['frame'] = [clip1['frame'][0], clip2['frame'][1]]
  clip['line'] = [clip1['line'][0], clip2['line'][1]]
  clip['caption'] = clip1['caption'] + ' ' + clip2['caption']
  return clip

def split_sentence(s):
  S_LEN = 128
  s_list = []
  for i in range(int(len(s.split()) / S_LEN) + 1):
    s_list.append(' '.join(s.split()[i * S_LEN:(i+1) * S_LEN]))
  return s_list

def config_model(model_mode):
  tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
  model = BertForNextSentencePrediction.from_pretrained('bert-base-uncased')
  if model_mode=='cpu' : model.to('cpu')
  else:
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    model.to('cuda')
  return tokenizer, model

def compute_next_sentence(s1, s2,model_mode):

  tokenizer, model = config_model(model_mode)
  s1_list = split_sentence(s1)
  s2_list = split_sentence(s2)
  scores = [] 
  for s1_clip in s1_list:
    for s2_clip in s2_list:
      token_list = tokenizer.encode(s1_clip + ',' + s2_clip, add_special_tokens = True)
      input_ids = torch.tensor(token_list).unsqueeze(0)
      segments_ids = [0] * (token_list.index(1010) + 1) + [1] * (len(token_list) - token_list.index(1010) - 1)
      segments_tensors = torch.tensor([segments_ids])
      outputs = model(input_ids, token_type_ids=segments_tensors)
      seq_relationship_scores = outputs[0]
      scores.append(seq_relationship_scores[0][0].detach().cpu())
  scores = np.array(scores, dtype = float)
  return np.mean(scores)

def group_clip(clips,model_mode):
  for i in range(len(clips)):
    if clips[i]['flag'] == 0:
      if i == 0:
        clips[1].update(merge_two(clips[0], clips[1]))
      elif i < len(clips) - 1:
        next_1 = compute_next_sentence(clips[i-1]['caption'], clips[i]['caption'],model_mode)
        next_2 = compute_next_sentence(clips[i]['caption'], clips[i+1]['caption'],model_mode)
        if next_1 > next_2:
          clips[i-1].update(merge_two(clips[i-1], clips[i]))
        else:
          clips[i+1].update(merge_two(clips[i], clips[i+1]))
      else:
          clips[-2].update(merge_two(clips[-2], clips[-1]))
  new_clips = []
  for clip in clips:
    if clip['flag']:
      new_clips.append(clip)
  for i in range(len(new_clips)):
    if new_clips[i]['action'] == 'select' or new_clips[i]['action'] == 'deselect':
      if len(new_clips[i]['code']):
        for j in range(i):
          code_sim = difflib.SequenceMatcher(None, new_clips[i]['code'], new_clips[j]['code']).quick_ratio()
          if code_sim > 0.9:
            new_clips[i]['parent'] = str(j + 1)
            #print (i, new_clips[i]['frame'], new_clips[i]['code'], j, new_clips[j]['frame'], new_clips[j]['code']) 
  return new_clips
        
def merge_caption(clips, captions):
  clip_list = []
  for clip in clips:
    start_index = end_index = len(captions) - 1
    for i in range(len(captions)):
      if clip['clip'][0] in range(int(captions[i].split()[0]), int(captions[i].split()[1])):
        start_index = i
      if clip['clip'][1] in range(int(captions[i].split()[0]), int(captions[i].split()[1])):
        end_index = i
    text_1 = ' '.join(list(map(lambda x: ' '.join(x.split()[2:]), captions[start_index:end_index + 1])))
    if len(clip_list) == 0 and start_index > 0:
      text_0 = ' '.join(list(map(lambda x: ' '.join(x.split()[2:]), captions[0:start_index])))
      clip_list.append({'frame': [0, clip['clip'][0]], 'line': [0, start_index], 'caption': text_0, 'flag': 0})
    elif len(clip_list) == 0 and start_index == 0:
      clip_list.append({'frame': clip['clip'], 'key_frame': clip['clip'], 'line': [start_index, end_index + 1], 'caption': text_1, 'flag': 1, 'action': clip['action'], 'code': clip['code'], 'parent': None})
    else:
      text_0 = ' '.join(list(map(lambda x: ' '.join(x.split()[2:]), captions[clip_list[-1]['line'][1]:start_index])))
      clip_list.append({'frame': [clip_list[-1]['frame'][1], clip['clip'][0]], 'line': [clip_list[-1]['line'][1], start_index], 'caption': text_0, 'flag': 0})
    clip_list.append({'frame': clip['clip'], 'key_frame': clip['clip'], 'line': [start_index, end_index + 1], 'caption': text_1, 'flag': 1, 'action': clip['action'], 'code': clip['code'], 'parent': None})
  if len(clip_list) == 0:
    return clip_list
  elif clip_list[-1]['line'][1] < len(captions) - 1:
    text_0 = ' '.join(list(map(lambda x: ' '.join(x.split()[2:]), captions[clip_list[-1]['line'][1]:len(captions)])))
    clip_list.append({'frame': [clip_list[-1]['frame'][1], int(captions[-1].split()[1])], 'line': [clip_list[-1]['line'][1], len(captions)], 'caption': text_0, 'flag': 0})
  return clip_list

def compute_clip(data_dir, actions, annotations, captions):
  label_dict = {'0':'others', '6':'enter_text', '7':'enter_text_popup_a', '8':'enter_text_popup_d', '9':'delete', '10':'popup', '11':'select', '12':'deselect', '13':'scroll', '14':'switch', '15':'enter'}
  action_overlaps = []
  old_ocr_box = bbox.BBox2D([0, 0, 0, 0], mode = bbox.XYXY)
  for i in range(len(actions)):
    action_box = bbox.BBox2D([int(annotations[i].split()[1]), int(annotations[i].split()[2]), int(annotations[i].split()[3]), int(annotations[i].split()[4])], mode = bbox.XYXY)
    s, e = rectify_workflow_captions(i, captions)
    with open(os.path.join(data_dir, 'OCR', actions[e-1].split()[0] + '.json'), 'r') as json_file:
      json_data = json.load(json_file)
    ocr_boxes = []
    ious = []
    codelines = []
    for line in json_data['lines']:
      ocr_boxes.append(bbox.BBox2D([line['vertice']['x_min'], line['vertice']['y_min'], line['vertice']['x_max'], line['vertice']['y_max']], mode = bbox.XYXY))
      codelines.append(line['text'])
    for ocr_box in ocr_boxes:
      ious.append(bbox.metrics.jaccard_index_2d(action_box, ocr_box))
    ious = np.array(ious)
    max_iou_index = np.argmax(ious)
    if ious[max_iou_index] > 0.05:
      if  actions[i].split()[1] == '6' or actions[i].split()[1] == '9' or actions[i].split()[1] == '11' or actions[i].split()[1] == '12':
        action_overlap = compute_action_overlap(int(ocr_boxes[max_iou_index].y1), int(ocr_boxes[max_iou_index].y2), int(old_ocr_box.y1), int(old_ocr_box.y2))
        old_ocr_box = ocr_boxes[max_iou_index]
        action_overlaps.append({'overlap': action_overlap, 'id': i, 'action': label_dict[actions[i].split()[1]], 'code': codelines[max_iou_index]})
  clips = find_clip(action_overlaps)
  return clips

def action_annotation_caption_reader (data_dir):
  with open(os.path.join(data_dir, 'Actions',  'actions.txt'), 'r') as action_file:
    actions = action_file.readlines()
  with open(os.path.join(data_dir, 'Annotations', 'annotations.txt'), 'r') as annotation_file:
    annotations = annotation_file.readlines()
  with open(os.path.join(data_dir, 'Captions', 'captions.txt'), 'r') as caption_file:
    captions = caption_file.readlines()
  return actions, annotations, captions

def save_to_file(clips,data_dir, workflow_filename):
    data_list = []
    for i, clip in enumerate(clips):
        fragment = ','.join(list(map(str, clip['frame'])))
        key_frame = '{:05},{:05}'.format(clip['key_frame'][0] + 1, clip['key_frame'][1] + 1)
        data = {
            'Caption': clip['caption'],
            'Step': i + 1,
            'Frame': key_frame,
            'Fragment': fragment,
            'Code': clip.get('code', ''),
            'Action': clip.get('action', ''),
            'Parent': clip.get('parent', '')
        }
        data_list.append(data)
    output_path = os.path.join(data_dir, workflow_filename)
    with open(output_path, 'w') as f:
        json.dump(data_list, f, indent=4)
    print(f"Data saved to {output_path}")

def extract_workflow(model_mode,data_dir,workflow_filename) :
  print('Workflow Generation Started...')
  actions, annotations, captions = action_annotation_caption_reader(data_dir)
  clips = compute_clip(data_dir, actions, annotations, captions)
  clips = merge_caption(clips, captions)
  clips = group_clip(clips,model_mode)
  save_to_file(clips,data_dir,workflow_filename)
  print('Workflow Generation Finished...')

def start():
  data_dir = 'Test_Dataset'
  video_path = f'{data_dir}/Videos/Android Application Development Tutorial - 12 - Setting up an Activity and Using SetContentView.mkv'
  vtt_file_path = f'{data_dir}/Videos/Android Application Development Tutorial - 12 - Setting up an Activity and Using SetContentView.en.vtt'
  images_path = f'{data_dir}/Images'
  caption_path = f'{data_dir}/Captions'
  annotations_path = f'{data_dir}/Annotations'
  actions_path = f'{data_dir}/Actions'
  model_path = f'{data_dir}/Model/model.ckpt-10'
  ocr_path = f'{data_dir}/OCR'
  model_mode='cpu'
  workflow_filename = 'workflow.json'
  # extract_images(video_path,images_path)
  # extract_caption(vtt_file_path,caption_path)
  # extract_annotations(images_path,annotations_path)
  # extract_action(images_path,annotations_path,model_path,actions_path)
  # extract_text(images_path,ocr_path)
  extract_workflow(model_mode,data_dir,workflow_filename)

if __name__ == '__main__':
  start()