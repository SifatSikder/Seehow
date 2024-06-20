import math
import os
import random
import sys
import glob
import matplotlib.pyplot as plt

import tensorflow as tf

from lib import dataset_utils

_NUM_VALIDATION = 0

# Seed for repeatability.
_RANDOM_SEED = 0

# The number of shards per dataset split.
_NUM_SHARDS = 4

class_num = 19

dir_list = '*'
dataset_name = 'jp_2s'
seq = False

'''
'java_0_0', 'java_0_1', 'java_0_2', 'java_0_3', 'java_0_4','java_1_0', 'java_1_1', 'java_1_2', 'java_1_3', 'java_1_4','java_2_0', 'java_2_1', 'java_2_2', 'java_2_3', 'java_2_4','java_3_0', 'java_3_1', 'java_3_2', 'java_3_3', 'java_3_4','java_4_0', 'java_4_1', 'java_4_2', 'java_4_3', 'java_4_4'

'python_0_0', 'python_0_1', 'python_0_2', 'python_0_3', 'python_0_4','python_1_0', 'python_1_1', 'python_1_2', 'python_1_3', 'python_1_4','python_2_0', 'python_2_1', 'python_2_2', 'python_2_3', 'python_2_4','python_3_0', 'python_3_1', 'python_3_2', 'python_3_3', 'python_3_4','python_4_0', 'python_4_1', 'python_4_2', 'python_4_3', 'python_4_4'
'''

dataset_path = '/home/cheer/Project/video_test/action_data'
output_path = '/home/cheer/Project/video_test/action_merge'

'''
T0 T1: 299*299, RGB, diff
T2 T3: 224*224, RGB, diff, black BG   jp_black
T4 T5: 224*224, graysacle, diff       jp_gray
T6 T7: 224*224, RGB, full             jp_full
T8 T9: 224*224, grayscale, full       jp_full_gray
T10 T11: 224*224, RGB, diff           jp
T12: 224*224, absolute, full          jp_a_o 
T13: 224*224, optical flow, full
T14: 224*224, diff optical flow
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


label_name = 'label.txt'
output_dir = os.path.join(output_path, dataset_name)


class ImageReader(object):
  """Helper class that provides TensorFlow image coding utilities."""

  def __init__(self):
    # Initializes function that decodes RGB JPEG data.
    self._decode_jpeg_data = tf.placeholder(dtype=tf.string)
    self._decode_jpeg = tf.image.decode_jpeg(self._decode_jpeg_data, channels=3)

  def read_image_dims(self, sess, image_data):
    image = self.decode_jpeg(sess, image_data)
    return image.shape[0], image.shape[1]

  def decode_jpeg(self, sess, image_data):
    image = sess.run(self._decode_jpeg,
                     feed_dict={self._decode_jpeg_data: image_data})
    assert len(image.shape) == 3
    assert image.shape[2] == 3
    return image


def _get_filenames_and_classes(dataset_path):
  file_list = []
  for folder in glob.glob(os.path.join(dataset_path, dir_list)):
    with open(os.path.join(dataset_path, folder, label_name)) as list_file:
      file_list += list_file.readlines()
  return file_list


def _get_dataset_filename(output_dir, split_name, shard_id):
  output_filename = 'clip_%s_%05d-of-%05d.tfrecord' % (
      split_name, shard_id, _NUM_SHARDS)
  return os.path.join(output_dir, output_filename)


def _convert_dataset(split_name, filenames, dataset_path):
  """Converts the given filenames to a TFRecord dataset.

  Args:
    split_name: The name of the dataset, either 'train' or 'validation'.
    filenames: A list of absolute paths to png or jpg images.
    class_names_to_ids: A dictionary from class names (strings) to ids
      (integers).
    dataset_dir: The directory where the converted datasets are stored.
  """
  assert split_name in ['train', 'validation']

  if not os.path.exists(output_dir):
    os.makedirs(output_dir)

  img_files_a = []
  img_files_b = []
  class_names = []

  for filename in filenames:
    img_files_a.append(os.path.join(dataset_path, filename.split('_')[0] + '_' + filename.split('_')[1] + '_' + filename.split('_')[2], parts[0], filename.strip().split()[0]))
    img_files_b.append(os.path.join(dataset_path, filename.split('_')[0] + '_' + filename.split('_')[1] + '_' + filename.split('_')[2], parts[1], filename.strip().split()[0]))
    if int(filename.strip().split()[1]) < class_num:
      class_names.append(filename.strip().split()[1])
    else:
      print(filename)
      exit(0)

  num_per_shard = int(math.ceil(len(filenames) / float(_NUM_SHARDS)))

  with tf.Graph().as_default():
    image_reader = ImageReader()

    with tf.Session('') as sess:

      for shard_id in range(_NUM_SHARDS):
        output_filename = _get_dataset_filename(
            output_dir, split_name, shard_id)

        with tf.python_io.TFRecordWriter(output_filename) as tfrecord_writer:
          start_ndx = shard_id * num_per_shard
          end_ndx = min((shard_id+1) * num_per_shard, len(filenames))
          for i in range(start_ndx, end_ndx):
            sys.stdout.write('\r>> Converting image %d/%d shard %d' % (
                i+1, len(filenames), shard_id))
            sys.stdout.flush()

            # Read the filename:
            image_a = tf.gfile.FastGFile(img_files_a[i], 'rb').read()
            image_b = tf.gfile.FastGFile(img_files_b[i], 'rb').read()
            class_id = int(class_names[i])

            example = dataset_utils.image_to_tfexample(
                image_a, image_b, b'jpg', class_id)
            tfrecord_writer.write(example.SerializeToString())

  sys.stdout.write('\n')
  sys.stdout.flush()


def _dataset_exists(output_dir):
  for split_name in ['train', 'validation']:
    for shard_id in range(_NUM_SHARDS):
      output_filename = _get_dataset_filename(
          output_dir, split_name, shard_id)
      if not tf.gfile.Exists(output_filename):
        return False
  return True


def main(_):

  file_list = _get_filenames_and_classes(dataset_path)

  # Divide into train and test:
  if seq:
    file_list.sort()
    training_filenames = file_list[_NUM_VALIDATION:]
  else:
    random.seed(_RANDOM_SEED)
    random.shuffle(file_list)
  training_filenames = file_list[_NUM_VALIDATION:]
  #validation_filenames = file_list[:_NUM_VALIDATION]

  # First, convert the training and validation sets.
  _convert_dataset('train', training_filenames, dataset_path)
  #_convert_dataset('validation', validation_filenames, dataset_path)

  #_clean_up_temporary_files(dataset_dir)
  print('\nFinished converting the Flowers dataset!')

if __name__ == '__main__':
  tf.app.run()
