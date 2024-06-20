import tensorflow as tf
import glob
import os

event_path = '/home/cheer/Project/ActionNet/model_c/jp_all'

def read_event():
  event_file = []
  total_loss = []
  for name in glob.glob(os.path.join(event_path, 'events.*')):
    event_file.append(name)
  event_file.sort()
  for file_name in event_file:
    total_loss += read_file(file_name)
  print (total_loss)
  print (len(total_loss))
  
def read_file(file_name):
  loss = []
  for e in tf.train.summary_iterator(file_name):
    for v in e.summary.value:
      if v.tag == 'total_loss_1':
        loss.append(v.simple_value)
  return loss

if __name__ == '__main__':
  read_event()
