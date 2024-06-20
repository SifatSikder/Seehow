from tensorflow.python import pywrap_tensorflow 
import numpy as np 
import matplotlib.pyplot as plt

net_name = 'action_vgg_e'

def convert(ckpt_param, npy_param):
  npy_param['conv1_1'] = [ckpt_param[net_name + '/conv1/conv1_1/weights'], ckpt_param[net_name + '/conv1/conv1_1/biases']]
  npy_param['conv1_2'] = [ckpt_param[net_name + '/conv1/conv1_2/weights'], ckpt_param[net_name + '/conv1/conv1_2/biases']]

  npy_param['conv2_1'] = [ckpt_param[net_name + '/conv2/conv2_1/weights'], ckpt_param[net_name + '/conv2/conv2_1/biases']]
  npy_param['conv2_2'] = [ckpt_param[net_name + '/conv2/conv2_2/weights'], ckpt_param[net_name + '/conv2/conv2_2/biases']]

  npy_param['conv3_1'] = [ckpt_param[net_name + '/conv3/conv3_1/weights'], ckpt_param[net_name + '/conv3/conv3_1/biases']]
  npy_param['conv3_2'] = [ckpt_param[net_name + '/conv3/conv3_2/weights'], ckpt_param[net_name + '/conv3/conv3_2/biases']]
  npy_param['conv3_3'] = [ckpt_param[net_name + '/conv3/conv3_3/weights'], ckpt_param[net_name + '/conv3/conv3_3/biases']]
  npy_param['conv3_4'] = [ckpt_param[net_name + '/conv3/conv3_4/weights'], ckpt_param[net_name + '/conv3/conv3_4/biases']]

  npy_param['conv4_1'] = [ckpt_param[net_name + '/conv4/conv4_1/weights'], ckpt_param[net_name + '/conv4/conv4_1/biases']]
  npy_param['conv4_2'] = [ckpt_param[net_name + '/conv4/conv4_2/weights'], ckpt_param[net_name + '/conv4/conv4_2/biases']]
  npy_param['conv4_3'] = [ckpt_param[net_name + '/conv4/conv4_3/weights'], ckpt_param[net_name + '/conv4/conv4_3/biases']]
  npy_param['conv4_4'] = [ckpt_param[net_name + '/conv4/conv4_4/weights'], ckpt_param[net_name + '/conv4/conv4_4/biases']]

  npy_param['conv5_1'] = [ckpt_param[net_name + '/conv5/conv5_1/weights'], ckpt_param[net_name + '/conv5/conv5_1/biases']]
  npy_param['conv5_2'] = [ckpt_param[net_name + '/conv5/conv5_2/weights'], ckpt_param[net_name + '/conv5/conv5_2/biases']]
  npy_param['conv5_3'] = [ckpt_param[net_name + '/conv5/conv5_3/weights'], ckpt_param[net_name + '/conv5/conv5_3/biases']]
  npy_param['conv5_4'] = [ckpt_param[net_name + '/conv5/conv5_4/weights'], ckpt_param[net_name + '/conv5/conv5_4/biases']]

  npy_param['fc6'] = [ckpt_param[net_name + '/fc6/weights'], ckpt_param[net_name + '/fc6/biases']]
  npy_param['fc7'] = [ckpt_param[net_name + '/fc7/weights'], ckpt_param[net_name + '/fc7/biases']]
  npy_param['fc8'] = [ckpt_param[net_name + '/fc8/weights'], ckpt_param[net_name + '/fc8/biases']]

  return npy_param


def readckpy():
  checkpoint_path="/home/cheer/Project/ActionNet/models/action_vgg_e/1/jp/model.ckpt-10000" 
  reader=pywrap_tensorflow.NewCheckpointReader(checkpoint_path) 
  var_to_shape_map=reader.get_variable_to_shape_map() 
  ckpt_param = {}
  npy_param = {}
  for key in var_to_shape_map: 
    print ("tensor_name",key) 
    ckpt_param[key] = reader.get_tensor(key) 
  print (ckpt_param[net_name + '/conv5/conv5_3/weights'].shape)
  #print (len(npy_param['conv1_1'][1]))
  #npy = convert(ckpt_param, npy_param)
  #print ('ckpt', npy['conv1_1'])
  #np.save('/home/cheer/Project/ActionNet/models/s/java/action_vgg_s.npy', npy)

def readnpy():
  npy_path = '/home/cheer/Project/ActionNet/models/s/java/action_vgg_s.npy'
  data = np.load(npy_path, encoding='latin1').item()
  print ('npy', data['conv1_1'])

def hist_weight():
  fig = plt.figure()

  checkpoint_path="/home/cheer/Project/ActionNet/models/action_vgg_e/0/jp/model.ckpt-10000" 
  reader=pywrap_tensorflow.NewCheckpointReader(checkpoint_path) 
  weight1 = reader.get_tensor(net_name + '/conv5/conv5_4/weights')
  weight1 = weight1.ravel()

  checkpoint_path="/home/cheer/Project/ActionNet/models/action_vgg_e/1/jp/model.ckpt-10000" 
  reader=pywrap_tensorflow.NewCheckpointReader(checkpoint_path) 
  weight2 = reader.get_tensor(net_name + '/conv5/conv5_4/weights')
  weight2 = weight2.ravel()

  euclidean = np.linalg.norm(weight1-weight2)
  cos_sim = np.dot(weight1, weight2)/(np.linalg.norm(weight1)*np.linalg.norm(weight2))
  print (euclidean, cos_sim)

  plt.subplot(2, 1, 1)
  plt.hist(weight1)
  plt.subplot(2, 1, 2)
  plt.hist(weight2)
  plt.show()

if __name__ == '__main__':
  readckpy()
  #readnpy()
  #hist_weight()


