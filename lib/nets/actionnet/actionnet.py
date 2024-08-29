from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


# from ..correlation import correlation
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import tf_slim as slim

def action_vgg_arg_scope(weight_decay=0.0005):
  with slim.arg_scope([slim.conv2d, slim.fully_connected],
                      activation_fn=tf.nn.relu,
                      weights_regularizer=slim.l2_regularizer(weight_decay),
                      biases_initializer=tf.zeros_initializer()):
    with slim.arg_scope([slim.conv2d], padding='SAME') as arg_sc:
      return arg_sc

def action_vgg_base(inputs, reuse, scope=None):
  with tf.variable_scope(scope, 'action_vgg_base', [inputs], reuse=reuse):
    with slim.arg_scope([slim.conv2d, slim.max_pool2d]):
      net = slim.repeat(inputs, 2, slim.conv2d, 64, [3, 3], scope='conv1')
      net = slim.max_pool2d(net, [2, 2], scope='pool1')
      net = slim.repeat(net, 2, slim.conv2d, 128, [3, 3], scope='conv2')
      net = slim.max_pool2d(net, [2, 2], scope='pool2')
      net = slim.repeat(net, 4, slim.conv2d, 256, [3, 3], scope='conv3')
      net = slim.max_pool2d(net, [2, 2], scope='pool3')
      net = slim.repeat(net, 4, slim.conv2d, 512, [3, 3], scope='conv4')
      net = slim.max_pool2d(net, [2, 2], scope='pool4')
      net = slim.repeat(net, 4, slim.conv2d, 512, [3, 3], scope='conv5')
      net = slim.max_pool2d(net, [2, 2], scope='pool5')
  return net

def action_vgg_3D_base(inputs, reuse, scope=None):
  with tf.variable_scope(scope, 'action_vgg_3D', [inputs], reuse=reuse):
    with tf.contrib.framework.arg_scope([tf.contrib.layers.conv3d, tf.contrib.layers.max_pool3d]):
      net = tf.contrib.layers.repeat(inputs, 2, tf.contrib.layers.conv3d, 64, [3, 3, 3], scope='conv1')
      net = tf.contrib.layers.max_pool3d(net, [2, 2, 2], scope='pool1')
      net = tf.contrib.layers.repeat(net, 2, tf.contrib.layers.conv3d, 128, [3, 3, 3], scope='conv2')
      net = tf.contrib.layers.max_pool3d(net, [1, 2, 2], scope='pool2')
      net = tf.contrib.layers.repeat(net, 4, tf.contrib.layers.conv3d, 256, [3, 3, 3], scope='conv3')
      net = tf.contrib.layers.max_pool3d(net, [1, 2, 2], scope='pool2')
      net = tf.contrib.layers.repeat(net, 4, tf.contrib.layers.conv3d, 512, [3, 3, 3], scope='conv4')
      net = tf.contrib.layers.max_pool3d(net, [1, 2, 2], scope='pool2')
      net = tf.contrib.layers.repeat(net, 4, tf.contrib.layers.conv3d, 512, [3, 3, 3], scope='conv5')
      net = tf.contrib.layers.max_pool3d(net, [1, 2, 2], scope='pool2')     
  return net

def lstm_a(image, scope=None):
  with tf.variable_scope(scope, 'lstm_layer', [image]):
    num_units = 4096
    lstm_layer=tf.nn.rnn_cell.LSTMCell(num_units,use_peepholes=True,forget_bias=1, name="lstm_cell")
    initial_state = lstm_layer.zero_state(1, dtype=tf.float32)
    outputs,state=tf.nn.dynamic_rnn(lstm_layer,image,initial_state=initial_state,time_major=False,dtype="float32")
  return outputs

def action_vgg_e(input_a,
           input_b, 
           input_mode,
           output_mode,
           num_classes=19, 
           is_training=True, 
           dropout_keep_prob=0.5,
           spatial_squeeze=True,
           scope='action_vgg_e',
           fc_conv_padding='VALID',
           global_pool=False,
           reuse=tf.AUTO_REUSE):

  with tf.variable_scope(scope, 'action_vgg_e', [input_a, input_b], reuse=reuse) as sc:
    end_points_collection = sc.original_name_scope + '_end_points'
    # Collect outputs for conv2d, fully_connected and max_pool2d.
    with slim.arg_scope([slim.fully_connected, slim.max_pool2d],
                        outputs_collections=end_points_collection):

      if input_mode == 0:
        inputs = input_a
      elif input_mode == 1:
        inputs = input_b
      elif input_mode == 2:
        inputs = tf.concat([input_a, input_b], axis=3)

      net = action_vgg_base(inputs, reuse, scope = sc)

      net = slim.conv2d(net, 4096, [7, 7], padding=fc_conv_padding, scope='fc6')
      net = slim.dropout(net, dropout_keep_prob, is_training=is_training,
                         scope='dropout6')

      if output_mode == 0 or output_mode == 2:
        net = slim.conv2d(net, 4096, [1, 1], scope='fc7')
      # Convert end_points_collection into a end_point dict.
        end_points = slim.utils.convert_collection_to_dict(end_points_collection)
        if global_pool:
          net = tf.reduce_mean(net, [1, 2], keep_dims=True, name='global_pool')
          end_points['global_pool'] = net

        net = slim.dropout(net, dropout_keep_prob, is_training=is_training,
                         scope='dropout7')
      elif output_mode == 1:
        end_points = slim.utils.convert_collection_to_dict(end_points_collection)
        if global_pool:
          net = tf.reduce_mean(net, [1, 2], keep_dims=True, name='global_pool')
          end_points['global_pool'] = net
        net = lstm_a(tf.transpose(tf.squeeze(net, 1), [1,0,2]), scope=sc)
        net = tf.transpose(net, [1,0,2])
        net = tf.expand_dims(net, 1)
      
      net = slim.conv2d(net, num_classes, [1, 1],
                        activation_fn=None,
                        normalizer_fn=None,
                        scope='fc8')
      if spatial_squeeze:
        net = tf.squeeze(net, [1, 2], name='fc8/squeezed')
      end_points[sc.name + '/fc8'] = net
      return net, end_points
action_vgg_e.default_image_size = 224

def action_vgg_l(input_a,
           input_b, 
           input_mode,
           output_mode,
           num_classes=19, 
           is_training=True, 
           dropout_keep_prob=0.5,
           spatial_squeeze=True,
           scope='action_vgg_l',
           fc_conv_padding='VALID',
           global_pool=False,
           reuse=tf.AUTO_REUSE):

  with tf.variable_scope(scope, 'action_vgg_l', [input_a, input_b], reuse=reuse) as sc:
    end_points_collection = sc.original_name_scope + '_end_points'
    # Collect outputs for conv2d, fully_connected and max_pool2d.
    with slim.arg_scope([slim.fully_connected, slim.max_pool2d],
                        outputs_collections=end_points_collection):

      net_a = action_vgg_base(input_a, reuse, scope = sc)
      net_b = action_vgg_base(input_b, reuse, scope = sc)
      net = tf.concat([net_a, net_b], axis=3)

      # Use conv2d instead of fully_connected layers.
      net = slim.conv2d(net, 4096, [7, 7], padding=fc_conv_padding, scope='fc6')
      net = slim.dropout(net, dropout_keep_prob, is_training=is_training,
                         scope='dropout6')

      if output_mode == 0 or output_mode == 2:
        net = slim.conv2d(net, 4096, [1, 1], scope='fc7')
      # Convert end_points_collection into a end_point dict.
        end_points = slim.utils.convert_collection_to_dict(end_points_collection)
        if global_pool:
          net = tf.reduce_mean(net, [1, 2], keep_dims=True, name='global_pool')
          end_points['global_pool'] = net

        net = slim.dropout(net, dropout_keep_prob, is_training=is_training,
                         scope='dropout7')
      elif output_mode == 1:
        end_points = slim.utils.convert_collection_to_dict(end_points_collection)
        if global_pool:
          net = tf.reduce_mean(net, [1, 2], keep_dims=True, name='global_pool')
          end_points['global_pool'] = net
        net = lstm_a(tf.transpose(tf.squeeze(net, 1), [1,0,2]), scope=sc)
        net = tf.transpose(net, [1,0,2])
        net = tf.expand_dims(net, 1)
      
      net = slim.conv2d(net, num_classes, [1, 1],
                        activation_fn=None,
                        normalizer_fn=None,
                        scope='fc8')
      if spatial_squeeze:
        net = tf.squeeze(net, [1, 2], name='fc8/squeezed')
      end_points[sc.name + '/fc8'] = net
      return net, end_points
action_vgg_l.default_image_size = 224

# def action_vgg_c(input_a,
#            input_b, 
#            input_mode,
#            output_mode,
#            num_classes=19, 
#            is_training=True, 
#            dropout_keep_prob=0.5,
#            spatial_squeeze=True,
#            scope='action_vgg_c',
#            fc_conv_padding='VALID',
#            global_pool=False,
#            reuse=tf.AUTO_REUSE):

#   with tf.variable_scope(scope, 'action_vgg_c', [input_a, input_b], reuse=reuse) as sc:
#     end_points_collection = sc.original_name_scope + '_end_points'
#     # Collect outputs for conv2d, fully_connected and max_pool2d.
#     with slim.arg_scope([slim.fully_connected, slim.max_pool2d],
#                         outputs_collections=end_points_collection):

#       conv_a = slim.repeat(input_a, 3, slim.conv2d, 64, [3,3], scope = 'conv_a')
#       conv_b = slim.repeat(input_b, 3, slim.conv2d, 64, [3,3], scope = 'conv_b')
#       cc = correlation(conv_a, conv_b, 1, 20, 1, 2, 20)
#       conv_a_4 = slim.conv2d(conv_a, 32, 1, scope='conv_redir')
#       inputs = tf.concat([conv_a_4, cc], axis=3)
#       net = action_vgg_base(inputs, reuse, scope = sc)

#       # Use conv2d instead of fully_connected layers.
#       net = slim.conv2d(net, 4096, [7, 7], padding=fc_conv_padding, scope='fc6')
#       net = slim.dropout(net, dropout_keep_prob, is_training=is_training,
#                          scope='dropout6')

#       if output_mode == 0 or output_mode == 2:
#         net = slim.conv2d(net, 4096, [1, 1], scope='fc7')
#       # Convert end_points_collection into a end_point dict.
#         end_points = slim.utils.convert_collection_to_dict(end_points_collection)
#         if global_pool:
#           net = tf.reduce_mean(net, [1, 2], keep_dims=True, name='global_pool')
#           end_points['global_pool'] = net

#         net = slim.dropout(net, dropout_keep_prob, is_training=is_training,
#                          scope='dropout7')
#       elif output_mode == 1:
#         end_points = slim.utils.convert_collection_to_dict(end_points_collection)
#         if global_pool:
#           net = tf.reduce_mean(net, [1, 2], keep_dims=True, name='global_pool')
#           end_points['global_pool'] = net
#         net = lstm_a(tf.transpose(tf.squeeze(net, 1), [1,0,2]), scope=sc)
#         net = tf.transpose(net, [1,0,2])
#         net = tf.expand_dims(net, 1)
      
#       net = slim.conv2d(net, num_classes, [1, 1],
#                         activation_fn=None,
#                         normalizer_fn=None,
#                         scope='fc8')
#       if spatial_squeeze:
#         net = tf.squeeze(net, [1, 2], name='fc8/squeezed')
#       end_points[sc.name + '/fc8'] = net
#       return net, end_points
# action_vgg_c.default_image_size = 224

def action_vgg_3D(input_a,
           input_b, 
           input_mode,
           output_mode,
           num_classes=19, 
           is_training=True, 
           dropout_keep_prob=0.5,
           spatial_squeeze=True,
           scope='action_vgg_3D',
           fc_conv_padding='VALID',
           global_pool=False,
           reuse=tf.AUTO_REUSE):

  with tf.variable_scope(scope, 'action_vgg_3D', [input_a, input_b], reuse=reuse) as sc:
    end_points_collection = sc.original_name_scope + '_end_points'
    with tf.contrib.framework.arg_scope([tf.contrib.layers.conv3d, tf.contrib.layers.max_pool3d], outputs_collections=end_points_collection):

      inputs = tf.stack([input_a, input_b], axis=1)

      net = action_vgg_3D_base(inputs, reuse, scope = sc)

      net = tf.squeeze(net, 1)
      net = tf.contrib.layers.conv2d(net, 4096, [7, 7], padding=fc_conv_padding, scope='fc6')
      net = tf.contrib.layers.dropout(net, dropout_keep_prob, is_training=is_training, scope='dropout6')

      net = tf.contrib.layers.conv2d(net, 4096, [1, 1], scope='fc7')
      end_points = slim.utils.convert_collection_to_dict(end_points_collection)

      net = tf.contrib.layers.dropout(net, dropout_keep_prob, is_training=is_training, scope='dropout7')
      
      net = tf.contrib.layers.conv2d(net, num_classes, [1, 1],
                        activation_fn=None,
                        normalizer_fn=None,
                        scope='fc8')
      if spatial_squeeze:
        net = tf.squeeze(net, [1, 2], name='fc8/squeezed')
      end_points[sc.name + '/fc8'] = net
      return net, end_points
action_vgg_3D.default_image_size = 224
