#!/usr/bin/env python
# -*- coding: utf-8 -*-
# File: guided_backpro.py
# Author: Qian Ge <geqian1001@gmail.com>

import tensorflow as tf

from tensorcv.models.layers import global_avg_pool


@tf.RegisterGradient("GuidedRelu")
def _GuidedReluGrad(op, grad):
    gate_g = tf.cast(grad > 0, "float32")
    gate_y = tf.cast(op.outputs[0] > 0, "float32")
    return grad * gate_g * gate_y


class GuideBackPro(object):
    def __init__(self, vis_model=None, class_id=None):
        assert vis_model is not None, 'vis_model cannot be None!'
        # assert not class_id is None, 'class_id cannot be None!'

        self._vis_model = vis_model
        if class_id is not None and not isinstance(class_id, list):
            class_id = [class_id]
        self._class_id = class_id

    def _create_model(self, image_a, image_b):
        keep_prob = 1
        self._vis_model.create_model([image_a, image_b, keep_prob])
        self.input_im_a = self._vis_model.layer['input_a']
        self.input_im_b = self._vis_model.layer['input_b']

        self._out_act = global_avg_pool(self._vis_model.layer['output'])
        self.pre_label = tf.nn.top_k(
            tf.nn.softmax(self._out_act), k=5, sorted=True)

    def _get_activation(self):
        with tf.name_scope('activation'):
            nclass = self._out_act.shape.as_list()[-1]
            act_list = []
            if self._class_id is None:
                class_list = [self.pre_label.indices[0][0]]
                act_list = [tf.reduce_max(self._out_act)]
            else:
                class_list = self._class_id
                for cid in class_list:
                    one_hot = tf.sparse_to_dense([[cid, 0]], [nclass, 1], 1.0)
                    self._out_act = tf.reshape(self._out_act, [1, nclass])
                    class_act = tf.matmul(self._out_act, one_hot)
                    act_list.append(class_act)

            return act_list, tf.convert_to_tensor(class_list)

    def get_visualization(self, image_a, image_b):
        g = tf.get_default_graph()

        with g.gradient_override_map({'Relu': 'GuidedRelu'}):
            try:
                self._create_model(image_a, image_b)
            except ValueError:
                with tf.variable_scope(tf.get_variable_scope()) as scope:
                    scope.reuse_variables()
                    self._create_model(image_a, image_b)
            act_list, class_list = self._get_activation()

            with tf.name_scope('guided_back_pro_map'):
                guided_back_pro_list_a = []
                guided_back_pro_list_b = []
                for class_act in act_list:
                    guided_back_pro_a = tf.gradients(
                        class_act, self._vis_model.layer['input_a'])
                    guided_back_pro_b = tf.gradients(
                        class_act, self._vis_model.layer['input_b'])
                    guided_back_pro_list_a.append(guided_back_pro_a)
                    guided_back_pro_list_b.append(guided_back_pro_b)

                self.visual_map_a = guided_back_pro_list_a
                self.visual_map_b = guided_back_pro_list_b
                self.class_list = class_list
                return guided_back_pro_list_a, guided_back_pro_list_b, class_list
