#!/usr/bin/env python
# -*- coding: utf-8 -*-
# File: gradcam.py
# Author: Qian Ge <geqian1001@gmail.com>

from itertools import count
import sys
import os
sys.path.append('lib/')

import tensorflow as tf
import numpy as np
from tensorcv.dataflow.image import ImageFromFile
from tensorcv.utils.viz import image_overlay, save_merge_images

from vgg import VGG19_FCN
from models.guided_backpro import GuideBackPro
from models.grad_cam import ClassifyGradCAM
from utils.viz import image_weight_mask

IM_PATH = '/home/cheer/Project/ActionNet/data'
SAVE_DIR = '/home/cheer/Project/ActionNet/results/'
VGG_PATH = '/home/cheer/Project/ActionNet/models/action_vgg_e/0/jp_full/model.ckpt-10000'


# def image_weight_mask(image, mask):
#     """
#     Args:
#         image: image with size [HEIGHT, WIDTH, CHANNEL]
#         mask: image with size [HEIGHT, WIDTH, 1] or [HEIGHT, WIDTH]
#     """
#     image = np.array(np.squeeze(image))
#     mask = np.array(np.squeeze(mask))
#     assert len(mask.shape) == 2
#     assert len(image.shape) < 4
#     mask.astype('float32')
#     mask = np.reshape(mask, (mask.shape[0], mask.shape[1]))
#     mask = mask / np.amax(mask)

#     if len(image.shape) == 2:
#         return np.multiply(image, mask)
#     else:
#         for c in range(0, image.shape[2]):
#             image[:, :, c] = np.multiply(image[:, :, c], mask)
#         return image


if __name__ == '__main__':

    # merge several output images in one large image
    merge_im = 1
    grid_size = np.ceil(merge_im**0.5).astype(int)

    # class label for Grad-CAM generation
    # 355 llama 543 dumbbell 605 iPod 515 hat 99 groose 283 tiger cat
    # 282 tabby cat 233 border collie 242 boxer
    # class_id = [355, 543, 605, 515]
    class_id = [0,1,2,3,4]

    # initialize Grad-CAM
    # using VGG19
    gcam = ClassifyGradCAM(
        vis_model=VGG19_FCN(is_load=True,
                            pre_train_path=VGG_PATH,
                            is_rescale=True))
    gbackprob = GuideBackPro(
        vis_model=VGG19_FCN(is_load=True,
                            pre_train_path=VGG_PATH,
                            is_rescale=True))

    # placeholder for input image
    image_a = tf.placeholder(tf.float32, shape=[None, None, None, 3])
    image_b = tf.placeholder(tf.float32, shape=[None, None, None, 3])

    # create VGG19 model
    gcam.create_model(image_a, image_b)
    gcam.setup_graph()

    # generate class map and prediction label ops
    map_op = gcam.get_visualization(class_id=class_id)
    label_op = gcam.pre_label

    back_pro_op = gbackprob.get_visualization(image_a, image_b)

    # initialize input dataflow
    # change '.png' to other image types if other types of images are used
    input_im_a = ImageFromFile('.jpg', data_dir=os.path.join(IM_PATH, 'T0'),
                             num_channel=3, shuffle=False)
    input_im_b = ImageFromFile('.jpg', data_dir=os.path.join(IM_PATH, 'T1'),
                             num_channel=3, shuffle=False)
    input_im_a.set_batch_size(1)
    input_im_b.set_batch_size(1)

    #writer = tf.summary.FileWriter(SAVE_DIR)
    with tf.Session() as sess:

        sess.run(tf.global_variables_initializer())
        #writer.add_graph(sess.graph)

        cnt = 0
        merge_cnt = 0
        #weight_im_list = [[] for i in range(len(class_id))]
        o_im_list_a = []
        o_im_list_b = []
        while input_im_a.epochs_completed < 1:
            im_a = input_im_a.next_batch()[0]
            im_b = input_im_b.next_batch()[0]
            gcam_map, b_map, label, o_im_a, o_im_b =\
                sess.run([map_op, back_pro_op, label_op, gcam.input_im_a, gcam.input_im_b],
                         feed_dict={image_a: im_a, image_b: im_b})
            print(label)
            o_im_list_a.extend(o_im_a)
            o_im_list_b.extend(o_im_b)
            for idx, cid, cmap in zip(count(), gcam_map[1], gcam_map[0]):
                overlay_im_a = image_overlay(cmap, o_im_a)
                overlay_im_b = image_overlay(cmap, o_im_b)
                weight_im_a = image_weight_mask(b_map[0], cmap)
                weight_im_b = image_weight_mask(b_map[1], cmap)
                try:
                    weight_im_list_a[idx].append(weight_im_a)
                    weight_im_list_b[idx].append(weight_im_b)
                    overlay_im_list_a[idx].append(overlay_im_a)
                    overlay_im_list_b[idx].append(overlay_im_b)
                except NameError:
                    gcam_class_id = gcam_map[1]
                    weight_im_list_a = [[] for i in range(len(gcam_class_id))]
                    weight_im_list_b = [[] for i in range(len(gcam_class_id))]
                    overlay_im_list_a = [[] for i in range(len(gcam_class_id))]
                    overlay_im_list_b = [[] for i in range(len(gcam_class_id))]
                    weight_im_list_a[idx].append(weight_im_a)
                    weight_im_list_b[idx].append(weight_im_b)
                    overlay_im_list_a[idx].append(overlay_im_a)
                    overlay_im_list_b[idx].append(overlay_im_b)
            merge_cnt += 1
            # Merging results
            if merge_cnt == merge_im:
                save_path_a = '{}oim_a_{}.png'.format(SAVE_DIR, cnt, cid)
                save_path_b = '{}oim_b_{}.png'.format(SAVE_DIR, cnt, cid)
                save_merge_images(np.array(o_im_list_a),
                                  [grid_size, grid_size],
                                  save_path_a)
                save_merge_images(np.array(o_im_list_b),
                                  [grid_size, grid_size],
                                  save_path_b)
                for w_im_a, w_im_b, over_im_a, over_im_b, cid in zip(weight_im_list_a,
                                              weight_im_list_b, 
                                              overlay_im_list_a, 
                                              overlay_im_list_b,
                                              gcam_class_id):
                    # save grad-cam results
                    save_path_a = '{}gradcam_a_{}_class_{}.png'.\
                        format(SAVE_DIR, cnt, cid)
                    save_path_b = '{}gradcam_b_{}_class_{}.png'.\
                        format(SAVE_DIR, cnt, cid)
                    save_merge_images(
                        np.array(over_im_a), [grid_size, grid_size], save_path_a)
                    save_merge_images(
                        np.array(over_im_b), [grid_size, grid_size], save_path_b)
                    # save guided grad-cam results
                    save_path_a = '{}guided_gradcam_a_{}_class_{}.png'.\
                        format(SAVE_DIR, cnt, cid)
                    save_path_b = '{}guided_gradcam_b_{}_class_{}.png'.\
                        format(SAVE_DIR, cnt, cid)
                    save_merge_images(
                        np.array(w_im_a), [grid_size, grid_size], save_path_a)
                    save_merge_images(
                        np.array(w_im_b), [grid_size, grid_size], save_path_b)
                weight_im_list_a = [[] for i in range(len(gcam_class_id))]
                weight_im_list_b = [[] for i in range(len(gcam_class_id))]
                overlay_im_list_a = [[] for i in range(len(gcam_class_id))]
                overlay_im_list_b = [[] for i in range(len(gcam_class_id))]
                o_im_list_a = []
                o_im_list_b = []
                merge_cnt = 0
                cnt += 1

        # Saving results
        if merge_cnt > 0:
            save_path_a = '{}oim_a_{}.png'.format(SAVE_DIR, cnt, cid)
            save_path_b = '{}oim_b_{}.png'.format(SAVE_DIR, cnt, cid)
            save_merge_images(np.array(o_im_list_a),
                              [grid_size, grid_size],
                              save_path_a)
            save_merge_images(np.array(o_im_list_b),
                              [grid_size, grid_size],
                              save_path_b)
            for w_im_a, w_im_b, over_im_a, over_im_b, cid in zip(weight_im_list_a, 
                                          weight_im_list_b,
                                          overlay_im_list_a, 
                                          overlay_im_list_b,
                                          gcam_class_id):
                # save grad-cam results
                save_path_a = '{}gradcam_a_{}_class_{}.png'.\
                    format(SAVE_DIR, cnt, cid)
                save_path_b = '{}gradcam_b_{}_class_{}.png'.\
                    format(SAVE_DIR, cnt, cid)
                save_merge_images(
                    np.array(over_im_a), [grid_size, grid_size], save_path_a)
                save_merge_images(
                    np.array(over_im_b), [grid_size, grid_size], save_path_b)
                # save guided grad-cam results
                save_path_a = '{}guided_gradcam_a_{}_class_{}.png'.\
                    format(SAVE_DIR, cnt, cid)
                save_path_b = '{}guided_gradcam_b_{}_class_{}.png'.\
                    format(SAVE_DIR, cnt, cid)
                save_merge_images(
                    np.array(w_im_a), [grid_size, grid_size], save_path_a)
                save_merge_images(
                    np.array(w_im_b), [grid_size, grid_size], save_path_b)
    #writer.close()
