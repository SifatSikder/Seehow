#!/bin/bash
set -e

SAVE_NAME=jp_full_gray
TRAINSET_NAME=jp_full_gray
MAX_STEP=10000

#1
MODEL_NAME=action_vgg_c
INPUT_MODE=2
OUTPUT_MODE=0
TRAIN_DIR=/home/cheer/Project/ActionNet/models/${MODEL_NAME}/${OUTPUT_MODE}/${SAVE_NAME}
TRAINSET_DIR=/home/cheer/Project/video_test/action_merge/${TRAINSET_NAME}

python3 train_image_classifier.py \
  --train_dir=${TRAIN_DIR} \
  --dataset_split_name=train \
  --dataset_dir=${TRAINSET_DIR} \
  --model_name=${MODEL_NAME} \
  --max_number_of_steps=${MAX_STEP} \
  --batch_size=8 \
  --learning_rate=0.01 \
  --learning_rate_decay_type=fixed \
  --save_interval_secs=600 \
  --save_summaries_secs=300 \
  --log_every_n_steps=10 \
  --optimizer=adam \
  --weight_decay=0.00004 \
  --input_mode=${INPUT_MODE} \
  --output_mode=${OUTPUT_MODE}

#2
MODEL_NAME=action_vgg_c
INPUT_MODE=2
OUTPUT_MODE=1
TRAIN_DIR=/home/cheer/Project/ActionNet/models/${MODEL_NAME}/${OUTPUT_MODE}/${SAVE_NAME}
TRAINSET_DIR=/home/cheer/Project/video_test/action_seq/${TRAINSET_NAME}

python3 train_image_classifier.py \
  --train_dir=${TRAIN_DIR} \
  --dataset_split_name=train \
  --dataset_dir=${TRAINSET_DIR} \
  --model_name=${MODEL_NAME} \
  --max_number_of_steps=${MAX_STEP} \
  --batch_size=8 \
  --learning_rate=0.001 \
  --learning_rate_decay_type=fixed \
  --save_interval_secs=600 \
  --save_summaries_secs=300 \
  --log_every_n_steps=10 \
  --optimizer=adam \
  --weight_decay=0.000004 \
  --input_mode=${INPUT_MODE} \
  --output_mode=${OUTPUT_MODE}

#3
MODEL_NAME=action_vgg_c
INPUT_MODE=2
OUTPUT_MODE=2
TRAIN_DIR=/home/cheer/Project/ActionNet/models/${MODEL_NAME}/${OUTPUT_MODE}/${SAVE_NAME}
TRAINSET_DIR=/home/cheer/Project/video_test/action_seq/${TRAINSET_NAME}

python3 train_image_classifier.py \
  --train_dir=${TRAIN_DIR} \
  --dataset_split_name=train \
  --dataset_dir=${TRAINSET_DIR} \
  --model_name=${MODEL_NAME} \
  --max_number_of_steps=${MAX_STEP} \
  --batch_size=8 \
  --learning_rate=0.001 \
  --learning_rate_decay_type=fixed \
  --save_interval_secs=600 \
  --save_summaries_secs=300 \
  --log_every_n_steps=10 \
  --optimizer=adam \
  --weight_decay=0.000004 \
  --input_mode=${INPUT_MODE} \
  --output_mode=${OUTPUT_MODE}
