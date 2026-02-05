#!/bin/bash

TEST_BATCH_SIZE=2
VIT_PRETRAIN_PATH="/root/HADetector/pretrained-weights/mae_pretrain_vit_base.pth"
EDGE_BROADEN=7
EDGE_LAMBDA=20
PREDICT_HEAD_NORM="BN"
CKPT_PATH="/root/autodl-tmp/HADataset_Ours_256_output_hadetector/best_checkpoint.pth"
SAVE_IMAGES=True
TEST_DATA_PATH="/root/autodl-tmp/HADataset-content-Ours-split/test"
OUTPUT_DIR="/root/autodl-tmp/HADataset_Ours_256_output_hadetector_test_1500"
LOG_DIR="/root/autodl-tmp/HADataset_Ours_256_output_hadetector_test_1500"
DEVICE="cuda"
NUM_WORKERS=8
PIN_MEM="--pin_mem"

python test_1500.py \
  --test_batch_size $TEST_BATCH_SIZE \
  --vit_pretrain_path $VIT_PRETRAIN_PATH \
  --edge_broaden $EDGE_BROADEN \
  --edge_lambda $EDGE_LAMBDA \
  --predict_head_norm $PREDICT_HEAD_NORM \
  --ckpt_path $CKPT_PATH \
  --save_images $SAVE_IMAGES \
  --test_data_path $TEST_DATA_PATH \
  --output_dir $OUTPUT_DIR \
  --log_dir $LOG_DIR \
  --device $DEVICE \
  --num_workers $NUM_WORKERS \
  $PIN_MEM