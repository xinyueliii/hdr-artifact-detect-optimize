python test.py \
  --world_size 1 \
  --test_batch_size 4 \
  --output_width 256 \
  --output_height 256 \
  --edge_lambda 20 \
  --predict_head_norm "BN" \
  --vit_pretrain_path "./pretrained-weights/mae_pretrain_vit_base.pth" \
  --ckpt_path "/root/autodl-tmp/HADataset_Ours_256_output_hatector/best_checkpoint.pth" \
  --test_data_path "/root/autodl-tmp/HADataset_Ours_256/test" \
  --output_dir /root/autodl-tmp/HADataset_Ours_256_output_hatector_test/ \
  --log_dir /root/autodl-tmp/HADataset_Ours_256_output_hatector_test/   \
  --seed 42 \
  --num_workers 8 \
2> >(tee -a train_error.log) 1> >(tee -a train_log.log)