python -m  pretrain.run \
  --output_dir checkpoint \
  --data_dir pretrain/wiki_data \
  --do_train True \
  --save_steps 20000 \
  --per_device_train_batch_size 64 \
  --model_name_or_path vinai/phobert-base \
  --fp16 True \
  --warmup_ratio 0.1 \
  --learning_rate 1e-4 \
  --num_train_epochs 8 \
  --overwrite_output_dir True \
  --dataloader_num_workers 6 \
  --weight_decay 0.01 \
  --encoder_mlm_probability 0.3 \
  --decoder_mlm_probability 0.5
  
# this script for training distributed

# torchrun --standalone --nnodes=1 --nproc_per_node=4 \
#   -m  pretrain.run \
#   --output_dir checkpoint \
#   --data_dir pretrain/sent-data \
#   --do_train True \
#   --save_steps 20000 \
#   --per_device_train_batch_size 64 \
#   --model_name_or_path vinai/phobert-base \
#   --fp16 True \
#   --warmup_ratio 0.1 \
#   --learning_rate 1e-4 \
#   --num_train_epochs 8 \
#   --overwrite_output_dir True \
#   --dataloader_num_workers 6 \
#   --weight_decay 0.01 \
#   --encoder_mlm_probability 0.3 \
#   --decoder_mlm_probability 0.5