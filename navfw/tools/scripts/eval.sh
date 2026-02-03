CUDA_VISIBLE_DEVICES='0'

cd ..

python main.py \
    --mode eval \
    --altitude 50 \
    --learning_rate 0.0001 \
    --train_batch_size 2 \
    --gsam_box_threshold 0.15 \
    --eval_batch_size 50 \
    --eval_max_timestep 20 \
    --gsam_use_map_cache \
    --progress_stop_val 0.75 \
    --checkpoint 

