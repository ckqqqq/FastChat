cd /home/ckq/CHATGPT/FastChat/fastchat/train/6_gpu_dp_lora_ft.sh
conda activate
conda activate accel
conda info --envs
# 原本的多卡不是ddp再跑的，现在ddp再跑，会激活那个 batchsize/gpu数量的if语句
 
# export 
# export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:32
WORLD_SIZE=1 CUDA_VISIBLE_DEVICES=1,2,3,4,5,6  deepspeed /home/ckq/CHATGPT/FastChat/fastchat/train/train_lora.py \
    --deepspeed /home/ckq/CHATGPT/FastChat/fastchat/train/dp_config.yaml \
    --lora_r 8 \
    --lora_alpha 16 \
    --lora_dropout 0.05 \
    --model_name_or_path /home/ckq/CHATGPT/0model_pretrained/vicuna-7b-delta-v1.1 \
    --data_path /home/ckq/CHATGPT/FastChat/playground/data/dummy.json \
    --bf16 True \
    --output_dir ./checkpoints \
    --num_train_epochs 3 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 1 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 1200 \
    --save_total_limit 100 \
    --learning_rate 2e-5 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --model_max_length 2048