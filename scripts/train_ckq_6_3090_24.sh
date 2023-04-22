mytime=$(date "+%Y%m%d-%H-%M-%S")
echo ${mytime}

project_root_path=$1
echo $project_root_path
export CUDA_VISIBLE_DEVICES='1,2,3,4,5,6'
echo $CUDA_VISIBLE_DEVICES
torchrun --nnodes=1 --nproc_per_node=6 --master_port=11140 \
    $project_root_path/fastchat/train/train_mem.py \
    --model_name_or_path $project_root_path/vicuna-7b \
    --data_path $project_root_path/playground/data/alpaca-data-conversation.json \
    --bf16 True \
    --output_dir ./output/checkpoints \
    --num_train_epochs 3 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 4 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 120 \
    --save_total_limit 10 \
    --learning_rate 2e-5 \
    --weight_decay 0. \
    --warmup_ratio 0.03 
