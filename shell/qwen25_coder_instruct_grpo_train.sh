project_root_dir="/root/wy/NPR-RL"
model_name="Qwen25-coder-ins"
model_dir_path="$project_root_dir/assets/model/qwen25_ins"
#model_dir_path="/root/wy/NPR-RL/assets/output/qwen25_ins/checkpoint-1000"
v100_gpu_train_data_dir="/root/wzy/NprRF-main/Generator/CodeBert/Train_Data/Train"
v100_gpu_valid_data_dir="/root/wzy/NprRF-main/Generator/CodeBert/Train_Data/Valid"
data_dir="$project_root_dir/assets/Train"
output_dir="$project_root_dir/assets/output/qwen25_ins/iter"
log_dir="$project_root_dir/assets/log/qwen25_ins"
rl_type="trl_grpo"
reward_formula_model_dir_path="$project_root_dir/assets/model/codebert_base"
is_deepspeed=True

cd $project_root_dir

if [[ "$is_deepspeed" == "True" ]]; then
    echo "use deepspeed"
    deepspeed --num_gpus=2 main.py \
        --model_name $model_name \
        --model_dir_path $model_dir_path \
        --data_dir $v100_gpu_train_data_dir \
        --output_dir $output_dir \
        --log_dir $log_dir \
        --rl_type $rl_type \
        --reward_formula_model_dir_path $reward_formula_model_dir_path \
        --is_finetune \
        --is_test \
        --test_number 96000 \
        --train_start_idx 0 \
        --is_deepspeed
else
    python -m main \
        --model_name $model_name \
        --model_dir_path $model_dir_path \
        --data_dir $v100_gpu_train_data_dir \
        --output_dir $output_dir \
        --log_dir $log_dir \
        --rl_type $rl_type \
        --reward_formula_model_dir_path $reward_formula_model_dir_path \
        --is_finetune \
        --is_test
fi