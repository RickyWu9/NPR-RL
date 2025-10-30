model_name="Qwen25-coder-ins"
model_dir_path="/home/wuyi/wy/NPR-RL/assets/model/qwen25_ins"
data_dir="/home/wuyi/wy/NPR-RL/assets/Train"
output_dir="/home/wuyi/wy/NPR-RL/assets/output/qwen25_ins"
log_dir="/home/wuyi/wy/NPR-RL/assets/log/qwen3_ins"
rl_type="trl_grpo"
reward_formula_model_dir_path="/home/wuyi/wy/NPR-RL/assets/model/codebert_base"

cd /home/wuyi/wy/NPR-RL
python -m main \
    --model_name $model_name \
    --model_dir_path $model_dir_path \
    --data_dir $data_dir \
    --output_dir $output_dir \
    --log_dir $log_dir \
    --rl_type $rl_type \
    --reward_formula_model_dir_path $reward_formula_model_dir_path \
    --is_finetune \
    --is_test \