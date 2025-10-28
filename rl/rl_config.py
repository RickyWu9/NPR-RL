import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from model.model_config import ModelConfig
from data.data_config import DataConfig
from reward.formula_score import FormulaScore
class RLConfig():
    def __init__(self, model_config: ModelConfig, data_config: DataConfig, 
                 rl_type, output_dir, log_dir, reward_formula_model_dir_path=None) -> None:
        self.model_config = model_config
        self.data_config = data_config
        self.rl_type = rl_type
        self.output_dir = output_dir
        self.log_dir = log_dir
        self.reward_formula_model_dir_path = reward_formula_model_dir_path
        self.train_config = None
        self.load_config()

    def load_config(self):
        if self.rl_type =="trl_grpo":
            from trl import GRPOConfig, GRPOTrainer
            self.train_config = GRPOConfig(
                output_dir=self.output_dir,
                logging_dir=self.log_dir,
                learning_rate=1e-5,
                lr_scheduler_type="cosine",
                warmup_ratio=0.03,
                logging_steps=10,
                max_steps=5000,
                per_device_train_batch_size=4,
                gradient_accumulation_steps=1,
                save_strategy="steps",
                save_steps=100,
                per_device_eval_batch_size=4,
      
                # train specific parameters
                max_prompt_length=512,
                max_completion_length=256,
                num_generations=4,
                beta=1e-3,
                fp16=True,  # 启用混合精度训练以加速和节省内存
                gradient_checkpointing=False,
                deepspeed=None,  # 可以设置为deepspeed配置文件路径
            )

            self.trainer = GRPOTrainer(
                model=self.model_config.model,
                reward_funcs=[FormulaScore.compute_reward],
                args=self.train_config,
                train_dataset=self.data_config.dataset,
                #processing_class=self.model_config.tokenizer,
                processing_class=self.model_config.tokenizer,
            )
        elif self.rl_type =="grpo":
            from rl.grpo_train import GRPOTrainer
            self.trainer = GRPOTrainer(
                model=self.model_config.model,
                tokenizer=self.model_config.tokenizer,
                dataset=self.data_config.dataset,
                epoch_num=1,
                sample_num=4,
                eps=1e-3,
                lr=1e-6,
                reward_formula_model_dir_path=self.reward_formula_model_dir_path,
                log_path=self.log_dir
                save_path=self.output_dir
            )
            

    def train(self):
        print("Start training......")
        self.trainer.train()

    def save_model(self, file_name):
        print("Start saving model......")
        self.trainer.save_model(file_name)
