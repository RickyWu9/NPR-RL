import sys, os
import argparse
import logging

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from data.data_config import DataConfig
from model.model_config import ModelConfig
from rl.rl_config import RLConfig

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # 模型名称
    parser.add_argument("--model_name", type=str, default="CodeT5")
    # base模型文件夹地址
    parser.add_argument("--model_dir_path", type=str, default="D://LLM//CodeT5")
    # fine tuned模型参数文件地址
    parser.add_argument("--param_file_path", type=str, default=None)
    # 训练数据文件地址
    parser.add_argument("--data_dir", type=str, default="D://Study//NPR&RL//ProjectV1//assets//Train")
    # 模型输出文件夹
    parser.add_argument("--output_dir", type=str, default="D:/Study/NPR&RL//ProjectV1//assets//grpo_res")
    # 日志文件夹
    parser.add_argument("--log_dir", type=str, default="D://Study//NPR&RL//ProjectV1//assets//log")
    # rl类型
    parser.add_argument("--rl_type", type=str, default="grpo")
    # reward formula计算模型
    parser.add_argument("--reward_formula_model_dir_path", type=str, default="D://LLM//codebert-base")
    # 是否微调模型
    parser.add_argument("--is_finetune", type=bool, default=False)
    # 是否开启测试
    parser.add_argument("--is_test", type=bool, default=True)
    args = parser.parse_args()
    logger.info(args)

    # 加载模型
    model_config = ModelConfig(args.model_name, args.model_dir_path, args.param_file_path, args.is_finetune)
    model, tokenizer = model_config.load_model()

    # 加载数据
    data_config = DataConfig(args.data_dir, tokenizer, args.is_test)
    dataset = data_config.load_dataset()

    # 加载rl模块
    rl_config = RLConfig(model_config, data_config, args.rl_type, args.output_dir, args.log_dir, args.reward_formula_model_dir_path)

    # 开始训练
    rl_config.train()  
    rl_config.save_model("last_checkpoint.bin")