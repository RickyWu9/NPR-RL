from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch 


class ModelConfig():
    def __init__(self, model_name, model_dir_path, param_file_path = None, is_finetune = False):
        self.model_name = model_name
        self.model_dir_path = model_dir_path
        self.param_file_path = param_file_path
        self.is_finetune = is_finetune
        self.model = None 
        self.tokenizer = None

    def load_model(self):
        print("Loading model......")
        if self.model_name == "CodeT5":
            from transformers import T5ForConditionalGeneration, RobertaTokenizer
            tokenizer = RobertaTokenizer.from_pretrained(self.model_dir_path)
            model = T5ForConditionalGeneration.from_pretrained(self.model_dir_path)
        else:
            tokenizer = AutoTokenizer.from_pretrained(self.model_dir_path)
            model = AutoModelForSeq2SeqLM.from_pretrained(self.model_dir_path)
        
        # 加载模型参数
        if self.param_file_path is not None:
            model.load_state_dict(torch.load(self.param_file_path))

        # 获取微调模型
        if self.is_finetune:
            from peft import get_peft_model, LoraConfig 
            config = LoraConfig(
                r=16,
                lora_alpha=32,
                lora_dropout=0.1,
                target_modules=["q", "k", "v", "o"],
            )
            model = get_peft_model(model, config)
        self.model = model
        self.tokenizer = tokenizer
        print("Model loaded......")
        return model, tokenizer 