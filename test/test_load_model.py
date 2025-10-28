import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from model.model_config import ModelConfig
from data.data_config import DataConfig


model_config = ModelConfig("qwen", "D://LLM//CodeT5")
model, tokenizer = model_config.load_model()
data_config = DataConfig("D://Study//NPR&RL//ProjectV1//assets//Train", tokenizer, is_test=True)
dataset = data_config.load_dataset()
idx = 2
input_txt = dataset[idx]["prompt"]
input_ids = tokenizer.encode(input_txt, return_tensors="pt")
print(input_ids.max())
print(input_ids.min())

print("input_ids shape:", input_ids.shape)
print("input_txt:", input_txt)

output = model.generate(
    dataset[idx]["input_ids"], 
    attention_mask = dataset[idx]["input_mask"],
    max_length=256,     
    return_dict_in_generate=True,
    output_scores=True
)
output_txt = tokenizer.decode(
    output.sequences[0], skip_special_tokens=True
)
print("output_ids shape:", output.sequences.shape)
print("output_txt:", output_txt)
print(len(output.scores))
print(output.sequences[0].shape)

