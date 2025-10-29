import torch
from torch.nn.utils import clip_grad_norm_
import sys, os
import numpy as np
from tqdm import tqdm
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from reward.formula_score import FormulaScore, RewardSmoother
class GRPOTrainer:
    def __init__(self, model, tokenizer, dataset, 
                 epoch_num, sample_num, eps, lr, 
                 reward_formula_model_dir_path=None,
                 log_path = None, save_path = None):
        self.model = model
        self.tokenizer = tokenizer
        self.dataset = dataset
        self.epoch_num = epoch_num
        self.sample_num = sample_num
        self.eps = eps
        self.lr = lr
        self.reward_formula_model_dir_path = reward_formula_model_dir_path
        FormulaScore.bert_score_model_dir_path = self.reward_formula_model_dir_path
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        self.log_path = log_path
        self.save_path = save_path
        if os.path.exists(self.log_path) is False:
            os.makedirs(self.log_path)
        self.log_file = open(os.path.join(self.log_path, 'RL.log'), 'a+')
        if os.path.exists(self.save_path) is False:
            os.makedirs(self.save_path)
        self.save_step = 1000

    def train(self):
        self.log_file.write(f"GRPO TRAIN LOG\n")
        reward_smoother = RewardSmoother()
        self.model.train()
        for epoch in tqdm(range(self.epoch_num), desc="Epoch"):
            skip_sample_num=0
            is_skip = False
            # 采样
            for idx, data in enumerate(self.dataset):
                input_len = data["input_ids"].shape[1]
                samples = []
                gen_tokens_list = []
                logprobs = []
                # 采样num_generation个样本
                with torch.no_grad():
                    for sample in range(self.sample_num):
                        output = self.model.generate(
                            input_ids = data["input_ids"], 
                            attention_mask = data["input_mask"],
                            do_sample=True,
                            top_p=0.98,
                            temperature=1,
                            max_new_tokens=256,
                            return_dict_in_generate=True,
                            output_scores=True
                        )
                        # 提取生成的 token 和 logprob
                        gen_ids = output.sequences[0]
                        gen_tokens_list.append(gen_ids)
                        gen_text = self.tokenizer.decode(gen_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)
                        print(gen_text)
                        samples.append(gen_text.replace("<FIXS>", "").replace("<FIXE>", ""))
                        # 计算 logprob
                        scores = torch.stack(output.scores, dim=1).squeeze(1)  # (gen_len, vocab)
                        probs = torch.log_softmax(scores, dim=-1)

                        logprob = probs.sum().item()
                        logprobs.append(logprob)

                if  is_skip: 
                    is_skip = False
                    continue

                rewards = FormulaScore.compute_reward(
                    samples, 
                    model_name_or_path = self.reward_formula_model_dir_path,
                    bug_method = data["bug_method"],
                    bug_line = data["bug_line"],
                    fix_line = data["fix_line"]
                )
                rewards = reward_smoother.smooth(np.array(rewards))
                avg_reward = sum(rewards) / self.sample_num

                # 计算策略梯度损失
                total_loss = 0.0
                for i in range(self.sample_num):
                    # 重新前向计算 logprob（训练模式）
                    gen_tokens = gen_tokens_list[i].unsqueeze(0)
                    full_input = torch.cat([data["input_ids"], gen_tokens], dim=1)
                    full_attention_mask = torch.cat([
                        data["input_mask"], 
                        torch.ones_like(gen_tokens)], 
                        dim=1
                    )
                    outputs = self.model(input_ids = full_input, attention_mask = full_attention_mask,labels=full_input)
                    
                    # 获取生成部分的logits
                    logits = outputs.logits[:, input_len-1:-1, :]  # 调整维度
                    gen_log_probs = torch.log_softmax(logits, dim=-1)
                    new_token_logprobs = gen_log_probs[0, torch.arange(len(gen_tokens_list[i])-1), gen_tokens_list[i][:-1]]
                    new_logprob = new_token_logprobs.sum()
                    
                    # 计算重要性采样比率
                    old_logprob_tensor = torch.tensor(logprobs[i], 
                                                      device=new_logprob.device, 
                                                      dtype=new_logprob.dtype,
                                                      requires_grad=False)
                    
                    # 添加数值稳定性保护
                    log_ratio = new_logprob - old_logprob_tensor
                    log_ratio = torch.clamp(log_ratio, -20, 20)  # 防止梯度爆炸[2](@ref)
                    rho = torch.exp(log_ratio)
                    
                    # 计算优势函数和损失
                    advantage = torch.tensor(
                        rewards[i] - avg_reward,
                        device=rho.device,
                        dtype=rho.dtype
                    )
                    advantage = torch.clamp(advantage, -10, 10)
                    
                    surr1 = rho * advantage
                    surr2 = torch.clamp(rho, 1 - self.eps, 1 + self.eps) * advantage
                    loss = -torch.min(surr1, surr2)
                    total_loss += loss
            
                if self.sample_num > 0:
                    total_loss /= self.sample_num
                    
                    # 梯度更新
                    self.optimizer.zero_grad()
                    total_loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                    self.optimizer.step()

                    # log
                    self.log_file.write(f"Epoch {epoch+1}/{self.epoch_num}, Sample {idx}/{len(self.dataset)}, Loss: {total_loss.item():.4f}, Reward: {avg_reward:.4f}\n")

                    if(idx%100==0):
                        print(f"Epoch {epoch+1}/{self.epoch_num}, Sample {idx}/{len(self.dataset)}, Loss: {total_loss.item():.4f}, Reward: {avg_reward:.4f}")
                

    def sample(self, input_ids):
        preds = self.generator_model.generate(
            input_ids,
            attention_mask=input_ids.ne(self.tokenizer.pad_token_id),
            use_cache=True,
            num_beams=5,
            early_stopping=True,
            max_length=256,
            pad_token_id=self.tokenizer.pad_token_id)



    
    def save_model(self, file_name):
        #self.model.save_pretrained(os.path.join(self.log_path, file_name))
        pass