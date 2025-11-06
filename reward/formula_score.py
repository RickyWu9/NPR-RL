from codebleu import calc_codebleu
import code_bert_score
import re

class FormulaScore(object):
    bert_score_model_dir_path = None
    

    @staticmethod
    def get_score(buggy, reference, prediction, model_name_or_path=None):
        if model_name_or_path:
            model_path = model_name_or_path
        else:
            model_path = FormulaScore.bert_score_model_dir_path
        #print(f"model_path: {model_path}")
        result = calc_codebleu(references=[reference], predictions=[prediction], lang='java')
        if result["dataflow_match_score"] != 0:
            codebleu = result["codebleu"]
        else:
            codebleu = result["ngram_match_score"]*0.3 + result["weighted_ngram_match_score"]*0.3 + result["syntax_match_score"]*0.4
        print(f"prediction: {prediction}")
        print(f"reference: {reference}")
        bert_score_rp = code_bert_score.score(cands=[prediction], refs=[reference], lang='java', model_type=model_path)[2].item()
        bert_score_bp = code_bert_score.score(cands=[prediction], refs=[buggy], lang='java', model_type=model_path)[2].item()
        return codebleu+bert_score_rp-bert_score_bp
        #return codebleu

    @staticmethod
    def compute_reward(completions, **kwargs):
        rewards = []
        # 从kwargs中获取参数
        
        bug_method = kwargs.get('bug_method', '')
        fix_line = kwargs.get('fix_line', '')
        bug_line = kwargs.get('bug_line', '')
        model_path = kwargs.get('reward_model','')

        # pre process
        if model_path is None or len(model_path)<=0:
            model_path = FormulaScore.bert_score_model_dir_path
        if isinstance(bug_method, list):
            bug_method = bug_method[0]
        if isinstance(fix_line, list):
            fix_line = fix_line[0]
        if isinstance(bug_line, list):
            bug_line = bug_line[0]

        # 遍历每个生成的修复代码，计算奖励分数
        for completion in completions:
            if isinstance(completion, list):
                completion = completion[0]["content"]
            completion = FormulaScore.extract_fix_code(completion)
            print(f"completion: {completion}\n--------------------")
            if completion is None:
                rewards.append(-10)
                continue
            
            ref = bug_method.split('\n')
            pre = bug_method.split('\n')
            for ind in range(len(ref)):
                if bug_line in ref[ind]:
                    ref[ind] = fix_line
                    pre[ind] = completion
            
            reference = '\n'.join(ref)
            prediction = '\n'.join(pre)
            score = FormulaScore.get_score(bug_method, reference, prediction, model_name_or_path = model_path)
            rewards.append(score)
        print(f"rewards: {rewards}")
        return rewards
    
    @staticmethod
    def extract_fix_code(completion):
        pattern = re.compile(r"```java\n(.*?)\n```", re.DOTALL)
        matches = pattern.findall(completion)
        if len(matches) > 0:
            return matches[0].strip()
        else:
            return None
    


class RewardSmoother:

    def __init__(self):
        self.reward_smooth_alpha = 0.9
        self.last_reward = None
    
    def smooth(self, reward):
        if self.last_reward is None:
            self.last_reward = reward
        else:
            self.last_reward = self.reward_smooth_alpha * self.last_reward + (1 - self.reward_smooth_alpha) * reward
        return self.last_reward