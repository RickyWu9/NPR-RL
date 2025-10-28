from codebleu import calc_codebleu
import code_bert_score


class FormulaScore(object):
    bert_score_model_dir_path = None
    

    @staticmethod
    def get_score(buggy, reference, prediction, model_name_or_path):
        #print(f"model_name_or_path: {model_name_or_path}")
        result = calc_codebleu(references=[reference], predictions=[prediction], lang='java')
        if result["dataflow_match_score"] != 0:
            codebleu = result["codebleu"]
        else:
            codebleu = result["ngram_match_score"]*0.3 + result["weighted_ngram_match_score"]*0.3 + result["syntax_match_score"]*0.4

        bert_score_rp = code_bert_score.score(cands=[prediction], refs=[reference], lang='java', model_type=model_name_or_path)[2].item()
        bert_score_bp = code_bert_score.score(cands=[prediction], refs=[buggy], lang='java', model_type=model_name_or_path)[2].item()
        return codebleu+bert_score_rp-bert_score_bp
        #return codebleu

    @staticmethod
    def compute_reward(completions, model_name_or_path, **kwargs):
        rewards = []
        # 从kwargs中获取参数
        bug_method = kwargs.get('bug_method', '')
        fix_line = kwargs.get('fix_line', '')
        bug_line = kwargs.get('bug_line', '')

        # 遍历每个生成的修复代码，计算奖励分数
        for completion in completions:
            ref = bug_method.split('\n')
            pre = bug_method.split('\n')
            for ind in range(len(ref)):
                if bug_line in ref[ind]:
                    ref[ind] = fix_line
                    pre[ind] = completion
            
            reference = '\n'.join(ref)
            prediction = '\n'.join(pre)
            score = FormulaScore.get_score(bug_method, reference, prediction, model_name_or_path)
            rewards.append(score)
        print(f"rewards: {rewards}")
        return rewards
    


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