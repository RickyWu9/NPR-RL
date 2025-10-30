import sys, os
import re
from tqdm import tqdm
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from model.model_config import ModelConfig
from data.rl_dataset import RLDataset
from data.rl_data_process import read_file


class DataConfig():
    def __init__(self, data_dir, model_config : ModelConfig, is_test, data_type="train", test_number=10):
        self.data_dir = data_dir
        self.model_name = model_config.model_name
        self.tokenizer = model_config.tokenizer
        self.is_test = is_test 
        self.data_type = data_type
        self.test_number = test_number
        self.data = None
        self.dataset: RLDataset = None

    def load_dataset(self):
        print("Loading data......")
        # 加载原始数据
        self.load_raw_data()
        # 生成prompt
        self.gen_prompt()
        # 数据分词
        self.tokenize_data()

        self.dataset = RLDataset(self.data)
        return self.dataset
    
    def load_raw_data(self):
            def is_blank(str):
                return (str is None) | (len(str.strip())==0)
            
            # 获取目标文件夹
            if self.data_type == "train":
                data_ids_file = os.path.join(self.data_dir, "trn.ids")
            else:
                data_ids_file = os.path.join(self.data_dir, "valid.ids")
            buggy_methods_dir = os.path.join(self.data_dir, "buggy_methods")
            buggy_lines_dir = os.path.join(self.data_dir, "buggy_lines")
            fix_lines_dir = os.path.join(self.data_dir, "fix_lines")

            # 获取训练数据索引文件
            filename_list = []
            filename_list = read_file(data_ids_file)
            print("total number of data:", len(filename_list))
            if self.is_test:
                filename_list = filename_list[:self.test_number]
            print("train number of data:", len(filename_list))
            # 通过索引读取数据
            self.data = []
            idx = 0
            for id in tqdm(filename_list):
                bug_line = open(os.path.join(buggy_lines_dir, id + ".txt"), 'r', encoding='utf8').read().strip()
                fix_line = open(os.path.join(fix_lines_dir, id + ".txt"), 'r', encoding='utf8').read().strip()
                bug_method = read_file(os.path.join(buggy_methods_dir, id + ".txt"))
                bug_method = '\n'.join(bug_method)
                # 判空
                if(is_blank(bug_line) | is_blank(fix_line) | is_blank(bug_method)):
                    continue
                self.data_process(idx, bug_method, bug_line, fix_line)
                idx += 1
    
    def data_process(self, idx, bug_method, bug_line, fix_line):
        if self.model_name=="CodeT5":
            bug_method_list = bug_method.split('\n')
            for ind in range(len(bug_method_list)):
                if bug_line in bug_method_list[ind]:
                    bug_method_list[ind] = " <BUGS> " + bug_line + " <BUGE> "
            format_bug_method = '\n'.join(bug_method_list)
            format_bug_method = re.sub(r'\s+', ' ', format_bug_method)
            format_bug_method = format_bug_method.replace('</s>', '<unk>')

            format_fix_line= re.sub(r'\s+', ' ', " <FIXS> " + fix_line.strip() + " <FIXE> ")
            format_fix_line = format_fix_line.replace('</s>', '<unk>')
            self.data.append(
                {   
                    "idx":idx,
                    "bug_method":format_bug_method.strip(),
                    "bug_line":bug_line,
                    "fix_line":format_fix_line,
                    "prompt":format_bug_method
                }
            )
        else:
            self.data.append(
                    {   
                        "idx":idx,
                        "bug_method":bug_method,
                        "bug_line":bug_line,
                        "fix_line":fix_line,
                    }
                )

        



    def gen_prompt(self):
        if self.model_name =="CodeT5":
            pass 
        else:
            # todo gen prompt
            system_txt = "You are a code repair assistant."
            "Your task is to modify bug line to fix line from a bug method given by user.Your reply only contains a fix line. "
            "Don't give any other useless info!!! Don't give analysis or explanation!!! Just give an anwser!!! Just give a code!!!"
            "Your reply is just the fix line code. Don't reply any others words!!! Give a brief anwser!!!"
            for data in self.data:
                bug_method = data["bug_method"]
                bug_line = data["bug_line"]
                prompt_txt = f"The bug method is below:\n{bug_method}\n"
                f"The bug line is below:\n{bug_line}"
                prompt = [{
                    "role":"system",
                    "content":system_txt
                }, {
                    "role":"user",
                    "content":prompt_txt
                }]
                data["prompt"] = prompt

    def tokenize_data(self):
        if self.model_name =="CodeT5":
            for data in self.data:
                input_ids = self.tokenizer.encode(data["bug_method"], max_length=512, padding='max_length', 
                                                truncation=True, return_tensors="pt")
                target_ids = self.tokenizer.encode(data["fix_line"], max_length=256, padding='max_length', 
                                                truncation=True, return_tensors="pt")
                input_mask = input_ids.ne(self.tokenizer.pad_token_id)
                target_mask = target_ids.ne(self.tokenizer.pad_token_id)
                data["input_ids"] = input_ids
                data["target_ids"] = target_ids
                data["input_mask"] = input_mask
                data["target_mask"] = target_mask
        else:
            pass