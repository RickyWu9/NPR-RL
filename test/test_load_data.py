import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from data.data_config import DataConfig

data_config = DataConfig("D://Study//NPR&RL//ProjectV1//assets//Train", None, is_test=True)
dataset = data_config.load_dataset()
print(dataset[0])