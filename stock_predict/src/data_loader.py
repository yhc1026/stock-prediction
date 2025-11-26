#1

import pandas as pd
import os
import sys
import importlib

current_dir = os.path.dirname(__file__)
project_root = os.path.dirname(current_dir)
config_path = os.path.join(project_root, 'config', 'config.py')
spec = importlib.util.spec_from_file_location("config_paths", config_path)
config_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(config_module)
DATA_PATHS = config_module.DATA_PATHS


class DataLoader:
    def __init__(self):
        self.data_paths = DATA_PATHS

    def load_and_preprocess_data(self):
        train_df = pd.read_csv(self.data_paths['train'])
        test_df = pd.read_csv(self.data_paths['test'])
        for df in [train_df, test_df]:
            df['日期'] = pd.to_datetime(df['日期'])
            df = df.sort_values(['股票代码', '日期']).reset_index(drop=True)
        return train_df, test_df

if __name__ == '__main__':
    data_loader = DataLoader()
    train_df, test_df=data_loader.load_and_preprocess_data()
    print(train_df.shape)