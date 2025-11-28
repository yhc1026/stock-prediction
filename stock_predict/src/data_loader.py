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


# class DataLoader:
#     def __init__(self):
#         self.data_paths = DATA_PATHS
#
#     def load_and_preprocess_data(self):
#         train_df = pd.read_csv(self.data_paths['train'])
#         test_df = pd.read_csv(self.data_paths['test'])
#         for df in [train_df, test_df]:
#             df['日期'] = pd.to_datetime(df['日期'])
#             df = df.sort_values(['股票代码', '日期']).reset_index(drop=True)
#         return train_df, test_df
#
# if __name__ == '__main__':
#     data_loader = DataLoader()
#     train_df, test_df=data_loader.load_and_preprocess_data()
#     print(train_df.shape)


class DataLoader:
    def __init__(self):
        self.data_paths = DATA_PATHS

    def _clean_abnormal_values(self, df, target_column='涨跌幅'):
        if target_column not in df.columns:
            return df
        lower_bound = df[target_column].quantile(0.001)
        upper_bound = df[target_column].quantile(0.999)
        stock_lower = -20
        stock_upper = 20
        final_lower = max(lower_bound, stock_lower)
        final_upper = min(upper_bound, stock_upper)
        mask = (df[target_column] >= final_lower) & (df[target_column] <= final_upper)
        cleaned_df = df[mask].copy()
        return cleaned_df

    def load_and_preprocess_data(self, clean_outliers=True):
        train_df = pd.read_csv(self.data_paths['train'])
        test_df = pd.read_csv(self.data_paths['test'])
        for df in [train_df, test_df]:
            df['日期'] = pd.to_datetime(df['日期'])
            df = df.sort_values(['股票代码', '日期']).reset_index(drop=True)
        if clean_outliers:
            train_df = self._clean_abnormal_values(train_df, '涨跌幅')
            test_df = self._clean_abnormal_values(test_df, '涨跌幅')
        return train_df, test_df

    def get_data_stats(self, df, name=""):
        print(f"shape: {df.shape}")
        print(f"time: {df['日期'].min()} - {df['日期'].max()}")
        print(f"num: {df['股票代码'].nunique()}")

        if '涨跌幅' in df.columns:
            print(f"涨跌幅")
            print(f"[{df['涨跌幅'].min():.4f},{df['涨跌幅'].max():.4f}]")
            print(f"{df['涨跌幅'].mean():.6f}±{df['涨跌幅'].std():.6f}")
            print(f"{df['涨跌幅'].median():.6f}")


if __name__ == '__main__':
    data_loader = DataLoader()
    train_df, test_df = data_loader.load_and_preprocess_data()
    data_loader.get_data_stats(train_df, "训练集")
    data_loader.get_data_stats(test_df, "测试集")