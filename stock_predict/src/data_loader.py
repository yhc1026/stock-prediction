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
        """清理异常值"""
        if target_column not in df.columns:
            return df

        print(f"=== 清理 {target_column} 异常值 ===")
        print(f"清理前数据量: {len(df):,}")
        print(f"原始范围: [{df[target_column].min():.2f}, {df[target_column].max():.2f}]")

        # 使用分位数清理极端值
        lower_bound = df[target_column].quantile(0.001)  # 0.1%
        upper_bound = df[target_column].quantile(0.999)  # 99.9%

        # 对于股票涨跌幅，也可以使用固定范围
        stock_lower = -20  # -20%
        stock_upper = 20  # +20%

        # 使用更严格的范围
        final_lower = max(lower_bound, stock_lower)
        final_upper = min(upper_bound, stock_upper)

        mask = (df[target_column] >= final_lower) & (df[target_column] <= final_upper)
        cleaned_df = df[mask].copy()

        print(f"使用范围: [{final_lower:.2f}, {final_upper:.2f}]")
        print(f"清理后数据量: {len(cleaned_df):,}")
        print(f"移除异常值: {len(df) - len(cleaned_df):,}")
        print(f"清理后范围: [{cleaned_df[target_column].min():.2f}, {cleaned_df[target_column].max():.2f}]")

        return cleaned_df

    def load_and_preprocess_data(self, clean_outliers=True):
        train_df = pd.read_csv(self.data_paths['train'])
        test_df = pd.read_csv(self.data_paths['test'])

        for df in [train_df, test_df]:
            df['日期'] = pd.to_datetime(df['日期'])
            df = df.sort_values(['股票代码', '日期']).reset_index(drop=True)

        # 清理异常值
        if clean_outliers:
            train_df = self._clean_abnormal_values(train_df, '涨跌幅')
            test_df = self._clean_abnormal_values(test_df, '涨跌幅')

        return train_df, test_df

    def get_data_stats(self, df, name=""):
        """获取数据统计信息"""
        print(f"\n=== {name} 数据统计 ===")
        print(f"数据形状: {df.shape}")
        print(f"时间范围: {df['日期'].min()} 到 {df['日期'].max()}")
        print(f"股票数量: {df['股票代码'].nunique()}")

        if '涨跌幅' in df.columns:
            print(f"涨跌幅统计:")
            print(f"  范围: [{df['涨跌幅'].min():.4f}, {df['涨跌幅'].max():.4f}]")
            print(f"  均值: {df['涨跌幅'].mean():.6f} ± {df['涨跌幅'].std():.6f}")
            print(f"  中位数: {df['涨跌幅'].median():.6f}")


if __name__ == '__main__':
    data_loader = DataLoader()
    train_df, test_df = data_loader.load_and_preprocess_data()

    # 显示统计信息
    data_loader.get_data_stats(train_df, "训练集")
    data_loader.get_data_stats(test_df, "测试集")