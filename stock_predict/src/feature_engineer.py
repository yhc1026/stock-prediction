#2
#3

import pandas as pd
#from ..config.config import DATA_PATHS

class FeatureEngineer:
    def __init__(self, train_df, test_df):
        self.train_df = train_df
        self.test_df = test_df

    def add_features(self,df):
        df = df.copy()
        df['价格变化'] = df['收盘'] - df['开盘']
        df['HL比例'] = (df['最高'] - df['最低']) / (df['收盘'] + 1e-8)
        df['价格位置'] = (df['收盘'] - df['最低']) / (df['最高'] - df['最低'] + 1e-8)
        df['MA5'] = df.groupby('股票代码')['收盘'].transform(lambda x: x.rolling(5).mean())
        df['MA10'] = df.groupby('股票代码')['收盘'].transform(lambda x: x.rolling(10).mean())
        df['MA20'] = df.groupby('股票代码')['收盘'].transform(lambda x: x.rolling(20).mean())
        df['volume_MA5'] = df.groupby('股票代码')['成交量'].transform(lambda x: x.rolling(5).mean())
        df['volume_ratio'] = df['成交量'] / (df['volume_MA5'] + 1e-8)
        df['volatility'] = df.groupby('股票代码')['收盘'].transform(lambda x: x.pct_change().rolling(10).std())
        df['day_of_week'] = df['日期'].dt.dayofweek
        df['month'] = df['日期'].dt.month
        return df

    def create_features(self):
        train_features = self.add_features(self.train_df)
        train_features.to_csv(r'D:\codeC\stock-prediction\stock_predict\data\feature\train_features.csv')
        test_features = self.add_features(self.test_df)
        test_features.to_csv(r'D:\codeC\stock-prediction\stock_predict\data\feature\test_features.csv')
        feature_names = ['开盘', '收盘', '最高', '最低', '成交量', '成交额', '换手率',
                           '价格变化', 'HL比例', '价格位置', 'MA5', 'MA10', 'MA20',
                           'volume_MA5', 'volume_ratio', 'volatility', 'day_of_week', 'month']
        return train_features, test_features, feature_names

    def prepare_training_data(self):
        # 创建目标变量：下一天的涨跌幅
        train_features, test_features, feature_names= self.create_features()
        train_features['target'] = train_features.groupby('股票代码')['涨跌幅'].shift(-1)
        train_features = train_features.dropna(subset=['target'])
        feature_names = [col for col in train_features.columns if col not in ['股票代码', '日期', '涨跌幅', 'target', '涨跌额', '振幅']]
        X_train = train_features[feature_names].fillna(0)
        y_train = train_features['target']
        X_test = test_features[feature_names].fillna(0)
        return X_train, y_train, X_test,test_features,feature_names

