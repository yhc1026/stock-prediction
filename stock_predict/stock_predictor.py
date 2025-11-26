import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error


class CompleteStockPredictor:
    def __init__(self):
        self.data_paths = {
            'train': r'D:\codeC\stock_predict\data\source\train.csv',
            'test': r'D:\codeC\stock_predict\data\source\test.csv',
            'output': 'outputs/predictions.csv'
        }


    def run(self):
        print("加载和预处理数据")
        train_df, test_df = self.load_and_preprocess_data()
        print("特征工程")
        train_features, test_features, feature_names = self.create_features(train_df, test_df)
        print("准备训练数据")
        X_train, y_train, X_test = self.prepare_training_data(train_features, test_features)
        print("训练模型")
        model = self.train_model(X_train, y_train, feature_names)
        print("进行预测")
        top_stocks, bottom_stocks = self.predict_stocks(model, X_test, test_features, feature_names)
        print("输出结果")
        self.save_results(top_stocks, bottom_stocks)
        print("done")
        return top_stocks, bottom_stocks


    def load_and_preprocess_data(self):
        train_df = pd.read_csv(self.data_paths['train'])
        test_df = pd.read_csv(self.data_paths['test'])
        for df in [train_df, test_df]:
            df['日期'] = pd.to_datetime(df['日期'])
            df = df.sort_values(['股票代码', '日期']).reset_index(drop=True)
        return train_df, test_df


    def create_features(self, train_df, test_df):
        def add_features(df):
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
            df.to_csv(r'D:\codeC\stock_predict\data\feature\features.csv')
            return df
        train_features = add_features(train_df)
        test_features = add_features(test_df)
        feature_names = ['开盘', '收盘', '最高', '最低', '成交量', '成交额', '换手率',
                           '价格变化', 'HL比例', '价格位置', 'MA5', 'MA10', 'MA20',
                           'volume_MA5', 'volume_ratio', 'volatility', 'day_of_week', 'month']
        return train_features, test_features, feature_names


    def prepare_training_data(self, train_features, test_features):
        # 创建目标变量：下一天的涨跌幅
        train_features['target'] = train_features.groupby('股票代码')['涨跌幅'].shift(-1)
        train_features = train_features.dropna(subset=['target'])
        feature_columns = [col for col in train_features.columns if col not in ['股票代码', '日期', '涨跌幅', 'target', '涨跌额', '振幅']]
        X_train = train_features[feature_columns].fillna(0)
        y_train = train_features['target']
        X_test = test_features[feature_columns].fillna(0)
        return X_train, y_train, X_test


    def train_model(self, X_train, y_train, feature_names):
        """LightGBM简化"""
        params = {
            'objective': 'regression',
            'metric': 'rmse',
            'learning_rate': 0.05,
            'num_leaves': 31,
            'max_depth': -1,
            'min_data_in_leaf': 20,
            'feature_fraction': 0.8,
            'bagging_fraction': 0.8,
            'verbosity': -1,
            'random_state': 42,
            'n_estimators': 1000  # 增加树的数量
        }
        #todo:交叉验证
        model = lgb.LGBMRegressor(**params)
        model.fit(X_train, y_train)
        train_pred = model.predict(X_train)
        train_rmse = np.sqrt(mean_squared_error(y_train, train_pred))
        print(f"RMSE: {train_rmse:.4f}")
        return model


    def predict_stocks(self, model, X_test, test_features, feature_names):
        predictions = model.predict(X_test)
        test_features['predicted_return'] = predictions
        latest_date = test_features['日期'].max()
        latest_data = test_features[test_features['日期'] == latest_date]
        top_10 = latest_data.nlargest(10, 'predicted_return')['股票代码'].tolist()
        bottom_10 = latest_data.nsmallest(10, 'predicted_return')['股票代码'].tolist()
        return top_10, bottom_10

    def save_results(self, top_stocks, bottom_stocks):
        import os
        os.makedirs('outputs', exist_ok=True)
        results = []
        for i, stock in enumerate(top_stocks, 1):
            results.append({'排名': i, '股票代码': stock, '类型': '涨幅最大'})
        for i, stock in enumerate(bottom_stocks, 1):
            results.append({'排名': i, '股票代码': stock, '类型': '涨幅最小'})
        result_df = pd.DataFrame(results)
        result_df.to_csv(self.data_paths['output'], index=False, encoding='utf-8-sig')


if __name__ == "__main__":
    predictor = CompleteStockPredictor()
    top_stocks, bottom_stocks = predictor.run()