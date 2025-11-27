#4
from datetime import datetime
import pickle
import time

import os
import importlib
import xgboost as xgb
import numpy as np
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import TimeSeriesSplit
import matplotlib.pyplot as plt
current_dir = os.path.dirname(__file__)
project_root = os.path.dirname(current_dir)
config_path = os.path.join(project_root, 'config', 'config.py')
spec = importlib.util.spec_from_file_location("config_paths", config_path)
config_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(config_module)
MODEL_CONFIGS = config_module.MODEL_CONFIGS


class ModelTrainer:
    def __init__(self, X_train, y_train, feature_names):
        self.X_train = X_train
        self.y_train = y_train
        self.feature_names = feature_names
        #print(f"X_train: {self.X_train.shape}, y_train: {self.y_train.shape}")

    def _get_params(self):
        return MODEL_CONFIGS

    def train_model(self, n_splits=5):
        #寻找最佳训练轮数
        tscv = TimeSeriesSplit(n_splits=n_splits)
        best_iterations = []
        for fold, (train_idx, val_idx) in enumerate(tscv.split(self.X_train)):
            X_tr, X_val = self.X_train.iloc[train_idx], self.X_train.iloc[val_idx]
            y_tr, y_val = self.y_train.iloc[train_idx], self.y_train.iloc[val_idx]
            dtrain = xgb.DMatrix(X_tr, label=y_tr)
            dval = xgb.DMatrix(X_val, label=y_val)
            params = self._get_params()
            params.pop('n_estimators', None)
            evals_result = {}
            model = xgb.train(
                params,
                dtrain,
                num_boost_round=4000,
                evals=[(dtrain, 'train'), (dval, 'val')],
                early_stopping_rounds=50,
                evals_result=evals_result,
                verbose_eval=False
            )
            best_iterations.append(model.best_iteration)
        print(f"Best iterations: {best_iterations}")
        #训练模型
        params = self._get_params()
        params['n_estimators'] = int(np.mean(best_iterations)) + 10
        final_model = xgb.XGBRegressor(**params)
        final_model.fit(self.X_train, self.y_train)
        # 保存模型
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_filename = f"xgb_model_{timestamp}.pkl"
        model_path = rf"D:\codeC\stock-prediction\stock_predict\outputs\models\{model_filename}"
        with open(model_path, 'wb') as f:
            pickle.dump(final_model, f)
        print(f"save model as: {model_filename}")
        return final_model


    def comprehensive_data_check(self):
        print("=== 数据全面检查 ===")
        print(f"总样本数: {len(self.X_train):,}")
        print(f"特征数: {len(self.X_train.columns)}")
        print(f"目标变量名称: {self.y_train.name if hasattr(self.y_train, 'name') else 'Unknown'}")

        # 检查目标变量
        print(f"\n目标变量统计:")
        print(f"  范围: [{self.y_train.min():.6f}, {self.y_train.max():.6f}]")
        print(f"  均值: {self.y_train.mean():.6f} ± {self.y_train.std():.6f}")
        print(f"  中位数: {self.y_train.median():.6f}")

        # 检查分位数
        quantiles = [0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99]
        for q in quantiles:
            print(f"  {q * 100:2.0f}%分位数: {self.y_train.quantile(q):.6f}")

        # 检查异常值
        q1, q3 = self.y_train.quantile(0.25), self.y_train.quantile(0.75)
        iqr = q3 - q1
        lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
        outliers = self.y_train[(self.y_train < lower) | (self.y_train > upper)]
        print(f"  异常值数量: {len(outliers):,} ({len(outliers) / len(self.y_train) * 100:.2f}%)")

        # 检查特征尺度
        print(f"\n特征尺度检查:")
        for col in self.X_train.columns[:3]:  # 只看前3个特征
            if self.X_train[col].dtype in ['float64', 'int64']:
                print(f"  {col}: [{self.X_train[col].min():.4f}, {self.X_train[col].max():.4f}]")


    def check_data_leakage(self):
        """检查是否存在数据泄露"""
        print("\n=== 数据泄露检查 ===")

        # 检查目标变量是否在特征中
        target_in_features = self.y_train.name in self.X_train.columns if hasattr(self.y_train, 'name') else False
        print(f"目标变量是否在特征中: {target_in_features}")

        # 检查特征与目标的相关性
        if len(self.X_train) > 10000:
            sample_idx = np.random.choice(len(self.X_train), 10000, replace=False)
            X_sample = self.X_train.iloc[sample_idx]
            y_sample = self.y_train.iloc[sample_idx]
        else:
            X_sample = self.X_train
            y_sample = self.y_train

        # 计算数值特征的相关性
        numeric_cols = X_sample.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 0:
            correlations = []
            for col in numeric_cols[:10]:  # 只看前10个特征
                corr = np.corrcoef(X_sample[col], y_sample)[0, 1]
                correlations.append((col, corr))

            # 按相关性绝对值排序
            correlations.sort(key=lambda x: abs(x[1]), reverse=True)
            print("特征与目标变量的相关性 (前5):")
            for col, corr in correlations[:5]:
                print(f"  {col}: {corr:.4f}")







if __name__ == "__main__":
    modeltrainer=ModelTrainer()
    modeltrainer.comprehensive_data_check()
