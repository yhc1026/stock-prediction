#4

import xgboost as xgb
import numpy as np
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import TimeSeriesSplit
import matplotlib.pyplot as plt


class ModelTrainer:
    def __init__(self, X_train, y_train, feature_names):
        self.X_train = X_train
        self.y_train = y_train
        self.feature_names = feature_names
        #print(f"X_train: {self.X_train.shape}, y_train: {self.y_train.shape}")

    def _get_params(self):
        """XGBoost参数配置"""
        return {
            'objective': 'reg:squarederror',  # XGBoost的回归目标
            'eval_metric': 'rmse',
            'learning_rate': 0.05,
            'max_depth': 6,  # 控制树深度，避免过拟合
            'min_child_weight': 1,
            'subsample': 0.8,  # 相当于bagging_fraction
            'colsample_bytree': 0.8,  # 相当于feature_fraction
            'reg_alpha': 0,  # L1正则化
            'reg_lambda': 1,  # L2正则化
            'random_state': 42,
            'n_estimators': 1000
        }

    # def train_model(self):
    #     """基础XGBoost训练"""
    #     model = xgb.XGBRegressor(**self._get_params())
    #     model.fit(self.X_train, self.y_train)
    #
    #     # 训练集表现
    #     train_pred = model.predict(self.X_train)
    #     train_rmse = np.sqrt(mean_squared_error(self.y_train, train_pred))
    #     print(f"训练集RMSE: {train_rmse:.4f}")
    #     return model
    #
    # def train_model_with_early_stopping(self, eval_ratio=0.1):
    #     """带早停的XGBoost训练"""
    #     # 分割训练集和验证集
    #     split_idx = int(len(self.X_train) * (1 - eval_ratio))
    #     X_tr, X_val = self.X_train[:split_idx], self.X_train[split_idx:]
    #     y_tr, y_val = self.y_train[:split_idx], self.y_train[split_idx:]
    #
    #     model = xgb.XGBRegressor(**self._get_params())
    #
    #     model.fit(
    #         X_tr, y_tr,
    #         eval_set=[(X_val, y_val)],
    #         early_stopping_rounds=50,
    #         verbose=False
    #     )
    #
    #     # 输出最佳迭代次数
    #     print(f"最佳迭代次数: {model.best_iteration}")
    #
    #     return model

    def train_model(self, n_splits=5):
        """修正版交叉验证训练"""
        tscv = TimeSeriesSplit(n_splits=n_splits)
        best_iterations = []

        for fold, (train_idx, val_idx) in enumerate(tscv.split(self.X_train)):
            X_tr, X_val = self.X_train.iloc[train_idx], self.X_train.iloc[val_idx]
            y_tr, y_val = self.y_train.iloc[train_idx], self.y_train.iloc[val_idx]

            # 使用原生XGBoost API
            dtrain = xgb.DMatrix(X_tr, label=y_tr)
            dval = xgb.DMatrix(X_val, label=y_val)

            params = self._get_params()
            # 从参数中移除n_estimators，因为它对原生API无效
            params.pop('n_estimators', None)

            evals_result = {}
            model = xgb.train(
                params,
                dtrain,
                num_boost_round=1000,  # 使用num_boost_round而不是n_estimators
                evals=[(dtrain, 'train'), (dval, 'val')],
                early_stopping_rounds=50,
                evals_result=evals_result,
                verbose_eval=False
            )
            best_iterations.append(model.best_iteration)

        # 用全部数据训练最终模型
        params = self._get_params()
        params['n_estimators'] = int(np.mean(best_iterations)) + 10

        final_model = xgb.XGBRegressor(**params)
        final_model.fit(self.X_train, self.y_train)

        return final_model

if __name__ == "__main__":
    pass