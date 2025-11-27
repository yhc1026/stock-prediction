DATA_PATHS = {
    'train': r'D:\codeC\stock-prediction\stock_predict\data\source\train.csv',
    'test': r'D:\codeC\stock-prediction\stock_predict\data\source\test.csv',
    'output': r'D:\codeC\stock-prediction\stock_predict\outputs\results\result.csv',
    'features': r'D:\codeC\stock-prediction\stock_predict\data\feature\features.csv'
}

MODEL_CONFIGS={
    'objective': 'reg:squarederror',
    'eval_metric': 'rmse',
    'learning_rate': 0.01,
    'max_depth': 3,
    'min_child_weight': 5,
    'subsample': 0.6,
    'colsample_bytree': 0.6,
    'reg_alpha': 0.5,
    'reg_lambda': 2,
    'gamma': 0.1,
    'random_state': 42,
    'n_estimators': 1000,
    'tree_method': 'hist',
    'device': 'cuda:0',
    #'predictor': 'gpu_predictor'
}