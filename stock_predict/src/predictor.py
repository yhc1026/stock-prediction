#5
#6

import os
import pandas as pd
import importlib
current_dir = os.path.dirname(__file__)
project_root = os.path.dirname(current_dir)
config_path = os.path.join(project_root, 'config', 'config.py')
spec = importlib.util.spec_from_file_location("config_paths", config_path)
config_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(config_module)
DATA_PATHS = config_module.DATA_PATHS

class Predictor:
    def __init__(self, model, X_test, test_features, feature_names):
        self.model = model
        self.X_test = X_test
        self.test_features = test_features
        self.feature_names = feature_names
        self.output=DATA_PATHS['output']


    def predict_stocks(self):
        predictions = self.model.predict(self.X_test)
        self.test_features['predicted_return'] = predictions
        latest_date = self.test_features['日期'].max()
        latest_data = self.test_features[self.test_features['日期'] == latest_date]
        top_10 = latest_data.nlargest(10, 'predicted_return')['股票代码'].tolist()
        bottom_10 = latest_data.nsmallest(10, 'predicted_return')['股票代码'].tolist()
        return top_10, bottom_10

    def save_results(self,top_stocks, bottom_stocks):
        os.makedirs('outputs', exist_ok=True)
        results = []
        results.append("top_stocks")
        for i, stock in enumerate(top_stocks, 1):
            results.append({'排名': i, '股票代码': stock})
        results.append("bottom_stocks")
        for i, stock in enumerate(bottom_stocks, 1):
            results.append({'排名': i, '股票代码': stock})
        result_df = pd.DataFrame(results)
        result_df.to_csv(self.output, index=False, encoding='utf-8-sig')