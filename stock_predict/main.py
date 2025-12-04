from src.data_loader import DataLoader
from src.feature_engineer import FeatureEngineer
from src.model_trainer import ModelTrainer
from src.predictor import Predictor
import numpy as np

class CompleteStockPredictor:
    def __init__(self):
        print("start")

    def run(self):
        print("加载和预处理数据")
        dataloader = DataLoader()
        train_df, test_df=dataloader.load_and_preprocess_data()
        print("done\n特征工程")
        print("done\n准备训练数据")
        fea_engineer = FeatureEngineer(train_df, test_df)
        X_train, y_train, X_test, test_features, feature_names = fea_engineer.prepare_training_data()
        print("done\n训练模型")
        trainer = ModelTrainer(X_train, y_train, feature_names)
        model=trainer.train_model()
        print("done\n进行预测")
        predictor = Predictor(model, X_test, test_features, feature_names)
        top_stocks, bottom_stocks = predictor.predict_stocks()
        predictor.save_results(top_stocks, bottom_stocks)
        print("done\n输出结果")
        print("done")
        return top_stocks, bottom_stocks

    def run2(self):
        dataloader = DataLoader()
        train_df, test_df = dataloader.load_and_preprocess_data()
        fea_engineer = FeatureEngineer(train_df, test_df)
        tem_X_train, tem_y_train, tem_X_test, use_test_features, use_feature_names = fea_engineer.prepare_training_data()
        results={}
        cnt=1
        print("加载和预处理数据")
        dataloader = DataLoader()
        train_df, test_df = dataloader.load_and_preprocess_data()
        for i in range(cnt):
            print("done\n特征工程")
            print("done\n准备训练数据")
            fea_engineer = FeatureEngineer(train_df, test_df)
            X_train, y_train, X_test, test_features, feature_names = fea_engineer.prepare_training_data()
            print("done\n训练模型")
            trainer = ModelTrainer(X_train, y_train, feature_names)
            model = trainer.train_model()
            print("done\n进行预测")
            predictor = Predictor(model, X_test, test_features, feature_names)
            predictions = predictor.return_predictions()
            results[i]=predictions
        arrays = np.array(list(results.values()))
        sum_by_column = np.sum(arrays, axis=0)
        average_by_column = sum_by_column/cnt
        print("average in main: ",average_by_column,"\n")
        predictor=Predictor(model=None, X_test=None, test_features=use_test_features, feature_names=use_feature_names)
        top,bottom = predictor.return_results(average_by_column)
        predictor.save_results(top, bottom)
        return top, bottom




    def check(self):
        print("加载和预处理数据")
        dataloader = DataLoader()
        train_df, test_df=dataloader.load_and_preprocess_data()
        print("done\n特征工程")
        print("done\n准备训练数据")
        fea_engineer = FeatureEngineer(train_df, test_df)
        X_train, y_train, X_test, test_features, feature_names = fea_engineer.prepare_training_data()
        print("done\n训练模型")
        trainer = ModelTrainer(X_train, y_train, feature_names)
        trainer.check_data_leakage()
        trainer.comprehensive_data_check()
        return

if __name__ == '__main__':
    num=input("run:1\ncheck:2\nrun2:3\n")
    if num=="1":
        predictor = CompleteStockPredictor()
        top, bottom=predictor.run()
        print(f"top: {top}\nbottom: {bottom}")
    elif num=="2":
        predictor = CompleteStockPredictor()
        predictor.check()
    elif num=="3":
        predictor = CompleteStockPredictor()
        top, bottom=predictor.run2()
        print(f"top: {top}\nbottom: {bottom}")
    else:
        print("1/2")