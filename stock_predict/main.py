from src.data_loader import DataLoader
from src.feature_engineer import FeatureEngineer
from src.model_trainer import ModelTrainer
from src.predictor import Predictor

class CompleteStockPredictor:
    def __init__(self):
        print("strat")


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


if __name__ == '__main__':
    predictor = CompleteStockPredictor()
    top, bottom=predictor.run()
    print(f"top: {top}\nbottom: {bottom}")