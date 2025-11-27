import os

def delete_file(num):
    if num == "1":
        paths=[r"D:\codeC\stock-prediction\stock_predict\outputs\results\result.csv",
               r"D:\codeC\stock-prediction\stock_predict\data\feature\test_features.csv",
               r"D:\codeC\stock-prediction\stock_predict\data\feature\train_features.csv"]
        for path in paths:
            try:
                if os.path.exists(path):
                    os.remove(path)
                    print(f"delete {path}")
                else:
                    print(f"no {path}")
            except Exception as e:
                print(f"{e}")
    elif num == "2":
        paths = [r"D:\codeC\stock-prediction\stock_predict\data\feature\test_features.csv",
                 r"D:\codeC\stock-prediction\stock_predict\data\feature\train_features.csv"]
        for path in paths:
            try:
                if os.path.exists(path):
                    os.remove(path)
                    print(f"delete {path}")
                else:
                    print(f"no {path}")
            except Exception as e:
                print(f"{e}")
    else:
        print("1/2")


if __name__ == "__main__":
    num=input("1:全删\n2:删feature\n")
    delete_file(num)