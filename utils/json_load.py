import os
import json



def loader(observer):
    pwd = os.environ["ANALYSIS_BOAT_HOME"]
    for file in os.listdir(pwd+"/dataes"):
        file_path = pwd + "/dataes/" + file
        print(file_path)

        # jsonファイルを読み込む
        f = open(file_path)
        # jsonデータを読み込んだファイルオブジェクトからPythonデータを作成
        observer.on_next(json.load(f))
        f.close()