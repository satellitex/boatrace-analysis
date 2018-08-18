from rx import Observable, Observer
from utils.json_load import loader


class CalcOddsExpectObserver(Observer):
    def __init__(self):
        self.cnt = 0
        self.cnt_manken = 0
        self.manken_yen = 0
        self.cnt_juni = {"first": [0 for i in range(7)], "second": [
            0 for i in range(7)], "third": [0 for i in range(7)]}

        self.odds_cnt = 0
        self.man_num = 0

    def on_next(self, value):
        for race_data in value:
            self.cnt += 1
            try:
                print(int(race_data["result"]["refund"]))
                odds = race_data["odds"]
                if odds != None:
                    self.odds_cnt += 1
                    for i in range(1, 4):
                        for j in range(1, 7):
                            for k in range(1, 7):
                                try:
                                    a = float(odds[i][j][k])
                                    if a >= 100:
                                        self.man_num += 1
                                except:
                                    continue

                first = int(race_data["result"]["first"])
                second = int(race_data["result"]["second"])
                third = int(race_data["result"]["third"])

                if int(race_data["result"]["refund"]) >= 10000 and first <= 3:
                    self.cnt_manken += 1
                    self.manken_yen += int(race_data["result"]["refund"])
                    self.cnt_juni["first"][first] += 1
                    self.cnt_juni["second"][second] += 1
                    self.cnt_juni["third"][third] += 1
            except:
                print("Except")

    def on_completed(self):
        print("Done!")
        print("レース数: ", self.cnt)
        print("万券の個数: ", self.cnt_manken)
        print("総万券額: ", self.manken_yen)
        print("平均万券額: ", self.manken_yen/self.cnt_manken)
        print("[順位][レーン]: ", self.cnt_juni)
        print("オッズの情報がとれた件数: ", self.odds_cnt)
        print("総合万馬券数: ", self.man_num)
        print("平均万馬券数: ", self.man_num / self.odds_cnt)

    def on_error(self, error):
        print("Error Occurred: {0}".format(error))


source = Observable.create(loader)
source.subscribe(CalcOddsExpectObserver())
