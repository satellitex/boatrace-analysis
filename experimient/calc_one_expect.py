from rx import Observable, Observer
from utils.json_load import loader


class CalcOneExpectObserver(Observer):
    def __init__(self):
        self.cnt = 0
        self.odds_cnt = 0
        self.cnt_juni = {"first": [0 for i in range(7)], "second": [
            0 for i in range(7)], "third": [0 for i in range(7)]}

    def on_next(self, value):
        for race_data in value:
            self.cnt += 1
            try:
                print(int(race_data["result"]["refund"]))

                first = int(race_data["result"]["first"])
                if first == 1:
                    second = int(race_data["result"]["second"])
                    third = int(race_data["result"]["third"])
                    self.cnt_juni["first"][first] += 1
                    self.cnt_juni["second"][second] += 1
                    self.cnt_juni["third"][third] += 1
            except:
                print("Except")

    def on_completed(self):
        print("Done!")
        print("レース数: ", self.cnt)
        print("[順位][レーン]: ", self.cnt_juni)
        print("オッズの情報がとれた件数: ", self.odds_cnt)

    def on_error(self, error):
        print("Error Occurred: {0}".format(error))


source = Observable.create(loader)
source.subscribe(CalcOneExpectObserver())
