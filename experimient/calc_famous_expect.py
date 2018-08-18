from rx import Observable, Observer
from utils.json_load import loader


class CalcFamousExpectObserver(Observer):
    def __init__(self):
        self.cnt = 0
        self.sum_found = 0
        self.cnt_juni = {"first": [0 for i in range(7)], "second": [
            0 for i in range(7)], "third": [0 for i in range(7)]}
        self.cnt_kbanme = [0 for i in range(6*6*6)]
        self.cnt_kfound = [0 for i in range(6*6*6)]

    def on_next(self, value):
        for race_data in value:
            try:
                print(int(race_data["result"]["refund"]))
                odds = race_data["odds"]
                oddslist = []
                if odds != None:
                    self.cnt += 1
                    for i in range(1, 7):
                        for j in range(1, 7):
                            for k in range(1, 7):
                                try:
                                    a = float(odds[i][j][k])
                                    oddslist.append((a, (i, j, k)))
                                except:
                                    continue
                else:
                    raise
                oddslist.sort()
                n = len(oddslist)

                first = int(race_data["result"]["first"])
                second = int(race_data["result"]["second"])
                third = int(race_data["result"]["third"])
                found = int(race_data["result"]["refund"])
                self.sum_found += found
                for i in range(n):
                    if oddslist[i][1] == (first, second, third):
                        self.cnt_kbanme[i] += 1
                        self.cnt_kfound[i] += found
                        break

            except:
                print("Except")

    def on_completed(self):
        print("Done!")
        print("レース数: ", self.cnt)
        print("合計額: ", self.sum_found)
        print("人気順当たり回数: ", self.cnt_kbanme)
        print("人気順当たり総額: ", self.cnt_kfound)

    def on_error(self, error):
        print("Error Occurred: {0}".format(error))


source = Observable.create(loader)
source.subscribe(CalcFamousExpectObserver())
