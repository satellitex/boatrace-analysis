from rx import Observable, Observer
from utils.json_load import loader

class CalcOddsExpectObserver(Observer):
    def __init__(self):
        self.cnt = 0
        self.cnt_manken = 0
        self.cnt_juni = {"first":[0 for i in range(7)],"second":[0 for i in range(7)],"third":[0 for i in range(7)]}

    def on_next(self, value):
        for race_data in value:
            self.cnt += 1
            print(int(race_data["result"]["refund"]))
            if int(race_data["result"]["refund"]) >= 10000:
                first = int(race_data["result"]["first"])
                second = int(race_data["result"]["second"])
                third = int(race_data["result"]["third"])
                self.cnt_manken += 1
                self.cnt_juni["first"][first] += 1
                self.cnt_juni["second"][second] += 1
                self.cnt_juni["third"][third] += 1





    def on_completed(self):
        print("Done!")
        print( self.cnt )
        print( self.cnt_manken )
        print( self.cnt_juni )

    def on_error(self, error):
        print("Error Occurred: {0}".format(error))


source = Observable.create(loader)
source.subscribe(CalcOddsExpectObserver())
