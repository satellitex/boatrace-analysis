from rx import Observable, Observer
from utils.json_load import loader

class CalcOddsExpectObserver(Observer):
    def __init__(self):
        self.cnt_manken = 0
        self.cnt_juni = [{},{},{}]

    def on_next(self, value):
        print(value)



    def on_completed(self):
        print("Done!")

    def on_error(self, error):
        print("Error Occurred: {0}".format(error))


source = Observable.create(loader)
source.subscribe(CalcOddsExpectObserver())
