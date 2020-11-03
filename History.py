import random
from ctypes import *
import json
from datetime import datetime, timedelta
import pandas as pd

# ZORRO_T6_FILE_TO_READ = 'History/GER30_2019.t6'


class AssetData:

    # name: spread, min_shares, leverage
    DEFAULT = [0, 0, 1]

    # index
    GER30 = [5, 0.001, 20]
    UK100 = [5, 0.001, 20]

    # forex
    EURUSD = [0.00013, 500, 30]  # threshold 0.0062, sl 115, ts 0.0055


class Trader:

    # balance, max_risk
    DEFAULT = [100, 0.12]
    FULL_RISK = [100, 1]
    HALF_RISK = [100, 0.5]

    FIVEHUN_LOWRISK = [500, 0.14]
    FIVEHUN_HALFRISK = [500, 0.5]
    FIVEHUN_FULLRISK = [500, 1]

    TEST = [1000000, 0.14]


def reduce(func, lis):
    if len(lis) == 0:
        print("Error in reduce")
        return
    if len(lis) == 1:
        return lis[0]
    else:
        return reduce(func, [func(lis[0], lis[1])] + lis[2:])


class ZorroT6(Structure):
    """
    struct {
      DATE  time; // timestamp of the end of the tick in UTC, OLE date/time format (double float)
      float fHigh,fLow;
      float fOpen,fClose;
      float fVal,fVol; // additional data, like ask-bid spread, volume etc.
    }
    """
    _fields_ = [('time', c_double),
                ('fHigh', c_float),
                ('fLow', c_float),
                ('fOpen', c_float),
                ('fClose', c_float),
                ('fVal', c_float),
                ('fVol', c_float)]


class Tick:

    def __init__(self, time, bid, ask):
        self.time = time
        self.bid = bid
        self.ask = ask

        self.price = ask

    def to_dict(self):
        return {"price": float(self.price), "time": self.time.strftime("%H:%M:%S %d.%m.%Y")}


class History:

    def __init__(self, asset_name, start, end=None, accuracy="M1"):

        if not end:
            end = start
        self.asset_name = asset_name
        self.hist = []
        self.accuracy = accuracy

        folder_name = "History/" + asset_name + "/" + accuracy + "/"
        file_name = asset_name + "_" + str(start) + "_" + str(end) + "_" + accuracy + ".csv"

        if accuracy == "M1":
            try:
                file = open(folder_name + file_name, "r")
                lines = file.readlines()
                file.close()
                for line in lines:
                    price, time = line.split(",")
                    time = self.datetime_from_string(time, accuracy)
                    price = float(price)
                    self.hist.append({"price": price, "time": time.strftime("%H:%M:%S %d.%m.%Y")})
            except Exception as e:
                print(e)
                print("No csv file could be found, creating a new one ...")
                self.hist = self.get_history_list_M1(asset_name, start, end)

        elif accuracy == "ticks":

            try:
                # use existent file
                file = open(folder_name + file_name)
                lines = file.readlines()
                file.close()
                self.hist = []
                for line in lines:
                    price, time = line.split(",")
                    time = self.datetime_from_string(time, accuracy)
                    price = float(price)
                    self.hist.append({"price": price, "time": time.strftime("%H:%M:%S %d.%m.%Y")})
            except Exception:
                # create new file from avail data
                print("No csv file could be found, creating a new one ...")
                self.hist = self.get_hist_list_ticks(asset_name, start, end)
        else:
            self.hist = self.get_history_list(asset_name, start, end)

        self.timeframe = self.hist[0]["time"] + "-" + self.hist[-1]["time"]
        self.prices = [elem["price"] for elem in self.hist]
        self.times = [elem["time"] for elem in self.hist]

    # <editor-fold desc="history reading stuff">
    def get_history_list(self, asset, start_year, end_year):

        # returns M1 by default

        history_list = []
        for i in range(end_year - start_year + 1):
            path = "History/" + asset + "/T6/" + asset + "_" + str(start_year + i) + ".t6"
            history_list += self.readT6(path)
        history_list.reverse()
        return history_list

    def get_hist_list_ticks(self, asset_name, start_year, end_year, all_time=False):

        print("Generating history from source tick files for", asset_name, start_year, "-", end_year)

        folder_name = "History/" + asset_name + "/tick_resources/"

        hist_list = []

        if all_time:
            start_year = 1990

        for i in range(start_year, end_year + 1):
            file_name = asset_name + "_" + str(i) + "_ticks.csv"

            try:
                out_folder_name = "History/" + self.asset_name + "/ticks/"

                self.hist = History.generic_tick_2_csv(folder_name, file_name, out_folder_name)

                """with open(folder_name + file_name) as file:
                    ticks = []
                    lines = file.readlines()
                    lines = lines[1:]
                    for line in lines:
                        date_time, ask, bid, _ = line.split(",")
                        ticks.append(Tick(self.datetime_from_string(date_time, "ticks"), bid, ask))
                    file.close()

                    to_ret = [tick.to_dict() for tick in filter(lambda x: start_year <= x.time.year <= end_year, ticks)]
                    self.create_csv_file(to_ret, self.asset_name, start_year, end_year, "ticks")  # todo path
                    hist_list.append(to_ret)"""
            except Exception as e:
                print(e, "Could not read tick data for year", i)

        return hist_list

    def get_history_list_M1(self, asset, start_year, end_year):
        """:returns list of dicts with price:p and time:t"""
        history_list = []
        for i in range(end_year - start_year + 1):
            folder = "History/" + asset + "/T6/"
            path = folder + asset + "_" + str(start_year + i) + ".t6"
            tmp = self.readT6(path)
            to_add = []
            for elem in tmp:
                to_add.append({"price": elem["meanPrice"], "time": elem["clearTime"]})
            to_add.reverse()
            history_list += to_add

        self.create_csv_file(history_list=history_list, asset=asset, start_year=start_year, end_year=end_year, acc="M1")

        return history_list

    @staticmethod
    def datetime_from_string(string, acc):
        string = string.replace("\n", "")
        if not acc:
            return datetime.strptime(string, "%H:%M:%S %d.%m.%Y")
        if acc == "ticks":
            return datetime.strptime(string, "%Y%m%d %H:%M:%S.%f")
        if acc == "M1":
            return datetime.strptime(string, "%H:%M %d.%m.%Y")
        else:
            print("Unknown accuracy, cannot create datetime object")
            return None

    @staticmethod
    def generic_tick_2_csv(in_folder, in_file, output_folder=""):

        history = []

        in_file_params = (in_folder+in_file, "r")
        out_file_params = (output_folder + in_file.replace("ticks", "custom_ticks"), "w")

        with open(*in_file_params) as in_file:
            with open(*out_file_params) as out_file:
                counter = 0
                line = in_file.readline()  # usually first line is desriptors
                while line:
                    line = in_file.readline()
                    if line:
                        if counter % 100000 == 0:
                            print(counter)
                        counter += 1
                        time_str, ask, bid, vol = line.split(",")
                        new_time = History.datetime_from_string(time_str, "ticks")
                        tick = Tick(new_time, bid, ask).to_dict()
                        history.append(tick)
                        out_file.writelines([str(tick["price"]) + "," + str(tick["time"]) + "\n"])
                    else:
                        break

        in_file.close()
        out_file.close()
        return history

    @staticmethod
    def create_csv_file(history_list, asset, start_year, end_year, acc):
        folder_name = "History/" + asset + "/" + acc + "/"
        file_name = asset + "_" + str(start_year) + "_" + str(end_year) + "_" + acc + ".csv"
        try:
            file = open(folder_name + file_name, "x")
            for quote in history_list:
                file.write(str(quote["price"]) + "," + str(quote["time"]) + "\n")
            file.close()

        except Exception as e:
            print(e, "csv file could not be created")

    @staticmethod
    def T62M1(h):
        dates = [e["clearTime"] for e in h]
        prices = [e["meanPrice"] for e in h]
        res = []
        for i in range(len(dates)):
            res.append({"time": dates[i], "price": prices[i]})
        return res

    def get_history_to_panda(self, asset, start_year, end_year):
        h = self.get_history_list(asset, start_year, end_year)
        dates = [e["clearTime"] for e in h]
        prices = [e["meanPrice"] for e in h]
        frame = pd.DataFrame(prices, index=dates, columns=["Price"])
        return frame

    @staticmethod
    def hist_to_pand(h):
        dates = [e["clearTime"] for e in h]
        prices = [e["meanPrice"] for e in h]
        frame = pd.DataFrame(prices, index=dates, columns=["Price"])
        return frame

    @staticmethod
    def m1_hist_to_pand(h):
        dates = [e["time"] for e in h]
        prices = [e["price"] for e in h]
        frame = pd.DataFrame(prices, index=dates, columns=["Price"])
        return frame

    def readT6(self, path, p=False):

        with open(path, 'rb') as file:
            result = []
            x = ZorroT6()
            while file.readinto(x) == sizeof(x):
                result.append({
                    'time': x.time,
                    'clearTime': self.ole2datetime(x.time).strftime('%H:%M %d.%m.%Y'),
                    'fHigh': x.fHigh,
                    'fLow': x.fLow,
                    'fOpen': x.fOpen,
                    'fClose': x.fClose,
                    'meanPrice': (x.fHigh + x.fLow) / 2,
                    'fVal': x.fVal,
                    'fVol': x.fVol})
        if p:
            print(json.dumps(result[:6], indent=2))
        return result

    @staticmethod
    def datetime2ole(date):
        date = datetime.strptime(date, '%d-%b-%Y')
        OLE_TIME_ZERO = datetime(1899, 12, 30)
        delta = date - OLE_TIME_ZERO
        return float(delta.days) + (float(delta.seconds) / 86400)  # 86,400 seconds in day

    @staticmethod
    def ole2datetime(oledt):
        OLE_TIME_ZERO = datetime(1899, 12, 30)
        return OLE_TIME_ZERO + timedelta(days=float(oledt))

    # </editor-fold>

    """
    Use with caution! In general, this is not what you want to do! Use indicators instead!
    """
    def __transform_prices(self, transformation_function, function_params):
        transformed_prices = transformation_function(self.prices, *function_params)
        self.hist = [{"price": transformed_prices[i], "time": self.times[i]} for i in range(len(transformed_prices))]
        self.prices = [p["price"] for p in self.hist]
