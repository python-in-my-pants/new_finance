from scipy.interpolate import interp2d
from scipy.signal import *
import matplotlib.pyplot as plt
from matplotlib import cm
import datetime
import numpy as np
import statistics
from History import AssetData, Trader, History
from scipy import *


class Trend:

    def __init__(self, data, t=None):
        self.data = data
        self.start_time = data[0]["time"]
        self.end_time = data[-1]["time"]
        self.prices = [e["price"] for e in data]

        self.len = len(self.prices)
        self.height = self.prices[-1] - self.prices[0]

        self.true_type = "Up  " if self.height > 0 else "Down"
        self.type = self.true_type if not t else t
        self.type_coherence = self.type == self.true_type

        # max movement in favored direction
        self.amplitude = max(self.prices) - self.prices[0] if self.type == "Up  " else self.prices[0] - min(self.prices)
        self.avg = sum(self.prices) / self.len
        self.med = statistics.median(self.prices)
        self.max_dd = self.get_max_dd()  # amplitude of highest move in diametral direction of the trend

        self.frac_dim = self.get_frac_dim()
        self.frad_dims = [Analyzer.get_frac_dim_static(self.prices[:i+1]) for i in range(len(self.prices))]

        self.momentum = self.height/self.len

        self.derivate1 = [0] + Analyzer.derive(self.prices)
        self.derivate2 = [0] + Analyzer.derive(self.derivate1)

    def print_prices(self):
        for d in self.data:
            print(d)

    def get_max_dd(self):
        curr_pivot = self.prices[0]
        max_dd = 0
        for p in self.prices:
            if self.type == "Up  ":
                if p > curr_pivot:
                    curr_pivot = p
                else:
                    if curr_pivot - p > max_dd:
                        max_dd = curr_pivot - p
            if self.type == "Down":
                if p < curr_pivot:
                    curr_pivot = p
                else:
                    if curr_pivot - p < max_dd:
                        max_dd = curr_pivot - p
        return max_dd if self.type == "Up  " else -max_dd

    def get_dds(self):

        class Drawdown:

            def __init__(self, data, trend_ending=False):
                # beware that these are the indices in the draw down, not in the trend!
                self.draw_down_data = data
                self.length = len(data)
                self.height = -min(data)-data[0] if data[-1] > data[0] else max(data) - data[0]
                self.height_index = self.draw_down_data.index(min(data) if data[-1] >= data[0] else max(data))
                self.trend_ending = trend_ending

            def __str__(self):
                return "Len: {:d}\t\tHeight: {:+8.2f}\t\tTrend ending: {}".\
                    format(self.length, self.height, self.trend_ending)

        start = self.prices[0]
        pivot = start
        dd = False
        dds = 0
        draw_downs = []

        if self.type == "Up  ":

            for i in range(len(self.prices)):
                p = self.prices[i]

                if p >= pivot:
                    pivot = p

                    # did a draw down end here?
                    if dd:
                        # create draw down object
                        draw_downs.append(Drawdown(self.prices[dds:i + 1]))

                    dd = False
                else:
                    # it's the start of a draw down
                    if i < len(self.prices) - 1:
                        if not dd:
                            dds = i  # draw down start index
                            dd = True
                    else:
                        # it's the last price of the trend and we're still in a draw down
                        draw_downs.append(Drawdown(self.prices[dds:], trend_ending=True))
        else:
            for i in range(len(self.prices)):
                p = self.prices[i]

                if p <= pivot:
                    pivot = p

                    # did a draw down end here?
                    if dd:
                        # create draw down object
                        draw_downs.append(Drawdown(self.prices[dds:i + 1]))

                    dd = False
                else:
                    # it's the start of a draw down
                    if i < len(self.prices) - 1:
                        if not dd:
                            dds = i  # draw down start index
                            dd = True
                    else:
                        # it's the last price of the trend and we're still in a draw down
                        draw_downs.append(Drawdown(self.prices[dds:], trend_ending=True))

        return draw_downs

    def get_frac_dim(self):
        p = self.len // 2
        h1 = self.prices[:self.len // 2]
        h2 = self.prices[self.len // 2:]
        n1 = (max(h1) - min(h1)) / p
        n2 = (max(h2) - min(h2)) / p
        n3 = (max(self.prices) - min(self.prices)) / self.len
        return 1 if (n1 + n2 <= 0 or n3 <= 0) else (np.log(n1 + n2) - np.log(n3)) / np.log(2)

    def __str__(self):
        return self.__repr__()

    def __repr__(self):
        return "Start: {}\n  End: {}\n\n  Len: {:5d}\t\tHeight: {:8.2f}\t\t Amp: {:8.2f}\n  Avg: {:8.2f}\t\t" \
               "   Med: {:8.2f}\t   MaxDD: {:8.2f}\t\tFracDim: {:f}\n\n Type: {}\t\t True type: {}\t  " \
               "Type coher:    {}\n\n  Mom: {:+8.2f}".\
            format(self.start_time, self.end_time, self.len, self.height, self.amplitude, self.avg, self.med,
                   self.max_dd, self.frac_dim, self.type, self.true_type, self.type_coherence, self.momentum)


class Analyzer:

    def __init__(self, hist_obj, min_trend_h=20, realistic=False, price_filter=None):

        self.hist_obj = hist_obj
        self.hist = hist_obj.hist
        self.volatility = self.get_hist_volatility(whole_timeframe=True)[0]
        self.trend_list = self.get_trends(self.hist, min_trend_h=min_trend_h, realistic=realistic)

    # <editor-fold desc="GETTER">
    def get_volatility(self, data):
        return self.get_std_dev(data)**0.5

    def get_hist_volatility(self, timeframe="M", whole_timeframe=False):
        if not whole_timeframe:
            f = -1
            if timeframe is "D":
                f = 894
            if timeframe is "W":
                f = 4469
            if timeframe is "M":
                f = 17878
            if timeframe is "Y":
                f = 214541
            if f < 0:
                return
        else:
            f = len(self.hist)

        p = [x["price"] for x in self.hist]
        std_devs = []

        for i in range(0, len(self.hist) // f):
            if (i + 1) * f < len(p):
                period_p = p[i * f:(i + 1) * f]
            else:
                period_p = p[i * f:]

            std_devs.append(self.get_std_dev(period_p))

        return [x ** 0.5 for x in std_devs]

    @staticmethod
    def get_frac_dim_static(prices):
        if len(prices) == 1:
            return 1
        prices_len = len(prices)
        p = prices_len // 2
        h1 = prices[:p]
        h2 = prices[p:]
        n1 = (max(h1) - min(h1)) / p
        n2 = (max(h2) - min(h2)) / p
        n3 = (max(prices) - min(prices)) / prices_len
        return 1 if (n1 + n2 <= 0 or n3 <= 0) else (np.log(n1 + n2) - np.log(n3)) / np.log(2)

    @staticmethod
    def get_std_dev(data):
        return np.std(data)

    @staticmethod
    def get_trends(data, min_trend_h=80, realistic=False):

        uptrend = -1
        begin = data[0]["price"]
        threshold = min_trend_h

        for i in range(len(data)):
            last = data[i]["price"]
            if last > begin + threshold:
                uptrend = True
                break
            if last < begin - threshold:
                uptrend = False
                break

        high = low = data[0]["price"]
        high_index = low_index = break_index = 0
        trends = []

        for i in range(len(data)):

            last = data[i]["price"]

            if uptrend:
                if last > high:
                    high = last
                    high_index = i
                else:
                    if last < high - threshold:
                        # count last peak as end of high trend and as start of low trend

                        # make trend for last up trend
                        if realistic:
                            trends.append(Trend(data[break_index:i + 1], t="Up  "))
                        else:
                            trends.append(Trend(data[low_index:high_index + 1]))

                        # change index
                        low = last
                        low_index = i
                        break_index = i
                        uptrend = False
            else:
                if last < low:
                    low = last
                    low_index = i
                else:
                    if last > low + threshold:
                        if realistic:
                            trends.append(Trend(data[break_index:i + 1], t="Down"))
                        else:
                            trends.append(Trend(data[high_index:low_index + 1]))

                        high = last
                        high_index = i
                        break_index = i
                        uptrend = True

        return trends

    def get_avg_heights(self, max_len=100):
        avg_heights = []
        for i in range(max_len):
            sum_heights = 0
            counter = 0
            for t in self.trend_list:
                if t.len == i:
                    sum_heights += t.height
                    counter += 1
            avg_h = sum_heights / counter if counter > 0 else 0
            avg_heights.append(avg_h)
        return avg_heights

    def get_med_heights(self, max_len=500):

        med_heights_up = [[] for _ in range(max_len)]
        med_heights_down = [[] for _ in range(max_len)]

        for t in self.trend_list:
            l = min(t.len, max_len - 1)
            if t.type == "Up  ":
                med_heights_up[l].append(t.height)
            if t.type == "Down":
                med_heights_down[l].append(t.height)

        for i in range(len(med_heights_up)):
            if med_heights_up[i]:
                med_heights_up[i] = statistics.median(med_heights_up[i])
            else:
                med_heights_up[i] = 0

        for i in range(len(med_heights_down)):
            if med_heights_down[i]:
                med_heights_down[i] = statistics.median(med_heights_down[i])
            else:
                med_heights_down[i] = 0

        return med_heights_up, med_heights_down

    def get_len_counts(self, max_len=500):
        counts = [0 for _ in range(max_len)]
        for t in self.trend_list:
            l = t.len if t.len <= max_len - 1 else max_len - 1
            counts[l] += 1
        return counts

    # </editor-fold>

    def find_trade_indicating_pattern(self, bin_size=3, min_pattern_len=3, max_pattern_len=8):

        def round_to_bin_size(n):
            rounded = np.round(n)
            rest = rounded % bin_size
            if rest >= bin_size:
                return (rounded//bin_size) + bin_size
            else:
                return rounded // bin_size

        rounded_prices = [round_to_bin_size(price) for price in self.hist_obj.prices]

        for pattern_len in range(min_pattern_len, max_pattern_len):

            for price_index in range(len(rounded_prices)):

                selected_prices = rounded_prices[price_index:price_index+pattern_len]
                normalized_selected_prices = [p-selected_prices[0] for p in selected_prices]

    def trend_len_height_cross_correlate(self):

        trend_heights = [trend.height for trend in self.trend_list]
        trend_lens = [trend.len for trend in self.trend_list]
        convolved = fftconvolve(trend_lens, trend_heights, mode="same")

        h_norm = trend_heights / np.linalg.norm(trend_heights)
        l_norm = trend_lens / np.linalg.norm(trend_lens)
        c_norm = convolved / np.linalg.norm(convolved)

        Plotter.plot_general_same_y(list(range(len(self.trend_list))),
                                    [l_norm, h_norm, c_norm],
                                    x_label="Trend #",
                                    y_labels=["Lengths", "Heights", "Cross correlation"])

    def autocorrelate(self, data):
        data_dict = {
            "trend len": [trend.len for trend in self.trend_list],
            "trend height": [trend.height for trend in self.trend_list]
        }

        if data in list(data_dict.keys()):
            signal = data_dict[data]
            x_label = data
        else:
            signal = data
            x_label = "Data"

        signal -= mean(signal)

        autocorrelated = correlate(signal, signal, mode="full")
        autocorrelated_norm = \
            autocorrelated[int(np.rint(autocorrelated.size/2)-1):] / np.var(autocorrelated)

        Plotter.plot_general(list(range(len(signal))), autocorrelated_norm)

    def get_win_loss_rythm(self, p=True):

        rythm = []
        counter = 0
        good_chain = True
        for t in self.trend_list:
            if t.type == t.true_type:
                if not good_chain:
                    rythm.append((counter, "good"))
                    counter = 0
                    good_chain = True
                counter += 1
            else:
                if good_chain:
                    rythm.append((counter, "bad"))
                    counter = 0
                    good_chain = False
                counter += 1

        if p:
            for elem in rythm:
                print(*elem) if elem[1] == "good" else print("        ", *elem)
            goods = list(filter(lambda a: a[1] == "good", rythm))
            bads = list(filter(lambda a: a[1] == "bad", rythm))
            print("avg + len:", sum([x[0] for x in goods])/len(goods),
                  "\navg - len:", sum([x[0] for x in bads])/len(bads))

        return rythm

    def print_trends(self, limit=10, print_prices=False, print_dds=False, direction_key=None, coherence_key=None):
        if direction_key and direction_key == "Up":
            direction_key = "Up  "
        for t in self.trend_list[:limit]:
            if direction_key is None or t.type == direction_key:
                if coherence_key is None or t.type_coherence == coherence_key:
                    print(t)
                    if print_dds:
                        for dd in t.get_dds():
                            print(dd)
                    if print_prices:
                        for i in range(len(t.prices)):
                            print("{:+8.2f} {:+8.2f} {:+8.2f}".format(t.prices[i], t.derivate1[i], t.derivate2[i]))
            print("---------------------")

    def SIM_trend_follow(self, p=False):
        longs = list(filter(lambda y: y.type == "Up  ", self.trend_list))
        shorts = list(filter(lambda y: y.type == "Down", self.trend_list))

        long_wins = sum([(x.height if x.height > 0 else 0) for x in longs])
        long_losses = sum([(x.height if x.height <= 0 else 0) for x in longs])

        short_losses = -sum([(x.height if x.height > 0 else 0) for x in shorts])
        short_wins = -sum([(x.height if x.height <= 0 else 0) for x in shorts])

        if p:
            print("Long wins:  {:12.2f}\t\tLong losses:  {:12.2f}\n"
                  "Short wins: {:12.2f}\t\tShort losses: {:12.2f}\n\n"
                  "\t   Sum: {:12.2f}\n  Buy&Hold: {:12.2f}".
                  format(long_wins, long_losses, short_wins, short_losses,
                         long_wins + long_losses + short_wins + short_losses,
                         self.hist[-1]["price"] - self.hist[0]["price"]))

        return long_wins, long_losses, short_wins, short_losses, long_wins + long_losses + short_wins + short_losses

    @staticmethod
    def derive(seq):
        return [seq[i+1]-seq[i] for i in range(len(seq)-1)]

    # <editor-fold desc="Probability getter">
    def GTEP_len(self, curr_len, max_len=500):
        longer_trends = list(filter(lambda x: x.len >= curr_len, self.trend_list))
        my_list = [0 for _ in range(max_len)]
        for trend in longer_trends:
            if trend.len > max_len:
                my_list[-1] += 1
            else:
                my_list[trend.len - curr_len] += 1
        return [elem / len(longer_trends) for elem in my_list]

    def GTCP_len_single(self, curr_len, expected_continuation):
        if expected_continuation < 0:
            print("Expected continuation must be >= 0!")
            return -1
        trend_base = list(filter(lambda x: x.len >= curr_len, self.trend_list))
        longer_trends = list(filter(lambda x: x.len >= curr_len+expected_continuation, trend_base))
        return len(longer_trends)/len(trend_base)

    def GTEP_len_single(self, curr_len, expected_continuation):
        return 1-self.GTCP_len_single(curr_len, expected_continuation)

    # ------------------------

    def GTEP_h(self, curr_h, bin_size=10):  # todo same for amplitude (for realistic trends used)
        """

        :param curr_h:
        :param bin_size:
        :return: prob of trend ending 'key' to 'nextKey'-1 price units, so for len 10:
        0-9 more units up: 0.05
        10-19 more units up: ...

        prob, dass ein trend der schon ... lang ist, noch ... hoch geht
        """
        if curr_h >= 0:
            higher_trends = list(filter(lambda x: x.height >= curr_h, self.trend_list))
            my_dict = {}  # holds height:prob
            # make bins for trend heights
            m = max([elem.height for elem in higher_trends])
            for i in range(int(m / bin_size)):
                my_dict[i * bin_size] = 0
                for trend in higher_trends:
                    if i * bin_size + curr_h <= trend.height < (i + 1) * bin_size + curr_h:
                        my_dict[i * bin_size] += 1

            return {k: my_dict[k] / len(higher_trends) for k in my_dict.keys()}
        else:
            lower_trends = list(filter(lambda x: x.height <= curr_h, self.trend_list))
            my_dict = {}  # holds height:prob
            # make bins for trend heights
            m = min([elem.height for elem in lower_trends])
            for i in range(int(m / bin_size), int((2*curr_h/bin_size)-0.5)+1):
                my_dict[i * bin_size] = 0
                for trend in lower_trends:
                    if i * bin_size - curr_h <= trend.height < (i + 1) * bin_size - curr_h:
                        my_dict[i * bin_size] += 1

            return {k: my_dict[k] / len(lower_trends) for k in my_dict.keys()}

    def GTCP_h_single(self, curr_h, expected_continuation):
        if curr_h >= 0:
            if expected_continuation < 0:
                print("Expected continuation must be >= 0!")
                return -1
            trend_base = list(filter(lambda x: x.height >= curr_h, self.trend_list))
            higher_trends = list(filter(lambda x: x.height >= curr_h + expected_continuation, trend_base))
        else:
            if expected_continuation > 0:
                print("Expected continuation must be <= 0!")
                return -1
            trend_base = list(filter(lambda x: x.height <= curr_h, self.trend_list))
            higher_trends = list(filter(lambda x: x.height <= curr_h + expected_continuation, trend_base))
        return len(higher_trends) / len(trend_base)

    def GTEP_h_single(self, curr_h, expected_continuation, bin_size=10):
        if curr_h >= 0:
            if expected_continuation < 0:
                print("Expected continuation must be >= 0!")
                return -1
            trend_base = list(filter(lambda x: x.height >= curr_h, self.trend_list))
            higher_trends = len(list(filter(lambda x: curr_h + expected_continuation - (bin_size/2) <= x.height <=
                                                      curr_h + expected_continuation + (bin_size/2), trend_base)))
        else:
            if expected_continuation > 0:
                print("Expected continuation must be <= 0!")
                return -1
            trend_base = list(filter(lambda x: x.height <= curr_h, self.trend_list))
            higher_trends = len(list(filter(lambda x: curr_h + expected_continuation - (bin_size/2) <= x.height <=
                                                      curr_h + expected_continuation + (bin_size/2), trend_base)))
        return higher_trends / len(trend_base)

    # -----------------

    def GTEP_len_h(self, curr_len, curr_h, max_dur=60, bin_size=5, interpolated=False):

        # todo mind negative height for down trends

        filt = (lambda x: x.height >= curr_h) if (curr_h > 0) else (lambda x: x.height <= curr_h)

        longer_trends = list(filter(lambda x: curr_len + max_dur >= x.len >= curr_len, self.trend_list))
        higher_trends = list(filter(filt, self.trend_list))  # respectively lower trends
        longer_higher = list(filter(lambda x: x in higher_trends, longer_trends))

        if not longer_trends:
            print("No equally long or longer trends in the data set")
            return 0, None, curr_len
        if not higher_trends:
            print("No equally high or higher trends in the data set")
            return 0, curr_h, None
        if not longer_higher:
            print("No trends that are longer and higher than the given one are in the data set")
            return 0, None, None

        try:
            max_h = max(longer_higher, key=lambda x: x.height).height // 1
        except Exception:
            max_h = curr_h
        max_h = min(max_h, curr_h*5)

        try:
            max_l = max(longer_higher, key=lambda x: x.len).len // 1
        except Exception:
            max_l = curr_len
        column_h = int((max_h - curr_h) / bin_size) + 1

        occurence_matrix = [[0 for _ in range(column_h)] for _ in range(curr_len, max_l + 1)]

        for column_index, certain_len_column in enumerate(occurence_matrix):
            for i in range(len(certain_len_column)):
                # count number of trends with that len and height in that range
                occurence_matrix[column_index][i] += \
                    len(list(filter(lambda x: curr_h + (i * bin_size) <= x.height < curr_h + ((i + 1) * bin_size),
                                    list(filter(lambda x: x.len == curr_len + column_index, longer_higher)))))

        if interpolated:
            interp_f = interp2d(list(range(column_h)), list(range(curr_len, max_l + 1)), occurence_matrix, kind="cubic")

            for column_index, certain_len_column in enumerate(occurence_matrix):
                for i in range(len(certain_len_column)):
                    if occurence_matrix[column_index][i] == 0:
                        occurence_matrix[column_index][i] = float(interp_f(column_index, i))
                    occurence_matrix[column_index][i] /= len(longer_higher)
                    occurence_matrix[column_index][i] *= 100

        else:
            for lis in occurence_matrix:
                for elem in lis:
                    elem /= len(longer_higher)
                    elem *= 100

        return occurence_matrix, max_h, max_l

    def GTEP_len_h_single(self, curr_len, curr_h, l, h, bin_size=5):

        mat, max_h, max_l = self.GTEP_len_h(curr_len, curr_h, max_dur=l-curr_len, bin_size=bin_size, interpolated=True)

        if type(mat) is list:
            return mat[l-curr_len][(h-curr_h)//bin_size]
        else:
            return mat

    # </editor-fold>


class Plotter:

    def __init__(self, analyzer):
        self.analyzer = analyzer
        plt.rc('figure', facecolor="#333333", edgecolor="#333333")
        plt.rc('axes', facecolor="#353535", edgecolor="#000000")
        plt.rc('lines', color="#393939")
        plt.rc('grid', color="#121212")

    def plot_history(self, max_len=1440, start_date=None, end_date=None, full_hist=False):

        # <editor-fold desc="setting history">
        if end_date:
            end_date += datetime.timedelta(days=1)

        if full_hist:
            pan = self.analyzer.hist
        else:
            if start_date or end_date:
                if start_date:

                    start_index = 0
                    for index, elem in enumerate(self.analyzer.hist):
                        elem_time = History.datetime_from_string(elem["time"], acc=None)
                        if elem_time.date() >= start_date:
                            start_index = index
                            break
                    if not end_date:
                        pan = self.analyzer.hist[start_index:]
                    else:
                        end_index = 0
                        for index, elem in enumerate(self.analyzer.hist[start_index:]):
                            elem_time = History.datetime_from_string(elem["time"], acc=None)
                            if elem_time.date() >= end_date:
                                end_index = index
                                break
                        pan = self.analyzer.hist[start_index:end_index]
                else:
                    end_index = 0
                    for index, elem in enumerate(self.analyzer.hist):
                        elem_time = History.datetime_from_string(elem["time"], acc=None)
                        if elem_time.date() >= end_date:
                            end_index = index
                            break
                    pan = self.analyzer.hist[:end_index]
            else:
                max_len = min(max_len, self.analyzer.hist.__len__())
                pan = self.analyzer.hist[:max_len]
        # </editor-fold>

        fig, ax1 = plt.subplots()

        # plot price curve
        ax1.plot(list(range(len(pan))), [x["price"] for x in pan], color="black")
        ax1.set_xticks((list(range(0, max_len, int(max_len / 96)))))

        plt.grid(True)
        plt.xticks(rotation=90)
        plt.tight_layout()
        plt.show()

    def plot_history_trends_profit(self, max_len=1440, start_date=None, end_date=None, full_hist=False):

        # <editor-fold desc="setting history">
        if end_date:
            end_date += datetime.timedelta(days=1)

        if full_hist:
            pan = self.analyzer.hist
        else:
            if start_date or end_date:
                if start_date:

                    start_index = 0
                    for index, elem in enumerate(self.analyzer.hist):
                        elem_time = History.datetime_from_string(elem["time"], acc=None)
                        if elem_time.date() >= start_date:
                            start_index = index
                            break
                    if not end_date:
                        pan = self.analyzer.hist[start_index:]
                    else:
                        end_index = 0
                        for index, elem in enumerate(self.analyzer.hist[start_index:]):
                            elem_time = History.datetime_from_string(elem["time"], acc=None)
                            if elem_time.date() >= end_date:
                                end_index = index
                                break
                        pan = self.analyzer.hist[start_index:end_index]
                else:
                    end_index = 0
                    for index, elem in enumerate(self.analyzer.hist):
                        elem_time = History.datetime_from_string(elem["time"], acc=None)
                        if elem_time.date() >= end_date:
                            end_index = index
                            break
                    pan = self.analyzer.hist[:end_index]
            else:
                max_len = min(max_len, self.analyzer.hist.__len__())
                pan = self.analyzer.hist[:max_len]
        # </editor-fold>

        trends = []
        lens = [t.len for t in self.analyzer.trend_list]
        for i in range(len(self.analyzer.trend_list)):
            if sum([x.len for x in trends]) + self.analyzer.trend_list[i].len <= len(pan):
                trends.append(self.analyzer.trend_list[i])
            else:
                break

        fig, ax1 = plt.subplots()

        # plot price curve
        ax1.plot(list(range(len(pan))), [x["price"] for x in pan], color="black")
        ax1.set_xticks((list(range(0, max_len, int(max_len/96)))))
        #ax1.set_yticks((list(range(min_price, max_price, (max_price-min_price)/10))))

        a = [sum([x - 1 for x in lens[:i]]) for i in range(len(trends)+1)]
        a_pairs = [a[i:i + 2] for i in range(len(a))][:-1]
        b = [[x.prices[0], x.prices[-1]] for x in trends]

        ax1.set_xlabel("Time " + pan[0]["time"] + " - " + pan[-1]["time"])
        ax1.set_ylabel("Price", color="tab:red")

        for i in range(len(a_pairs)):
            ax1.plot(a_pairs[i], b[i], color=("green" if trends[i].type == "Up  " else "#8B0000"))

        ax1.tick_params(axis="y")
        plt.xticks(rotation=90)

        # -----------------------

        profits = []
        for trend in trends:
            if trend.type == "Up  ":
                if trend.height >= 0:
                    profits.append(trend.height)
                else:
                    profits.append(trend.height)
            else:
                if trend.height < 0:
                    profits.append(-trend.height)
                else:
                    profits.append(-trend.height)

        summed_profits = [sum(profits[:i]) for i in range(len(profits))]

        ax2 = ax1.twinx()
        ax2.set_ylabel('Profit', color='tab:blue')
        ax2.plot([x[0] for x in a_pairs], summed_profits)
        #ax2.tick_params(axis="y")

        plt.grid(True)
        plt.xticks(rotation=90)
        plt.tight_layout()
        plt.show()

    @staticmethod
    def plot_general(x, y, z=None, label1="X", label2="Y", plt_type="plt"):
        plt.rc('figure', facecolor="#333333", edgecolor="#333333")
        plt.rc('axes', facecolor="#353535", edgecolor="#000000")
        plt.rc('lines', color="#393939")
        plt.rc('grid', color="#121212")
        if plt_type == "plt":
            plt.plot(x, y)
        if plt_type == "scatter" and z:
            plt.scatter(x, y, s=0.5, c=z, cmap=cm.jet)
        if plt_type == "bar":
            plt.bar(x, y)
        plt.grid(True)
        plt.xlabel(label1)
        plt.ylabel(label2)
        plt.tight_layout()
        plt.show()

    @staticmethod
    def plot_general_same_y(x, ys, x_label="X", y_labels=["Y"]):

        plt.rc('figure', facecolor="#333333", edgecolor="#333333")
        plt.rc('axes', facecolor="#353535", edgecolor="#000000")
        plt.rc('lines', color="#393939")
        plt.rc('grid', color="#121212")

        if len(ys) < 1:
            return

        y_len = len(ys[0])
        for i in range(len(ys)):
            if len(ys[i]) != y_len:
                print("y dimensions must be all equal but have lengths: ", [len(y) for y in ys])
                return

        fig, ax1 = plt.subplots()

        for y in range(0, len(ys[1:])+1):
            ax1.plot(x, ys[y], label=y_labels[y])

        ax1.set_xlabel(x_label)
        ax1.tick_params(axis="y")
        plt.xticks(rotation=90)
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()

    @staticmethod
    def plot_general_multi_y(x, ys, x_label="X", y_labels=["Y"]):
        if len(ys) < 1:
            return

        y_len = len(ys[0])
        for i in range(len(ys)):
            if len(ys[i]) != y_len:
                print("y dimensions must be all equal but have lengths: ", [len(y) for y in ys])
                return

        fig, ax1 = plt.subplots()

        ax1.plot(x, ys[0])

        ax1.set_xlabel(x_label, color="tab:red")
        ax1.set_ylabel(y_labels[0])

        ax1.tick_params(axis="y")
        plt.xticks(rotation=90)

        cmap = {
            1: "red",
            2: "blue",
            3: "green",
            4: "yellow",
            5: "purple"
        }

        for y in range(1, len(ys[1:])+1):
            print(y)
            ax2 = ax1.twinx()
            ax2.set_ylabel(y_labels[y], color="tab:" + cmap[y])
            ax2.plot(x, ys[y], color=cmap[y])
            ax2.tick_params(axis="y")

        plt.grid(True)
        plt.xticks(rotation=90)

        plt.tight_layout()
        plt.show()

    def scatter_trends(self):
        x = [t.len for t in self.analyzer.trend_list]
        y = [t.height for t in self.analyzer.trend_list]
        colors = [("#ff0000" if t.height < 0 else "#00ff00") for t in self.analyzer.trend_list]
        p = plt.scatter(x, y, s=1, c=colors)
        ax = p.figure.add_subplot(111)
        ax.set_aspect('equal', adjustable='box')
        plt.grid(True)
        plt.xlabel("Trend length")
        plt.ylabel("Trend height")
        plt.subplots_adjust(0.17, 0.10, 0.99, 0.99)
        plt.show()

    def PSIM_trend_follow(self, min_trend_len, max_trend_len, realistic=False, step=1):

        tmp = self.analyzer.trend_list
        profits = []
        lws = []
        sws = []
        lls = []
        sls = []
        for i in range(min_trend_len, max_trend_len, step):
            self.analyzer.trend_list = self.analyzer.get_trends(self.analyzer.hist, min_trend_h=i, realistic=realistic)
            lw, sw, ll, sl, s = self.analyzer.SIM_trend_follow()
            profits.append(s)
            lws.append(lw)
            sws.append(sw)
            lls.append(ll)
            sls.append(sl)
            if i % 10 == 0:
                print("{:04.2f}".format((100 * (i - min_trend_len) / (max_trend_len - min_trend_len))), "% progress")

        self.analyzer.trend_list = tmp
        lls = list(map(lambda x: -x, lls))
        sls = list(map(lambda x: -x, sls))
        width = step/2

        plt.plot(list(range(min_trend_len, max_trend_len, step)), profits)
        plt.bar(list(range(min_trend_len, max_trend_len, step)), lws, color="#32cd32", width=width)
        plt.bar(list(map(lambda x: x + width, range(min_trend_len, max_trend_len, step))), sws, color="#008000", width=width)

        plt.bar(list(range(min_trend_len, max_trend_len, step)), lls, color="#dc143c", width=width)
        plt.bar(list(map(lambda x: x + width, range(min_trend_len, max_trend_len, step))), sls, color="#800000", width=width)

        plt.grid(True)
        plt.subplots_adjust(0.17, 0.10, 0.99, 0.99)
        plt.show()

    # <editor-fold desc="Plot trend distribution">
    def PTD_len_medh(self):
        max_len = 500
        med_heights_up, med_heights_down = self.analyzer.get_med_heights(max_len)

        plt.bar(list(range(max_len)), med_heights_up)
        plt.bar(list(range(max_len)), med_heights_down)
        plt.grid(True)
        plt.xlabel("Trend length")
        plt.ylabel("Trend med height")
        plt.subplots_adjust(0.17, 0.10, 0.99, 0.99)
        plt.show()

    def PTD_len_avgh(self):
        max_len = 100
        avg_heights = self.analyzer.get_avg_heights(max_len)

        plt.plot(list(range(max_len)), avg_heights)
        plt.grid(True)
        plt.xlabel("Trend length")
        plt.ylabel("Trend avg height")
        plt.subplots_adjust(0.17, 0.10, 0.99, 0.99)
        plt.show()

    def PTD_len_num(self):
        max_len = max([x.len for x in self.analyzer.trend_list])
        counts = self.analyzer.get_len_counts(max_len)

        plt.bar(list(range(max_len)), [100 * c / len(self.analyzer.trend_list) for c in counts])
        plt.grid(True)
        plt.xlabel("Trend length")
        plt.ylabel("Percentage of trends with that length")
        plt.subplots_adjust(0.17, 0.10, 0.99, 0.99)
        plt.show()

    def PTD_len_num_rel(self):
        max_len = max([x.len for x in self.analyzer.trend_list])
        counts = self.analyzer.get_len_counts(max_len)
        counts = counts[1:]

        plt.bar(list(range(1, max_len)), [c/(i+1) for i, c in enumerate(counts)])
        plt.grid(True)
        plt.xlabel("Trend length")
        plt.ylabel("trends with that length / length")
        plt.xticks(list(range(1, max_len, 10)))
        plt.tight_layout()
        plt.show()

    def PTD_len_num_bins(self, bins_w=10):
        max_len = 500
        counts = self.analyzer.get_len_counts(max_len)
        s_counts = []

        for i in range((max_len // bins_w)):
            s_counts.append(sum(counts[bins_w * i:(bins_w + 1) * i]))

        plt.bar(list(range(0, max_len, bins_w)), [100 * c / len(self.analyzer.trend_list) for c in s_counts], width=5)
        plt.grid(True)
        plt.xlabel("Trend length")
        plt.ylabel("Percentage of trends with that length")
        plt.xticks(list(range(0, max_len, bins_w * 5)))
        plt.subplots_adjust(0.17, 0.10, 0.99, 0.99)
        plt.show()
    # </editor-fold>

    # <editor-fold desc="Plot trend end probability">
    def PTEP_h(self, curr_h, bin_size=10):

        # mind negative trends with negative h
        d = self.analyzer.GTEP_h(curr_h, bin_size=bin_size)
        x = list(d.keys())
        y = [100*x for x in list(d.values())]
        y_summed = [sum(y[:i+1]) for i in range(len(y))]

        fig, ax1 = plt.subplots()

        ax1.plot(x, y, color="blue")

        ax1.set_xlabel("Price units from current height")
        ax1.set_ylabel("Prob of trend ending in that range from current height", color="blue")

        ax1.set_xticks(x)
        ax1.set_yticks(y)
        plt.xticks(rotation=90)
        plt.grid(True)

        ax2 = ax1.twinx()

        ax2.plot(x, y_summed, color="red")
        ax2.set_ylabel("Summed probability", color="red")
        ax2.set_yticks(list(range(0, 101, 5)))

        plt.grid(True)
        plt.title("{} TEP Height (height: {}, bin size {})".format(self.analyzer.hist_obj.asset_name, curr_h, bin_size), fontsize=21)
        fig.tight_layout()
        plt.show()

    # plot trend continuation prob
    def PTCP_h(self, curr_h, bin_size=10):

        # mind negative trends with negative h
        d = self.analyzer.GTEP_h(curr_h, bin_size=bin_size)
        x = list(d.keys())
        y = [100*x for x in list(d.values())]
        rev_y = [100-g for g in y]
        y_summed = [sum(y[:i+1]) for i in range(len(y))]#
        rev_sums = [100-g for g in y_summed]

        fig, ax1 = plt.subplots()

        ax1.plot(x, rev_y, color="blue")

        ax1.set_xlabel("Price units from current height")
        ax1.set_ylabel("Prob of trend continuing in that range from current height", color="blue")

        ax1.set_xticks(x)
        ax1.set_yticks(rev_y)
        plt.xticks(rotation=90)
        plt.grid(True)

        ax2 = ax1.twinx()

        ax2.plot(x, rev_sums, color="red")
        ax2.set_ylabel("Summed probability", color="red")
        ax2.set_yticks(list(range(0, 101, 5)))

        plt.grid(True)
        plt.title("{} TCP Height (height: {}, bin size {})".format(self.analyzer.hist_obj.asset_name, curr_h, bin_size), fontsize=21)
        fig.tight_layout()
        plt.show()

    def PTEP_len(self, curr_len, max_len=2000):

        """
        What is the probability for the trend to end in n bars?
        """

        if type(curr_len) is not list:

            probs = self.analyzer.GTEP_len(curr_len=curr_len, max_len=max_len)
            probs_summed = [sum(probs[:i + 1]) for i in range(len(probs))]

            fig, ax1 = plt.subplots()

            ax1.plot(list(range(curr_len, max_len + curr_len)), [x * 100 for x in probs],
                     color="blue", label="prob")
            ax1.set_xticks(list(range(curr_len, max_len + curr_len, 20)))

            ax1.set_xlabel("Trend length")
            ax1.set_ylabel("Probability of trend ending in %", color="blue")
            plt.xticks(rotation=90)
            plt.grid(True)

            ax2 = ax1.twinx()

            ax2.plot(list(range(curr_len, max_len + curr_len)), [x * 100 for x in probs_summed],
                     color="red", label="summed prob")
            ax2.set_ylabel("Summed probabilty in %", color="red")
            ax2.set_yticks(list(range(0, 101, 5)))

            plt.legend()
            plt.grid(True)
            plt.xticks(rotation=90)
            plt.title("{} TEP Length (length: {})".format(self.analyzer.hist_obj.asset_name, curr_len), fontsize=21)
            fig.tight_layout()
            plt.show()
        else:
            probs = []
            for l in curr_len:
                probs.append(self.analyzer.GTEP_len(curr_len=l, max_len=max_len))

            for prob in probs:
                pass

    def PTCP_len(self, curr_len, max_len=2000):

        """
        What is the probability for the trend to end in n bars?
        """

        if type(curr_len) is not list:

            probs = self.analyzer.GTEP_len(curr_len=curr_len, max_len=max_len)
            rev_probs = [1-x for x in probs]
            probs_summed = [sum(probs[:i + 1]) for i in range(len(probs))]
            rev_sums = [1-x for x in probs_summed]

            fig, ax1 = plt.subplots()

            ax1.plot(list(range(curr_len, max_len + curr_len)), [x * 100 for x in rev_probs],
                     color="blue", label="prob")
            ax1.set_xticks(list(range(curr_len, max_len + curr_len, 20)))

            ax1.set_xlabel("Trend length")
            ax1.set_ylabel("Probability of trend continuing in %", color="blue")
            plt.xticks(rotation=90)
            plt.grid(True)

            ax2 = ax1.twinx()

            ax2.plot(list(range(curr_len, max_len + curr_len)), [x * 100 for x in rev_sums],
                     color="red", label="summed prob")
            ax2.set_ylabel("Summed probabilty in %", color="red")
            ax2.set_yticks(list(range(0, 101, 5)))

            plt.legend()
            plt.grid(True)
            plt.xticks(rotation=90)
            plt.title("{} TCP Length (length: {})".format(self.analyzer.hist_obj.asset_name, curr_len), fontsize=21)
            fig.tight_layout()
            plt.show()
        else:
            probs = []
            for l in curr_len:
                probs.append(self.analyzer.GTEP_len(curr_len=l, max_len=max_len))

            for prob in probs:
                pass

    def PTEP_len_h(self, curr_len, curr_h, max_dur=60, bin_size=5, interpol="bilinear"):

        occurence_matrix, max_h, max_l = self.analyzer.GTEP_len_h(curr_len, curr_h, max_dur, bin_size)

        """# interpolate all 0 values
        interp_f = interp2d(list(range(column_h)), list(range(curr_len, max_l+1)), occurence_matrix, kind="cubic")

        for column_index, certain_len_column in enumerate(occurence_matrix):
            for i in range(len(certain_len_column)):
                if occurence_matrix[column_index][i] == 0:
                    occurence_matrix[column_index][i] = float(interp_f(column_index, i))
                occurence_matrix[column_index][i] /= len(longer_higher)
                occurence_matrix[column_index][i] *= 100

        # ----------- plotting ---------------

        y, x = np.meshgrid(np.array([curr_h + x*bin_size for x in list(range(column_h))]),
                           np.array(list(range(curr_len, max_l+1))))"""

        fig, ax = plt.subplots(figsize=(8, 8))
        asp = min(max_l-curr_len, max_h-curr_h)/max(max_h-curr_h, max_l-curr_len)
        ax.imshow(np.transpose(np.array(occurence_matrix)), origin='lower', interpolation=interpol, aspect=asp,
                  extent=[curr_len, max_l, curr_h, max_h])
        plt.subplots_adjust(0.17, 0.10, 0.99, 0.99)

        #plt.pcolor(x, y, np.array(occurence_matrix))
        plt.show()

        """fig = plt.figure()
        ax = fig.add_subplot(111, projection="3d")
        surf = ax.plot_surface(x, y, np.array(occurence_matrix), cmap=cm.coolwarm)
        ax.set_zlim(0, 1)
        fig.colorbar(surf, aspect=5)"""


        """
        # <editor-fold desc="old">
        # make a list that holds for every len how often a trend with that len occurs in longer_trends
        occs = [0 for _ in range(max([x.len for x in longer_higher]) + 1)]
        for t in longer_higher:
            occs[t.len] += 1
        certain_len_probs = [x / len(longer_higher) for x in occs]

        bin_list = [[] for _ in range(max([x.len for x in longer_higher]) + 1)]

        # get all trends with a <= h <= b, how many are there? maybe none, so interpolate before requesting
        for l in range(max([x.len for x in longer_higher]) + 1):
            trends_w_len_l = list(filter(lambda x: x.len == curr_len + l, longer_higher))
            max_h = max([x.height for x in longer_higher])

            # curr_height <-> max_h => make bins of size ...5?
            bin_size = 5
            height_column = [0 for _ in range(int((max_h-curr_h)/bin_size)+1)]
            for t in trends_w_len_l:
                # todo for down trends
                height_column[int((t.height - curr_h)/bin_size)] += 1

            # for each bin check if 0, if so: interpolate ...int(curr_h), int(max_h + 0.5) + 1, bin_size
            for i in range(len(height_column)):

                if height_column[i] == 0:
                    # interpolate

                    # find lower index value
                    lower_index = i
                    lower_index_val = None
                    while (not lower_index_val) and lower_index >= 0:
                        lower_index -= 1
                        if height_column[lower_index // bin_size] > 0:
                            lower_index_val = height_column[lower_index // bin_size]

                    # find higher index value
                    higher_index = i
                    higher_index_val = None
                    while (not higher_index_val) and higher_index <= len(height_column)-1:
                        higher_index += 1
                        if height_column[higher_index // bin_size] > 0:
                            higher_index_val = height_column[higher_index // bin_size]

                    if higher_index_val and lower_index_val:
                        height_column[i] = lower_index_val + (lower_index * ((lower_index_val - higher_index_val) /
                                                                          (lower_index - higher_index)))
                    elif higher_index_val and not lower_index_val:
                        height_column[i // bin_size] = higher_index_val
                    elif lower_index_val and not higher_index_val:
                        height_column[i // bin_size] = lower_index_val

            bin_list[l] = height_column

            print("interpolated for len", l, ":", height_column)"""
        # </editor-fold>
    # </editor-fold>

    def complete_plot(self, max_len=500, bins_w=10):  # TODO

        fig, axs = plt.subplots(2, 2)

        p1 = axs[0, 0]
        p2 = axs[0, 1]
        p3 = axs[1, 0]
        p4 = axs[1, 1]

        # trend end probs
        probs = self.analyzer.GTEP_len(0)
        p1.plot(list(range(max_len)), probs)
        p1.grid(True)
        p1.set_title("Trend end prob")
        p1.set_xticks(list(range(0, max_len + 1, 20)))
        p1.set_yticks([x / 100 for x in list(range(0, 100, 5))])
        p1.set_xlabel("Bars from current")
        p1.set_ylabel("Probability of trend ending")

        # len vs h
        p2.scatter([t.len for t in self.analyzer.trend_list], [t.height for t in self.analyzer.trend_list], s=1)
        # p2.set_aspect('equal', adjustable='box')
        p2.set_xlabel("Trend length")
        p2.set_ylabel("Trend height")
        p2.set_title("Height vs length")

        # len vs avg h
        avg_heights = self.analyzer.get_avg_heights(max_len)
        p3.plot(list(range(max_len)), avg_heights)
        p3.grid(True)
        p3.set_xlabel("Trend length")
        p3.set_ylabel("Trend avg height")
        p3.set_title("Length vs avg h")

        # bins
        counts = self.analyzer.get_len_counts(max_len)
        s_counts = []

        for i in range((max_len // bins_w)):
            s_counts.append(sum(counts[bins_w * i:(bins_w + 1) * i]))

        p4.bar(list(range(0, max_len, bins_w)), [100 * c / len(self.analyzer.trend_list) for c in s_counts], width=5)
        p4.grid(True)
        p4.set_xlabel("Trend length")
        p4.set_ylabel("Percentage of trends with that length")
        p4.set_xticks(list(range(0, max_len, bins_w * 5)))
        p4.set_title("Trend distribution")

        fig.suptitle(self.analyzer.hist_obj.asset_name)
        plt.subplots_adjust(0.17, 0.10, 0.99, 0.99)
        plt.show()


class Backtester:

    def __init__(self, strat, hist_obj, use_balance=False,
                 asset_data=AssetData.DEFAULT, trader_data=Trader.DEFAULT,
                 tp_in_pips=None, sl_in_pips=None, ts_in_pips=None,
                 tp_in_value=None, sl_in_value=None):

        # unpack objects
        balance, max_risk_per_position = trader_data
        cost, min_shares, leverage = asset_data

        self.strat = strat
        self.hist_obj = hist_obj

        self.open_long_trades = []
        self.open_short_trades = []
        self.closed_trades = []

        self.base_balance = balance
        self.available_margin = balance
        self.use_balance = use_balance
        self.leverage = leverage
        self.risk = max_risk_per_position
        self.min_shares = min_shares
        self.trade_cost = cost
        self.max_balance = self.base_balance
        self.max_dd = 0
        self.max_dd_perc = 0

        self.sl_in_pips = sl_in_pips
        self.tp_in_pips = tp_in_pips
        self.ts_in_pips = ts_in_pips

        self.tp_in_value = tp_in_value
        self.sl_in_value = sl_in_value

        self.actions = {"enterLong":  self.enterLong,
                        "exitLong":   self.exitLong,
                        "enterShort": self.enterShort,
                        "exitShort":  self.exitShort}

    def test(self, p=True, full_print=False, single_step=False, use_sl_for_risk=False):

        """
        :param p: print or not
        :param full_print: print additional info like trades
        :param single_step: simulate strategy step by step with user interaction
        :param use_sl_for_risk: opens positions with whole available margin and uses sl_in_value for limiting the
                                risk per position to a percentage of the available_margin
        :return: the profit of this strategy with these parameters over this time span
        """

        # todo lower risk over time, begin high risk, lower to the end

        prices = self.hist_obj.prices
        times = self.hist_obj.times
        lookback = self.strat.lookback

        self.open_long_trades = []
        self.open_short_trades = []
        self.closed_trades = []

        self.available_margin = self.base_balance
        self.max_dd = 0
        self.max_dd_perc = 0
        self.max_balance = self.base_balance
        margin_call = False

        self.strat.init_indicators(self.hist_obj.prices, self.hist_obj.times)
        self.strat.initrun(self.hist_obj.hist[:lookback+1])

        if p:
            print("Starting backtest for strat", self.strat.name, "on", self.hist_obj.asset_name,
                  "over", self.hist_obj.timeframe)

        for index in range(len(prices)):

            if index >= lookback:

                price = prices[index]
                time = times[index]

                # test SL and TP and TS for all open trades
                for trade in self.open_long_trades+self.open_short_trades:
                    trade.check(price, time)

                # check for automatically closed trades
                for trade in self.open_long_trades:
                    if not trade.currently_open:
                        # handle margin for those trades, that closed themselves, as they did not go through
                        # "exitLong"/"exitShort"
                        self.available_margin += trade.profit
                        self.available_margin += trade.margin
                        self.closed_trades.append(trade)
                        self.open_long_trades.remove(trade)

                # check for automatically closed trades
                for trade in self.open_short_trades:
                    if not trade.currently_open:
                        # handle margin for those trades, that closed themselves, as they did not go through
                        # "exitLong"/"exitShort"
                        self.available_margin += trade.profit
                        self.available_margin += trade.margin
                        self.closed_trades.append(trade)
                        self.open_short_trades.remove(trade)

                total_positions_worth = sum([t.get_value(price) for t in self.open_long_trades+self.open_short_trades])

                # broker asks for refinancing open positions, cannot enter trades under this condition
                if self.use_balance and self.available_margin/2 + total_positions_worth <= 0:  # <-- margin call

                    # all positions are automatically closed, you fucked up badly
                    if self.available_margin + total_positions_worth <= 0:
                        print("Closing positions, account worth is 0!")
                        self.exitLong(price, time)
                        self.exitShort(price, time)
                        self.print_trades()
                        return self.get_results(p=p)

                    # trigger margin call if not yet set
                    if full_print and not margin_call:
                        """
                        Margin call only occurs, when the profit of a certain position is negative and its absolute 
                        value is bigger than the positions margin, this causes the broker to call for a bigger 
                        margin to keep the position open
                        
                        bound capital in a position is: price*position_size/leverage = margin
                        """

                        print("Margin call due to automated trade close (SL/TP/TS)!")
                        margin_call = True

                if self.available_margin > self.max_balance:
                    self.max_balance = self.available_margin
                if self.max_balance-self.available_margin > self.max_dd:
                    self.max_dd = self.max_balance-self.available_margin
                    self.max_dd_perc = 100*(1 - (self.available_margin / self.max_balance))

                # exec strat
                action_list = self.strat.run(self.hist_obj.hist[index-lookback:index+1])
                if action_list:
                    for entry in action_list:

                        if self.use_balance and self.available_margin/2 + total_positions_worth > 0:
                            margin_call = False

                        action, price, time, trade_params = entry
                        if not trade_params or trade_params is None:
                            trade_params = [self.tp_in_pips, self.sl_in_pips, self.ts_in_pips,
                                            self.tp_in_value, self.sl_in_value]
                        if not trade_params[0]:
                            trade_params[0] = self.tp_in_pips
                        if not trade_params[1]:
                            trade_params[1] = self.sl_in_pips
                        if not trade_params[2]:
                            trade_params[2] = self.ts_in_pips
                        if not trade_params[3]:
                            trade_params[3] = self.tp_in_value
                        if not trade_params[4]:
                            trade_params[4] = self.sl_in_value

                        # open or close trades
                        if self.use_balance:
                            if action == "enterLong" or action == "enterShort":

                                if self.available_margin/2 + total_positions_worth <= 0:

                                    if full_print and not margin_call:
                                        # trigger margin call if not yet
                                        print("Margin call! Cannot open new positions!")
                                        margin_call = True

                                    continue

                                if use_sl_for_risk:
                                    position_size = self.available_margin * self.leverage / price
                                else:
                                    position_size = self.risk * self.available_margin * self.leverage / price

                                if position_size < self.min_shares:
                                    continue
                                else:
                                    if use_sl_for_risk:
                                        self.sl_in_value = self.risk * self.available_margin
                                        trade_params[4] = self.sl_in_value

                            else:
                                position_size = -1
                        else:
                            position_size = 1
                        self.actions[action](price, time, position_size, trade_params)

                if full_print and p:
                    if index % len(prices)//20 == 0:  #
                        self.get_results(p=True)

                if action_list and single_step:
                    self.print_test_status(price, time, index/len(prices))
                    if input() == "exit":
                        single_step = False

        # close all trades that are still open on last price in history
        for trade in self.open_long_trades+self.open_short_trades:
            trade.close(prices[-1], times[-1])

        if p and full_print:
            self.print_trades()
        if p:
            print("\nFinal results:\n---------------------------------------------------------------\n")
        results = self.get_results(p=p)
        if p:
            print("Best possible results:\n")
            Analyzer(hist_obj=self.hist_obj, min_trend_h=0, realistic=False).SIM_trend_follow(True)

        return results

    def profit_with_tax(self, start, end, p=True, sl_for_risk=False):

        use_bal = self.use_balance
        self.use_balance = True

        bal = self.base_balance
        running_bal = bal

        orig_hist = self.hist_obj
        tax_sum = 0

        if p:
            print("\nStarting backtest for strat", self.strat.name, "on", self.hist_obj.asset_name, "over",
                  start, "-", end, "with base balance", self.base_balance, "and risk of", self.risk)
            print("\n!!!   !!!   !!!   Scholz rule is NOT applied!   !!!   !!!   !!!\n")
            print("                                  balance: {:12.2f}".format(bal))

        for i in range(start, end+1):

            self.hist_obj = History(self.hist_obj.asset_name, i)
            abs_p = self.test(p=False, use_sl_for_risk=sl_for_risk)
            p = (0.75 * abs_p)
            tax_sum += 0.25*abs_p
            perc_gr = (((running_bal+p)/running_bal)-1)*100
            running_bal += p
            if p:
                print("Profit in {}:\t\t{:12.2f}, balance: {:12.2f}, growth in %: {} %, tax: {:12.2f}, trades: {}".
                      format(i, p, running_bal, "{:12.2f}".format(perc_gr).zfill(5), 0.25*abs_p, len(self.closed_trades)))
            self.base_balance = running_bal

        if p:
            print("\nProfit in {}-{}:{:12.2f}, balance: {:12.2f}, % growth: {:12.2f}%\nrisk free:\t\t\t{:12.2f}".
                  format(start, end, self.base_balance-bal, self.base_balance, ((self.base_balance/bal)-1)*100,
                         bal*(1.07**(end-start+1))))
            # risk free is based off of all-weather portfolio with 9% gain/year -> 0.75*9% ~7%/year

        self.base_balance = bal

        if p:
            self.hist_obj = History(self.hist_obj.asset_name, start, end)
            pure_profit = self.test(p=False)
            print("\nPayed tax:\t\t\t{:12.2f}, prof\\tax:{:12.2f}\n".format(tax_sum, pure_profit))

        self.use_balance = use_bal
        self.hist_obj = orig_hist

        return running_bal-self.base_balance

    def optimize_sl(self, low=0, high=500, steps=1, p=False, in_pips=True):

        factor = 1
        if type(steps) is not int:
            factor = 1 / steps

            low = int(low * factor + 0.5)
            high = int(high * factor + 0.5)
            steps = int(steps * factor + 0.5)

        former_sl = self.sl_in_pips if in_pips else self.sl_in_value
        ran = list(range(low, high, steps))

        if in_pips:
            self.sl_in_pips = ran[0] / factor
        else:
            self.sl_in_value = ran[0] / factor
        best_res = self.test(p=False)  # todo risk sl
        best_sl = ran[0]/factor
        if p:
            print("Optimisation of stop loss started ...")
        p_counter = 0

        for sl in ran:
            if p:
                p_counter += 1
                if p_counter % (len(ran)//10) == 0:
                    print("{:5.2f} % progress, best result so far: {} with stop loss of {}".format(
                        100*p_counter/len(ran), best_res, best_sl/factor))
            if in_pips:
                self.sl_in_pips = sl / factor
            else:
                self.sl_in_value = sl / factor
            test_res = self.test(p=False)  # todo
            if test_res > best_res:
                best_res = test_res
                best_sl = sl/factor

        if in_pips:
            self.sl_in_pips = former_sl
        else:
            self.sl_in_value = former_sl
        if p:
            print("Optimised stop loss to", best_sl, "with profit of", best_res)
        return best_sl

    def optimize_ts(self, low=0, high=500, steps=1, p=False):

        factor = 1
        if type(steps) is not int:
            factor = 1 / steps

            low = int(low * factor + 0.5)
            high = int(high * factor + 0.5)
            steps = int(steps * factor + 0.5)

        former_ts = self.ts_in_pips
        ran = list(range(low, high, steps))

        self.ts_in_pips = ran[0] / factor
        best_res = self.test(p=False)
        best_ts = ran[0]/factor

        if p:
            print("Optimisation of trailing stop started ...")
            p_counter = 0

        for ts in ran:
            if p:
                p_counter += 1
                if p_counter % (len(ran)//10) == 0:
                    print("{:5.2f} % progress, best results so far: {} with trailing stop of {}".
                          format(100*p_counter/len(ran), best_res, best_ts/factor))
            self.ts_in_pips = ts / factor
            test_res = self.test(p=False)
            if test_res > best_res:
                best_res = test_res
                best_ts = ts/factor

        self.ts_in_pips = former_ts
        if p:
            print("Optimised trailing stop to", best_ts, "with profit of", best_res)
        return best_ts

    def optimize_strat_param(self, param_name, low=0, high=500, steps=1, p=False, to_plot=False):

        if param_name not in self.strat.parameter_names:
            print("The parameter you want to optimize doesn't exist in this strategy!")
            return -1

        try:
            factor = 1
            if type(steps) is not int:
                factor = 1 / steps

                low = int(low*factor+0.5)
                high = int(high*factor+0.5)
                steps = int(steps*factor+0.5)

            init_value = vars(self.strat)[param_name]
            profits = []
            param_range = range(low, high, steps)
            vars(self.strat)[param_name] = param_range[0]/factor
            best_result = self.test(p=False)
            best_param = param_range[0]/factor

            if p:
                print("Optimization of parameter", "'" + param_name + "' started for strat", self.strat.name,
                      "on asset", self.hist_obj.asset_name)
                p_counter = 0

            for i in param_range:
                if p:
                    p_counter += 1
                    m = max(1, (len(param_range) // 10))
                    if p_counter % m == 0:
                        print("{:5.2f} % progress, best result so far: {:5.2f} with {} of {}".format(
                            100 * p_counter / len(param_range), best_result, param_name, best_param/factor))

                vars(self.strat)[param_name] = i/factor
                test_res = self.test(p=False)
                profits.append(test_res)
                if test_res > best_result:
                    best_result = test_res
                    best_param = i/factor

            vars(self.strat)[param_name] = init_value

            if not to_plot:
                if p:
                    print("Optimised ", param_name, "to", best_param, "with profit of", best_result)
                return best_param
            else:
                return [i/factor for i in list(param_range)], profits

        except Exception as e:
            print("Could not optimize strategy parameterr")
            print(e)

    def plot_param_profile(self, param_name, low=0, high=500, steps=1):
        plt.rc('figure', facecolor="#333333", edgecolor="#333333")
        plt.rc('axes', facecolor="#353535", edgecolor="#000000")
        plt.rc('lines', color="#393939")
        plt.rc('grid', color="#121212")
        x, y = self.optimize_strat_param(param_name, low, high, steps, p=True, to_plot=True)
        Plotter.plot_general(x, y, label1=param_name, label2="Profit")

    def print_test_status(self, price, time, progress):

        print("\nBalance: {:12.2f} / {}\t\tPrice: {:12.2f}\t\tTime: {}\t\tProgress: {:12.2f} %\n".
              format(self.available_margin, self.base_balance, price, time, progress))

        if self.open_long_trades:
            print("Open long trades:\n")
            for t in self.open_long_trades:
                print(t)

        if self.open_short_trades:
            print("Open short trades:\n")
            for t in self.open_short_trades:
                print(t)

        print("Results:\n-------------------------------------------------")
        self.get_results()
        print("#################################################")

    def get_results(self, p=True):  # maybe optimise if needed

        if not p:
            return sum([x.profit for x in self.closed_trades])

        trades = len(self.closed_trades)

        if trades:
            long_wins_num = len(list(filter(lambda x: x.direction == 1 and x.profit >= 0, self.closed_trades)))
            long_wins = sum([(x.profit if x.profit >= 0 else 0) for x in
                             list(filter(lambda x: x.direction == 1, self.closed_trades))])

            long_losses_num = len(list(filter(lambda x: x.direction == 1 and x.profit < 0, self.closed_trades)))
            long_losses = sum([(x.profit if x.profit < 0 else 0) for x in
                               list(filter(lambda x: x.direction == 1, self.closed_trades))])

            short_wins_num = len(list(filter(lambda x: x.direction == -1 and x.profit > 0, self.closed_trades)))
            short_wins = sum([(x.profit if x.profit >= 0 else 0) for x in
                              list(filter(lambda x: x.direction == -1, self.closed_trades))])

            short_losses_num = len(list(filter(lambda x: x.direction == -1 and x.profit <= 0, self.closed_trades)))
            short_losses = sum([(x.profit if x.profit < 0 else 0) for x in
                                list(filter(lambda x: x.direction == -1, self.closed_trades))])

            # TODO sharpe & ulcer ratio ... maybe alpha if it is what I think?
            if p:
                print(" Long wins: {:12.2f} ({:5.2f} %)\t\t"
                      "Long losses:  {:12.2f} ({:5.2f} %)\n"
                      "Short wins: {:12.2f} ({:5.2f} %)\t\t"
                      "Short losses: {:12.2f} ({:5.2f} %)\n\n\t   "
                      "Sum: {:12.2f}\t\t"
                      "MaxDD:\t\t{:12.2f}\n"
                      "  # trades: {:9d}\t\t\t"
                      "MaxDD in %: {:12.2f}\n"
                      "  Buy&Hold: {:12.2f}\n".
                      format(long_wins, 100*long_wins_num/trades, long_losses, 100*long_losses_num/trades,
                             short_wins, 100*short_wins_num/trades, short_losses, 100*short_losses_num/trades,
                             long_wins + long_losses + short_wins + short_losses, self.max_dd, trades, self.max_dd_perc,
                             self.hist_obj.hist[-1]["price"] - self.hist_obj.hist[0]["price"]))
            return long_wins + long_losses + short_wins + short_losses
        else:
            return 0

    def print_trades(self):
        print("\n-------------------------Trades-------------------------\n")
        print("Init. balance: ", self.base_balance)
        s = self.base_balance
        for t in self.closed_trades:
            print(t)
            s += t.profit
            print("Balance: ", s, "\n")

    def plot_trades(self, bar_amount=900, start_bar=0):

        """
        todo
        this does not work bc the trades do not follow each other due to ts and stuff unlike trends did
        so you have to find the index of the starting point of each trend by comparing the dates of the price bars
        and the entry time of the trend
        :param bar_amount:
        :param start_bar:
        :return:
        """

        plt.rc('figure', facecolor="#333333", edgecolor="#333333")
        plt.rc('axes', facecolor="#353535", edgecolor="#000000")
        plt.rc('lines', color="#393939")
        plt.rc('grid', color="#121212")

        # visually in graph with price, like trends
        max_len = min(start_bar+bar_amount, self.hist_obj.hist.__len__())
        pan = self.hist_obj.hist[start_bar:max_len]

        first_trend_len = 0
        print(self.closed_trades[0].open_time)
        for i in range(start_bar, max_len+start_bar):
            if pan[i]["time"] != self.closed_trades[0].open_time:
                first_trend_len += 1
            else:
                break
        print(first_trend_len)

        trades = []
        lens = [t.len for t in self.closed_trades]
        for i in range(len(self.closed_trades)):
            if first_trend_len + sum([x.len for x in trades]) + self.closed_trades[i].len <= len(pan):
                trades.append(self.closed_trades[i])
            else:
                break

        fig, ax1 = plt.subplots()

        # plot price curve
        ax1.plot(list(range(start_bar, bar_amount+start_bar)), [x["price"] for x in pan], color="black")
        #ax1.set_xticks((list(range(0, max_len, int(max_len / 96)))))
        ax1.set_xlabel("Time " + pan[0]["time"] + " - " + pan[-1]["time"])
        ax1.set_ylabel("Price", color="tab:red")

        a = [sum([x - 1 for x in lens[:i]]) for i in range(len(trades) + 1)]
        a_pairs = [a[i:i + 2] for i in range(len(a))][:-1]
        b = [[x.open_price, x.close_price] for x in trades]

        for i in range(len(a_pairs)):
            ax1.plot([pair+first_trend_len for pair in a_pairs[i]], b[i],
                     color=("green" if trades[i].direction == 1 else "#8B0000"))

        ax1.tick_params(axis="y")
        plt.xticks(rotation=90)

        # -----------------------

        profits = [t.profit for t in trades]
        summed_profits = [sum(profits[:i]) for i in range(len(profits))]

        ax2 = ax1.twinx()
        ax2.set_ylabel('Profit', color='tab:blue')
        ax2.plot([x[0] for x in a_pairs], summed_profits)
        ax2.tick_params(axis="y")

        plt.xticks(rotation=90)
        plt.tight_layout()
        plt.show()

    def scatter_trades(self):
        plt.rc('figure', facecolor="#333333", edgecolor="#333333")
        plt.rc('axes', facecolor="#353535", edgecolor="#000000")
        plt.rc('lines', color="#393939")
        plt.rc('grid', color="#121212")
        x = [len(t.prices) for t in self.closed_trades]
        y = [t.profit for t in self.closed_trades]
        colors = [("#ff0000" if t.direction <= 0 else "#00ff00") for t in self.closed_trades]
        p = plt.scatter(x, y, s=1, c=colors)
        plt.grid(True)
        plt.xlabel("Trade length")
        plt.ylabel("Trade profit")
        plt.tight_layout()
        plt.show()

    def plot_trade_distribution(self, bin_size=1):
        plt.rc('figure', facecolor="#333333", edgecolor="#333333")
        plt.rc('axes', facecolor="#353535", edgecolor="#000000")
        plt.rc('lines', color="#393939")
        plt.rc('grid', color="#121212")
        min_prof = min([trade.profit for trade in self.closed_trades])
        max_prof = max([trade.profit for trade in self.closed_trades])

        profit_bins = [[] for _ in range(int(min_prof), int(max_prof)+1, bin_size)] # int(minProf) is index of 0 in profit bins
        profit_sums = [[] for _ in range(int(min_prof), int(max_prof)+1, bin_size)] # int(minProf) is index of 0 in profit bins

        long_trades = list(filter(lambda x: x.direction == 1, self.closed_trades))
        short_trades = list(filter(lambda x: x.direction == -1, self.closed_trades))

        # counting numbers
        for index in range(len(profit_bins)):
            # profit range min_prof -> min_prof + bin_size
            lower_bin_border = (index+int(min_prof))*bin_size
            upper_bin_border = lower_bin_border + bin_size

            num_of_long_trades_w_that_profit = len(list(filter(
                lambda trade: lower_bin_border <= trade.profit < upper_bin_border, long_trades)))
            num_of_short_trades_w_that_profit = len(list(filter(
                lambda trade: lower_bin_border <= trade.profit < upper_bin_border, short_trades)))

            profit_bins[index] += [num_of_long_trades_w_that_profit, num_of_short_trades_w_that_profit]

        # summing profits of trades with certain len
        for index in range(len(profit_bins)):
            # profit range min_prof -> min_prof + bin_size
            lower_bin_border = (index+int(min_prof))*bin_size
            upper_bin_border = lower_bin_border + bin_size

            sum_of_long_trades_w_that_profit = sum(x.profit for x in list(filter(
                lambda trade: lower_bin_border <= trade.profit < upper_bin_border, long_trades)))
            sum_of_short_trades_w_that_profit = sum(x.profit for x in list(filter(
                lambda trade: lower_bin_border <= trade.profit < upper_bin_border, short_trades)))

            profit_sums[index] += [sum_of_long_trades_w_that_profit, sum_of_short_trades_w_that_profit]

        fig, (ax1, ax2) = plt.subplots(1, 2)

        width = 0.45

        # plot 1
        ax1.bar(list(range(int(min_prof), int(max_prof)+1, bin_size)),
                [x[0] for x in profit_bins],
                color="#32cd32", width=width)
        ax1.bar(list(map(lambda x: x + width, list(range(int(min_prof), int(max_prof)+1, bin_size)))),
                [x[1] for x in profit_bins],
                color="#dc143c", width=width)
        ax1.set_xticks((list(map(lambda x: int(x+width/2),
                             list(range(int(min_prof/2), int(max_prof/2)+1, bin_size*2))))))
        #ax1.grid(True)

        # plot 2
        ax2.bar(list(range(int(min_prof), int(max_prof) + 1, bin_size)),
                [x[0] for x in profit_sums],
                color="#32cd32", width=width)
        ax2.bar(list(map(lambda x: x + width, list(range(int(min_prof), int(max_prof) + 1, bin_size)))),
                [x[1] for x in profit_sums],
                color="#dc143c", width=width)
        ax2.set_xticks((list(map(lambda x: int(x + width / 2),
                             list(range(int(min_prof / 2), int(max_prof / 2) + 1, bin_size * 2))))))

        fig.subplots_adjust(0.05, 0.05, 0.99, 0.99)
        #ax2.grid(True)
        plt.show()

    # <editor-fold desc="Enter & exit trades">
    def enterLong(self, price, time, position_size=1, trade_params=None):

        if self.use_balance:
            self.available_margin -= (position_size * price / self.leverage)

        if not trade_params:
            trade_params = [self.tp_in_pips, self.sl_in_pips, self.ts_in_pips, self.tp_in_value, self.sl_in_value]

        self.open_long_trades.append(Trade(price, time, 1, *trade_params, position_size=position_size,
                                           spread=self.trade_cost, leverage=self.leverage))

    def enterShort(self, price, time, position_size=1, trade_params=None):

        if self.use_balance:
            self.available_margin -= (position_size * price / self.leverage)

        if not trade_params:
            trade_params = [self.tp_in_pips, self.sl_in_pips, self.ts_in_pips, self.tp_in_value, self.sl_in_value]

        self.open_short_trades.append(Trade(price, time, -1, *trade_params, position_size=position_size,
                                            spread=self.trade_cost, leverage=self.leverage))

    def exitLong(self, price, time, position_size=1, trade_params=None, trade_id=None):
        if not trade_id:
            for trade in self.open_long_trades:
                trade.close(price, time)
                # handle account margin here
                self.available_margin += trade.profit
                self.available_margin += trade.margin

                self.closed_trades.append(trade)
                self.open_long_trades.remove(trade)
        else:
            for trade in self.open_long_trades:
                if trade.trade_id == trade_id:
                    trade.close(price, time)

                    self.available_margin += trade.profit
                    self.available_margin += trade.margin

                    self.closed_trades.append(trade)
                    self.open_long_trades.remove(trade)
                    return

    def exitShort(self, price, time, position_size=1, trade_params=None, trade_id=None):
        if not trade_id:
            for trade in self.open_short_trades:
                trade.close(price, time)
                self.available_margin += trade.profit
                self.available_margin += trade.margin
                self.closed_trades.append(trade)
                self.open_short_trades.remove(trade)
        else:
            for trade in self.open_short_trades:
                if trade.trade_id == trade_id:
                    trade.close(price, time)
                    self.available_margin += trade.profit
                    self.available_margin += trade.margin
                    self.closed_trades.append(trade)
                    self.open_short_trades.remove(trade)
                    return
    # </editor-fold>


class Trade:

    # todo built in trail lock (break even price +- best price * trail lock)
    def __init__(self, open_price, o_time, direction,
                 tp_in_pips=None, sl_in_pips=None, ts_in_pips=None,
                 tp_in_value=None, sl_in_value=None,
                 position_size=1, spread=0,
                 leverage=1, currently_open=True):

        self.currently_open = currently_open
        self.direction = direction  # 1 = up, -1 = down
        self.len = 1

        self.open_price = open_price
        self.close_price = None

        self.open_time = o_time
        self.close_time = None

        self.max_profit = 0

        # cost & balance
        self.spread = spread
        self.profit = 0
        self.position_size = position_size
        self.margin = position_size * open_price / leverage
        self.prices = [(open_price, o_time)]

        # management parameters
        # stop loss in pips
        if sl_in_pips:
            self.sl_in_pips = self.open_price-sl_in_pips if direction == 1 else self.open_price+sl_in_pips
        else:
            self.sl_in_pips = sl_in_pips
        # take profit in pips
        if tp_in_pips:
            self.tp_in_pips = self.open_price+tp_in_pips if direction == 1 else self.open_price-tp_in_pips
        else:
            self.tp_in_pips = tp_in_pips

        self.ts_in_pips = ts_in_pips  # trailing stop in pips

        self.tp_in_value = tp_in_value
        self.sl_in_value = sl_in_value

        if self.ts_in_pips:
            if self.direction == 1:
                self.trail = self.open_price - self.ts_in_pips
            if self.direction == -1:
                self.trail = self.open_price + self.ts_in_pips
        else:
            self.trail = None

        self.trade_id = id(self)

    def get_profit(self, p):

        if self.direction == 1:
            return self.position_size * (p - self.open_price - self.spread)
        if self.direction == -1:
            return self.position_size * (self.open_price - p - self.spread)

    def get_value(self, p):
        return self.margin + self.get_profit(p)

    def check(self, updated_price, time):

        if self.currently_open:
            self.len += 1

        if self.currently_open:
            self.prices.append((updated_price, time))

            """
            do not automatically close underfinanced positions, as only the sum of the worth of the accounts positions
            is relevant to a margin call
            
            # close position if used margin is used up
            if self.get_value(updated_price) <= 0:
                self.close(updated_price, time)
            """

            if self.sl_in_value:
                if self.get_profit(updated_price) <= -self.sl_in_value:
                    #print("Closing", self.open_price, "long" if self.direction == 1 else "short",
                     #     "position on", updated_price, "with sl of", self.sl_in_value, "at a value of", self.get_profit(updated_price))
                    self.close(updated_price, time)

            if self.sl_in_pips:
                if self.direction == 1:
                    if updated_price <= self.sl_in_pips:
                        #print("Closing", self.open_price, "long position on", updated_price, "with sl of", self.sl_in_pips)
                        self.close(updated_price, time)
                if self.direction == -1:
                    if updated_price >= self.sl_in_pips:
                        #print("Closing", self.open_price, "short position on", updated_price, "with sl of", self.sl_in_pips)
                        self.close(updated_price, time)

            if self.tp_in_value:
                if self.get_profit(updated_price) >= self.tp_in_value:
                    self.close(updated_price, time)

            if self.tp_in_pips:
                if self.direction == 1:
                    if updated_price >= self.tp_in_pips:
                        self.close(updated_price, time)
                if self.direction == -1:
                    if updated_price <= self.tp_in_pips:
                        self.close(updated_price, time)

            if self.ts_in_pips:

                # check exit for trailing stop in pips
                if self.direction == 1:
                    if self.open_price+self.spread <= updated_price <= self.trail:
                        self.close(updated_price, time)
                if self.direction == -1:
                    if self.open_price-self.spread >= updated_price >= self.trail:
                        self.close(updated_price, time)

                prof = self.get_profit(updated_price)
                if prof > self.max_profit:
                    self.max_profit = prof
                    if self.direction == 1:
                        # long
                        self.trail = updated_price-self.ts_in_pips
                    if self.direction == -1:
                        # short
                        self.trail = updated_price+self.ts_in_pips

    def close(self, close_price, time):
        if self.currently_open:
            self.close_price = close_price
            self.prices.append((close_price, time))
            self.currently_open = False
            self.close_time = time
            self.profit = self.get_profit(self.close_price)

    def __str__(self):
        return "Open price:   {:12.6f}\t\tOpen time:    {}\n" \
               "Close price:  {:12.6f}\t\tClose time:   {}\n" \
               "Profit:       {:12.6f}\t\tDirection:    {:+d}\t\t\t\t\tCurrently open: {}\n" \
               "Position size:{:12.6f}\n".\
            format(self.open_price, self.open_time, self.close_price if self.close_price else 0,
                   self.close_time if self.close_time else "NA",
                   self.profit, self.direction, self.currently_open, self.position_size)



