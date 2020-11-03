import inspect
from abc import ABC, abstractmethod
from History import History
import datetime


class Strategy(ABC):

    @abstractmethod
    def __init__(self, lookback):
        """
        :param lookback: assures you always get fed in data lists of this length+1
        """
        self.lookback = lookback
        self.name = "EmptyStrat"

    @abstractmethod
    def run(self, data):
        """
        :param data:
        :return: list of actions to do, action is [action_type, price, time]
        """
        pass

    @abstractmethod
    def initrun(self, data):
        """
        should be executed only once -> after first batch of data comes in, e.g. to init certain params

        :param data:
        :return:
        """
        pass


# todo
class TrendFollowStratWithProbModell(Strategy):
    """
    how likely is it that the last high/low was the optimal exit point?

    check this condition only if a certain threshold from the peak is reached bc else you would always end on the
    average length trend
    """

    def __init__(self, analyzer, threshold=80, threshold2=40, len_prob_threshold=0.8, h_prob_threshold=0.8):
        super().__init__(lookback=1)
        self.uptrend = -1
        self.analyzer = analyzer
        self.high = self.low = self.begin = 0
        self.high_index = self.low_index = self.break_index = 0
        self.hist = []

        self.threshold = threshold
        self.second_threshold = threshold2
        self.len_prob_threshold = len_prob_threshold
        self.h_prob_threshold = h_prob_threshold
        self.parameter_names = ["threshold",
                                "threshold2",
                                "len_prob_threshold",
                                "h_prob_threshold"]  # everything that can be used to init the strat

        self.initializing = True
        self.current_trend_len = 0
        self.name = "TrendFollowStratWithProbModell"

    def initrun(self, data):

        """
        wait for first trend (up or down) but don't act on it,
        enter when first trend ends and make the kind of
        your entry (long/short) dependent on the former trend

        :param data:
        :return:
        """

        self.hist = []
        self.high = self.low = self.begin = data[0]["price"]

    def run(self, data):

        self.hist.append(data[0])
        i = len(self.hist) - 1
        last = data[0]["price"]

        actions = []

        if self.initializing:

            # wait out first trend to end
            for i in range(len(self.hist)):
                last = self.hist[i]["price"]
                if last > self.begin + self.threshold:
                    self.uptrend = True
                    self.initializing = False
                    break
                if last < self.begin - self.threshold:
                    self.uptrend = False
                    self.initializing = False
                    break

        # <editor-fold desc="strategy part">
        else:

            if self.uptrend:
                if last > self.high:
                    self.high = last
                    self.high_index = i
                    self.current_trend_len += 1
                else:

                    if last < self.high - self.threshold:
                        # count last peak as end of high trend and as start of low trend

                        # up trend ends here
                        actions.append(["exitLong", last, data[0]["time"], ()])
                        actions.append(["enterShort", last, data[0]["time"], ()])

                        # change index
                        self.low = last
                        self.low_index = i
                        self.uptrend = False
                        self.current_trend_len = 0
                    else:

                        self.current_trend_len += 1

                        # todo other exit conditions go here
                        if last < self.high - self.second_threshold:
                            ...

            else:
                if last < self.low:
                    self.low = last
                    self.low_index = i
                    self.current_trend_len += 1
                else:

                    if last > self.low + self.threshold:

                        # down trend ends here
                        actions.append(["exitShort", last, data[0]["time"], ()])
                        actions.append(["enterLong", last, data[0]["time"], ()])

                        self.high = last
                        self.high_index = i
                        self.uptrend = True
                        self.current_trend_len = 0

                    else:

                        self.current_trend_len += 1

                        # todo other exit conditions go here

            return actions
        # </editor-fold>


# todo strat: "expecto momentum" use moving average over last 50 (or so) trend momentums as estimate for next trend
#  momentum

# idea: frac dim is higher on non type coherent trends, check this
# for 80 threshold, 1.3583 is just above the avg frac dim for positive coherence, negative is 1.5
# so assume neg coherence for frac_dim > 1.36 and exit out on drawdown?
# maybe combine with typical trend len & height


class MarketOpenSpike(Strategy):

    # todo strat: always buy with tight trailing stop just before market opening (08:50 or 08:55) and use first swing,
    #  if it is negative, immideately reverse your position, if again negative go out ...... run some analysis to back
    #  up on periodic momentum spikes

    def __init__(self):
        super().__init__(lookback=1)
        self.name = "MarketOpenSpike"
        self.parameter_names = []

    def initrun(self, data):
        pass

    def run(self, data):

        price = data[0]["price"]
        time = data[0]["time"]

        datetime_time = History.datetime_from_string(time, None)

        actions = []

        if datetime.time(hour=8, minute=55, second=00) <= datetime_time.time() < \
                datetime.time(hour=8, minute=55, second=1):
            actions.append(["enterLong", price, time, [None, None, 45, None, None]])  # tp sl ts

        return actions


# done
class TrendFollowStrat(Strategy):

    def __init__(self, threshold=80):
        super().__init__(lookback=1)
        self.uptrend = -1
        self.high = self.low = self.begin = 0
        self.high_index = self.low_index = self.break_index = 0
        self.hist = []
        self.threshold = threshold
        self.initializing = True
        self.name = "TrendFollowStrat"
        self.parameter_names = ["threshold"]  # everything that can be used to init the strat

    def init_indicators(self, prices, times):
        pass

    def initrun(self, data):

        """
        wait for first trend (up or down) but don't act on it,
        enter when first trend ends and make the kind of
        your entry (long/short) dependent on the former trend

        :param data:
        :return:
        """

        self.hist = []
        self.initializing = True
        self.high = self.low = self.begin = data[0]["price"]

    def run(self, data):

        self.hist.append(data[0])
        i = len(self.hist) - 1
        last = data[0]["price"]

        actions = []

        if self.initializing:

            if last > self.high:
                self.high = last
            if last < self.low:
                self.low = last

            if last > self.low + self.threshold:
                self.uptrend = True
                self.initializing = False
            if last < self.high - self.threshold:
                self.uptrend = True
                self.initializing = False

        # <editor-fold desc="strategy part">
        else:

            if self.uptrend:
                if last > self.high:
                    self.high = last
                    self.high_index = i
                else:
                    if last < self.high - self.threshold:
                        # count last peak as end of high trend and as start of low trend

                        # todo: up trend ends here
                        actions.append(["exitLong", last, data[0]["time"], []])
                        actions.append(["enterShort", last, data[0]["time"], []])

                        # change index
                        self.low = last
                        self.low_index = i
                        self.uptrend = False
            else:
                if last < self.low:
                    self.low = last
                    self.low_index = i
                else:
                    if last > self.low + self.threshold:
                        # todo: down trend ends here
                        actions.append(["exitShort", last, data[0]["time"], []])
                        actions.append(["enterLong", last, data[0]["time"], []])

                        self.high = last
                        self.high_index = i
                        self.uptrend = True

            return actions
        # </editor-fold>


class FilteredTrendFollowStrat(Strategy):

    def __init__(self, threshold, filter_class, *filter_args):
        super().__init__(lookback=1)  # must be >= length of test period quotes
        self.uptrend = -1
        self.high = self.low = self.begin = 0
        self.high_index = self.low_index = self.break_index = 0
        self.hist = []
        self.initializing = True

        self.name = "FilteredTrendFollowStrat"
        # everything that can be used to init the strat
        filter_class_var_names = list(inspect.signature(filter_class.__init__).parameters.keys())[1:]
        self.parameter_names = ["threshold"] + filter_class_var_names

        self.threshold = threshold
        self.filter = filter_class(*filter_args)
        self.filter_values = []

    def init_indicators(self, prices, times):
        self.filter.initialize(prices)
        self.iterator = self.filter()

    def initrun(self, data):

        """
        wait for first trend (up or down) but don't act on it,
        enter when first trend ends and make the kind of
        your entry (long/short) dependent on the former trend

        :param data: first price quotes
        """

        self.high = self.low = self.begin = next(self.iterator)

    def run(self, data):

        self.hist.append(data[0])
        i = len(self.hist) - 1
        last_filtered = next(self.iterator)
        last_price = data[0]["price"]

        actions = []

        if self.initializing:

            if last_filtered > self.high:
                self.high = last_filtered
            if last_filtered < self.low:
                self.low = last_filtered

            if last_filtered > self.low + self.threshold:
                self.uptrend = True
                self.initializing = False
            if last_filtered < self.high - self.threshold:
                self.uptrend = True
                self.initializing = False

        # <editor-fold desc="strategy part">
        else:

            if self.uptrend:
                if last_filtered > self.high:
                    self.high = last_filtered
                    self.high_index = i
                else:

                    # down move bigger than threshold
                    if last_filtered < self.high - self.threshold:
                        # count last peak as end of high trend and as start of low trend

                        # up trend ends here
                        actions.append(["exitLong", last_price, data[0]["time"], ()])
                        actions.append(["enterShort", last_price, data[0]["time"], ()])

                        # change index
                        self.low = last_filtered
                        self.low_index = i
                        self.uptrend = False
            else:
                if last_filtered < self.low:
                    self.low = last_filtered
                    self.low_index = i
                else:
                    # up move bigger than threshold
                    if last_filtered > self.low + self.threshold:
                        # down trend ends here
                        actions.append(["exitShort", last_price, data[0]["time"], ()])
                        actions.append(["enterLong", last_price, data[0]["time"], ()])

                        self.high = last_filtered
                        self.high_index = i
                        self.uptrend = True

            return actions
        # </editor-fold>


strat_dict = {"trend follow": TrendFollowStrat,
              "trend follow prob": TrendFollowStratWithProbModell,
              "market open spike": MarketOpenSpike,
              "filtered trend follow": FilteredTrendFollowStrat,
              }
