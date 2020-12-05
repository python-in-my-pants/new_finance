import inspect
from abc import ABC, abstractmethod
from History import History
import datetime
from Indicators import *
from Triggers import *
from Analyzer import Trend


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


# todo
class MarketOpenSpike(Strategy):

    # todo strat: always buy with tight trailing stop just before market opening (08:50 or 08:55) and use first swing,
    #  if it is negative, immediately reverse your position, if again negative go out ...... run some analysis to back
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
class TrendFollowStrat(Strategy):  # todo still not working

    def __init__(self, threshold=80, threshold_func=None):
        super().__init__(lookback=1)
        self.uptrend = -1
        self.high = self.low = self.begin = 0
        self.high_index = self.low_index = self.break_index = 0
        self.hist = []
        self.threshold = threshold
        self.threshold_func = threshold_func
        self.initializing = True
        self.name = "TrendFollowStrat"
        self.parameter_names = ["threshold"]  # everything that can be used to init the strat

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
        if self.threshold_func:
            self.threshold_func.initialize(data[:self.threshold_func.lookback])

    def run(self, data):

        self.hist.append(data[0])
        i = len(self.hist) - 1
        last = data[0]["price"]

        if self.threshold_func and len(self.hist) >= self.threshold_func.lookback:
            self.threshold = 5 * self.threshold_func([q["price"] for q in self.hist])

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
                self.uptrend = False
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
                        actions.append(["exitLong", []])
                        actions.append(["enterShort", []])

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
                        actions.append(["exitShort", []])
                        actions.append(["enterLong", []])

                        self.high = last
                        self.high_index = i
                        self.uptrend = True

            return actions
        # </editor-fold>


# done
class FilteredTrendFollowStrat(Strategy):

    """
    Trend follow strategy, but prices are run through filter before we look for trends
    """

    def __init__(self, threshold, filter_class, *filter_args):
        super().__init__(lookback=100000000)  # must be >= length of test period quotes
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

    def initrun(self, data):

        """
        wait for first trend (up or down) but don't act on it,
        enter when first trend ends and make the kind of
        your entry (long/short) dependent on the former trend

        :param data: first price quotes
        """

        self.filter.initialize([d["price"] for d in data])
        self.high = self.low = self.begin = self.filter()

    def run(self, data):

        self.hist.append(data[0])
        i = len(self.hist) - 1
        last_filtered = self.filter()
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


# todo
class TrendPredictionStrat(Strategy):

    """
    IDEA:

    only enter on a trend start if
     - its predicted height * pred_percentage_buffer >= min_trend_h (+ additional_abs_buffer)
     - its predicted height - pred_abs_buffer  >= min_trend_h (+ additional_abs_buffer)

     exit when
      - trend_h >= predicted height * pred_percentage_buffer OR
      - trend_h >= predicted height - pred_abs_buffer is reached

    """

    def __init__(self, analyzer,
                 pred_percentage_buffer=0.7, pred_abs_buffer=25, additional_abs_buffer=5,
                 pred_mode="avg", number_of_similar_trends_used=None, similarity_threshold=0, min_trends_used=None):
        super().__init__(lookback=1)

        print("\nWarning! Used analyzer object should not be based on the same history as the backtest!")

        self.uptrend = -1
        self.high = self.low = self.begin = 0
        self.high_index = self.low_index = self.break_index = 0

        self.hist = []
        self.threshold = analyzer.min_trend_h
        self.anal = analyzer
        self.pred_mode = pred_mode
        self.number_of_similar_trends_used = number_of_similar_trends_used
        self.similarity_threshold = similarity_threshold
        self.min_trends_used = min_trends_used

        self.pred_percentage_buffer = pred_percentage_buffer
        self.pred_abs_buffer = pred_abs_buffer

        self.entry_pred_threshold = self.threshold + additional_abs_buffer

        self.predicted_h = None

        self.initializing = True
        self.name = "TrendPredictionStrat"
        self.parameter_names = ["pred_percentage_buffer",
                                "pred_abs_buffer",
                                "additional_abs_buffer"]  # everything that can be used to init the strat

    def initrun(self, data):

        """
        wait for first trend (up or down) but don't act on it,
        enter when first trend ends and make the kind of
        your entry (long/short) dependent on the former trend

        :param data:
        :return:
        """

        self.hist = []
        self.hist_prices = []
        self.initializing = True
        self.high = self.low = self.begin = data[0]["price"]

    def run(self, data):

        debug = True
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
                self.uptrend = False
                self.initializing = False

        # <editor-fold desc="strategy part">
        else:

            if self.uptrend:

                # exit condition long
                if self.predicted_h:

                    true_uptrend_start = self.begin - self.threshold
                    current_trend_h = last - true_uptrend_start

                    if current_trend_h >= self.predicted_h * self.pred_percentage_buffer or \
                       current_trend_h >= self.predicted_h - self.pred_abs_buffer:

                        actions.append(["exitLong", []])

                ###############################################################################

                if last > self.high:
                    self.high = last
                    self.high_index = i
                else:
                    if last < self.high - self.threshold:
                        # count last peak as end of high trend and as start of low trend

                        # todo: up trend ended min_trend_h bars ago,
                        #       down trend started min_threshold bars ago

                        # if not already done, exit long trends now
                        actions.append(["exitLong", []])

                        past_uptrend = Trend(self.hist[self.low_index:self.high_index + 1])
                        if debug:
                            print("\n---Past uptrend was:", past_uptrend.height)
                        _, pred_h = self.anal.predict_next_trend(past_uptrend,
                                                                 mode=self.pred_mode,
                                                                 number_of_similar_trends_used=self.number_of_similar_trends_used,
                                                                 similarity_threshold=self.similarity_threshold,
                                                                 min_trends_used=self.min_trends_used,
                                                                 p=debug)
                        if pred_h:
                            self.predicted_h = pred_h
                            if debug:
                                #print("    Predicted down height:", pred_h)
                                print("\n---Predicted downtrend height:", pred_h)

                        if pred_h and \
                           abs(self.predicted_h) * self.pred_percentage_buffer >= self.entry_pred_threshold and \
                           abs(self.predicted_h) - self.pred_abs_buffer >= self.entry_pred_threshold:

                            actions.append(["enterShort", []])

                        # change index
                        if debug:
                            #print("Actual up height:      ", last - self.begin + 2*self.threshold)
                            pass
                        self.low = last
                        self.low_index = i
                        self.uptrend = False
                        self.begin = last
            else:

                # exit condition short
                if self.predicted_h:

                    true_downtrend_start = self.begin + self.threshold
                    current_trend_h = true_downtrend_start - last

                    if current_trend_h <= self.predicted_h * self.pred_percentage_buffer or \
                       current_trend_h <= self.predicted_h + self.pred_abs_buffer:
                        actions.append(["exitShort", []])

                ###############################################################################

                if last < self.low:
                    self.low = last
                    self.low_index = i
                else:
                    if last > self.low + self.threshold:

                        # todo: down trend ends here

                        actions.append(["exitShort", []])

                        # entry conditions for long trades
                        past_downtrend = Trend(self.hist[self.high_index:self.low_index + 1])
                        if debug:
                            print("\n---Past downtrend was:", past_downtrend.height)
                        _, pred_h = self.anal.predict_next_trend(past_downtrend,
                                                                 mode=self.pred_mode,
                                                                 number_of_similar_trends_used=self.number_of_similar_trends_used,
                                                                 similarity_threshold=self.similarity_threshold,
                                                                 min_trends_used=self.min_trends_used,
                                                                 p=debug)

                        if pred_h:
                            self.predicted_h = pred_h
                            if debug:
                                #print("Predicted up height:   ", pred_h)
                                print("\n---Predicted uptrend height:", pred_h)

                        if pred_h and \
                           abs(self.predicted_h) * self.pred_percentage_buffer >= self.entry_pred_threshold and \
                           abs(self.predicted_h) - self.pred_abs_buffer >= self.entry_pred_threshold:

                            actions.append(["enterLong", []])

                        if debug:
                            #print("    Actual down height:    ", last - self.begin - 2*self.threshold)
                            pass
                        self.high = last
                        self.high_index = i
                        self.uptrend = True
                        self.begin = last

            return actions
        # </editor-fold>


# done
class SMACrossover(Strategy):

    def __init__(self, short, long):

        super().__init__(long)
        self.name = "SMACrossoverStrat"

        self.fast_sma = SMA(short)
        self.slow_sma = SMA(long)

    def initrun(self, data):
        if len(data) < self.lookback:
            raise AssertionError("too few values provided")

        self.slow_sma.initialize(data)
        self.fast_sma.initialize(data)

    def run(self, data):

        actions = []
        prices = [d["price"] for d in data]

        self.fast_sma(prices)
        self.slow_sma(prices)

        if cross_up(self.fast_sma, self.slow_sma):
            actions.append(["exitShort", ()])
            actions.append(["enterLong", ()])

        if cross_down(self.fast_sma, self.slow_sma):
            actions.append(["exitLong", ()])
            actions.append(["enterShort", ()])

        return actions


# done
class LPCrossover(Strategy):

    def __init__(self, short, long):

        super().__init__(0)
        self.name = "LPCrossoverStrat"

        self.fast_lp = Lowpass(short)
        self.smooth_lp = Lowpass(long)

    def initrun(self, data):
        if len(data) < self.lookback:
            raise AssertionError("too few values provided")

        prices = [q["price"] for q in data]

        self.smooth_lp.initialize(prices)
        self.fast_lp.initialize(prices)

    def run(self, data):

        actions = []
        prices = [d["price"] for d in data]

        self.fast_lp(prices)
        self.smooth_lp(prices)

        if cross_up(self.fast_lp, self.smooth_lp):
            actions.append(["exitShort", ()])
            actions.append(["enterLong", ()])

        if cross_down(self.fast_lp, self.smooth_lp):
            actions.append(["exitLong", ()])
            actions.append(["enterShort", ()])

        return actions


"""
class IndicatorSignalStrat(Strategy):

    def __init__(self, entry_signal_long, exit_signal_long,
                       entry_signal_short, exit_signal_short):

        self.lookback = max(entry_signal_long.lookback, exit_signal_long.lookback,
                            entry_signal_short.lookback, exit_signal_short.lookback)
        print("Lookback:", self.lookback)
        super().__init__(self.lookback)

        self.name = "IndicatorSignalStrat"

        self.entry_signal_long = entry_signal_long
        self.exit_signal_long = exit_signal_long

        self.entry_signal_short = entry_signal_short
        self.exit_signal_short = exit_signal_short

        self.counter = 0

    def initrun(self, data):

        prices = [d["price"] for d in data]
        self.entry_signal_long.initialize(prices)
        self.exit_signal_long.initialize(prices)
        self.entry_signal_short.initialize(prices)
        self.exit_signal_short.initialize(prices)

    def run(self, data):
        
        actions = []

        price = data[0]["price"]
        time = data[0]["time"]

        prices = [d["price"] for d in data]

        enter_long = self.entry_signal_long(prices)
        exit_long = self.exit_signal_long(prices)

        enter_short = self.entry_signal_short(prices)
        exit_short = self.exit_signal_short(prices)

        if enter_long and not exit_long:
            actions.append(["enterLong", price, time, ()])
        if exit_long:
            actions.append(["exitLong", price, time, ()])

        if enter_short and not exit_short:
            actions.append(["enterShort", price, time, ()])
        if exit_short:
            actions.append(["exitShort", price, time, ()])

        return actions
"""

strat_dict = {"trend follow": TrendFollowStrat,
              "trend follow prob": TrendFollowStratWithProbModell,
              "market open spike": MarketOpenSpike,
              "filtered trend follow": FilteredTrendFollowStrat,
              "SMACrossoverStrat": SMACrossover,
              "LPCrossoverStrat": LPCrossover,
              "trend pred": TrendPredictionStrat,
              }
