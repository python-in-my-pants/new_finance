import numpy as np
from SignalTransformation import *
from abc import ABC, abstractmethod
from Analyzer import Analyzer


class Indicator(ABC):

    def __init__(self):
        self.lookback = 0

    def __call__(self, prices):
        self.assert_lookback(prices)
        return self.call(prices)

    def assert_lookback(self, prices):
        if self.lookback == 1 and type(prices) is not list:
            return
        if len(prices) < self.lookback:
            raise AssertionError("Not enough values for indicator provided: {}, please supply {} values"
                                 .format(len(prices), self.lookback))

    @abstractmethod
    def call(self, prices):
        """

        :param prices:
        :return: single value of the indicator based on "prices" and maybe indicator state
        """
        pass

    @abstractmethod
    def initialize(self, prices):
        pass

    @staticmethod
    @abstractmethod
    def get_full(prices, *args, **kwargs):
        pass


def makeIndicator(indicator_name, *indicator_params):
    return indicator_name(*indicator_params)


class TrendIndicator(Indicator):

    def __init__(self, threshold):
        super().__init__()
        self.lookback = 1

        self.threshold = threshold
        self.initializing = True

        self.uptrend = -1
        self.high = self.low = self.begin = 0
        self.high_index = self.low_index = self.break_index = 0

        self.call_index = 0

    def call(self, price):

        self.call_index += 1

        i = self.call_index
        last = price

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

                        # up trend ends here
                        # TODO former trend (down) was from lowindex:highindex+1 which we know only now

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

                        # TODO former trend (up) was from highindex:lowindex+1 which we know only now
                        # down trend ends here
                        self.high = last
                        self.high_index = i
                        self.uptrend = True

        if self.initializing:
            return 0
        else:
            if self.uptrend:
                return 1
            else:
                return -1

    def initialize(self, prices):
        self.initializing = True
        self.high = self.low = self.begin = prices[0]
        self.call_index = 0

    @staticmethod
    def get_full(prices, *args, **kwargs):

        if args:
            threshold = args[0]
        elif kwargs:
            threshold = kwargs["threshold"]
        else:
            raise AssertionError("You must provide a threshold for the TrendIndicator!")

        trend_indicator_obj = TrendIndicator(threshold)
        trend_indicator_obj.initialize(prices)
        results = []
        for price in prices:
            tmp = trend_indicator_obj(price)
            print(tmp)
            results.append(tmp)

        return results


class SMA(Indicator):

    def __init__(self, period):
        Indicator.__init__(self)
        self.lookback = period
        self.period = period
        self.values = []
        self.name = "SMA {}".format(self.period)

    def initialize(self, prices):
        self.values = [prices[0] for _ in range(self.period)]

    def call(self, prices):
        self.values = prices #self.values[1:] + [prices]
        return sum(self.values)/len(self.values)

    @staticmethod
    def get_full(prices, *args, **kwargs):
        if args:
            return np.concatenate(
                (np.asarray([prices[0] for _ in range(args[0] - 1)]),
                 np.convolve(prices, np.repeat(1.0, args[0]) / args[0], 'valid')))

        return np.concatenate(
            (np.asarray([prices[0] for _ in range(kwargs["period"]-1)]),
             np.convolve(prices, np.repeat(1.0, kwargs["period"]) / kwargs["period"], 'valid')))


class Lowpass(Indicator):

    def __init__(self, f, sampling_frequency=1 / 60):
        super().__init__()
        self.freq = f
        self.prices = list()
        self.sampling_freq = sampling_frequency  # change this if quotes are not per minute!
        self.lookback = 0
        self.values = []

    def call(self, price):
        for v in self.values:
            yield v

    def initialize(self, prices):
        self.values = lowpass(prices, self.freq, sampling_frequency=self.sampling_freq)

    @staticmethod
    def get_full(prices, *args, **kwargs):
        """
        :param prices: the prices to use
        :param args: nothing
        :param kwargs: cutoff_freq as the cutoff frequency for the lowpass filter
        :return: lowpass filtered prices
        """
        if args:
            return lowpass(prices, *args)
        return lowpass(prices, kwargs["cutoff_freq"])


class Highpass(Indicator):

    def __init__(self, f, sampling_frequency=1 / 60):
        super().__init__()
        self.freq = f
        self.prices = list()
        self.sampling_freq = sampling_frequency  # change this if quotes are not per minute!
        self.lookback = 0
        self.values = []

    def call(self, price):
        for v in self.values:
            yield v

    def initialize(self, prices):
        self.values = lowpass(prices, self.freq, sampling_frequency=self.sampling_freq)

    @staticmethod
    def get_full(prices, *args, **kwargs):
        """
        :param prices: the prices to use
        :param args: nothing
        :param kwargs: cutoff_freq as the cutoff frequency for the lowpass filter
        :return: lowpass filtered prices
        """
        return highpass(prices, kwargs["cutoff_freq"])


# todo
class Bandpass:

    def __init__(self, f1, f2, lookback=100, order=10, sampling_frequency=1/60):
        self.freq1 = f1
        self.freq2 = f2
        self.lookback = lookback
        self.prices = list()
        self.order = order
        self.sampling_freq = sampling_frequency  # change this if quotes are not per minute!

    def __call__(self, price):
        self.prices.append(price)
        filtered = bandpass(self.prices[-self.lookback:], self.freq1, self.freq2,
                            order=self.order, sampling_frequency=self.sampling_freq)
        # only return last value here ... is there a more perfomant way of getting this?
        return filtered[-1]

    def initialize(self, prices):
        # return bandpass(prices, self.freq1, self.freq2, order=self.order, sampling_frequency=self.sampling_freq)
        return [self(p) for p in prices]


# todo
class Bandstop:

    def __init__(self, f1, f2, lookback=100, order=10, sampling_frequency=1/60):
        self.freq1 = f1
        self.freq2 = f2
        self.lookback = lookback
        self.prices = list()
        self.order = order
        self.sampling_freq = sampling_frequency  # change this if quotes are not per minute!

    def __call__(self, price):
        self.prices.append(price)
        filtered = bandstop(self.prices[-self.lookback:], self.freq1, self.freq2,
                            order=self.order, sampling_frequency=self.sampling_freq)
        # only return last value here ... is there a more perfomant way of getting this?
        return filtered[-1]

    def initialize(self, prices):
        # return bandstop(prices, self.freq1, self.freq2, order=self.order, sampling_frequency=self.sampling_freq)
        return [self(p) for p in prices]


class FunnyIndicator:

    def __init__(self):
        pass

    @staticmethod
    def get_full(prices):
        lowp_raw10 = Lowpass.get_full(prices, cutoff_freq=10)
        derivative = Analyzer.derive_same_len(lowp_raw10, times=2)  # 2
        mader = [i for i in Lowpass.get_full(derivative, cutoff_freq=5)]
        return Lowpass.get_full(mader, cutoff_freq=10)

    def __call__(self, *args, **kwargs):
        pass

    def initialize(self, prices):
        pass
