import numpy as np
from SignalTransformation import *
from abc import ABC, abstractmethod


class Indicator(ABC):

    def __init__(self):
        self.lookback = 0

    def __call__(self, prices):
        self.assert_lookback(prices)
        self.call(prices)

    def assert_lookback(self, prices):
        if len(prices) < self.lookback:
            raise AssertionError("Not enough values for indicator provided: {}, please supply {} values"
                                 .format(len(prices), self.lookback))

    @abstractmethod
    def call(self, price):
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


class SMA(Indicator):

    def __init__(self, period):
        Indicator.__init__(self)
        self.lookback = period
        self.period = period
        self.values = []

    def initialize(self, prices):
        self.values = [prices[0] for _ in range(self.period)]

    def call(self, prices):
        self.values = prices #self.values[1:] + [prices]
        return sum(self.values)/len(self.values)

    @staticmethod
    def get_full(prices, *args, **kwargs):
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



