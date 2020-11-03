import numpy as np
from SignalTransformation import *


class MA:

    def __init__(self, period, price):
        self.period = period
        self.values = [price for _ in range(period)]

    def __call__(self, price):
        self.values = self.values[1:] + [price]
        return np.average(self.values)

    def get_full(self, prices):
        return np.convolve(prices, np.repeat(1.0, self.period) / self.period, 'valid')


class Lowpass:

    def __init__(self, f, order=10, sampling_frequency=1/60):
        self.freq = f
        self.prices = list()
        self.order = order
        self.sampling_freq = sampling_frequency  # change this if quotes are not per minute!

        self.values = []

    def __call__(self):

        for v in self.values:
            yield v

    def initialize(self, prices):
        self.values = lowpass(prices, self.freq, order=self.order, sampling_frequency=self.sampling_freq)


class Highpass:

    def __init__(self, f, lookback=100, order=10, sampling_frequency=1/60):
        self.freq = f
        self.lookback = lookback
        self.prices = list()
        self.order = order
        self.sampling_freq = sampling_frequency  # change this if quotes are not per minute!

    def __call__(self, price):
        self.prices.append(price)
        filtered = highpass(self.prices[-self.lookback:], self.freq,
                            order=self.order, sampling_frequency=self.sampling_freq)
        # only return last value here ... is there a more perfomant way of getting this?
        return filtered[-1]

    def initialize(self, prices):
        # return highpass(prices, self.freq, order=self.order, sampling_frequency=self.sampling_freq)
        return [self(p) for p in prices]


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