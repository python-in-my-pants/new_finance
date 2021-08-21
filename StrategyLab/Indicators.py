import numpy as np
from SignalTransformation import *
from abc import ABC, abstractmethod
from Analyzer import Analyzer


class Indicator(ABC):

    def __init__(self):
        self.lookback = 0
        self.prior_values = []

    def __call__(self, prices):
        self._assert_lookback(prices)
        v = self._call(prices)
        self.prior_values.append(v)
        return v

    def _assert_lookback(self, prices):
        if self.lookback == 1 and type(prices) is not list:
            return
        if len(prices) < self.lookback:
            raise AssertionError("Not enough values for indicator provided: {}, please supply {} values"
                                 .format(len(prices), self.lookback))

    @abstractmethod
    def _call(self, prices):
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


# todo make atr for dynamic threshold trend follow strat


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

    def _call(self, prices):

        uptrend = 1
        downtrend = -1
        undecided = 0
        stagnating_uptrend = 0.5
        stagnating_downtrend = -0.5

        self.call_index += 1

        i = self.call_index
        last = prices if type(prices) is not list else prices[-1]

        if self.initializing:

            if last > self.high:
                self.high = last
            if last < self.low:
                self.low = last

            if last > self.low + self.threshold:
                self.uptrend = True
                self.initializing = False
                return uptrend
            if last < self.high - self.threshold:
                self.uptrend = False
                self.initializing = False
                return downtrend

            return undecided

        else:

            if self.uptrend:
                if last > self.high:
                    self.high = last
                    self.high_index = i
                    return uptrend
                else:
                    if last < self.high - self.threshold:
                        # count last peak as end of high trend and as start of low trend

                        # up trend ends here
                        # TODO former trend (down) was from lowindex:highindex+1 which we know only now

                        # change index
                        self.low = last
                        self.low_index = i
                        self.uptrend = False
                        return downtrend
                    return stagnating_uptrend
            else:
                if last < self.low:
                    self.low = last
                    self.low_index = i
                    return downtrend
                else:
                    if last > self.low + self.threshold:

                        # TODO former trend (up) was from highindex:lowindex+1 which we know only now
                        # down trend ends here
                        self.high = last
                        self.high_index = i
                        self.uptrend = True
                        return uptrend
                    return stagnating_downtrend

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
            results.append(tmp)

        return results


class SMA(Indicator):

    def __init__(self, period):
        Indicator.__init__(self)
        self.lookback = period
        self._period = period
        self.name = "SMA {}".format(self._period)

    def initialize(self, prices):
        pass

    def _call(self, prices):
        return sum(prices[-self.lookback:])/self._period

    @staticmethod
    def get_full(prices, *args, **kwargs):
        lookback = args[0]
        values = []
        for i in range(len(prices)):
            if i >= lookback:
                ps = prices[i-lookback:i]
                values.append(sum(ps)/lookback)
            else:
                values.append(prices[0])
        return values

        """if args:
            return np.concatenate(
                (np.asarray([prices[0] for _ in range(args[0] - 1)]),
                 np.convolve(prices, np.repeat(1.0, args[0]) / args[0], 'valid')))

        return np.concatenate(
            (np.asarray([prices[0] for _ in range(kwargs["period"]-1)]),
             np.convolve(prices, np.repeat(1.0, kwargs["period"]) / kwargs["period"], 'valid')))"""


class ATR(Indicator):

    def __init__(self, period):
        Indicator.__init__(self)
        self.lookback = period
        self._period = period
        self.name = "ATR {}".format(self._period)

    def initialize(self, prices):
        pass

    def _call(self, prices):
        der = Analyzer.derive_same_len(prices[-self.lookback:])[1:]
        return sum([abs(x) for x in der])/(self.lookback-1)

    @staticmethod
    def get_full(prices, *args, **kwargs):
        lookback = args[0]
        values = []
        for i in range(len(prices)):
            if i >= lookback:
                ps = prices[i-lookback:i]
                der = Analyzer.derive_same_len(ps[-lookback:])[1:]
                values.append(sum([abs(x) for x in der])/(lookback-1))
            else:
                values.append(0)
        return values


class Lowpass(Indicator):

    def __init__(self, f, sampling_frequency=1 / 60):
        super().__init__()
        self.freq = f
        self.prices = list()
        self.sampling_freq = sampling_frequency  # change this if quotes are not per minute!
        self.lookback = 0
        self.values = []
        self.call_counter = 0

    def _call(self, price):
        self.call_counter += 1
        return self.values[self.call_counter-1]

    def initialize(self, prices):
        self.values = lowpass(prices, self.freq, sampling_frequency=self.sampling_freq)
        self.call_counter = 0

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
        self.call_counter = 0

    def _call(self, price):
        self.call_counter += 1
        return self.values[self.call_counter - 1]

    def initialize(self, prices):
        self.values = lowpass(prices, self.freq, sampling_frequency=self.sampling_freq)
        self.call_counter = 0

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
