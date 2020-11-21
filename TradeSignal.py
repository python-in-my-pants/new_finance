class TradeSignal:

    def __init__(self, trigger, indicators):
        """
        example call:

        trade_signal = TradeSignal(cross_up,
                                   [make_indicator(SMA, 12), make_indicator(SMA, 26)])

        trade_signal(prices) now gives the output of cross_up for SMA12 and SMA26, while automatically feeding
        the indicators with the prices and updating their states
        """

        if type(indicators) is not list:
            indicators = [indicators]

        self.indicators = indicators
        # holds all the values returned from its indicators
        self.indicator_value_histories = [[] for _ in range(len(self.indicators))]
        self.lookback = max([indicator.lookback for indicator in self.indicators])
        self.trigger = trigger

    def initialize(self, prices):

        for indicator in self.indicators:
            indicator.initialize(prices)

        for i, indicator_value_hist in enumerate(self.indicator_value_histories):
            indicator_value_hist.append(self.indicators[i](prices))

    def __call__(self, prices):

        if len(prices) < self.lookback:
            print("Error in TradeSignal call, too few prices provided, look back not reached! "
                  "Provided {}, lookback is {}".format(len(prices), self.lookback))
            return

        for i, indicator_value_hist in enumerate(self.indicator_value_histories):
            relevant_prices = prices[-self.indicators[i].lookback:]
            indicator_value = self.indicators[i](relevant_prices)
            indicator_value_hist.append(indicator_value)

        for entry in self.indicator_value_histories:
            print("indicator values from trade signal:", entry)

        return self.trigger(*self.indicator_value_histories)
