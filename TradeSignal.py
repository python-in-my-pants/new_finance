class TradeSignal:

    def __init__(self, trigger, indicators, direction="up"):
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
        self.lookback = max([indicator.lookback+1 for indicator in self.indicators])
        self.trigger = trigger
        self.direction = direction

    def __call__(self, prices):

        if len(prices) < self.lookback:
            print("Error in TradeSignal call, too few prices provided, look back not reached! "
                  "Provided {}, lookback is {}".format(len(prices), self.lookback))
            return

        for indicator in self.indicators:
            indicator(prices[-indicator.lookback:])

        return self.trigger(*self.indicators)

    def initialize(self, prices):

        for indicator in self.indicators:
            indicator.initialize(prices)
