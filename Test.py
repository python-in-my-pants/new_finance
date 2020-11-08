from Analyzer import *
from Strats import *
from History import *
from ArtificialHistory import *
from Indicators import *
from TradeSignal import *
from Triggers import *
import numpy as np


def func():

    dax_hist = History("GER30", 2017, use_cached=True)

    def autocor_trend_mom():

        anal16 = Analyzer(dax_hist, min_trend_h=82, realistic=False)

        moms = np.asarray([abs(trend.momentum) for trend in anal16.trend_list])

        Plotter.sm_autocor(moms)

    def anal_trends():
        anal16 = Analyzer(dax_hist)
        plotter = Plotter(anal16)

        trends = anal16.get_trends(anal16.hist, min_trend_h=10, realistic=False)
        trend_heights = [abs(x.height) for x in trends]
        trend_len = [abs(x.len) for x in trends]
        mom = [abs(x.height/x.len)*10 for x in trends]

        plotter.plot_general_same_y(list(range(len(trends))),
                                    [trend_heights, trend_len, mom],
                                    x_label="Trends",
                                    y_labels=["heights", "lens", "momentum"])

    def backtest():
        t = Trader.HALF_RISK
        trend_follow = strat_dict["trend follow"](82)  # 82

        bt2 = Backtester(trend_follow, dax_hist, use_balance=True, asset_data=AssetData.GER30,
                         trader_data=t, ts_in_pips=45, sl_in_pips=45)

        bt2.test(use_sl_for_risk=False, full_print=True)
        # bt2.profit_with_tax(2016, 2020, sl_for_risk=False)

    def plot_some_indicators():
        prices = dax_hist.prices

        lowp_raw10 = Lowpass.get_full(prices, cutoff_freq=10)
        ma = SMA.get_full(lowp_raw10, period=5)
        derivative = Analyzer.derive_same_len(lowp_raw10, times=2)  # 2
        mader = [12000 + i * 10000 for i in Lowpass.get_full(derivative, cutoff_freq=5)]
        lpmader = Lowpass.get_full(mader, cutoff_freq=10)

        # to prevent zigzag trades from 0-crosses or other crosses, use rolling window of fixed size over integral of both
        # monitored graphs, so small negative dips don't trigger the integral to be negative, of course at the price
        # of bigger lag the bigger the window

        Plotter.plot_general_same_y(x=list(range(len(dax_hist.prices))),
                                    ys=[prices,
                                        ma,
                                        lpmader,
                                        [12000 for _ in range(len(dax_hist.prices))]],
                                    y_labels=["prices",
                                              "ma",
                                              "lp der",
                                              "12k"],
                                    crosshair=False)

    def test_singal_strat():
        trader = Trader.HALF_RISK

        short_period = 50
        long_period = 82

        out_short = TradeSignal(cross_up,
                                [makeIndicator(SMA, short_period),
                                 makeIndicator(SMA, long_period)])
        in_long = TradeSignal(cross_up,
                              [makeIndicator(SMA, short_period),
                               makeIndicator(SMA, long_period)])

        out_long = TradeSignal(cross_down,
                               [makeIndicator(SMA, short_period),
                                makeIndicator(SMA, long_period)])
        in_short = TradeSignal(cross_down,
                               [makeIndicator(SMA, short_period),
                                makeIndicator(SMA, long_period)])

        signal_strat = strat_dict["indicator signal"](in_long, out_long, in_short, out_short)

        bt2 = Backtester(signal_strat, dax_hist, use_balance=True,
                         asset_data=AssetData.GER30, trader_data=trader)

        bt2.test(use_sl_for_risk=False, full_print=True)

    autocor_trend_mom()


func()
