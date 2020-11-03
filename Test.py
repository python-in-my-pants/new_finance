from Analyzer import *
from Strats import *
from History import *
from ArtificialHistory import *
from SignalTransformation import *
from Indicators import *


def func():

    dax_hist = History("GER30", 2017)
    #lp = Lowpass(10)
    #lp.initialize(dax_hist.prices)
    #filtered_slow = lp.values

    def anal_trends():

        anal16 = Analyzer(dax_hist, realistic=True, min_trend_h=40)
        plotter = Plotter(anal16)

        trends = anal16.get_trends(anal16.hist, min_trend_h=82, realistic=False)
        trend_heights = [x.height for x in trends]
        trend_len = [x.len for x in trends]

        plotter.plot_general_same_y(list(range(len(trends))), [trend_heights, trend_len],
                                    x_label="Trends", y_labels=["heights", "lens"])

    def backtest():
        t = Trader.DEFAULT
        trend_follow = strat_dict["filtered trend follow"](40, Lowpass, 40) #82

        bt2 = Backtester(trend_follow, dax_hist, use_balance=True, asset_data=AssetData.GER30,
                         trader_data=t)#, ts_in_pips=45, sl_in_pips=45)

        bt2.test(use_sl_for_risk=False, full_print=True)
        # bt2.profit_with_tax(2016, 2020, sl_for_risk=False)

    backtest()
    #Plotter.plot_general_same_y(list(range(len(filtered_slow))), [filtered_slow, dax_hist.prices], y_labels=["filtered prices", "raw prices"])


func()
