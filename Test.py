from Analyzer import *
from Backtester import Backtester
from Strats import *
from History import *
from ArtificialHistory import *
from Indicators import *
from TradeSignal import *
from Triggers import *
import numpy as np


def func():

    dax_hist = History("GER30", 2016, use_cached=True)

    def autocor_trend_mom():

        anal16 = Analyzer(dax_hist, min_trend_h=5, realistic=False)

        moms = np.asarray([abs(trend.momentum) for trend in anal16.trend_list])

        Plotter.sm_autocor(moms)

    def anal_trends():
        anal16 = Analyzer(dax_hist)
        plotter = Plotter(anal16)

        trends = anal16.get_trends(anal16.hist, min_trend_h=5, realistic=False)
        trend_heights = [abs(x.height) for x in trends]
        trend_len = [abs(x.len) for x in trends]
        mom = [abs(x.height/x.len)*10 for x in trends]

        plotter.plot_general_same_y(list(range(len(trends))),
                                    [trend_heights, trend_len, mom],
                                    x_label="Trends",
                                    y_labels=["heights", "lens", "momentum"])

    def backtest():
        t = Trader.DEFAULT

        """entry_sig_l = TradeSignal(cross_up, [makeIndicator(SMA, 5), makeIndicator(SMA, 10)])
        exit_sig_l = TradeSignal(cross_down, [makeIndicator(SMA, 5), makeIndicator(SMA, 10)])

        entry_sig_s = TradeSignal(cross_down, [makeIndicator(SMA, 5), makeIndicator(SMA, 10)])
        exit_sig_s = TradeSignal(cross_up, [makeIndicator(SMA, 5), makeIndicator(SMA, 10)])

        lp_co = strat_dict["indicator signal"](entry_sig_l, exit_sig_l, entry_sig_s, exit_sig_s)  # 82"""

        trend_follow = strat_dict["trend follow"](35)

        bt1 = Backtester(trend_follow, dax_hist,
                         use_balance=True, asset_data=AssetData.GER30,
                         trader_data=t)#, ts_in_pips=45, sl_in_pips=45)
        # bt1.deep_test(deepness=100)
        bt1.test()

        """for i in range(5, 200, 5):
            ts = anal.get_trends(dax_hist.hist, min_trend_h=i)
            avg_h = avg([abs(t.height) for t in ts])
            print(i, avg_h, avg_h/i)"""

        bt1.plot_trades(crosshair=False, plot_trends=True, min_trend_h=35,
                        indicators=[[1, TrendIndicator, 35]])

        """,
                        indicators=[[0, SMA, 5],
                                    [0, SMA, 10]])"""
        # bt2.profit_with_tax(2016, 2020, sl_for_risk=False)

    def plot_some_indicators():
        prices = dax_hist.prices

        lowp_raw10 = Lowpass.get_full(prices, cutoff_freq=10)
        ma = SMA.get_full(lowp_raw10, period=5)
        derivative = Analyzer.derive_same_len(lowp_raw10, times=2)  # 2
        mader = [12000 + i * 10000 for i in Lowpass.get_full(derivative, cutoff_freq=5)]
        lpmader = Lowpass.get_full(mader, cutoff_freq=10)

        # TODO
        #  to prevent zigzag trades from 0-crosses or other crosses, use rolling window of fixed size over integral of
        #  both monitored graphs, so small negative dips don't trigger the integral to be negative, of course at the
        #  price of bigger lag the bigger the window

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

        # TODO does not trade anymore, why?

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

        bt2 = Backtester(signal_strat, dax_hist,
                         use_balance=False,
                         asset_data=AssetData.GER30, trader_data=trader)

        bt2.test(use_sl_for_risk=False, full_print=True)
        bt2.plot_trades(plot_trends=False,
                        indicators=[[0, SMA, short_period], [0, SMA, long_period]],
                        bar_amount=1000, crosshair=False)

    def test_artif_hist():

        arti = ArtificialHistory(dax_hist, h=40)

        Plotter.plot_general_same_y(list(range(len(dax_hist.prices))),
                                    [dax_hist.prices, arti.prices],
                                    y_labels=["real", "arti"])

    def test_trend_forecast():

        analyzer = Analyzer(dax_hist, min_trend_h=82)

        plen, ph = 0, 0
        d_len_sum, d_h_sum = 0, 0

        delta_len_percentual, delta_h_percentual = 0, 0
        delta_len_percentual_sum, delta_h_percentual_sum = 0, 0

        for trend in analyzer.trend_list:

            print("Len: {} \t\t\t\t\t\t\tHeight: {}".format(trend.len, trend.height))

            d_len, d_h = abs(plen - trend.len), abs(ph - trend.height)
            d_len_sum += d_len
            d_h_sum += d_h_sum

            if not (d_len == 0 and d_h == 0):

                delta_len_percentual, delta_h_percentual = abs(d_len/trend.len), abs(d_h/trend.height)
                delta_len_percentual_sum += delta_len_percentual
                delta_h_percentual_sum += delta_h_percentual

                print("Diff len: {:12.2f}\t\t\t\tDiff height: {:12.2f}".format(d_len, d_h))
                print("Diff len percentual: {:12.2f} % Diff height percentual: {:12.2f} %"
                      .format(delta_len_percentual*100, delta_h_percentual*100))

            plen, ph = analyzer.expected_following_trend_from_trend(trend, similarity_threshold=0.5)

            if not (plen == 0 and ph == 0):
                print("\n                         Pred len: {:12.2f} Pred height: {:12.2f}".format(plen, ph))

            print()

        print("Avg d len: {:12.2f} Avg d h: {:12.2f}"
              .format(d_len_sum/(len(analyzer.trend_list)-1), d_h_sum/(len(analyzer.trend_list)-1)))
        print("Avg d len percentual: {:12.2f} % Avg d h percentual: {:12.2f} %"
              .format(100*delta_len_percentual_sum / (len(analyzer.trend_list) - 1),
                      100*delta_h_percentual_sum/(len(analyzer.trend_list)-1)))

    def test_trend_forecasting_ability(n=20, min_trend_h=50):

        np.set_printoptions(suppress=True)

        avg_rand_difference = 0

        anal = Analyzer(dax_hist, min_trend_h=min_trend_h)

        r = len(anal.trend_list) - 1

        for _ in range(r):
            random_list_a = (np.random.random(n) * 2) - 1
            random_list_b = (np.random.random(n) * 2) - 1

            avg_rand_difference += np.sum(np.absolute(random_list_a - random_list_b))/n

        avg_rand_difference /= r

        print("Avg rand difference:               ", avg_rand_difference)

        # ########################################

        avg_avg_sim_difference = 0

        for base_trend in anal.trend_list[:-1]:

            base_similar_trend_containers, base_sum_sim = anal.get_similar_trends(base_trend, n, -1)
            next_similar_trend_containers, next_sum_sim = anal.get_similar_trends(base_trend.next_trend, n, -1)

            base_similarities = np.asarray([container.sim for container in base_similar_trend_containers])
            next_similarities = np.asarray([container.sim for container in next_similar_trend_containers])

            avg_avg_sim_difference += np.sum(np.absolute(base_similarities - next_similarities))/n

        avg_avg_sim_difference /= r

        print("Average sim difference:            ", avg_avg_sim_difference)

        print("Average following trend similarity:", anal.get_avg_following_trend_similarity())

        return avg_rand_difference, avg_avg_sim_difference

    def plot_trend_forecasting_ability(low=5, high=180, steps=1):
        xs = list(range(low, high, steps))
        diffs = list(zip(*[test_trend_forecasting_ability(min_trend_h=i) for i in xs]))
        rand_diff = diffs[0]
        sim_diff = diffs[1]
        abs_diffs = [abs(r-s) for r, s in zip(rand_diff, sim_diff)]

        Plotter.plot_general_same_y(xs, [rand_diff, sim_diff, abs_diffs],
                                    x_label="Min trend h", y_labels=["Rand diff", "Sim diff", "Abs diff"])

    # todo 25 pips before predicted height end and 80% of predicted height as exit points, whatever is earlier

    def test_trend_forecasting_model(test_years,
                                     min_trend_h,
                                     model_years=None,
                                     model_hist=None,
                                     strict_mode=False):

        if not model_years and not model_hist:
            raise Exception("You must provide a model history or year for a model history!")

        if model_years:
            if type(model_years) is not list:
                model_years = list(model_years)

            if len(model_years) > 2:
                model_years = [model_years[0], model_years[-1]]

        if type(test_years) is not list:
            test_years = list(test_years)

        if len(test_years) > 2:
            test_years = [test_years[0], test_years[-1]]

        if model_hist:
            anal = Analyzer(model_hist, min_trend_h=min_trend_h, realistic=False)
        else:
            h = History("GER30", *model_years)
            anal = Analyzer(h, min_trend_h=min_trend_h, realistic=False)

        anal.get_intern_trend_prediction_error(p=True, use_strict_mode=strict_mode)
        test_anal = Analyzer(History("GER30", *test_years), min_trend_h=min_trend_h, realistic=False)

        anal.get_extern_trend_prediction_error(test_anal.trend_list, p=True, use_strict_mode=strict_mode)

    #test_trend_forecasting_model([2017, 2018, 2019], 35, model_hist=dax_hist)
    test_singal_strat()


func()
