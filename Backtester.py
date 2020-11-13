from Utility import *
from Analyzer import Trade, Analyzer, Plotter
from ArtificialHistory import ArtificialHistory
from History import AssetData, Trader, History
import matplotlib.pyplot as plt
import functools as ft


class Backtester:

    def __init__(self, strat, hist_obj, use_balance=False,
                 asset_data=AssetData.DEFAULT, trader_data=Trader.DEFAULT,
                 tp_in_pips=None, sl_in_pips=None, ts_in_pips=None,
                 tp_in_value=None, sl_in_value=None):

        # unpack objects
        balance, max_risk_per_position = trader_data
        cost, min_shares, leverage = asset_data

        self.strat = strat
        self.hist_obj = hist_obj

        self.open_long_trades = []
        self.open_short_trades = []
        self.closed_trades = []

        self.base_balance = balance
        self.available_margin = balance
        self.use_balance = use_balance
        self.leverage = leverage
        self.risk = max_risk_per_position
        self.min_shares = min_shares
        self.trade_cost = cost
        self.max_balance = self.base_balance
        self.max_dd = 0
        self.max_dd_perc = 0

        self.sl_in_pips = sl_in_pips
        self.tp_in_pips = tp_in_pips
        self.ts_in_pips = ts_in_pips

        self.tp_in_value = tp_in_value
        self.sl_in_value = sl_in_value

        self.actions = {"enterLong": self.enterLong,
                        "exitLong": self.exitLong,
                        "enterShort": self.enterShort,
                        "exitShort": self.exitShort}

    def test(self, p=True, full_print=False, single_step=False, use_sl_for_risk=False,
             only_long=False, only_short=False, fast=True):

        """
        :param only_long:
        :param only_short:
        :param fast:
        :param p: print or not
        :param full_print: print additional info like trades
        :param single_step: simulate strategy step by step with user interaction
        :param use_sl_for_risk: opens positions with whole available margin and uses sl_in_value for limiting the
                                risk per position to a percentage of the available_margin
        :return: the profit of this strategy with these parameters over this time span
        """

        # todo lower risk over time, begin high risk, lower to the end

        prices = self.hist_obj.prices
        times = self.hist_obj.times
        lookback = self.strat.lookback

        self.open_long_trades = []
        self.open_short_trades = []
        self.closed_trades = []

        self.available_margin = self.base_balance
        self.max_dd = 0
        self.max_dd_perc = 0
        self.max_balance = self.base_balance
        margin_call = False

        self.strat.initrun(self.hist_obj.hist[:lookback + 1])

        if p:
            print("\nStarting backtest for strat", self.strat.name, "on", self.hist_obj.asset_name,
                  "over", self.hist_obj.timeframe)

        for index in range(len(prices)):

            if index >= lookback:

                price = prices[index]
                time = times[index]

                # test SL and TP and TS for all open trades
                for trade in self.open_long_trades + self.open_short_trades:
                    trade.check(price, time)

                # check for automatically closed trades
                for trade in self.open_long_trades:
                    if not trade.currently_open:
                        # handle margin for those trades, that closed themselves, as they did not go through
                        # "exitLong"/"exitShort"
                        self.available_margin += trade.profit
                        self.available_margin += trade.margin
                        self.closed_trades.append(trade)
                        self.open_long_trades.remove(trade)

                # check for automatically closed trades
                for trade in self.open_short_trades:
                    if not trade.currently_open:
                        # handle margin for those trades, that closed themselves, as they did not go through
                        # "exitLong"/"exitShort"
                        self.available_margin += trade.profit
                        self.available_margin += trade.margin
                        self.closed_trades.append(trade)
                        self.open_short_trades.remove(trade)

                total_positions_worth = sum(
                    [t.get_value(price) for t in self.open_long_trades + self.open_short_trades])

                # broker asks for refinancing open positions, cannot enter trades under this condition
                if self.use_balance and self.available_margin / 2 + total_positions_worth <= 0:  # <-- margin call

                    # all positions are automatically closed, you fucked up badly
                    if self.available_margin + total_positions_worth <= 0:
                        print("Closing positions, account worth is 0!")
                        self.exitLong(price, time)
                        self.exitShort(price, time)
                        if full_print:
                            self.print_trades()
                        return self.get_results(p=p)

                    # trigger margin call if not yet set
                    if full_print and not margin_call:
                        """
                        Margin call only occurs, when the profit of a certain position is negative and its absolute 
                        value is bigger than the positions margin, this causes the broker to call for a bigger 
                        margin to keep the position open

                        bound capital in a position is: price*position_size/leverage = margin
                        """

                        print("Margin call due to automated trade close (SL/TP/TS)!")
                        margin_call = True

                if self.available_margin > self.max_balance:
                    self.max_balance = self.available_margin
                if self.max_balance - self.available_margin > self.max_dd:
                    self.max_dd = self.max_balance - self.available_margin
                    self.max_dd_perc = 100 * (1 - (self.available_margin / self.max_balance))

                # exec strat
                action_list = self.strat.run(self.hist_obj.hist[index - lookback:index + 1])
                if action_list:
                    for entry in action_list:

                        if self.use_balance and self.available_margin / 2 + total_positions_worth > 0:
                            margin_call = False

                        action, price, time, trade_params = entry

                        if only_long and (action == "enterShort" or action == "exitShort"):
                            continue

                        if only_short and (action == "enterLong" or action == "exitLong"):
                            continue

                        if not trade_params or trade_params is None:
                            trade_params = [self.tp_in_pips, self.sl_in_pips, self.ts_in_pips,
                                            self.tp_in_value, self.sl_in_value]
                        if not trade_params[0]:
                            trade_params[0] = self.tp_in_pips
                        if not trade_params[1]:
                            trade_params[1] = self.sl_in_pips
                        if not trade_params[2]:
                            trade_params[2] = self.ts_in_pips
                        if not trade_params[3]:
                            trade_params[3] = self.tp_in_value
                        if not trade_params[4]:
                            trade_params[4] = self.sl_in_value

                        # open or close trades
                        if self.use_balance:
                            if action == "enterLong" or action == "enterShort":

                                if self.available_margin / 2 + total_positions_worth <= 0:

                                    if full_print and not margin_call:
                                        # trigger margin call if not yet
                                        print("Margin call! Cannot open new positions!")
                                        margin_call = True

                                    continue

                                if use_sl_for_risk:
                                    position_size = self.available_margin * self.leverage / price
                                else:
                                    position_size = self.risk * self.available_margin * self.leverage / price

                                if position_size < self.min_shares:
                                    continue
                                else:
                                    if use_sl_for_risk:
                                        self.sl_in_value = self.risk * self.available_margin
                                        trade_params[4] = self.sl_in_value

                            else:
                                position_size = -1
                        else:
                            position_size = 1
                        self.actions[action](price, time, position_size, trade_params)

                if full_print and p:
                    if index % len(prices) // 20 == 0:  #
                        self.get_results(p=True)

                if action_list and single_step:
                    self.print_test_status(price, time, index / len(prices))
                    if input() == "exit":
                        single_step = False

        # close all trades that are still open on last price in history
        for trade in self.open_long_trades + self.open_short_trades:
            trade.close(prices[-1], times[-1])

        if p and full_print:
            self.print_trades()
        if p:
            print("\nFinal results:\n---------------------------------------------------------------\n")
        results = self.get_results(p=p)
        if p and not fast:
            print("Best possible results:\n")
            Analyzer(hist_obj=self.hist_obj, min_trend_h=0, realistic=False).SIM_trend_follow(True)

        return results

    @timeit
    def deep_test(self, p=True, full_print=False, single_step=False, use_sl_for_risk=False,
                  only_long=False, only_short=False, fast=True, deepness=10, min_trend_h=40):

        print("\nStarting test #1")
        results = [self.test(p, full_print, single_step, use_sl_for_risk, only_long, only_short, fast)]
        og_hist = self.hist_obj

        for i in range(deepness-1):
            print("\nStarting test #{}".format(i+2))
            self.hist_obj = ArtificialHistory(og_hist, h=min_trend_h)
            results.append(self.test(p, full_print, single_step, use_sl_for_risk, only_long, only_short, fast))

        print("Average profit: ", avg(results))
        self.hist_obj = og_hist

    def profit_with_tax(self, start, end, p=True, sl_for_risk=False):

        use_bal = self.use_balance
        self.use_balance = True

        bal = self.base_balance
        running_bal = bal

        orig_hist = self.hist_obj
        tax_sum = 0

        if p:
            print("\nStarting backtest for strat", self.strat.name, "on", self.hist_obj.asset_name, "over",
                  start, "-", end, "with base balance", self.base_balance, "and risk of", self.risk)
            print("\n!!!   !!!   !!!   Scholz rule is NOT applied!   !!!   !!!   !!!\n")
            print("                                  balance: {:12.2f}".format(bal))

        for i in range(start, end + 1):

            self.hist_obj = History(self.hist_obj.asset_name, i)
            abs_p = self.test(p=False, use_sl_for_risk=sl_for_risk)
            p = (0.75 * abs_p)
            tax_sum += 0.25 * abs_p
            perc_gr = (((running_bal + p) / running_bal) - 1) * 100
            running_bal += p
            if p:
                print("Profit in {}:\t\t{:12.2f}, balance: {:12.2f}, growth in %: {} %, tax: {:12.2f}, trades: {}".
                      format(i, p, running_bal, "{:12.2f}".format(perc_gr).zfill(5), 0.25 * abs_p,
                             len(self.closed_trades)))
            self.base_balance = running_bal

        if p:
            print("\nProfit in {}-{}:{:12.2f}, balance: {:12.2f}, % growth: {:12.2f}%\nrisk free:\t\t\t{:12.2f}".
                  format(start, end, self.base_balance - bal, self.base_balance, ((self.base_balance / bal) - 1) * 100,
                         bal * (1.07 ** (end - start + 1))))
            # risk free is based off of all-weather portfolio with 9% gain/year -> 0.75*9% ~7%/year

        self.base_balance = bal

        if p:
            self.hist_obj = History(self.hist_obj.asset_name, start, end)
            pure_profit = self.test(p=False)
            print("\nPayed tax:\t\t\t{:12.2f}, prof\\tax:{:12.2f}\n".format(tax_sum, pure_profit))

        self.use_balance = use_bal
        self.hist_obj = orig_hist

        return running_bal - self.base_balance

    def optimize_sl(self, low=0, high=500, steps=1, p=False, in_pips=True):

        factor = 1
        if type(steps) is not int:
            factor = 1 / steps

            low = int(low * factor + 0.5)
            high = int(high * factor + 0.5)
            steps = int(steps * factor + 0.5)

        former_sl = self.sl_in_pips if in_pips else self.sl_in_value
        ran = list(range(low, high, steps))

        if in_pips:
            self.sl_in_pips = ran[0] / factor
        else:
            self.sl_in_value = ran[0] / factor
        best_res = self.test(p=False)  # todo risk sl
        best_sl = ran[0] / factor
        if p:
            print("Optimisation of stop loss started ...")
        p_counter = 0

        for sl in ran:
            if p:
                p_counter += 1
                if p_counter % (len(ran) // 10) == 0:
                    print("{:5.2f} % progress, best result so far: {} with stop loss of {}".format(
                        100 * p_counter / len(ran), best_res, best_sl / factor))
            if in_pips:
                self.sl_in_pips = sl / factor
            else:
                self.sl_in_value = sl / factor
            test_res = self.test(p=False)  # todo
            if test_res > best_res:
                best_res = test_res
                best_sl = sl / factor

        if in_pips:
            self.sl_in_pips = former_sl
        else:
            self.sl_in_value = former_sl
        if p:
            print("Optimised stop loss to", best_sl, "with profit of", best_res)
        return best_sl

    def optimize_ts(self, low=0, high=500, steps=1, p=False):

        factor = 1
        if type(steps) is not int:
            factor = 1 / steps

            low = int(low * factor + 0.5)
            high = int(high * factor + 0.5)
            steps = int(steps * factor + 0.5)

        former_ts = self.ts_in_pips
        ran = list(range(low, high, steps))

        self.ts_in_pips = ran[0] / factor
        best_res = self.test(p=False)
        best_ts = ran[0] / factor

        if p:
            print("Optimisation of trailing stop started ...")
            p_counter = 0

        for ts in ran:
            if p:
                p_counter += 1
                if p_counter % (len(ran) // 10) == 0:
                    print("{:5.2f} % progress, best results so far: {} with trailing stop of {}".
                          format(100 * p_counter / len(ran), best_res, best_ts / factor))
            self.ts_in_pips = ts / factor
            test_res = self.test(p=False)
            if test_res > best_res:
                best_res = test_res
                best_ts = ts / factor

        self.ts_in_pips = former_ts
        if p:
            print("Optimised trailing stop to", best_ts, "with profit of", best_res)
        return best_ts

    def optimize_strat_param(self, param_name, low=0, high=500, steps=1, p=True, to_plot=False):

        if param_name not in self.strat.parameter_names:
            print("The parameter you want to optimize doesn't exist in this strategy!")
            return -1

        try:
            factor = 1
            if type(steps) is not int:
                factor = 1 / steps

                low = int(low * factor + 0.5)
                high = int(high * factor + 0.5)
                steps = int(steps * factor + 0.5)

            init_value = vars(self.strat)[param_name]
            profits = []
            param_range = range(low, high, steps)
            vars(self.strat)[param_name] = param_range[0] / factor
            best_result = self.test(p=False)
            best_param = param_range[0] / factor

            if p:
                print("Optimization of parameter", "'" + param_name + "' started for strat", self.strat.name,
                      "on asset", self.hist_obj.asset_name)
                p_counter = 0

            for i in param_range:
                if p:
                    p_counter += 1
                    m = max(1, (len(param_range) // 10))
                    if p_counter % m == 0:
                        print("{:5.2f} % progress, best result so far: {:5.2f} with {} of {}".format(
                            100 * p_counter / len(param_range), best_result, param_name, best_param / factor))

                vars(self.strat)[param_name] = i / factor
                test_res = self.test(p=False)
                profits.append(test_res)
                if test_res > best_result:
                    best_result = test_res
                    best_param = i / factor

            vars(self.strat)[param_name] = init_value

            if not to_plot:
                if p:
                    print("Optimised ", param_name, "to", best_param, "with profit of", best_result)
                return best_param
            else:
                return [i / factor for i in list(param_range)], profits

        except Exception as e:
            print("Could not optimize strategy parameterr")
            print(e)

    def plot_param_profile(self, param_name, low=0, high=500, steps=1):
        plt.rc('figure', facecolor="#333333", edgecolor="#333333")
        plt.rc('axes', facecolor="#353535", edgecolor="#000000")
        plt.rc('lines', color="#393939")
        plt.rc('grid', color="#121212")
        xi, y = self.optimize_strat_param(param_name, low, high, steps, p=True, to_plot=True)
        Plotter.plot_general(xi, y, label1=param_name, label2="Profit")

    def print_test_status(self, price, time, progress):

        print("\nBalance: {:12.2f} / {}\t\tPrice: {:12.2f}\t\tTime: {}\t\tProgress: {:12.2f} %\n".
              format(self.available_margin, self.base_balance, price, time, progress))

        if self.open_long_trades:
            print("Open long trades:\n")
            for t in self.open_long_trades:
                print(t)

        if self.open_short_trades:
            print("Open short trades:\n")
            for t in self.open_short_trades:
                print(t)

        print("Results:\n-------------------------------------------------")
        self.get_results()
        print("#################################################")

    def get_results(self, p=True):  # maybe optimise if needed

        if not p:
            return sum([x.profit for x in self.closed_trades])

        trades = len(self.closed_trades)

        if trades:
            long_wins_num = len(list(filter(lambda x: x.direction == 1 and x.profit >= 0, self.closed_trades)))
            long_wins = sum([(x.profit if x.profit >= 0 else 0) for x in
                             list(filter(lambda x: x.direction == 1, self.closed_trades))])

            long_losses_num = len(list(filter(lambda x: x.direction == 1 and x.profit < 0, self.closed_trades)))
            long_losses = sum([(x.profit if x.profit < 0 else 0) for x in
                               list(filter(lambda x: x.direction == 1, self.closed_trades))])

            short_wins_num = len(list(filter(lambda x: x.direction == -1 and x.profit > 0, self.closed_trades)))
            short_wins = sum([(x.profit if x.profit >= 0 else 0) for x in
                              list(filter(lambda x: x.direction == -1, self.closed_trades))])

            short_losses_num = len(list(filter(lambda x: x.direction == -1 and x.profit <= 0, self.closed_trades)))
            short_losses = sum([(x.profit if x.profit < 0 else 0) for x in
                                list(filter(lambda x: x.direction == -1, self.closed_trades))])

            # TODO sharpe & ulcer ratio ... maybe alpha if it is what I think?
            if p:
                print(" Long wins: {:12.2f} ({:5.2f} %)\t\t"
                      "Long losses:  {:12.2f} ({:5.2f} %)\n"
                      "Short wins: {:12.2f} ({:5.2f} %)\t\t"
                      "Short losses: {:12.2f} ({:5.2f} %)\n\n\t   "
                      "Sum: {:12.2f}\t\t"
                      "MaxDD:\t\t{:12.2f}\n"
                      "  # trades: {:9d}\t\t\t"
                      "MaxDD in %: {:12.2f}\n"
                      "  Buy&Hold: {:12.2f}\n".
                      format(long_wins, 100 * long_wins_num / trades, long_losses, 100 * long_losses_num / trades,
                             short_wins, 100 * short_wins_num / trades, short_losses, 100 * short_losses_num / trades,
                             long_wins + long_losses + short_wins + short_losses, self.max_dd, trades, self.max_dd_perc,
                             self.hist_obj.hist[-1]["price"] - self.hist_obj.hist[0]["price"]))
            return long_wins + long_losses + short_wins + short_losses
        else:
            return 0

    def print_trades(self):
        print("\n-------------------------Trades-------------------------\n")
        print("Init. balance: ", self.base_balance)
        s = self.base_balance
        for t in self.closed_trades:
            print(t)
            s += t.profit
            print("Balance: ", s, "\n")

    def plot_trades(self, min_trend_h=5, realistic=False, plot_trends=True, indicators=None,
                    bar_amount=10000, start_bar=0, crosshair=True):

        plt.rc('figure', facecolor="#333333", edgecolor="#333333")
        plt.rc('axes', facecolor="#353535", edgecolor="#000000")
        plt.rc('lines', color="#393939")
        plt.rc('grid', color="#121212")

        # visually in graph with price, like trends
        max_len = min(start_bar + bar_amount, self.hist_obj.hist.__len__())
        used_history = self.hist_obj.hist[start_bar:max_len]
        used_history_prices = [quote["price"] for quote in used_history]

        time_index_dict = {used_history[i]["time"]: i for i in range(len(used_history))}

        # all trades that happend in used history
        trades = [trade for trade in self.closed_trades if
                  trade.open_time in time_index_dict.keys() and
                  trade.close_time in time_index_dict.keys()]

        found_trends = Analyzer.get_trends(used_history, min_trend_h=min_trend_h, realistic=realistic)

        trends = [trend for trend in found_trends if
                  trend.start_time in time_index_dict.keys() and
                  trend.end_time in time_index_dict.keys()]

        fig, ax1 = plt.subplots()

        # plot price curve
        line, = ax1.plot(list(range(start_bar, bar_amount + start_bar)),
                         [quote["price"] for quote in used_history],
                         color="black", label="Price")
        ax1.set_xlabel("Time " + used_history[0]["time"] + " - " + used_history[-1]["time"])
        ax1.set_ylabel("Price", color="tab:red")

        if plot_trends:
            # plot trends
            trend_x_values = [[time_index_dict[trend.start_time], time_index_dict[trend.end_time]] for trend in trends]
            trend_y_values = [[trend.prices[0], trend.prices[-1]] for trend in trends]

            for i in range(len(trends)):
                ax1.plot(trend_x_values[i],
                         trend_y_values[i],
                         ":",
                         color=("green" if trends[i].type == "Up  " else "#8B0000"),
                         #color=("#003333" if trends[i].type == "Up  " else "#170000"),
                         label="Trends")

        if indicators:

            for entry in indicators:

                plot_type, indicator_class, indicator_params = entry[0], entry[1], entry[2:]
                indicator_data = indicator_class.get_full(used_history_prices, *indicator_params)
                parameter_strings = ft.reduce(lambda a, b: a + " " + b, [str(p) for p in indicator_params])
                indicator_name = indicator_class.__name__ + " " + parameter_strings

                if plot_type == 0:
                    ax1.plot(list(range(start_bar, bar_amount + start_bar)),
                             [value for value in indicator_data[start_bar:bar_amount + start_bar]],
                             label=indicator_name)
                if plot_type == 1:
                    ax1.twinx().plot(list(range(start_bar, bar_amount + start_bar)),
                                     [value for value in indicator_data[start_bar:bar_amount + start_bar]],
                                     label=indicator_name)

        # plot trades
        trade_x_values = [[time_index_dict[trade.open_time], time_index_dict[trade.close_time]] for trade in trades]
        trade_y_values = [[trade.open_price, trade.close_price] for trade in trades]

        for i in range(len(trades)):
            ax1.plot(trade_x_values[i],
                     trade_y_values[i],
                     color=("green" if trades[i].direction == 1 else "#8B0000"))

        # -----------------------

        ax1.tick_params(axis="y")
        plt.xticks(rotation=90)

        """
        profits = [t.profit for t in trades]
        summed_profits = [sum(profits[:i]) for i in range(len(profits))]

        ax2 = ax1.twinx()
        ax2.set_ylabel('Profit', color='tab:blue')

        ax2.plot([x[0] for x in trade_x_values],
                 summed_profits)

        ax2.tick_params(axis="y")
        """

        plt.xticks(rotation=90)
        plt.tight_layout()
        plt.grid(True)
        if indicators:
            plt.legend()

        if crosshair:
            snap_cursor = Plotter.SnappingCursor(ax1, line)
            fig.canvas.mpl_connect('motion_notify_event', snap_cursor.on_mouse_move)

        plt.show()

    def scatter_trades(self):
        plt.rc('figure', facecolor="#333333", edgecolor="#333333")
        plt.rc('axes', facecolor="#353535", edgecolor="#000000")
        plt.rc('lines', color="#393939")
        plt.rc('grid', color="#121212")
        x = [len(t.prices) for t in self.closed_trades]
        y = [t.profit for t in self.closed_trades]
        colors = [("#ff0000" if t.direction <= 0 else "#00ff00") for t in self.closed_trades]
        p = plt.scatter(x, y, s=1, c=colors)
        plt.grid(True)
        plt.xlabel("Trade length")
        plt.ylabel("Trade profit")
        plt.tight_layout()
        plt.show()

    def plot_trade_distribution(self, bin_size=1):
        plt.rc('figure', facecolor="#333333", edgecolor="#333333")
        plt.rc('axes', facecolor="#353535", edgecolor="#000000")
        plt.rc('lines', color="#393939")
        plt.rc('grid', color="#121212")
        min_prof = min([trade.profit for trade in self.closed_trades])
        max_prof = max([trade.profit for trade in self.closed_trades])

        profit_bins = [[] for _ in
                       range(int(min_prof), int(max_prof) + 1, bin_size)]  # int(minProf) is index of 0 in profit bins
        profit_sums = [[] for _ in
                       range(int(min_prof), int(max_prof) + 1, bin_size)]  # int(minProf) is index of 0 in profit bins

        long_trades = list(filter(lambda x: x.direction == 1, self.closed_trades))
        short_trades = list(filter(lambda x: x.direction == -1, self.closed_trades))

        # counting numbers
        for index in range(len(profit_bins)):
            # profit range min_prof -> min_prof + bin_size
            lower_bin_border = (index + int(min_prof)) * bin_size
            upper_bin_border = lower_bin_border + bin_size

            num_of_long_trades_w_that_profit = len(list(filter(
                lambda trade: lower_bin_border <= trade.profit < upper_bin_border, long_trades)))
            num_of_short_trades_w_that_profit = len(list(filter(
                lambda trade: lower_bin_border <= trade.profit < upper_bin_border, short_trades)))

            profit_bins[index] += [num_of_long_trades_w_that_profit, num_of_short_trades_w_that_profit]

        # summing profits of trades with certain len
        for index in range(len(profit_bins)):
            # profit range min_prof -> min_prof + bin_size
            lower_bin_border = (index + int(min_prof)) * bin_size
            upper_bin_border = lower_bin_border + bin_size

            sum_of_long_trades_w_that_profit = sum(x.profit for x in list(filter(
                lambda trade: lower_bin_border <= trade.profit < upper_bin_border, long_trades)))
            sum_of_short_trades_w_that_profit = sum(x.profit for x in list(filter(
                lambda trade: lower_bin_border <= trade.profit < upper_bin_border, short_trades)))

            profit_sums[index] += [sum_of_long_trades_w_that_profit, sum_of_short_trades_w_that_profit]

        fig, (ax1, ax2) = plt.subplots(1, 2)

        width = 0.45

        # plot 1
        ax1.bar(list(range(int(min_prof), int(max_prof) + 1, bin_size)),
                [x[0] for x in profit_bins],
                color="#32cd32", width=width)
        ax1.bar(list(map(lambda x: x + width, list(range(int(min_prof), int(max_prof) + 1, bin_size)))),
                [x[1] for x in profit_bins],
                color="#dc143c", width=width)
        ax1.set_xticks((list(map(lambda x: int(x + width / 2),
                                 list(range(int(min_prof / 2), int(max_prof / 2) + 1, bin_size * 2))))))
        # ax1.grid(True)

        # plot 2
        ax2.bar(list(range(int(min_prof), int(max_prof) + 1, bin_size)),
                [x[0] for x in profit_sums],
                color="#32cd32", width=width)
        ax2.bar(list(map(lambda x: x + width, list(range(int(min_prof), int(max_prof) + 1, bin_size)))),
                [x[1] for x in profit_sums],
                color="#dc143c", width=width)
        ax2.set_xticks((list(map(lambda x: int(x + width / 2),
                                 list(range(int(min_prof / 2), int(max_prof / 2) + 1, bin_size * 2))))))

        fig.subplots_adjust(0.05, 0.05, 0.99, 0.99)
        # ax2.grid(True)
        plt.show()

    # <editor-fold desc="Enter & exit trades">
    def enterLong(self, price, time, position_size=1, trade_params=None):

        if self.use_balance:
            self.available_margin -= (position_size * price / self.leverage)

        if not trade_params:
            trade_params = [self.tp_in_pips, self.sl_in_pips, self.ts_in_pips, self.tp_in_value, self.sl_in_value]

        self.open_long_trades.append(Trade(price, time, 1, *trade_params, position_size=position_size,
                                           spread=self.trade_cost, leverage=self.leverage))

    def enterShort(self, price, time, position_size=1, trade_params=None):

        if self.use_balance:
            self.available_margin -= (position_size * price / self.leverage)

        if not trade_params:
            trade_params = [self.tp_in_pips, self.sl_in_pips, self.ts_in_pips, self.tp_in_value, self.sl_in_value]

        self.open_short_trades.append(Trade(price, time, -1, *trade_params, position_size=position_size,
                                            spread=self.trade_cost, leverage=self.leverage))

    def exitLong(self, price, time, position_size=1, trade_params=None, trade_id=None):
        if not trade_id:
            for trade in self.open_long_trades:
                trade.close(price, time)
                # handle account margin here
                self.available_margin += trade.profit
                self.available_margin += trade.margin

                self.closed_trades.append(trade)
                self.open_long_trades.remove(trade)
        else:
            for trade in self.open_long_trades:
                if trade.trade_id == trade_id:
                    trade.close(price, time)

                    self.available_margin += trade.profit
                    self.available_margin += trade.margin

                    self.closed_trades.append(trade)
                    self.open_long_trades.remove(trade)
                    return

    def exitShort(self, price, time, position_size=1, trade_params=None, trade_id=None):
        if not trade_id:
            for trade in self.open_short_trades:
                trade.close(price, time)
                self.available_margin += trade.profit
                self.available_margin += trade.margin
                self.closed_trades.append(trade)
                self.open_short_trades.remove(trade)
        else:
            for trade in self.open_short_trades:
                if trade.trade_id == trade_id:
                    trade.close(price, time)
                    self.available_margin += trade.profit
                    self.available_margin += trade.margin
                    self.closed_trades.append(trade)
                    self.open_short_trades.remove(trade)
                    return
    # </editor-fold>

