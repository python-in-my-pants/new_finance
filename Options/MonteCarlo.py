import warnings

import numpy as np
import pandas as pd
from pandas_datareader import data as wb
from scipy.stats import norm, gmean, cauchy
import pickle
from DDict import DDict
from eu_option import EuroOption
from CustomDict import CustomDict
from Utility import timeit, median, StopWatch
from Option import Option
from Option_utility import datetime_to_dte, round_cut
from option_greeks import get_greeks
import matplotlib.pyplot as plt
import random
from pprint import pprint as pp
from Utility import flatten
from pandasgui import show
from datetime import datetime, timedelta, date
from typing import List

debug = False

"""
options for IV:
- use first exp month options IV (not only of position but of option chain)
- use positions option IV (higher / lower ?)
- historic IV (all data)
- 1 year IV

option for supposed price prob dist:
- sum all implied price dists from options of certain month and weight them by open interest
"""


class MonteCarloSimulator(object):
    """
    Give list of tickers
    Downloads data and calculates MC Sim for each ticker using 10k iterations and 1k days
    When requesting probs, use precomputed data

    before exiting object, safe stock_data_dict to file


    ticker, days, iterations -> pop, p_n, p_sl
    """

    def __init__(self, tickers: list, days: int = 1000, iterations: int = 10 ** 3):

        if debug:
            print("Building monte carlo simulator ...")
        tickers.append("msft")
        tickers.sort()
        self.tickers = [t.lower() for t in tickers]
        self.daily_close_prices_all_tickers = self.load_stock_data_dict(self.tickers)  # ticker ->

        """for ticker in self.tickers:
            df = self.daily_close_prices_all_tickers[ticker.upper()].dropna()
            print(f'Ticker: {ticker} === Var: {df.var()}, Avg: {df.mean()}, Median: {df.median()}')"""

        self.sim_df = self.monte_carlo(self.tickers, days_forecast=days, iterations=iterations)
        self.days = days
        if debug:
            print("Finished building monte carlo simulator!")

    # <editor-fold desc="file handling">
    def safe_stock_data_dict(self, data_df: pd.DataFrame):
        filename = "stock_data_dict"  # todo datestamp

        try:
            if debug:
                print(f'Creating file:\n\t{filename}.pickle ...')

            with open(filename + ".pickle", "wb") as file:
                pickle.dump(data_df, file)

            if debug:
                print("File created successfully!")

        except FileNotFoundError as e:
            print("While creating the file", filename, "an exception occurred:", e)

    @staticmethod
    def download_stock_data(tickers: list) -> pd.DataFrame:
        years = 1
        data = pd.DataFrame()
        one_year_earlier = (date.today() - timedelta(days=365 * years)).strftime('%Y-%m-%d')
        if debug:
            print("Downloading history for", tickers)
        if len(tickers) == 1:
            data[tickers] = wb.DataReader(tickers, data_source='yahoo', start=one_year_earlier)['Adj Close']
        else:
            data = wb.DataReader(tickers, data_source='yahoo', start=one_year_earlier)['Adj Close']
        return data

    def load_stock_data_dict(self, tickers: list = None) -> pd.DataFrame:
        try:
            if debug:
                print("Reading stock data dict file ...")
            f = pickle.load(open("stock_data_dict.pickle", "rb"))
            avail_tickers = f.columns.values.tolist()
            # print(set(tickers), set(avail_tickers), set(tickers) - set(avail_tickers))
            if not set([t.upper() for t in tickers]) - set(avail_tickers):
                if debug:
                    print("File read successfully!")
                return f

        except FileNotFoundError as e:
            print("While reading the file stock_data_dict.pickle an exception occurred:", e)

        data = self.download_stock_data(tickers)
        self.safe_stock_data_dict(data)
        return data

    # </editor-fold>

    # <editor-fold desc="Interm calcs">
    def log_returns(self, data) -> pd.DataFrame:
        return np.log(1 + data.pct_change())

    def simple_returns(self, data):
        return (data / data.shift(1)) - 1

    def rel_returns(self, data):
        return data / data.shift(1)

    def drift_calc(self, data, return_type='log'):
        if return_type == 'log':
            lr = self.log_returns(data)
        elif return_type == 'simple':
            lr = self.simple_returns(data)
        else:
            print("Unknown return type", return_type)
            return None
        u = lr.mean()
        var = lr.var()
        drift = u - (0.5 * var)
        try:
            return drift.values
        except:
            return drift

    def daily_returns(self, _data, days, iterations, return_type='log'):

        data = _data.dropna()
        if return_type == "rel":
            rel_returns = self.rel_returns(data)
            try:
                stv = self.rel_returns(data).std().values
                mean = self.rel_returns(data).mean().values
            except:
                stv = self.rel_returns(data).std()
                mean = self.rel_returns(data).mean()

            norm_samples = np.reshape(norm(loc=mean, scale=stv).rvs(size=days * iterations), (days, iterations))

            return norm_samples

        ft = self.drift_calc(data, return_type)
        if return_type == 'log':
            try:
                stv = self.log_returns(data).std().values
            except:
                stv = self.log_returns(data).std()
        elif return_type == 'simple':
            try:
                stv = self.simple_returns(data).std().values
            except:
                stv = self.simple_returns(data).std()

        # this is where the random magic happens
        dr = np.exp(ft + stv * norm.ppf(np.random.rand(days, iterations)))

        # data frame with "days" rows and "iterations" columns filled with
        # percentual daily returns (like 1.05 for +5%)
        return dr

    # </editor-fold>

    def get_stock_data(self, ticker: str):
        if ticker not in list(self.daily_close_prices_all_tickers.keys()):
            self.daily_close_prices_all_tickers[ticker] = self.download_stock_data([ticker])
        return self.daily_close_prices_all_tickers[ticker]

    def simulate_mc(self, stock_close_prices_single_ticker: pd.DataFrame, days: int, iterations: int):
        """
        Returns list of simulated prices of length days using iterations

        #Example use
        simulate_mc(data, 252, 1000)
        """

        # Generate daily returns
        returns = self.daily_returns(stock_close_prices_single_ticker, days, iterations)

        # Create empty matrix
        price_list = np.zeros_like(returns)

        # Put the last actual price in the first row of matrix.
        price_list[0] = stock_close_prices_single_ticker.iloc[-1]

        # Calculate the price of each day (for all trials in parallel)
        for t in range(1, days):
            price_list[t] = price_list[t - 1] * returns[t]

        return pd.DataFrame(price_list)

    # @timeit
    def monte_carlo(self, tickers: list, days_forecast: int, iterations: int):

        simulated_dfs = []

        for t in range(len(tickers)):
            try:
                sim_df = self.simulate_mc(self.daily_close_prices_all_tickers[tickers[t].upper()],
                                          days_forecast + 1,
                                          iterations)
                sim_df['ticker'] = tickers[t]
                cols = sim_df.columns.tolist()
                cols = cols[-1:] + cols[:-1]
                sim_df = sim_df[cols]
                simulated_dfs.append(sim_df)

            except IndexError as e:
                print("Index Error:", e)

        return pd.concat(simulated_dfs)

    def p_greater_n_end(self, ticker: str, threshold: float, on: str = 'value'):
        """
        Example use (probability our investment will return at least 20% over the days specified in our prediction
        probs_find(predicted, 0.2, on = 'return')

        :param ticker:
        :param threshold: threshold
        :param on:
        :return:
        """
        predicted = self.sim_df.loc[self.sim_df["ticker"] == ticker]

        """if on == 'return':
            # on how many days do we have a percentual return greater than threshold?

            predicted0 = predicted.iloc[0, 0]
            predicted = predicted.iloc[-1]
            pred_list = list(predicted)

            over = [(i * 100) / predicted0 for i in pred_list if ((i - predicted0) * 100) / predicted0 >= threshold]
            less = [(i * 100) / predicted0 for i in pred_list if ((i - predicted0) * 100) / predicted0 < threshold]"""

        if on == 'value':

            predicted = predicted.iloc[-1]  # just checks end value
            pred_list = list(predicted)[1:]

            over = [i for i in pred_list if i > threshold]
            less = [i for i in pred_list if i <= threshold]

        else:
            print("'on' must be either value or return")

        return len(over) / (len(over) + len(less))

    def plot_simulated_prices(self, ticker: str, bin_size: float = 1):

        print(f'Plotting simulated prices for ticker {ticker}')

        prices = \
            flatten(
                [self.sim_df.loc[self.sim_df["ticker"] == ticker.lower()].iloc[d, 1:].to_list() for d in range(100)])

        bins = [i for i in range(int((max(prices) / bin_size) + 1))]

        binned_prices = [sum([1 for p in prices if i * bin_size <= p < (i + 1) * bin_size]) / len(prices) for i in bins]

        plt.bar(bins, binned_prices)
        plt.show()

    def get_pop(self, ticker, days, break_even, best_u_price, delta):

        try:
            end_stock_prices = self.sim_df.loc[self.sim_df["ticker"] == ticker.lower()].iloc[days, 1:].astype('float')
        except IndexError as e:
            print(self.sim_df.loc[self.sim_df["ticker"] == ticker.lower()].head(10))
            print(f'Index Error: {e}, Days: {days}, Ticker: {ticker}, \nDf["ticker"]: {set(self.sim_df["ticker"])},\n'
                  f'Shape: {self.sim_df.loc[self.sim_df["ticker"] == ticker.lower()].shape}')
            raise e
        # print(f'Var: {end_stock_prices.var()}, Avg: {end_stock_prices.mean()}, Median: {end_stock_prices.median()}')

        # print(f'break even: {break_even}, {ticker} end prices on day {days} std dev: {end_stock_prices.std()}, f'mean: {end_stock_prices.mean()}')
        if best_u_price > break_even:  # directional long
            return len(end_stock_prices.loc[end_stock_prices > break_even]) / len(end_stock_prices)
        if best_u_price < break_even:  # directional short
            return len(end_stock_prices.loc[end_stock_prices < break_even]) / len(end_stock_prices)
        if best_u_price == break_even:  # todo does this even make sense? NO, delta can be shit due to far OTM
            if delta > 0:  # directional long
                return len(end_stock_prices.loc[end_stock_prices > break_even]) / len(end_stock_prices)
            if delta < 0:  # directional short
                return len(end_stock_prices.loc[end_stock_prices < break_even]) / len(end_stock_prices)
        else:
            print(f'Something went wrong getting pop: '
                  f'ticker={ticker}, best_u_price={best_u_price}, break_even={break_even}, days={days}')
            ...
            # todo complex curves like butterflies & condors & such
            return -1

    # for debugging purposes only
    def get_pop_dist(self, ticker, days, break_even, best_u_price, delta):
        df = self.sim_df.loc[self.sim_df["ticker"] == ticker.lower()].iloc[days, 1:].astype('float')
        if best_u_price > break_even:
            return df, df.loc[df > break_even]
        if best_u_price < break_even:
            return df, df.loc[df < break_even]
        if best_u_price == break_even:
            if delta > 0:
                return df, df.loc[df > break_even]
            else:
                return df, df.loc[df < break_even]

    @timeit
    def get_pn_psl(self, option_strat, risk_free_rate,
                   force_iterations=None, mode="bjerksund", with_plots=False, epochs=10) \
            -> [float, float, float, float, float, float]:
        """
        TODO use bid/ask dependent on long/short leg, so the spread is incorporated into the calc
         -> already done by assuming fill at nat instead of mid, we pay the full spread on entry

        TODO shorts make this faulty

        TODO return underlying prices building the Pn curve & the break even curve each day
         (u price of opt price closest to 0 / closest to tp)

        :param epochs:
        :param with_plots:
        :param mode:
        :param force_iterations:
        :param option_strat:
        :param risk_free_rate:
        :return: prob of reaching profit target or hitting stop loss
        """

        # ################################################################################################# #
        # constants

        if debug:
            stop_watch = StopWatch("get_pn_psl")

        stock_price_resolution = 100  # height of matrix

        if debug:
            print(f'Mode: {mode} Iterations: {epochs}')

        # ############################################################################################################ #

        # <editor-fold desc="Set up parameters">
        first_dte = option_strat.positions.dte_until_first_exp() + 1
        close_days = datetime_to_dte(option_strat.close_date) + 1

        if debug:
            print(f'First dte: {first_dte}, close days: {close_days}')

        ticker = option_strat.env_container.ticker
        imp_vol = option_strat.env_container.env.IV
        current_stock_price = option_strat.env_container.u_ask

        tp = option_strat.tp_percentage / 100 * option_strat.positions.max_profit
        if option_strat.sl_percentage >= 100:
            sl = -float('inf')
        else:
            sl = -option_strat.sl_percentage / 100 * option_strat.positions.risk

        # 0 day is now, 1 day is end of today, 2 days is end of tomorrow etc
        if force_iterations:
            simulated_stock_prices = self.sim_df.loc[self.sim_df["ticker"] == ticker.lower()].iloc[:first_dte + 1,
                                     1:force_iterations + 1]
        else:
            simulated_stock_prices = self.sim_df.loc[self.sim_df["ticker"] == ticker.lower()].iloc[:first_dte + 1, 1:]
        iterations = len(simulated_stock_prices.iloc[0, :])
        # </editor-fold>

        if debug:
            stop_watch.take_time("set up parameters")

        # ############################################################################################################ #

        # <editor-fold desc="Set IV and increments">
        inc_mode = 2

        if inc_mode == 0:  # todo remove 'false' if iv is overstated in monte carlo
            # option 2: use iv percentile adjusted to dte and compute outliers separately
            deviation = (first_dte + 1) / 365.0 * imp_vol * current_stock_price
            min_stock = max(option_strat.env_container.u_ask - deviation, 0.01)
            max_stock = option_strat.env_container.u_ask + deviation
        elif inc_mode == 1:
            min_stock = max(0.01, current_stock_price / 10)
            max_stock = current_stock_price * 2
        elif inc_mode == 2:
            n_std_dev = 2
            std_dev = np.asarray(simulated_stock_prices).std()
            min_stock = max(0.01, current_stock_price - n_std_dev * std_dev)
            max_stock = current_stock_price + n_std_dev * std_dev
        else:
            raise RuntimeError

        stock_price_increment = (max_stock - min_stock) / (stock_price_resolution + 1)
        stock_price_increment = min(stock_price_increment, current_stock_price / 20)
        """
        print(f'\n'
              f'Stock price: {current_stock_price: >5.2f}\n'
              f'Increment:   {stock_price_increment: >5.2f}\n'
              f'max stock:   {max_stock: >5.2f}\n'
              f'min stock:   {min_stock: >5.2f}\n'
              f'std dev:    ±{std_dev: >5.2f}\n')
        #"""

        if debug:
            print(f'Min stock: {min_stock}, Max stock: {max_stock}, Increment: {stock_price_increment}')

        # </editor-fold>

        if debug:
            stop_watch.take_time("set IV and increments")

        def precompute_strat_gains() -> List[CustomDict]:
            # (stock_price_res+1) * (first_dte+1) entries
            _strat_gains = [CustomDict(
                {round(min_stock + i * stock_price_increment, 2): 0 for i in range(stock_price_resolution + 1)})
                for _ in range(first_dte + 1)]

            # precompute gains for certain u_prices and until first expiration
            for day in range(len(_strat_gains)):  # iterate over days
                for stock_price in list(_strat_gains[day].keys()):
                    # todo maybe time -1
                    _strat_gains[day][stock_price] = \
                        option_strat.positions.get_profit_n_dte(first_dte - day, stock_price, risk_free_rate, mode=mode)

            return _strat_gains

        strat_gains = precompute_strat_gains()

        if debug:
            stop_watch.take_time("precompute strat gains")

        def get_break_even_on_day(day):
            best_stock_price = -1
            min_dist = float('inf')
            for stock_price, gain in strat_gains[day].items():
                if abs(gain) < min_dist:
                    min_dist = abs(gain)
                    best_stock_price = stock_price
            return best_stock_price

        if with_plots:
            pp(strat_gains[first_dte - 1].items())
            # """
            # todo debug plotting
            # todo draw zero in different color for break even line over time&price
            plt.contourf(
                [round(min_stock + i * stock_price_increment, 2) for i in range(stock_price_resolution + 1)],
                [d for d in range(first_dte)],
                [[strat_gains[d][i] for i in range(stock_price_resolution + 1)] for d in range(first_dte)],

                list(np.linspace(int(-option_strat.positions.risk), int(option_strat.positions.max_profit),
                                 int(20 * (option_strat.positions.max_profit + option_strat.positions.risk)))),
                extend="neither",
                cmap=plt.cm.get_cmap("plasma"))
            # plt.imshow([[strat_gains[d][i] for i in range(stock_price_resolution+1)] for d in range(first_dte)],
            #           cmap=plt.cm.get_cmap("plasma"), aspect="auto", origin="lower")
            plt.xticks(list(strat_gains[0].keys()))
            plt.show()
            # """

        # ############################################################################################################ #

        def track_price_paths(lower_iteration_limit, upper_iteration_limit):
            """

            :param lower_iteration_limit: start at this iteration
            :param upper_iteration_limit: end
            :return:
            """

            # now we track the paths of mc simulations along strat_gains
            done_iterations = set()
            done_iter_until_close = set()

            tp_hit_days = 0
            tp_hit_days_list = []
            tp_hits_before_close = 0

            sl_hit_days = 0
            sl_hits_before_close = 0

            gains_at_close = []
            gains_at_day = [[0 for _ in range(upper_iteration_limit - lower_iteration_limit)] for _ in range(first_dte + 1)]
            anomaly_day = 0

            for d in range(first_dte + 1):  # iterate over days, day 0 is now, day 1 is todays close etc

                # only go while there are undecided iterations left (tp hit / sl hit?)
                if tp_hit_days + sl_hit_days >= upper_iteration_limit - lower_iteration_limit:
                    break

                tp_hits_b4 = tp_hit_days
                tp_hits_this_day = 0

                if d == close_days + 1:  # day after close
                    done_iter_until_close = done_iterations

                for i, sim_stock_price in enumerate(
                        simulated_stock_prices.iloc[d, lower_iteration_limit:upper_iteration_limit].to_list()):

                    # result of this iteration is already clear (sl hit/tp hit)
                    if i in done_iterations:
                        continue

                    if tp_hits_this_day > 150:
                        # print(f'TP hits on day {d}: {tp_hits_this_day}')
                        anomaly_day = d

                    sim_stock_price = round_cut(sim_stock_price, 2)  # random.randint(0, 50)

                    try:
                        gain = strat_gains[d][sim_stock_price]
                    except KeyError:
                        gain = option_strat.positions.get_profit_n_dte(first_dte - d, sim_stock_price, risk_free_rate)

                    gains_at_day[d][i] = gain

                    if close_days == d:
                        gains_at_close.append(gain)

                    if gain >= tp:
                        tp_hits_this_day += 1
                        done_iterations.add(i)
                        tp_hit_days += 1
                        tp_hit_days_list.append(d)
                        if d < close_days:  # has to be < instead of <= bc for close pop, we already count gains_at_close,
                            # otherwise tp hits on close day would be counted twice
                            tp_hits_before_close += 1
                        continue
                    if gain <= sl:
                        done_iterations.add(i)
                        sl_hit_days += 1
                        if d < close_days:
                            sl_hits_before_close += 1
                        continue

                if debug and tp_hit_days > 0 and (tp_hit_days - tp_hits_b4) / tp_hit_days > 0.30:
                    # hit more than 10% of TPs today alone
                    print(f'Weird day: d={d},'
                          f'\n\tAvg gain: {sum([gains_at_day[d][i] for i in range(upper_iteration_limit - lower_iteration_limit)]) / (upper_iteration_limit - lower_iteration_limit)},'
                          f'\n\tAvg price: {sum(simulated_stock_prices.loc[d, :].to_list()) / (upper_iteration_limit - lower_iteration_limit)}')

                if debug:
                    stop_watch.take_time(f"iterate over day {d}")

            # <editor-fold desc="stuff">
            # ######################################################################################################## #

            if debug:
                long_dir = option_strat.positions.max_profit_point > option_strat.positions.break_even
                cases = sum(
                    [1 for x in simulated_stock_prices.iloc[close_days, lower_iteration_limit:upper_iteration_limit] if x > option_strat.positions.break_even]) \
                    if long_dir else \
                    sum([1 for x in simulated_stock_prices.iloc[close_days, lower_iteration_limit:upper_iteration_limit] if
                         x < option_strat.positions.break_even])

                print(f'Max gain: {max(gains_at_close)}, Min gain: {min(gains_at_close)}')
                print(
                    f'Stock price {"above" if long_dir else "below"} {option_strat.positions.break_even} @ exp in {cases} '
                    f'cases bc max_profit u is {option_strat.positions.max_profit_point}')
                print(f'Gains at close >0 in {sum([1 for x in gains_at_close if x > 0])} cases')

            if with_plots:
                bin_size = 1
                bins = [i for i in range(int((max_stock / bin_size) + 1))]
                print(f'Anomaly day: {anomaly_day}')
                anomaly_bins = \
                    [sum([1 for p in simulated_stock_prices.loc[anomaly_day, lower_iteration_limit:upper_iteration_limit] if
                          i * bin_size <= p < (i + 1) * bin_size])
                     for i in bins]
                comp_bins = \
                    [sum([1 for p in simulated_stock_prices.loc[anomaly_day + 3, lower_iteration_limit:upper_iteration_limit] if
                          i * bin_size <= p < (i + 1) * bin_size])
                     for i in bins]
                plt.bar(bins, anomaly_bins)
                plt.bar(bins, comp_bins)
                plt.show()

            if with_plots:
                # """
                # plt.imshow(gains_at_day, aspect="auto", origin="lower", cmap=plt.cm.get_cmap("plasma"))
                plt.contourf(
                    [i for i in range(upper_iteration_limit - lower_iteration_limit)],
                    [d for d in range(first_dte)],
                    [[gains_at_day[d][i] for i in range(upper_iteration_limit - lower_iteration_limit)] for d in range(first_dte)],
                    list(np.linspace(int(-option_strat.positions.risk), int(option_strat.positions.max_profit),
                                     int(20 * (option_strat.positions.max_profit + option_strat.positions.risk)))),
                    cmap=plt.cm.get_cmap("plasma"),
                    extend="neither"
                )
                plt.show()
                # """

            # todo weight green curve with respective probs to get an expected value curve

            if with_plots:
                # """
                # summed prob of hitting tp up to this day (inclusive)
                plt.plot(range(first_dte),
                         [100 * sum([1 for x in tp_hit_days_list if x <= d]) / (upper_iteration_limit - lower_iteration_limit) for d in range(first_dte)])
                # % of tp hits on each day
                plt.plot(range(first_dte),
                         [100 * sum([1 for x in tp_hit_days_list if x == d]) / (tp_hit_days + 0.01) for d in
                          range(first_dte)])
                # avg gains on this day
                plt.plot(range(first_dte),
                         [sum([gains_at_day[d][i] for i in range(upper_iteration_limit - lower_iteration_limit)]) / (upper_iteration_limit - lower_iteration_limit) for d in range(first_dte)])
                plt.plot([close_days, close_days], [0, 50])
                plt.show()
                # """

            # ######################################################################################################## #
            # </editor-fold>

            if tp_hit_days_list:
                _tp_d_avg = sum(tp_hit_days_list) / len(tp_hit_days_list)
                _tp_d_med = median(tp_hit_days_list)
            else:
                _tp_d_med = -1
                _tp_d_avg = -1

            # todo (len(gains_at_close) + tp_hit_days_b4_close + sl_hit_days_b4_close)
            #  ZeroDivisionError: division by zero
            pop_hits_at_close = [1 for i, x in enumerate(gains_at_close) if x > 0]# and i not in done_iter_until_close]
            _close_pop = (sum(pop_hits_at_close) + tp_hits_before_close - sl_hits_before_close) / \
                         (upper_iteration_limit - lower_iteration_limit) #(len(gains_at_close) + tp_hits_before_close + sl_hits_before_close)

            # todo this overstates bc it counts tp_hit_days_b4 close double bc they may also be above tp in gains@close
            tp_hits_at_close = [1 for i, x in enumerate(gains_at_close) if x >= tp]# and i not in done_iter_until_close]
            _close_pn = (sum(tp_hits_at_close) + tp_hits_before_close - sl_hits_before_close) / \
                        (upper_iteration_limit - lower_iteration_limit) #(len(gains_at_close) + tp_hits_before_close + sl_hits_before_close)

            if debug:
                print(f'\n'
                      f'     Iteration diff: { (upper_iteration_limit - lower_iteration_limit)}\n\n'
                      f' Gains at close > 0: {sum([1 for x in gains_at_close if x > 0])},\n'
                      f'   TP hits b4 close: {tp_hits_before_close}\n'
                      f'   SL hits b4 close: {sl_hits_before_close}\n'
                      f' Gains at close len: {len(gains_at_close)}')
                print(f'          Close PoP: {_close_pop:.5f}\n')
                print(f'G@clo >= TP of {tp:.2f}: {sum([1 for x in gains_at_close if x >= tp]):.5f},\n'
                      f'   TP hits b4 close: {tp_hits_before_close}\n'
                      f' Gains at close len: {len(gains_at_close)}')
                print(f'           Close PN: {_close_pn:.5f}\n')

            return tp_hit_days / (upper_iteration_limit - lower_iteration_limit), sl_hit_days / (upper_iteration_limit - lower_iteration_limit), \
                   _tp_d_med, _tp_d_avg, \
                   _close_pop, _close_pn, \
                   get_break_even_on_day(close_days)

        # <editor-fold desc="Compute probs">

        # split simulated stock prices in this many batches and compute probs for each batch, then average over results
        batch_size = int(iterations / epochs)
        if batch_size < 10:
            epochs = 1

        # todo use median instead of avg
        p_tp, p_sl, tp_d_med, tp_d_avg, close_pop, close_pn, close_be = 0, 0, 0, 0, 0, 0, 0

        for outer_loop_iter in range(epochs):

            in_p_tp, in_p_sl, in_tp_d_med, in_tp_d_avg, in_close_pop, in_close_pn, in_close_be = \
                track_price_paths(outer_loop_iter * batch_size, (outer_loop_iter + 1) * batch_size)

            p_tp += in_p_tp
            p_sl += in_p_sl
            tp_d_med += in_tp_d_med
            tp_d_avg += in_tp_d_avg
            close_pop += in_close_pop
            close_pn += in_close_pn
            close_be += in_close_be

            if debug:
                stop_watch.take_time("inner iteration")

        p_tp /= epochs
        p_sl /= epochs
        tp_d_med /= epochs
        tp_d_avg /= epochs
        close_pop /= epochs
        close_pn /= epochs
        close_be /= epochs

        if debug:
            print(f'\nPn_Psl results with iterations={epochs}, simulation size={iterations}:\n'
                  f'\n\tAt expiration:\n'
                  f'\t\tP50: {p_tp * 100: >2.2f} %\n'
                  f'\t\tPSL: {p_sl * 100: >2.2f} %\n'
                  f'\n'
                  f'\tAt close:\n'
                  f'\t\tPoP: {close_pop * 100: >2.2f} %\n'
                  f'\t\tP50: {close_pn * 100: >2.2f} %\n'
                  f'\t\tAvg: {tp_d_avg:.2f}\n'
                  f'\t\tMed: {tp_d_med:.2f}')

        # </editor-fold>

        return p_tp, p_sl, tp_d_med, tp_d_avg, close_pop, close_pn, close_be

    def get_pop_pn_sl(self, option_strat, risk_free_rate):

        # for calls: (for puts inverted <>)
        # if long strike < short strike => debit => max gain = strike_diff - debit paid
        # if long strike > short strike => credit => max gain = credit received
        prob_of_prof = self.get_pop(option_strat.env_container.ticker,
                                    option_strat.positions.dte_until_first_exp(),
                                    option_strat.positions.break_even,
                                    option_strat.positions.max_profit_point,
                                    option_strat.positions.greeks["delta"])

        ptp, psl, tp_med, tp_avg, close_pop, close_pn, close_be = self.get_pn_psl(option_strat,
                                                                                  risk_free_rate)  # , mode="bjerksund")
        if debug:
            print(f'pop: {prob_of_prof:.2f}, ptp {ptp:.2f}, psl {psl:.2f}, tp_med {tp_med}, tp_avg {tp_avg}, '
                  f'close_pop {close_pop:.2f}, close_pn {close_pn:.2f}')

        return DDict({
            # prob of being above break even at expiration day close
            "prob_of_profit": round(prob_of_prof, 5),
            # prof of hitting TP until expiration day close
            "p_tp": round(ptp, 5),
            # prof of hitting SL until expiration day close
            "p_sl": round(psl, 5),
            # median days needed to hit TP
            "tp_med": tp_med,
            # avg days needed to hit SL
            "tp_avg": tp_avg,
            # prob of being profitable until close day, inclusive (including prior hit TPs and SLs)
            "close_pop": close_pop,
            # prob of hitting TP until close day, inclusive (including prior hit SLs)
            "close_pn": close_pn,
            # break even on close day EOD
            "close_be": close_be,
        })


@timeit
def test_mc_accu(option_strat, risk_free_rate, outer_iter=1, sim_size=10 ** 4):
    ptp = 0
    prob_of_profit = 0
    psl = 0
    tp_med_d = 0
    tp_avg_d = 0
    close_pop = 0
    close_pn = 0

    for i in range(outer_iter):
        mcs = MonteCarloSimulator(["amc"], iterations=sim_size)
        prob_dict = mcs.get_pop_pn_sl(option_strat=option_strat, risk_free_rate=risk_free_rate)

        ptp += prob_dict.p_tp
        prob_of_profit += prob_dict.prob_of_profit
        psl += prob_dict.p_sl
        tp_med_d += prob_dict.tp_med
        tp_avg_d += prob_dict.tp_avg
        close_pop += prob_dict.close_pop
        close_pn += prob_dict.close_pn

    ptp /= outer_iter
    prob_of_profit /= outer_iter
    psl /= outer_iter
    tp_med_d /= outer_iter
    tp_avg_d /= outer_iter
    close_pop /= outer_iter
    close_pn /= outer_iter

    print(f'\nAccuracy test with iterations={outer_iter}, simulation size={sim_size}:\n'
          f'\n\tAt expiration:\n'
          f'\t\tPoP: {prob_of_profit * 100: >2.2f} %\n'
          f'\t\tP50: {ptp * 100: >2.2f} %\n'
          f'\t\tPSL: {psl * 100: >2.2f} %\n'
          f'\n'
          f'\tAt close:\n'
          f'\t\tPoP: {close_pop * 100: >2.2f} %\n'
          f'\t\tP50: {close_pn * 100: >2.2f} %\n'
          f'\t\tAvg: {tp_avg_d:.2f}\n'
          f'\t\tMed: {tp_med_d:.2f}')


if __name__ == "__main__":
    pd.set_option('display.float_format', lambda x: '%.8f' % x)
