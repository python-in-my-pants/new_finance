import warnings

import numpy as np
import pandas as pd
from pandas_datareader import data as wb
from scipy.stats import norm, gmean, cauchy
import pickle
from DDict import DDict
from eu_option import EuroOption
from CustomDict import CustomDict
from Utility import timeit, median
from Option import Option
from Option_utility import datetime_to_dte, round_cut
from option_greeks import get_greeks
import matplotlib.pyplot as plt
import random
from pprint import pprint as pp


class MonteCarloSimulator(object):

    """
    Give list of tickers
    Downloads data and calculates MC Sim for each ticker using 10k iterations and 1k days
    When requesting probs, use precomputed data

    before exiting object, safe stock_data_dict to file


    ticker, days, iterations -> pop, p_n, p_sl
    """

    def __init__(self, tickers: list, days: int = 1000, iterations: int = 10000):

        tickers.append("msft")
        tickers.sort()
        self.tickers = [t.lower() for t in tickers]
        self.daily_close_prices_all_tickers = self.load_stock_data_dict(tickers)  # ticker ->
        self.sim_df = self.monte_carlo(tickers, days_forecast=days, iterations=iterations)
        self.days = days

    # <editor-fold desc="file handling">
    def safe_stock_data_dict(self, data_df: pd.DataFrame):
        filename = "stock_data_dict"  # todo datestamp

        try:
            print(f'Creating file:\n\t{filename}.pickle ...')

            with open(filename + ".pickle", "wb") as file:
                pickle.dump(data_df, file)

            print("File created successfully!")

        except FileNotFoundError as e:
            print("While creating the file", filename, "an exception occurred:", e)

    @staticmethod
    def download_stock_data(tickers: list) -> pd.DataFrame:
        data = pd.DataFrame()
        print("Downloading history for", tickers)
        if len(tickers) == 1:
            data[tickers] = wb.DataReader(tickers, data_source='yahoo', start='2010-1-1')['Adj Close']
            data = pd.DataFrame(data)
        else:
            data = wb.DataReader(tickers, data_source='yahoo', start='2010-1-1')['Adj Close']
        # data = data["ticker"].apply(lambda x: x.lower())
        return data  # .fillna(value=0)

    def load_stock_data_dict(self, tickers: list = None) -> pd.DataFrame:
        try:
            print("Reading stock data dict file ...")
            f = pickle.load(open("stock_data_dict.pickle", "rb"))
            avail_tickers = f.columns.values.tolist()
            # print(set(tickers), set(avail_tickers), set(tickers) - set(avail_tickers))
            if not set([t.upper() for t in tickers]) - set(avail_tickers):
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

    def drift_calc(self, data, return_type='simple'):
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

    def daily_returns(self, _data, days, iterations, return_type='simple'):

        data = _data.dropna()
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

        # data frame with "days" rows and "iterations" columns filled with percentual daily returns (like 1.05 for +5%)
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

    def monte_carlo(self, tickers: list, days_forecast: int, iterations: int):

        simulated_dfs = []

        for t in range(len(tickers)):
            try:
                sim_df = self.simulate_mc(self.daily_close_prices_all_tickers.iloc[:, t],
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

    def get_pop(self, ticker, days, break_even, best_u_price):

        end_stock_prices = self.sim_df.loc[self.sim_df["ticker"] == ticker.lower()].iloc[days+1, 1:].astype('float')

        if best_u_price > break_even:  # directional long
            return len(end_stock_prices.loc[end_stock_prices > break_even]) / len(end_stock_prices)
        if best_u_price < break_even:  # directional short
            return len(end_stock_prices.loc[end_stock_prices < break_even]) / len(end_stock_prices)
        else:
            print("Something went wrong!")
            ...
            # todo complex curves like butterflies & condors & such
            return -1

    @timeit
    def get_pn_psl(self, option_strat, risk_free_rate, force_iterations=None, mode="bjerksund") -> [float, float, int, int, float, float]:
        """
        TODO use bid/ask dependent on long/short leg, so the spread is incorporated into the calc
         -> already done by assuming fill at nat instead of mid, we pay the full spread on entry

        TODO shorts make this faulty

        :param option_strat:
        :param risk_free_rate:
        :return: prob of reaching profit target or hitting stop loss
        """

        # ################################################################################################# #
        # constants

        binomial_iterations = 10  # good trade off between accuracy and time
        stock_price_resolution = 100  # height of matrix

        print(f'Mode: {mode} Iterations: {binomial_iterations}')

        # ################################################################################################# #

        first_dte = option_strat.positions.dte_until_first_exp()+1
        close_days = datetime_to_dte(option_strat.close_date)+1
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
            simulated_stock_prices = self.sim_df.loc[self.sim_df["ticker"] == ticker.lower()].iloc[:first_dte, 1:force_iterations+1]
        else:
            simulated_stock_prices = self.sim_df.loc[self.sim_df["ticker"] == ticker.lower()].iloc[:first_dte, 1:]
        iterations = len(simulated_stock_prices.iloc[0, :])

        if False and imp_vol > 0:  # todo remove 'false' if iv is overstated in monte carlo
            # option 2: use iv percentile adjusted to dte and compute outliers separately
            deviation = (first_dte+1)/365.0 * imp_vol * current_stock_price
            min_stock = max(option_strat.env_container.u_ask - deviation, 0.01)
            max_stock = option_strat.env_container.u_ask + deviation
        elif True:
            # option 1: use min & max from monte sim_stock_prices
            min_stock, max_stock = simulated_stock_prices.min(), simulated_stock_prices.max()
            if type(min_stock) is pd.Series:
                min_stock, max_stock = min_stock.min(), max_stock.max()
        else:
            min_stock = max(0.01, current_stock_price/10)
            max_stock = current_stock_price*2

        stock_price_increment = (max_stock - min_stock) / (stock_price_resolution+1)

        print(f'Min stock: {min_stock}, Max stock: {max_stock}, Increment: {stock_price_increment}')

        # (stock_price_res+1) * (first_dte+1) entries
        strat_gains = [CustomDict(
            {round(min_stock + i * stock_price_increment, 2): 0 for i in range(stock_price_resolution+1)})
            for _ in range(first_dte)]

        # precompute gains for certain u_prices and until first expiration
        for day in range(len(strat_gains)):  # iterate over days
            for stock_price in list(strat_gains[day].keys()):

                strat_gains[day][stock_price] = \
                    option_strat.positions.get_profit_n_dte(first_dte-day, stock_price, risk_free_rate)

                """
                # all leg gains + stock gains at that day with that stock price
                strat_gains[day][stock_price] += sum([(
                    EuroOption(stock_price,
                          leg.asset.strike,
                          risk_free_rate,
                          (len(strat_gains) - day) / 365.0,  # dte then

                          binomial_iterations,
                          {'is_call': leg.asset.opt_type == "c",
                           'eu_option': False,
                           'sigma': leg.asset.iv}).price() * 100
                     - abs(leg.cost)) * (1 if leg.cost > 0 else -1)
                     for leg in legs]) + (stock_price - current_stock_price) * stock_quantity
                     """

        pp(strat_gains[first_dte-1].items())

        """
        # todo debug plotting
        # todo draw zero in different color for break even line over time&price
        '''plt.contourf(
            [round(min_stock + i * stock_price_increment, 2) for i in range(stock_price_resolution+1)],
            [d for d in range(first_dte)],
            [[strat_gains[d][i] for i in range(stock_price_resolution+1)] for d in range(first_dte)],

            list(np.linspace(int(-option_strat.positions.risk), int(option_strat.positions.max_profit),
                             int(20*(option_strat.positions.max_profit+option_strat.positions.risk)))),
            extend="both",
            cmap=plt.cm.get_cmap("plasma"))'''
        plt.imshow([[strat_gains[d][i] for i in range(stock_price_resolution+1)] for d in range(first_dte)],
                   cmap=plt.cm.get_cmap("plasma"), aspect="auto", origin="lower")
        plt.xticks(list(strat_gains[0].keys()))
        plt.show()
        #"""

        # now we track the paths of mc simulations along strat_gains

        done_iterations = set()
        tp_hit_days = 0
        tp_hit_days_list = []
        tp_hit_days_b4_close = 0
        sl_hit_days = 0
        gains_at_close = []
        sl_hit_days_b4_close = 0

        gains_at_day = [[0 for _ in range(iterations)] for _ in range(first_dte)]

        for d in range(first_dte):  # iterate over days, day 0 is now, day 1 is todays close etc

            # only go while there are undecided iterations left (tp hit / sl hit?)
            if tp_hit_days+sl_hit_days >= iterations:
                break

            tp_hits_b4 = tp_hit_days
            tp_hits_this_day = 0

            for i, sim_stock_price in enumerate(sorted(simulated_stock_prices.iloc[d, :].to_list())): # iterate over iterations

                # result of this iteration is already clear
                if i in done_iterations:
                    continue

                if tp_hits_this_day > 150:
                    print(f'TP hits on day {d}: {tp_hits_this_day}')
                    ...

                sim_stock_price = round_cut(sim_stock_price, 2)  # random.randint(0, 50)

                try:
                    gain = strat_gains[d][sim_stock_price]
                except KeyError:
                    gain = option_strat.positions.get_profit_n_dte(first_dte-d, sim_stock_price, risk_free_rate)

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
                        tp_hit_days_b4_close += 1
                    continue
                if gain <= sl:
                    done_iterations.add(i)
                    sl_hit_days += 1
                    if d < close_days:
                        sl_hit_days_b4_close += 1
                    continue

            if tp_hit_days > 0 and (tp_hit_days-tp_hits_b4) / tp_hit_days > 0.30:
                # hit more than 10% of TPs today alone
                print(f'Weird day: d={d},'
                      f'\n\tAvg gain: {sum([gains_at_day[d][i] for i in range(iterations)])/iterations},'
                      f'\n\tAvg price: {sum(simulated_stock_prices.loc[d, :].to_list())/iterations}')

        """
        long_dir = option_strat.positions.max_profit_point > option_strat.positions.break_even
        cases = sum([1 for x in simulated_stock_prices.iloc[close_days, :] if x > option_strat.positions.break_even]) \
            if long_dir else \
            sum([1 for x in simulated_stock_prices.iloc[close_days, :] if x < option_strat.positions.break_even])

        print(f'Max gain: {max(gains_at_close)}, Min gain: {min(gains_at_close)}')
        print(f'Stock price {"above" if long_dir else "below"} {option_strat.positions.break_even} @ exp in {cases} '
              f'cases bc max_profit u is {option_strat.positions.max_profit_point}')
        print(f'Gains at close >0 in {sum([1 for x in gains_at_close if x > 0])} cases')
        """

        df = pd.DataFrame(gains_at_day)
        print(df.head())

        #"""
        #plt.imshow(gains_at_day, aspect="auto", origin="lower", cmap=plt.cm.get_cmap("plasma"))
        plt.contourf(
            [i for i in range(iterations)],
            [d for d in range(first_dte)],
            [[gains_at_day[d][i] for i in range(iterations)] for d in range(first_dte)],

            list(np.linspace(int(-option_strat.positions.risk), int(option_strat.positions.max_profit),
                             int(20 * (option_strat.positions.max_profit + option_strat.positions.risk)))),
            cmap=plt.cm.get_cmap("plasma"))
        plt.show()
        #"""

        # todo weight green curve with respective probs to get an expected value curve

        #"""
        # summed prob of hitting tp up to this day (inclusive)
        plt.plot(range(first_dte),
                 [100*sum([1 for x in tp_hit_days_list if x <= d])/iterations for d in range(first_dte)])
        # % of tp hits on each day
        plt.plot(range(first_dte),
                 [100*sum([1 for x in tp_hit_days_list if x == d])/(tp_hit_days+0.01) for d in range(first_dte)])
        # avg gains on this day
        plt.plot(range(first_dte),
                 [sum([gains_at_day[d][i] for i in range(iterations)])/iterations for d in range(first_dte)])
        plt.plot([close_days, close_days], [0, 50])
        plt.show()
        #"""

        if tp_hit_days_list:
            tp_d_avg = sum(tp_hit_days_list) / len(tp_hit_days_list)
            tp_d_med = median(tp_hit_days_list)
        else:
            tp_d_med = -1
            tp_d_avg = -1

        print(f' Gains at close > 0: {sum([1 for x in gains_at_close if x > 0])},\n'
              f'   TP hits b4 close: {tp_hit_days_b4_close}\n'
              f'   SL hits b4 close: {sl_hit_days_b4_close}\n'
              f' Gains at close len: {len(gains_at_close)}')
        close_pop = (sum([1 for x in gains_at_close if x > 0]) + tp_hit_days_b4_close - sl_hit_days_b4_close) / \
                    (len(gains_at_close) + tp_hit_days_b4_close + sl_hit_days_b4_close)
        print(f'          Close PoP: {close_pop}\n')
        print(f'Gains >= TP of {tp}: {sum([1 for x in gains_at_close if x >= tp])},\n'
              f'   TP hits b4 close: {tp_hit_days_b4_close}\n'
              f' Gains at close len: {len(gains_at_close)}')
        close_pn = (sum([1 for x in gains_at_close if x >= tp]) + tp_hit_days_b4_close) / \
                   (len(gains_at_close) + tp_hit_days_b4_close)
        print(f'           Close PN: {close_pn}\n')

        return tp_hit_days / iterations, sl_hit_days / iterations, tp_d_med, tp_d_avg, close_pop, close_pn

    def get_pop_pn_sl(self, option_strat, risk_free_rate):

        # for calls: (for puts inverted <>)
        # if long strike < short strike => debit => max gain = strike_diff - debit paid
        # if long strike > short strike => credit => max gain = credit received

        prob_of_prof = self.get_pop(option_strat.env_container.ticker,
                                    option_strat.positions.dte_until_first_exp(),
                                    option_strat.positions.break_even,
                                    option_strat.positions.max_profit_point)
        ptp, psl, tp_med, tp_avg, close_pop, close_pn = self.get_pn_psl(option_strat, risk_free_rate)#, mode="bjerksund")

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
            "close_pn": close_pn
        })


if __name__ == "main":
    pd.set_option('display.float_format', lambda x: '%.8f' % x)
    mcs = MonteCarloSimulator(["abt"])
    # print(mcs.p_greater_n_end("expr", 4))
    print(mcs.get_pop_pn_sl_old("abt", opt_price=0.30, delta=-0.29, gamma=0.23, theta=-0.00578,
                                tp=0.15, sl=-0.10, days=44))

