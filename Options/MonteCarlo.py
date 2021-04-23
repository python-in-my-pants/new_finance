import numpy as np
import pandas as pd
from pandas_datareader import data as wb
from scipy.stats import norm, gmean, cauchy
import pickle
from DDict import DDict
import time

pd.set_option('display.float_format', lambda x: '%.8f' % x)


class MonteCarloSimulator(object):

    """
    Give list of tickers
    Downloads data and calculates MC Sim for each ticker using 10k iterations and 1k days
    When requesting probs, use precomputed data

    before exiting object, safe stock_data_dict to file


    ticker, days, iterations -> pop, p_n, p_sl
    """

    def __init__(self, tickers: list, days: int = 1000, iterations: int = 1000):
        tickers.append("msft")
        self.tickers = tickers
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
        if len(tickers) == 1:
            data[tickers] = wb.DataReader(tickers, data_source='yahoo', start='2010-1-1')['Adj Close']
            data = pd.DataFrame(data)
        else:
            data = wb.DataReader(tickers, data_source='yahoo', start='2010-1-1')['Adj Close']
        return data  # .fillna(value=0)

    def stock_df_to_dict(self, df: pd.DataFrame, tickers: list) -> dict:
        return {t: df[t] for t in tickers}

    def load_stock_data_dict(self, tickers: list = None) -> pd.DataFrame:
        try:
            print("Reading stock data dict file ...")
            f = pickle.load(open("stock_data_dict.pickle", "rb"))
            avail_tickers = f.columns.values.tolist()
            if not set(tickers) - set(avail_tickers):
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
            sim_df = self.simulate_mc(self.daily_close_prices_all_tickers.iloc[:, t],
                                      days_forecast + 1,
                                      iterations)

            sim_df['ticker'] = tickers[t]
            cols = sim_df.columns.tolist()
            cols = cols[-1:] + cols[:-1]
            sim_df = sim_df[cols]
            simulated_dfs.append(sim_df)

        simulatedDF = pd.concat(simulated_dfs)
        return simulatedDF

        """for t in range(len(tickers)):

            y = self.simulate_mc(data.iloc[:, t], (days_forecast + 1), iterations)

            y['ticker'] = tickers[t]
            cols = y.columns.tolist()
            cols = cols[-1:] + cols[:-1]
            y = y[cols]
            simulated_df.append(y)
    
        simulated_df = pd.concat(simulated_df)
        return simulated_df"""

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

    def get_pop(self, ticker):
        return self.p_greater_n_end(ticker, 0)

    def get_p_n(self, ticker: str, n: int):
        return self.p_greater_n_end(ticker, n)

    # todo sanity check, does this make sense at all?
    # todo test for naked puts, covered calls
    def get_pop_pn_sl(self, ticker: str, opt_price: float,
                      delta: float, gamma: float, theta: float,
                      days: int, tp: float, sl: float = -float('inf'), stock_quantity: float = 0):
        """
        todo build a smart model for future option price prediction with greeks and monte carlo & strike & dte

        assume:
            static theta
            static gamma
            0 vega

        return pop (gain >= 0)
        pn using tp (prob of making threshold percent of profit)
        :param stock_quantity:
        :param theta: todo divide by 100?
        :param gamma:
        :param delta:
        :param opt_price:
        :param ticker:
        :param tp: underlying price value at which you take profit / 100
        :param sl: if supplied, don't count samples where sl is hit before profit target / 100
        :param days:
        :return:
        """

        # TODO adjust for (naked) shorts, currently pop=100 theta??
        # TODO purely negative spreads have pop > 0, why?
        # tODO do we get rid of zero returns for days where no price data was available yet?

        # whats the diff between option price calc here and break even calc in options strat calc?
        # ergebnis muss iwo auf exp p/l graph liegen bei exp, abhängig vom stock price

        print("\nGetting probabilities ...")
        start = time.time()

        days += 1
        if days > self.days:
            ...  # warn user or calc sim with given days anew
            return

        if stock_quantity != 0:
            delta -= stock_quantity

        invert = False
        if opt_price < 0:
            opt_price = -opt_price
            invert = True

        # get prices of all iterations until n days in the future
        simulated_stock_prices = self.sim_df.loc[self.sim_df["ticker"] == ticker.lower()].iloc[:days, 1:]

        if stock_quantity != 0:
            # stock gains
            stock_gains = simulated_stock_prices - simulated_stock_prices.iloc[0]
            stock_gains *= stock_quantity

        tmp = simulated_stock_prices.iloc[-1]
        print("kamino", len(tmp[tmp > 122.5]))

        # df[day, iteration]
        future_option_prices = pd.DataFrame(np.zeros_like(simulated_stock_prices))
        _iterations = len(future_option_prices.columns.values)

        # set first row to current option price
        future_option_prices.iloc[0, :] = opt_price

        def get_option_price(stock_price, ):
            ...

        # calculate future option prices based on simulated underlying prices and delta and gamma
        # todo add vega influence
        # opt_price_tomorrow = opt_price_today + u_diff * delta
        # next_delta = old_delta + u_diff * gamma
        for d in range(1, days):  # this doesn t end up on P/L graph on exp, why?
            # theta change over time
            # gamma change on moneyness
            # theta change on moneyness
            # vega influence
            # IV change on u price
            u_diff = simulated_stock_prices.iloc[d] - simulated_stock_prices.iloc[d-1]
            future_option_prices.iloc[d] = future_option_prices.iloc[d-1] + u_diff * delta + theta
            delta += u_diff * gamma
            future_option_prices.iloc[d][future_option_prices.iloc[d] < 0] = 0

        # substract current option price from all future option prices
        future_option_prices -= opt_price

        if invert:
            future_option_prices = -future_option_prices

        if stock_quantity != 0:
            future_option_prices += stock_gains

        # ############################################################################################################ #

        # for each iteration, check if
        # a) gain on expiry is > 0
        # b) tp was hit
        # c) sl was hit
        # and save numbers of occurrences

        # a) check end result > 0
        end_prices = list(future_option_prices.iloc[-1])
        above_zero = sum(1 for i in end_prices if i > 0)

        # b) check if tp/sl was hit todo vectorize
        tp_hit = 0
        sl_hit = 0
        for i in range(_iterations):
            for p in future_option_prices[i]:
                if p >= tp:
                    tp_hit += 1
                    break
                if p <= sl:
                    sl_hit += 1
                    break

        print(f"Getting probs took {time.time() - start:.2f} s")

        return DDict({
            "prob_of_profit": round(above_zero / _iterations, 5),
            "p_tp": round(tp_hit / _iterations, 5),
            "p_sl": round(sl_hit / _iterations, 5),
        })


if __name__ == "main":
    mcs = MonteCarloSimulator(["abt"])
    # print(mcs.p_greater_n_end("expr", 4))
    print(mcs.get_pop_pn_sl("abt", opt_price=0.30, delta=-0.29, gamma=0.23, theta=-0.00578,
                            tp=0.15, sl=-0.10, days=44))

