import numpy as np
import pandas as pd
from pandas_datareader import data as wb
from scipy.stats import norm, gmean, cauchy
import pickle


class MCCapsule:

    def __init__(self, tickers):
        ...

class MonteCarloSimulator(object):

    """
    Give list of tickers
    Downloads data and calculates MC Sim for each ticker using 10k iterations and 1k days
    When requesting probs, use precomputed data


    ticker, days, iterations -> pop, p_n, p_sl
    """

    def __init__(self, tickers):
        self.tickers = tickers
        self.stock_data_dict = self.load_stock_data_dict()  # ticker ->
        self.sim_df = self.monte_carlo(tickers, 50, 10000)

    # <editor-fold desc="File">
    def safe_stock_data_dict(self):
        filename = "stock_data_dict"

        try:
            print(f'Creating file:\n\t{filename}.pickle ...')

            with open(filename + ".pickle", "wb") as file:
                pickle.dump(self, file)

            print("File created successfully!")

        except Exception as e:
            print("While creating the file", filename, "an exception occurred:", e)

    @staticmethod
    def load_stock_data_dict():
        try:
            f = pickle.load(open("stock_data_dict.pickle", "rb"))
            return f
        except Exception as e:
            print("While reading the file stock_data_dict.pickle an exception occurred:", e)
            return dict()
    # </editor-fold>

    def get_stock_data(self, ticker):
        if ticker not in list(self.stock_data_dict.keys()):
            self.stock_data_dict[ticker] = self.import_stock_data([ticker])
        return self.import_stock_data([ticker])

    @staticmethod
    def import_stock_data(tickers):
        data = pd.DataFrame()
        if len([tickers]) == 1:
            data[tickers] = wb.DataReader(tickers, data_source='yahoo', start='2010-1-1')['Adj Close']
            data = pd.DataFrame(data)
        else:
            for t in tickers:
                data[t] = wb.DataReader(t, data_source='yahoo', start='2010-1-1')['Adj Close']
        return data

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

    def daily_returns(self, data, days, iterations, return_type='log'):
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

        # Oftentimes, we find that the distribution of returns is a variation of the normal distribution where it has a fat tail
        # This distribution is called cauchy distribution

        # this is where the random magic happens
        dr = np.exp(ft + stv * norm.ppf(np.random.rand(days, iterations)))

        # data frame with "days" rows and "iterations" columns filled with percentual daily returns (like 1.05 for +5%)
        return dr
    # </editor-fold>

    def simulate_mc(self, ticker: str, days: int, iterations: int):
        """
        Returns list of simulated prices of length days using iterations

        #Example use
        simulate_mc(data, 252, 1000)
        """

        hist_df = self.get_stock_data(ticker).iloc[:, 0]

        # Generate daily returns
        returns = self.daily_returns(hist_df, days, iterations)

        # Create empty matrix
        price_list = np.zeros_like(returns)

        # Put the last actual price in the first row of matrix.
        price_list[0] = hist_df.iloc[-1]

        # Calculate the price of each day (for all trials in parallel)
        for t in range(1, days):
            price_list[t] = price_list[t - 1] * returns[t]

        return pd.DataFrame(price_list)

    def monte_carlo(self, tickers, days_forecast, iterations):

        data = self.simulate_mc(tickers, (days_forecast + 1), iterations)
        simulated_df = []

        for t in range(len(tickers)):

            y = self.simulate_mc(data.iloc[:, t], (days_forecast + 1), iterations)

            y['ticker'] = tickers[t]
            cols = y.columns.tolist()
            cols = cols[-1:] + cols[:-1]
            y = y[cols]
            simulated_df.append(y)
    
        simulated_df = pd.concat(simulated_df)
        return simulated_df

    def probs_find(self, threshold, on='value'):
        """
        Example use (probability our investment will return at least 20% over the days specified in our prediction
        probs_find(predicted, 0.2, on = 'return')

        :param threshold: threshold
        :param on:
        :return:
        """
        predicted = self.sim_df

        if on == 'return':
            # on how many days do we have a percentual return greater than threshold?

            predicted0 = predicted.iloc[0, 0]
            predicted = predicted.iloc[-1]
            pred_list = list(predicted)

            over = [(i * 100) / predicted0 for i in pred_list if ((i - predicted0) * 100) / predicted0 >= threshold]
            less = [(i * 100) / predicted0 for i in pred_list if ((i - predicted0) * 100) / predicted0 < threshold]

        elif on == 'value':

            predicted = predicted.iloc[-1]  # just checks end value
            pred_list = list(predicted)

            over = [i for i in pred_list if i > threshold]
            less = [i for i in pred_list if i <= threshold]

        else:
            print("'on' must be either value or return")

        return len(over) / (len(over) + len(less))

    def get_pop(self):
        return self.probs_find(0, on="value")

    def lem(self, prices, opt_price, delta, gamma, tp, sl):
        # tp hit? sl hit?
        init_price = prices[0]
        for price in prices[1:]:
            ...

    def get_pop_pn_sl(self, opt_price, delta, gamma, tp, days, sl=None, iterations=10000):
        """
        return pop (gain >= 0)
        pn using tp (prob of making threshold percent of profit)
        :param iterations:
        :param tp: underlying price value at which you take profit
        :param sl: if supplied, don't count samples where sl is hit before profit target
        :param days:
        :return:
        """
        n = 0
        mc_sim = self.simulate_mc(self.tickers, days=days, iterations=iterations)
        for col_index in range(1, iterations):
            prices = list(mc_sim.loc[:, col_index])[:days]
            ...


mcs = MonteCarloSimulator("AMC")
print(mcs.probs_find(3))
