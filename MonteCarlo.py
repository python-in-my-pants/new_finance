import numpy as np
import pandas as pd
from pandas_datareader import data as wb
from scipy.stats import norm, gmean, cauchy


class MonteCarloSimulator:

    def __init__(self, ticker):
        self.sim_df = self.monte_carlo([ticker], 50, 10000)

    def import_stock_data(self, tickers, start='2010-1-1'):  #, end = datetime.today().strftime('%Y-%m-%d')):
        data = pd.DataFrame()
        if len([tickers]) == 1:
            data[tickers] = wb.DataReader(tickers, data_source='yahoo', start = start)['Adj Close']
            data = pd.DataFrame(data)
        else:
            for t in tickers:
                data[t] = wb.DataReader(t, data_source='yahoo', start = start)['Adj Close']
        return data

    def log_returns(self, data):
        return np.log(1+data.pct_change())

    def simple_returns(self, data):
        return ((data/data.shift(1))-1)

    def drift_calc(self, data, return_type='log'):
        if return_type=='log':
            lr = self.log_returns(data)
        elif return_type=='simple':
            lr = self.simple_returns(data)
        else:
            print("Unknown return type", return_type)
            return None
        u = lr.mean()
        var = lr.var()
        drift = u-(0.5*var)
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
        elif return_type=='simple':
            try:
                stv = self.simple_returns(data).std().values
            except:
                stv = self.simple_returns(data).std()
        #Oftentimes, we find that the distribution of returns is a variation of the normal distribution where it has a fat tail
        # This distribution is called cauchy distribution
        dr = np.exp(ft + stv * norm.ppf(np.random.rand(days, iterations)))

        # data frame with "days" rows and "iterations" columns filled with percentual daily returns (like 1.05 for +5%)
        return dr

    def simulate_mc(self, hist_df: pd.DataFrame, days: int, iterations: int):
        """
        Returns list of simulated prices of length days using iterations

        #Example use
        simulate_mc(data, 252, 1000)

        :param hist_df: historical price data from yahoo
        :param days:
        :param iterations: used iterations
        :return:
        """
        # Generate daily returns
        returns = self.daily_returns(hist_df, days, iterations)

        # Create empty matrix
        price_list = np.zeros_like(returns)

        # Put the last actual price in the first row of matrix.
        price_list[0] = hist_df.iloc[-1]

        # Calculate the price of each day (for all trials in parallel)
        for t in range(1, days):
            price_list[t] = price_list[t - 1] * returns[t]

        """
        # Printing information about stock
        try:
            [print(nam) for nam in hist_df.columns]
        except:
            print(hist_df.name)
    
        print(f"Days: {days - 1}")
        print(f"Expected Value: ${round(pd.DataFrame(price_list).iloc[-1].mean(), 2)}")
        print(f"Return: {round(100 * (pd.DataFrame(price_list).iloc[-1].mean() - price_list[0, 1]) / pd.DataFrame(price_list).iloc[-1].mean(),2)}%")
        print(f"Probability of Breakeven: {probs_find(pd.DataFrame(price_list), 0, on='return')}")
        """

        my_df = pd.DataFrame(price_list)

        return my_df

    def monte_carlo(self, tickers, days_forecast, iterations, start_date='2000-1-1'):
        data = self.import_stock_data(tickers, start=start_date)
        # simulated_df = []
        return self.simulate_mc(data.iloc[:, 0], (days_forecast+1), iterations)
        """
        for t in range(len(tickers)):
            y = self.simulate_mc(data.iloc[:, t], (days_forecast+1), iterations)

            #y['ticker'] = tickers[t]
            cols = y.columns.tolist()
            cols = cols[-1:] + cols[:-1]
            y = y[cols]
            simulated_df.append(y)

        simulated_df = pd.concat(simulated_df)
        return simulated_df
        """

    def probs_find(self, threshold, on='value'):
        """
        Example use (probability our investment will return at least 20% over the days specified in our prediction
        probs_find(predicted, 0.2, on = 'return')

        :param predicted: Dataframe of predicted prices using simulate_mc
        :param threshold: threshold
        :param on:
        :return:
        """
        predicted = self.sim_df

        if on == 'return':
            # on how many days do we have a percentual return greater than threshold?

            predicted0 = predicted.iloc[0, 0]
            predicted = predicted.iloc[-1]
            predList = list(predicted)

            over = [(i*100) / predicted0 for i in predList if ((i-predicted0)*100) / predicted0 >= threshold]
            less = [(i*100) / predicted0 for i in predList if ((i-predicted0)*100) / predicted0 < threshold]

        elif on == 'value':

            predicted = predicted.iloc[-1]
            predList = list(predicted)

            over = [i for i in predList if i >= threshold]
            less = [i for i in predList if i < threshold]

        else:
            print("'on' must be either value or return")

        return len(over)/(len(over)+len(less))


mcs = MonteCarloSimulator("AMC")
print(mcs.probs_find(3))
