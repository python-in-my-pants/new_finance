from pandas_datareader import data as dr
import pandas as pd
from datetime import date, timedelta
from typing import List
import numpy as np
from scipy.stats import norm, lognorm
from matplotlib import pyplot as plt

pd.set_option("display.max_rows", None, "display.max_columns", None)
pd.set_option('display.width', 1000)
pd.set_option('display.max_colwidth', 500)
pd.set_option('display.float_format', lambda x: '%.5f' % x)
pd.options.mode.chained_assignment = None  # default='warn'

def outer_get(tickers, threshold=12, with_plot=False):

    def download_stock_data() -> pd.DataFrame:
        years = 10
        data = pd.DataFrame()
        one_year_earlier = (date.today() - timedelta(days=365 * years)).strftime('%Y-%m-%d')
        if len(tickers) == 1:
            data[tickers] = dr.DataReader(tickers, data_source='yahoo', start=one_year_earlier)['Adj Close']
        else:
            data = dr.DataReader(tickers, data_source='yahoo', start=one_year_earlier)['Adj Close']
        return data

    stock_price_hist = download_stock_data().dropna()
    current_stock_price = stock_price_hist.iloc[-1, 0]
    days_to_sim = 90

    def get_mc_sim(mode = "simple", get_dist=False):

        def get_abs_returns(data: pd.DataFrame) -> pd.DataFrame:
            return data - data.shift(1)

        def get_log_returns(data) -> pd.DataFrame:
            return np.log(1 + data.pct_change())

        def get_simple_returns(data) -> pd.DataFrame:
            return data / data.shift(1)

        price_var = stock_price_hist.var().values[0]
        price_mean = stock_price_hist.mean().values[0]  # arithmetic mean

        abs_returns = get_abs_returns(stock_price_hist)
        abs_ret_var = abs_returns.var().values[0]
        abs_ret_mean = abs_returns.mean().values[0]

        log_returns = get_log_returns(stock_price_hist)
        log_ret_var = log_returns.var().values[0]
        log_ret_mean = log_returns.mean().values[0]

        simple_ret = get_simple_returns(stock_price_hist)
        simple_ret_var = simple_ret.var().values[0]
        simple_ret_mean = simple_ret.mean().values[0]
        """
        print(f'Simple ret var: {simple_ret_var}, '
              f'Simple ret mean: {simple_ret_mean}, '
              f'Simple ret std dev: {simple_ret_var**0.5}')
        """

        # scale is std dev = root(var)
        if mode == "simple":
            norm_distr = norm(loc=simple_ret_mean, scale=simple_ret_var**0.5)
        if mode == "log":
            norm_distr = norm(loc=log_ret_mean, scale=log_ret_var ** 0.5)
        if mode == "abs":
            norm_distr = norm(loc=abs_ret_mean, scale=abs_ret_var ** 0.5)
        #print(f'MCS using STD of {simple_ret_var**0.5:.5f} and mean of {simple_ret_mean:.5f}')

        if with_plot:
            x1 = np.linspace(norm_distr.ppf(0.01), norm_distr.ppf(0.99), 100)
            plt.plot(x1, norm_distr.pdf(x1), label="Historic distribution")

            x2 = np.linspace(norm.ppf(0.01), norm.ppf(0.99), 100)
            plt.plot(x2, norm(loc=1).pdf(x2), label="Standard normal dist")

            plt.legend()
            plt.show()

        def use_normal_dist_of_returns():

            gen_prices = [current_stock_price] + [0 for _ in range(days_to_sim)]
            norm_dist_randoms = norm_distr.rvs(size=days_to_sim)

            for i in range(days_to_sim):
                # scale=std, loc=mean
                gen_prices[i+1] = gen_prices[i] * norm_dist_randoms[i]

            return gen_prices

        iterations = 1000

        sim_prices = pd.DataFrame([use_normal_dist_of_returns() for _ in range(iterations)])
        end_prices = sim_prices.iloc[:, -1]
        p_mc = len(end_prices.loc[end_prices >= threshold])/len(end_prices)

        #print(f'MCS end price STD {end_prices.var()**0.5:.5f} and mean of {end_prices.mean():.5f}')

        if with_plot:
            sim_prices.transpose().plot(legend=False)
            plt.show()

            end_prices_mean = end_prices.mean()
            end_prices_var = end_prices.var()

            log_std_dev = end_prices_var ** 0.5
            x3 = np.linspace(0, 100, 10000)

            fig, ax = plt.subplots()
            ax.hist(end_prices.values, bins=iterations, density=True, histtype="stepfilled", label="End prices")
            ax.plot(np.exp(x3) - 1, lognorm.pdf(np.exp(x3), log_std_dev, end_prices_mean), label="Log norm")
            ax.plot(x3, norm.pdf(x3, end_prices_mean, log_std_dev), label="Normal dist")

            plt.xlim([-5, 100])
            plt.legend()
            plt.show()

        if not get_dist:
            return p_mc

        x = np.linspace(0, current_stock_price*5, 1000)

        def cdf(thresh):
            return len(end_prices.loc[end_prices >= thresh]) / len(end_prices)

        return x, [cdf(y) for y in x]

    def get_iv_calc(get_dist=False):

        iv = 1.69
        std_dev = current_stock_price * iv * ((days_to_sim + 0.1) / 365.0) ** 0.5

        def use_iv():

            #print(f'IVC using STD of {std_dev:.5f} and mean of {current_stock_price:.5f}')
            return norm.cdf(threshold, loc=current_stock_price, scale=std_dev)

        if not get_dist:
            return 1 - use_iv()

        x = np.linspace(current_stock_price-3*std_dev, current_stock_price+3*std_dev, 1000)
        return x, 1-norm.cdf(x, loc=current_stock_price, scale=std_dev)

    print(f'P_mc: {100 * sum([get_mc_sim(mode="simple") for _ in range(10)])/10:.2f} %,'
          f'\tP_iv: {get_iv_calc()*100:.2f} %\n')

    plt.plot(*get_mc_sim(get_dist=True), label="Monte carlo CDF 90", color="r")
    plt.plot(*get_iv_calc(get_dist=True), label="IV CDF 90", color="b")

    days_to_sim = 30

    plt.plot(*get_mc_sim(get_dist=True), label="Monte carlo CDF 30")
    plt.plot(*get_iv_calc(get_dist=True), label="IV CDF 30")

    plt.plot([current_stock_price, current_stock_price], [1, 0])
    plt.xlim([0, 50])
    plt.legend()
    plt.show()

def lognorm_test():
    p = np.random.randint(0, 10, 1000)
    var = p.var()
    mean = p.mean()
    std = var**0.5

    mu = mean
    sd = std
    print(f'Std: {std}, Mean: {mean}')
    x = np.linspace(mu - 3 * sd, mu + 3 * sd, 100)
    plt.plot(x, norm.pdf(x, mu, sd), label="Normal")
    plt.plot(np.exp(x) - 1, lognorm.pdf(np.exp(x), sd, mu), label="Log-Normal")

    plt.xlim([-10, 10])
    plt.legend()
    plt.show()

outer_get(["AMC"], 8)

#lognorm_test()
