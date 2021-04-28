import numpy as np
import pandas as pd
from pandas_datareader import data as wb
from scipy.stats import norm, gmean, cauchy
import pickle
from DDict import DDict
from eu_option import EuroOption
from CustomDict import CustomDict
from Utility import timeit
from Option import Option


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
            tickers = [t.lower() for t in tickers]
            # print(set(tickers), set(avail_tickers), set(tickers) - set(avail_tickers))
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

    @timeit
    def get_pop(self, ticker, days, break_even, price_dir):
        # return self.p_greater_n_end(ticker, 0)

        # todo days-1 or days?
        end_stock_prices = self.sim_df.loc[self.sim_df["ticker"] == ticker.lower()].iloc[days, 1:].astype('float')

        if price_dir > 0:  # directional long
            return len(end_stock_prices[end_stock_prices > break_even]) / len(end_stock_prices)
        if price_dir < 0:  # directional short
            return len(end_stock_prices[end_stock_prices < break_even]) / len(end_stock_prices)
        else:
            ...
            # todo butterflies & condors & such
            return -1


    '''
    # todo sanity check, does this make sense at all?
    # todo test for naked puts, covered calls
    @timeit
    def get_pop_pn_sl_old(self, ticker: str, opt_price: float,
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

        # df[day, iteration]
        future_option_prices = pd.DataFrame(np.zeros_like(simulated_stock_prices))
        _iterations = len(future_option_prices.columns.values)

        # set first row to current option price
        future_option_prices.iloc[0, :] = opt_price

        # calculate future option prices based on simulated underlying prices and delta and gamma
        # todo add vega influence
        # opt_price_tomorrow = opt_price_today + u_diff * delta
        # next_delta = old_delta + u_diff * gamma
        for d in range(1, days):  # this doesnt end up on P/L graph on exp, why?
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
    '''

    @timeit
    def get_pn_psl(self, option_strat, risk_free_rate) -> [float, float]:
        """
        TODO use bid/ask dependent on long/short leg, so the spread is incorporated into the calc

        :param option_strat:
        :param risk_free_rate:
        :return: prob of reaching profit target or hitting stop loss
        """

        # ################################################################################################# #
        # constants

        binomial_iterations = 5  # good tradeoff between accuracy and time
        stock_price_resolution = 60  # height of matrix

        # ################################################################################################# #

        first_dte = option_strat.positions.dte_until_first_exp()
        legs = [pos for pos in list(option_strat.positions.pos_dict.values()) if type(pos.asset) is Option]
        ticker = option_strat.env_container.ticker
        imp_vol = option_strat.env_container.env.IV
        current_stock_price = option_strat.env_container.u_ask
        stock_quantity = option_strat.positions.stock.quantity

        tp = option_strat.tp_percentage / 100 * option_strat.positions.max_profit
        if option_strat.sl_percentage >= 100:
            sl = -float('inf')
        else:
            sl = -option_strat.sl_percentage / 100 * option_strat.positions.risk

        simulated_stock_prices = self.sim_df.loc[self.sim_df["ticker"] == ticker.lower()].iloc[:first_dte, 1:]
        iterations = len(simulated_stock_prices.iloc[0, :])

        if False and imp_vol > 0:  # todo remove 'false' if iv is overstated in monte carlo
            # option 2: use iv percentile adjusted to dte and compute outliers separately
            deviation = first_dte/365.0 * imp_vol * current_stock_price
            min_stock = max(option_strat.env_container.u_ask - deviation, 0.01)
            max_stock = option_strat.env_container.u_ask + deviation
        else:
            # option 1: use min & max from monte sim_stock_prices
            min_stock, max_stock = simulated_stock_prices.min(), simulated_stock_prices.max()
            if type(min_stock) is pd.Series:
                min_stock, max_stock = min_stock.min(), max_stock.max()

        stock_price_increment = min((max_stock - min_stock) / stock_price_resolution, 1)

        # (stock_price_res+1) * (first_dte+1) entries
        strat_gains = [CustomDict(
            {round(min_stock + i * stock_price_increment, 2): 0 for i in range(stock_price_resolution+1)})
            for _ in range(first_dte+1)]

        for day in range(len(strat_gains)):  # iterate over days
            for stock_price in list(strat_gains[day].keys()):

                # This is a for-version of below list comprehension
                
                leg_gains = 0
                for leg in legs:
                    future_leg_price = EuroOption(stock_price,
                                                  leg.asset.strike,
                                                  risk_free_rate,
                                                  (len(strat_gains) - day) / 365.0,  # dte then

                                                  binomial_iterations,
                                                  {'is_call': leg.asset.opt_type == "c",
                                                   'eu_option': False,
                                                   'sigma': leg.asset.iv}).price()

                    leg_gain = future_leg_price*100 - abs(leg.cost)

                    if leg.cost < 0:  # short leg
                        leg_gain = -leg_gain

                    leg_gains += leg_gain

                leg_gains += (stock_price - current_stock_price) * stock_quantity
                strat_gains[day][stock_price] = leg_gains

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

        # now we track the paths of mc simulations along strat_gains

        done_iterations = set()
        tp_hit_days = 0
        tp_hit_days_b4_close = 0
        sl_hit_days = 0
        for d in range(1, first_dte):  # iterate over days, start from tomorrow bc todays sim prices are all equal

            # only go while there are uncertain iterations left (tp hit / sl hit?)
            if tp_hit_days+sl_hit_days >= iterations:
                break

            for i, sim_stock_price in enumerate(simulated_stock_prices.iloc[d, :]):  # iterate over iterations

                # result of this iteration is already clear
                if i in done_iterations:
                    continue

                try:
                    gain = strat_gains[d][sim_stock_price]
                except KeyError:
                    # calc missing value manually
                    gain = sum([(EuroOption(sim_stock_price,
                                            leg.asset.strike,
                                            risk_free_rate,
                                            (len(strat_gains) - d) / 365.0,  # dte then

                                            binomial_iterations,
                                            {'is_call': leg.asset.opt_type == "c",
                                             'eu_option': False,
                                             'sigma': leg.asset.iv}).price() * 100
                                 - abs(leg.cost)) * (1 if leg.cost > 0 else -1)
                                for leg in legs]) + (sim_stock_price - current_stock_price) * stock_quantity
                    ...

                if gain >= tp:
                    done_iterations.add(i)
                    tp_hit_days += 1
                    if d <= first_dte:
                        tp_hit_days_b4_close += 1
                    continue
                if gain <= sl:
                    done_iterations.add(i)
                    sl_hit_days += 1
                    continue

        return tp_hit_days / iterations, sl_hit_days / iterations

    '''
    @timeit
    def get_p50(self, comb_pos, risk_free_rate) -> float:
        """

                :param comb_pos:
                :param risk_free_rate:
                :return:
                """

        from Option_strategy_selector import Option

        # ################################################################################################# #
        # constants

        binomial_iterations = 5  # good tradeoff between accuracy and time
        stock_price_resolution = 60  # height of matrix

        # ################################################################################################# #

        first_dte = comb_pos.dte_until_first_exp()
        legs = [pos for pos in comb_pos.pos_dict.values() if type(pos) is Option]
        ticker = comb_pos.underlying
        # iv_percentile = option_strat.env_container.env.ivp
        current_stock_price = comb_pos.u_ask
        stock_quantity = comb_pos.stock.quantity

        tp = 0.5 * comb_pos.max_profit

        simulated_stock_prices = self.sim_df.loc[self.sim_df["ticker"] == ticker.lower()].iloc[:first_dte, 1:]
        iterations = len(simulated_stock_prices.iloc[0, :])

        # option 1: use min & max from monte sim_stock_prices
        min_stock, max_stock = min(simulated_stock_prices), max(simulated_stock_prices)

        # option 2: use iv percentile adjusted to dte and compute outliers separately
        """
        deviation = first_dte / 365.0 * iv_percentile * current_stock_price
        min_stock = current_stock_price - deviation
        max_stock = min(current_stock_price + deviation, 0.01)
        #"""

        stock_price_increment = (max_stock - min_stock) / stock_price_resolution

        # (stock_price_res+1) * (first_dte+1) entries
        strat_gains = [CustomDict(
            {min_stock + i * stock_price_increment: 0 for i in range(stock_price_resolution + 1)})
            for _ in range(first_dte + 1)]

        for day in range(len(strat_gains)):  # iterate over days
            for stock_price in list(strat_gains[day].keys()):
                """
                This is a for-version of below list comprehension

                leg_gains = 0
                for leg in legs:
                    future_leg_price = EuroOption(stock_price,
                          leg.asset.strike,
                          risk_free_rate,
                          len(strat_gains) - day,  # dte then

                          binomial_iterations,
                          {'is_call': leg.asset.opt_type == "c",
                           'eu_option': False,
                           'sigma': leg.asset.iv}).price()
                    leg_gain = future_leg_price - leg.cost
                    if leg.cost < 0:  # short leg
                        leg_gain = -leg_gain

                    leg_gains += leg_gain
                leg_gains += (stock_price - current_stock_price) * stock_quantity
                """

                # all leg gains + stock gains at that day with that stock price
                strat_gains[day][stock_price] += sum([(
                                                              EuroOption(stock_price,
                                                                         leg.asset.strike,
                                                                         risk_free_rate,
                                                                         len(strat_gains) - day,  # dte then

                                                                         binomial_iterations,
                                                                         {'is_call': leg.asset.opt_type == "c",
                                                                          'eu_option': False,
                                                                          'sigma': leg.asset.iv}).price()
                                                              - leg.cost) * (1 if leg.cost > 0 else -1)
                                                      for leg in legs]) \
                                                 + (stock_price - current_stock_price) * stock_quantity

        # now we track the paths of mc simulations along strat_gains

        done_iterations = set()
        tp_hit_days = []
        for d in range(1, first_dte):  # iterate over days, start from tomorrow bc todays sim prices are all equal

            # only go while there are uncertain iterations left (tp hit / sl hit?)
            if len(tp_hit_days) >= iterations:
                break

            for i, sim_stock_price in enumerate(simulated_stock_prices.iloc[d, :]):  # iterate over iterations

                # result of this iteration is already clear
                if i in done_iterations:
                    continue

                gain = strat_gains[d][sim_stock_price]
                if gain >= tp:
                    done_iterations.add(i)
                    tp_hit_days.append(d)
                    continue

        return len(tp_hit_days) / iterations
    '''

    @timeit
    def get_pop_pn_sl(self, option_strat, risk_free_rate):

        prob_of_prof = self.get_pop(option_strat.env_container.ticker,
                                    option_strat.positions.dte_until_first_exp(),
                                    option_strat.positions.break_even,
                                    option_strat.positions.greeks["delta"])
        ptp, psl = self.get_pn_psl(option_strat, risk_free_rate)

        return DDict({
            "prob_of_profit": round(prob_of_prof, 5),
            "p_tp": round(ptp, 5),
            "p_sl": round(psl, 5)
        })


if __name__ == "main":
    pd.set_option('display.float_format', lambda x: '%.8f' % x)
    mcs = MonteCarloSimulator(["abt"])
    # print(mcs.p_greater_n_end("expr", 4))
    print(mcs.get_pop_pn_sl_old("abt", opt_price=0.30, delta=-0.29, gamma=0.23, theta=-0.00578,
                                tp=0.15, sl=-0.10, days=44))

