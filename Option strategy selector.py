import urllib.request
from bs4 import BeautifulSoup
import yfinance as yf
import pandas as pd
from pprint import pprint as pp
from datetime import datetime, date, timedelta
import numpy as np
from math import sqrt, exp, log
from scipy.stats import norm
from option_greeks import get_greeks
import warnings
import copy
import pickle

pritn = print
online = False

pd.set_option("display.max_rows", None, "display.max_columns", None)
pd.set_option('display.width', 1000)
pd.set_option('display.max_colwidth', 500)
pd.set_option('display.float_format', lambda x: '%.5f' % x)
pd.options.mode.chained_assignment = None  # default='warn'

ACCOUNT_BALANCE = 2000


class RiskWarning(Warning):
    pass


def get_libor_rates():
    libor_url = "https://www.finanzen.net/zinsen/libor/usd"

    source = urllib.request.urlopen(libor_url).read()
    soup = BeautifulSoup(source, 'lxml')

    table = soup.find('tbody', attrs={'id': 'InterestRateYieldList'})
    table_rows = table.find_all('tr')

    rates = []
    for tr in table_rows:
        td = tr.find_all('td')
        row = [tr.text.strip() for tr in td]
        rates.append(float(row[1].replace(",", ".")))

    return rates


def dte_to_date(dte):
    return datetime.now() + timedelta(days=dte)


libor_rates = get_libor_rates() if online else [0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01]
libor_expiries = (1, 7, 30, 61, 92, 182, 365)


def get_risk_free_rate(annualized_dte):
    """

    :param annualized_dte:
    :return:
    """
    return libor_rates[min_closest_index(libor_expiries, annualized_dte*365)]


# barchart.com
def get_options_meta_data(symbol):
    headers = {
        'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_11_5) AppleWebKit/537.36 (KHTML, like Gecko) '
                      'Chrome/50.0.2661.102 Safari/537.36'}
    req = urllib.request.Request('https://www.barchart.com/stocks/quotes/' + symbol, headers=headers)
    with urllib.request.urlopen(req) as response:
        resp_text = response.read()
        soup = BeautifulSoup(resp_text, 'html.parser')
        text = soup.find_all(text=True)

        ivr, vol30, oi30 = 0, 0, 0
        next_earnings = ""

        for i, t in enumerate(text):
            if "IV Rank" in t and "%" in text[i + 2]:
                ivr = float(text[i + 2].replace("%", ""))
            if "Volume Avg (30-Day)" in t and vol30 == 0:  # option volume
                vol30 = int(text[i + 2].replace(",", ""))
            if "Open Int (30-Day)" in t and oi30 == 0:
                oi30 = int(text[i + 2].replace(",", ""))
            if "Next Earnings Date" in t and not next_earnings:
                # print(text[i+1], text[i+2], text[i+3])
                next_earnings = text[i + 2].replace(" ", "")

        return ivr, vol30, oi30, next_earnings


def get_underlying_outlook(symbol):
    """
    :param symbol:
    :return: a, b, c: short - mid - long term outlook, -1 = bearish, 1 = bullish
    """
    ...
    # barchart.com ? (seems shitty)
    # RSI, VWAP, Bollinger
    # use option chain data to get prob distribution of future price
    #   https://financial-hacker.com/the-mechanical-turk/#more-2974
    # yf.Ticker("AAPL").get_recommendations()
    return 1, 1, 1


# type, positions, credit/debit, max risk, break even on exp, max prof, bpr, return on margin with 50% tp, close by date
# use monte carlo sim for P50
def propose_strategies(ticker, defined_risk_only=True, avoid_events=True,
                       min_vol30=25000, min_oi30=5000,
                       strict_mode=False, assignment_must_be_possible=False):
    # todo get short term price outlook (technical analysis, resistances from barchart.com etc)

    print(f'Settings:\n\n'
          f'          Defined risk only: {defined_risk_only}\n'
          f'               Avoid events: {avoid_events}\n'
          f'                 Min vol 30: {min_vol30}\n'
          f'       Min open interest 30: {min_oi30}\n'
          f'                Strict mode: {strict_mode}\n'
          f'Assignment must be possible: {assignment_must_be_possible}\n')

    # <editor-fold desc="Get info">
    if online:
        yf_obj = yf.Ticker(ticker)
        u_price = yf_obj.info["ask"]
        sector = yf_obj.info["sector"]
    else:
        class Dummy:
            def __init__(self):
                self.ticker = ticker

        yf_obj = Dummy()
        u_price = 10
        sector = "ABC"

    option_chain = get_option_chain(yf_obj)

    expirations = option_chain.expirations

    print(option_chain.options.head(5))

    #next_puts = option_chain.expiry_next().puts()
    next_calls = option_chain.expiry_next().calls()

    pritn(next_calls)

    """print(next_calls.head(300))
    pritn()
    print(next_puts.head(300))"""

    if online:
        ivr, vol30, oi30, next_earnings = get_options_meta_data(ticker)
    else:
        ivr, vol30, oi30, next_earnings = 0.5, 26000, 7000, "01/01/30"
    short, mid, long = get_underlying_outlook(ticker)

    # </editor-fold>

    limit_date = datetime.now() + timedelta(days=365 * 5)

    def binary_events_present():

        """
        returns true if undefined binary events can occur,
        returns the date of the earliest event as datetime object if any otherwise
        else returns false
        """

        # don't trade healthcare
        if sector == "Healthcare":
            return True

        if next_earnings and next_earnings != "N/A" and avoid_events:
            return datetime.strptime(next_earnings, '%M/%d/%y')  # .strftime("%Y-%m-%d")

        # todo check for anticipated news

    def is_liquid():

        if vol30 < min_vol30:
            warnings.warn(f'n\n\tWarning! Average volume < {min_vol30}: {vol30}\n')
            if strict_mode:
                return False

        if oi30 < min_oi30:
            warnings.warn(f'n\n\tWarning! Average open interest < {min_oi30}: {oi30}\n')
            if strict_mode:
                return False

        put_spr_r, put_spr_abs = get_atm_bid_ask(next_puts, u_price)
        call_spr_r, call_spr_abs = get_atm_bid_ask(next_calls, u_price)

        if put_spr_r >= 10 and call_spr_r >= 10:
            warnings.warn(f'\n\n\tWarning! Spread ratio is very wide: Puts = {put_spr_abs}, Calls = {call_spr_abs}\n')
            if strict_mode:
                return False

        return True

    def assingment_risk_tolerance_exceeded():

        if u_price * 100 > ACCOUNT_BALANCE / 2:
            s = f'Warning! Underlying asset price exceeds account risk tolerance! {u_price * 100} > {ACCOUNT_BALANCE / 2}'
            warnings.warn(s, RiskWarning)
            if assignment_must_be_possible:
                return

    # -----------------------------------------------------------------------------------------------------------------

    # <editor-fold desc="1 Binary events">
    binary_event_date = binary_events_present()

    if type(binary_event_date) is bool and binary_event_date:
        warnings.warn("Warning! Underlying may be subject to undefined binary events!", RiskWarning)
        if strict_mode:
            return

    if binary_event_date and type(binary_event_date) is not bool:
        limit_date = binary_event_date

    # </editor-fold>

    # <editor-fold desc="2 liquidity">
    """
    only rough check, options to trade must be checked seperately when chosen
    """

    if not True and is_liquid():  # todo
        warnings.warn(f'Warning! Underlying seems illiquid!')
        if strict_mode:
            return
    # </editor-fold>

    # <editor-fold desc="3 Price">
    assingment_risk_tolerance_exceeded()
    # </editor-fold>

    print("\nAll mandatory checks done! Starting strategy selection ...")

    # -----------------------------------------------------------------------------------------------------------------

    # 4 # IV rank

    if high(ivr):

        # <editor-fold desc="defined risk">

        # covered call
        #print(f'Covered call (tasty): {CoveredCall.get_tasty_variation(option_chain)}')

        # Credit spreads (Bear/Bull)

        # Short condors

        # </editor-fold>

        if not defined_risk_only:
            ...
            # Strangle
            # Naked Puts
            # Covered Calls

    if low(ivr):
        ...

        # defined risk
        ...
        # debit spread
        # diagonals


def get_sigma(option_type, option_price, s, k, r, T):
    # option price (mid), underlying price, strike, interest, dte annualized

    def bsm_price(option_type, sigma, s, k, r, T, q):
        # calculate the bsm price of European call and put options
        sigma = float(sigma)
        d1 = (np.log(s / k) + (r - q + sigma ** 2 * 0.5) * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)
        if option_type == 'c':
            price = np.exp(-r * T) * (s * np.exp((r - q) * T) * norm.cdf(d1) - k * norm.cdf(d2))
            return price
        elif option_type == 'p':
            price = np.exp(-r * T) * (k * norm.cdf(-d2) - s * np.exp((r - q) * T) * norm.cdf(-d1))
            return price
        else:
            print('No such option type %s') % option_type

    def implied_vol(option_type, option_price, s, k, r, T, q):
        # apply bisection method to get the implied volatility by solving the BSM function
        precision = 0.00001
        upper_vol = 500.0
        max_vol = 500.0
        min_vol = 0.0001
        lower_vol = 0.0001
        iteration = 0

        while 1:
            iteration += 1
            mid_vol = (upper_vol + lower_vol) / 2.0
            price = bsm_price(option_type, mid_vol, s, k, r, T, q)

            if option_type == 'c':

                lower_price = bsm_price(option_type, lower_vol, s, k, r, T, q)

                if (lower_price - option_price) * (price - option_price) > 0:
                    lower_vol = mid_vol
                else:
                    upper_vol = mid_vol

                if abs(price - option_price) < precision:
                    break

                if mid_vol > max_vol - 5:
                    mid_vol = 0.000001
                    break

            elif option_type == 'p':
                upper_price = bsm_price(option_type, upper_vol, s, k, r, T, q)

                if (upper_price - option_price) * (price - option_price) > 0:
                    upper_vol = mid_vol
                else:
                    lower_vol = mid_vol

                if abs(price - option_price) < precision:
                    break

                if iteration > 50: break

        return mid_vol

    return implied_vol(option_type, option_price, s, k, r, T, 0.01)

"""
def old_get_greeks(contract_type, _C0, _S, _K, _t, _r, iv=None):
    # contract type is "c" or "p"

    # print(contract_type, _C0, _S, _K, _t, _r)

    # todo theta still wrong, way too high/low
    #  delta inaccurate

    # todo not calculating IV for puts!?!?!

    def get_sigma(option_type, option_price, s, k, r, T):

        # option price (mid), underlying price, strike, interest, dte annualized

        def bsm_price(option_type, sigma, s, k, r, T, q):
            # calculate the bsm price of European call and put options
            sigma = float(sigma)
            d1 = (np.log(s / k) + (r - q + sigma ** 2 * 0.5) * T) / (sigma * np.sqrt(T))
            d2 = d1 - sigma * np.sqrt(T)
            if option_type == 'c':
                price = np.exp(-r * T) * (s * np.exp((r - q) * T) * norm.cdf(d1) - k * norm.cdf(d2))
                return price
            elif option_type == 'p':
                price = np.exp(-r * T) * (k * norm.cdf(-d2) - s * np.exp((r - q) * T) * norm.cdf(-d1))
                return price
            else:
                print('No such option type %s') % option_type

        def implied_vol(option_type, option_price, s, k, r, T, q):
            # apply bisection method to get the implied volatility by solving the BSM function
            precision = 0.00001
            upper_vol = 500.0
            max_vol = 500.0
            min_vol = 0.0001
            lower_vol = 0.0001
            iteration = 0

            while 1:
                iteration += 1
                mid_vol = (upper_vol + lower_vol) / 2.0
                price = bsm_price(option_type, mid_vol, s, k, r, T, q)

                if option_type == 'c':

                    lower_price = bsm_price(option_type, lower_vol, s, k, r, T, q)

                    if (lower_price - option_price) * (price - option_price) > 0:
                        lower_vol = mid_vol
                    else:
                        upper_vol = mid_vol

                    if abs(price - option_price) < precision:
                        break

                    if mid_vol > max_vol - 5:
                        mid_vol = 0.000001
                        break

                elif option_type == 'p':
                    upper_price = bsm_price(option_type, upper_vol, s, k, r, T, q)

                    if (upper_price - option_price) * (price - option_price) > 0:
                        upper_vol = mid_vol
                    else:
                        lower_vol = mid_vol

                    if abs(price - option_price) < precision:
                        break

                    if iteration > 50: break

            return mid_vol

        return implied_vol(option_type, option_price, s, k, r, T, 0.01)

    def d(sigma, S, K, r, t):
        d1 = 1 / (sigma * sqrt(t)) * (log(S / K) + (r + sigma ** 2 / 2) * t)
        d2 = d1 - sigma * sqrt(t)
        return d1, d2

    def call_price(sigma, S, K, r, t, d1, d2):
        C = norm.cdf(d1) * S - norm.cdf(d2) * K * exp(-r * t)
        return C

    def put_price(sigma, S, K, r, t, d1, d2):
        P = -norm.cdf(-d1) * S + norm.cdf(-d2) * K * exp(-r * t)
        return P

    # <editor-fold desc="Greeks">
    def delta(d_1):
        if contract_type == 'c':
            return norm.cdf(d_1)
        if contract_type == 'p':
            return -norm.cdf(-d_1)

    def gamma(d2, S, K, sigma, r, t):
        return K * np.exp(-r * t) * (norm.pdf(d2) / (S ** 2 * sigma * np.sqrt(t)))

    def theta(d1, d2, S, K, sigma, r, t):
        if contract_type == 'c':
            _theta = -S * sigma * norm.pdf(d1) / (2 * np.sqrt(t)) - r * K * np.exp(-r * t) * norm.cdf(d2)
        if contract_type == 'p':
            _theta = -S * sigma * norm.pdf(-d1) / (2 * np.sqrt(t)) + r * K * np.exp(-r * t) * norm.cdf(-d2)

        return _theta / 100

    def vega(sigma, S, K, r, t):
        d1, d2 = d(sigma, S, K, r, t)
        v = S * norm.pdf(d1) * np.sqrt(t)
        return v

    # </editor-fold>

    if not iv:
        iv = get_sigma(contract_type, _C0, _S, _K, _r, _t)

    _d1, _d2 = d(iv, _S, _K, _r, _t)

    return iv, \
           delta(_d1), \
           gamma(_d2, _S, _K, iv, _r, _t), \
           vega(iv, _S, _K, _r, _t), \
           theta(_d1, _d2, _S, _K, iv, _r, _t)
"""


def get_option_chain(yf_obj):
    # contractSymbol  lastTradeDate  strike  lastPrice    bid    ask  change  percentChange   volume  openInterest
    # ... impliedVolatility  inTheMoney contractSize currency

    start = datetime.now()

    if not online:
        return OptionChain.from_file(filename=yf_obj.ticker + "_chain")

    option_chain = DDict()
    ticker_data = yf_obj
    underlying_ask = float(yf_obj.info["ask"])
    expirations = [datetime.strptime(t, "%Y-%m-%d") for t in ticker_data.options]
    relevant_fields = ["name", "strike", "bid", "ask", "mid", "volume", "OI",
                       "IV", "delta", "gamma", "vega", "theta", "rho"]

    def populate_greeks(contracts, _dte, contract_type):
        contracts["delta"] = 0.0
        contracts["gamma"] = 0.0
        contracts["vega"] = 0.0
        contracts["theta"] = 0.0
        contracts["rho"] = 0.0
        contracts["mid"] = 0.0

        risk_free_interest = get_risk_free_rate(_dte)

        for index, contract in contracts.iterrows():
            contracts.loc[index, "mid"] = (contract["bid"] + contract["ask"]) / 2
            contract_price = contract["mid"]
            iv = max(min(float(contract["IV"]), 100), 0.005)

            opt_price, iv, delta, gamma, theta, vega, rho = get_greeks(contract_type,
                                                                       underlying_ask,
                                                                       float(contract["strike"]),
                                                                       max(_dte, 0.001),
                                                                       risk_free_interest,
                                                                       contract_price,
                                                                       iv=iv)

            contracts.loc[index, "delta"] = -delta if contract_type == "p" else delta
            contracts.loc[index, "gamma"] = gamma
            contracts.loc[index, "vega"] = vega
            contracts.loc[index, "theta"] = theta / 100
            contracts.loc[index, "rho"] = rho
            contracts.loc[index, "IV"] = iv

            """
            opt_price, iv, delta, gamma, theta, vega, rho = get_greeks(contract_type,
                                                                       underlying_ask,
                                                                       float(contract["strike"]),
                                                                       _dte,
                                                                       risk_free_interest,
                                                                       contract_price,
                                                                       dividend=0,
                                                                       iv=min(max(0.005, orig_iv), 10))

            contracts["o delta"] = delta
            contracts["o gamma"] = gamma
            contracts["o vega"] = vega
            contracts["o theta"] = theta"""

    for expiration in expirations:
        exp_string = expiration.strftime("%Y-%m-%d")

        print(f'Collecting option chain for {ticker_data.info["symbol"]} {exp_string} ...')

        chain = ticker_data.option_chain(exp_string)
        dte = float((expiration.date() - date.today()).days) / 365

        chain.puts.rename(columns={"impliedVolatility": "IV", "openInterest": "OI", "contractSymbol": "name"},
                          inplace=True)
        chain.calls.rename(columns={"impliedVolatility": "IV", "openInterest": "OI", "contractSymbol": "name"},
                           inplace=True)

        chain.puts.fillna(0).astype({"volume": "int32"}, copy=False)
        chain.calls.fillna(0).astype({"volume": "int32"}, copy=False)

        populate_greeks(chain.calls, dte, "c")
        populate_greeks(chain.puts, dte, "p")

        option_chain[exp_string] = DDict({"puts": chain.puts[relevant_fields], "calls": chain.calls[relevant_fields]})

    print("Gathering options data took", (datetime.now() - start).total_seconds(), "seconds")

    return OptionChain(chain_dict=option_chain)


class DDict(dict):
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__
    __hash__ = dict.__hash__


def get_timestamp():
    return datetime.now().strftime("%H:%M:%S.%f")


# <editor-fold desc="Requirements">

def high(x):  # x 0 to 100, high if > 50; returns scale from 0 to 1 indicating highness
    return 0 if x <= 50 else (x / 50 if x > 0 else 0) - 1


def low(x):  # x 0 to 100
    return 0 if x >= 50 else 1 - (x / 50 if x > 0 else 0)


def calendar_spread_check(front_m_vol, back_m_vol):
    return front_m_vol >= back_m_vol * 1.2


def get_closest_date(expiries, dte):
    """
    :param expiries: list of strings in format year-mon-day
    :param dte: days to expiration
    :return: closest date from expiries to dte
    """
    best_exp = None
    best_diff = 10000
    for expiration in expiries:
        diff = abs(dte - date_to_dte(expiration))
        if diff < best_diff:
            best_diff = abs(dte - date_to_dte(expiration))
            best_exp = expiration
    return best_exp


def date_to_dte(date_str):
    return abs((datetime.now().date() - exp_to_date(date_str)).days)


def exp_to_date(expiry):
    return datetime.strptime(expiry, "%Y-%m-%d").date()


def date_to_opt_format(d):
    return exp_to_date(d).strftime("%b %d")


def get_delta_option_strike(chain, delta):  # chain must contain deltas (obviously onii-chan <3)
    return chain["strike"][min_closest_index(list(chain["delta"]), delta)]


def min_closest_index(a, v):
    """
    :param a: list of values
    :param v: target value
    :return: index of object in list that is closest to target value
    """
    return min(range(len(a)), key=lambda i: abs(abs(a[i]) - v))


def get_atm_strike(chain, _ask):
    return chain["strike"][min_closest_index(list(chain["strike"]), _ask)]


def get_atm_bid_ask(chain, _ask):
    """
    returns relative and absolute spread
    """

    return 0,0  # todo

    b = float(chain.loc[chain["strike"] == get_atm_strike(chain, _ask)]["bid"])
    a = float(chain.loc[chain["strike"] == get_atm_strike(chain, _ask)]["ask"])

    if a == b == 0:
        return 0, 0

    return 1 - b / a, a - b


# </editor-fold>


# <editor-fold desc="Strats">

# todo TEST!
class Option:

    def __init__(self, opt_type, expiration, strike, bid, ask,
                 delta, gamma, theta, vega, rho, iv):

        """
        :param opt_type:
        :param expiration: as string
        :param strike:
        :param premium:

        :param delta:
        :param gamma:
        :param theta:
        :param vega:
        :param rho:
        :param iv:
        """

        # TODO take single DF line to extract data

        self.opt_type, self.expiration, self.strike = opt_type, expiration, strike
        self.bid = bid
        self.ask = ask

        self.expiration_dt = exp_to_date(self.expiration)
        self.dte = date_to_dte(self.expiration_dt)

        self.greeks = DDict({"delta": delta, "gamma": gamma, "theta": theta, "vega": vega, "rho": rho})

        self.iv = iv

    def parse_option(self, opt_string):

        lot, month, day, strike, opt_type, _, premium = opt_string.split()
        expiration = datetime.strptime(month + " " + day + " 2021", '%b %d %Y').strftime("%Y-%m-%d")

        if "-" in lot:
            lot = -1 * float(lot.replace("-", ""))
        else:
            lot = float(lot)

        strike = float(strike)

        if opt_type == "P":
            opt_type = "put"
        if opt_type == "C":
            opt_type = "call"

        premium = float(premium)

        self.opt_type, self.expiration, self.strike, self.ask = opt_type, expiration, strike, premium
        self.bid = self.ask

    def __str__(self):
        return f'{date_to_opt_format(self.expiration)} {self.opt_type.upper()} {self.strike} @ {self.bid}/{self.ask}'


# todo TEST!
class Stock:

    def __init__(self, ticker, bid, ask):
        self.name = ticker
        self.bid = bid
        self.ask = ask
        self.greeks = DDict({"delta": 1, "gamma": 0, "theta": 0, "vega": 0, "rho": 0})
        self.risk = self.get_risk()

    def __str__(self):
        return f'{self.name} @ {self.bid}/{self.ask}'


# todo TEST!
class Position:

    def __init__(self, asset, quantiy):  # , underlying_price):
        self.asset = asset
        self.quantity = quantiy
        self.cost = self.get_cost()
        self.greeks = DDict({"delta": 0, "gamma": 0, "theta": 0, "vega": 0, "rho": 0})
        self.set_greeks()
        self.risk = self.get_risk()
        # self.underlying_price = underlying_price

    def set_greeks(self):

        self.greeks.delta = self.quantity * self.asset.delta
        self.greeks.gamma = self.quantity * self.asset.gamma
        self.greeks.vega = self.quantity * self.asset.vega
        self.greeks.theta = self.quantity * self.asset.theta
        self.greeks.rho = self.quantity * self.asset.tho

        if type(self.asset) is Option:
            self.greeks.delta *= 100
            self.greeks.gamma *= 100
            self.greeks.vega *= 100
            self.greeks.theta *= 100
            self.greeks.rho *= 100

        if self.quantity < 0:
            self.greeks.delta *= -1
            self.greeks.gamma *= -1
            self.greeks.vega *= -1
            self.greeks.theta *= -1
            self.greeks.rho *= -1

    def get_risk(self):

        # long stock / option
        if self.quantity > 0:
            return self.asset.bid * self.quantity * (100 if type(self.asset) is Option else 1)

        # short options / stocks
        if self.quantity < 0:

            if type(self.asset) is Option and self.asset.opt_type == "p":
                return (-self.asset.bid + self.asset.strike) * -self.quantity

            return float('inf')

            """# http://tastytradenetwork.squarespace.com/tt/blog/buying-power

            if type(self.asset) is Option:
                a = (0.2 * self.underlying_price - self.asset.strike + self.underlying_price + self.asset.bid) \
                        * (-self.quantity) * 100
                b = self.asset.strike * 10
                c = 50 * (-self.quantity) + self.asset.bid * 100
                return max(a, b, c)

            # https://support.tastyworks.com/support/solutions/articles/43000435243-short-stock-or-etfs-

            if type(self.asset) is Stock:
                if self.underlying_price > 5:
                    return max(-self.quantity * 5, -self.quantity * self.underlying_price)
                else:
                    return max(-self.quantity * self.underlying_price, -self.quantity * 2.5)"""

        return 0

    def get_cost(self):
        if self.quantity > 0:
            c = self.quantity * self.asset.ask
        else:
            c = -self.quantity * self.asset.bid
        if type(self.asset) is Option:
            return c*100
        return c

    def change_quantity_to(self, x):
        self.quantity = x
        self.cost = self.get_cost()
        self.set_greeks()

    def add_x(self, x):
        self.change_quantity_to(self.quantity + x)


# todo TEST!
class CombinedPosition:

    def __init__(self, pos_dict, u_bid, u_ask):
        """

        todo always have min 0 lot stock position to be able to refer to underlying price

        :param pos_dict:  (option symbol/stock ticker): position
        """
        self.pos_dict = pos_dict
        self.cost = self.get_cost()
        self.greeks = self.get_greeks()
        self.risk = self.get_risk()
        self.u_bid = u_bid
        self.u_ask = u_ask

        # todo add mandatory (possible 0 lot) stock position
        """if not any(type(pos.asset) is Stock for pos in self.pos_dict.values()):
            ticker = list(self.pos_dict.values())[0].name[:6]
            # remove numbers
            ticker = ''.join([i for i in ticker if not i.isdigit()])"""

    def get_risk(self):
        positions = copy.deepcopy(self.pos_dict.values())

        shorts = [p for p in positions if p.quantity < 0]
        longs = [p for p in positions if p.quantity > 0]

        stock = [p for p in positions if type(p.asset) is Stock][0]  # there is 1 stock position

        # BEWARE all you do to stock if quantity < 0 must be also done to stock in shorts

        shorts.sort(key=lambda x: x.risk, reverse=True)

        risk = 0  # based on 1 share, so in cents or small dollar values

        """if stock.quantity > 0:  # long stock
            long_puts = [p for p in longs if p.asset.opt_type == "p"]

            # select highest strike put
            long_puts.sort(key=lambda x: x.asset.strike, reverse=True)
            high_put_pos = long_puts[0]

            to_cover = stock.quantity
            
            while to_cover > 0 and long_puts:"""

        for short_pos in shorts:

            to_cover = -short_pos.quantity

            def get_long_covering_score(lp):

                theta_decay_start_dte = 90  # assume no relevant change to theta before this period
                extr_projection = lambda x: sqrt(1 - x ** 2)  # circle upper right for extrinsic 90 dte to 0 dte
                scale = lambda x: -x / theta_decay_start_dte + 1  # maps 90...0 to 0...1

                strike_diff = abs(short_pos.asset.strike - lp.asset.strike)

                # forecasted extrinsic value of long option when short option expires
                current_intr = lp.asset.strike - stock.asset.bid \
                    if lp.asset.opt_type == "p" else stock.asset.ask - lp.asset.strike
                current_intr = max(0, current_intr)
                current_extr = lp.asset.bid - current_intr
                l_dte = lp.asset.dte
                s_dte = short_pos.asset.dte
                dte_diff = l_dte - s_dte
                # todo test
                given_up_extr_by_exe = current_extr + lp.asset.theta * dte_diff \
                    if dte_diff > theta_decay_start_dte else \
                    extr_projection(scale(dte_diff)) * current_extr

                return strike_diff + given_up_extr_by_exe

            if type(short_pos.asset) is Option:

                if short_pos.asset.opt_type == "c":  # can only cover short calls with long stock

                    # <editor-fold desc="cover with long stock">

                    # is there any long stock?
                    if stock.quantity > 0:

                        long_q = stock.quantity
                        stock.add_x(-min(to_cover*100, long_q))
                        to_cover -= min(to_cover, long_q/100)

                    # </editor-fold>

                if short_pos.asset.opt_type == "p":  # can only cover short puts with short stock

                    # <editor-fold desc="cover with short stock">

                    # is there any short stock?
                    if stock.quantity < 0:

                        short_stock_q = -stock.quantity

                        stock.add_x(min(to_cover*100, short_stock_q))
                        stock_in_shorts = [s for s in shorts if type(s) is Stock][0]
                        stock_in_shorts.asset.add_x(min(to_cover*100, short_stock_q))

                        to_cover -= min(to_cover, short_stock_q/100)

                    # </editor-fold>

                # <editor-fold desc="or with long option">

                same_type_longs = [p for p in longs if p.asset.opt_type == short_pos.asset.opt_type]
                longs_w_cov_score = [(long_p, get_long_covering_score(long_p)) for long_p in same_type_longs]
                longs_w_cov_score.sort(key=lambda x: x[1])  # sort by covering score

                while to_cover > 0 and longs:  # go until short position is fully covered or no longs remain

                    # update long quantities
                    long_q = longs_w_cov_score[0][0].quantity
                    longs_w_cov_score[0][0].add_x(-min(to_cover, long_q))

                    # stock.add_x(-min(to_cover, long_q))  todo why was this in??
                    to_cover -= min(to_cover, long_q)

                    # update risk
                    risk += longs_w_cov_score[0][0].risk + abs(short_pos.asset.strike - longs_w_cov_score[0][0].strike)

                    if longs_w_cov_score[0][0].quantity == 0:
                        longs.remove(longs_w_cov_score[0][0])
                        longs_w_cov_score.remove(longs_w_cov_score[0])

                # </editor-fold>

                if to_cover > 0:

                    if short_pos.risk == float('inf'):
                        return float('inf')

                    risk += to_cover * short_pos.risk

            if type(short_pos.asset) is Stock:

                to_cover /= 100

                # receive stock bid upfront
                risk -= short_pos.asset.bid * -short_pos.quantity

                long_calls = [p for p in longs if p.asset.opt_type == "c"]
                longs_w_cov_score = [(long_p, get_long_covering_score(long_p)) for long_p in long_calls]
                longs_w_cov_score.sort(key=lambda x: x[1])  # sort by covering score

                while to_cover > 0 and longs:  # go until short position is fully covered or no longs remain

                    # adjust long position quantity
                    long_q = longs_w_cov_score[0][0].quantity
                    used_long_options = min(to_cover, long_q)
                    longs_w_cov_score[0][0].add_x(-used_long_options)

                    # adjust short position quantity
                    stock.add_x(min(to_cover*100, -short_pos.asset.quantity))

                    stock_in_shorts = [s for s in shorts if type(s) is Stock][0]
                    stock_in_shorts.asset.add_x(min(to_cover * 100, -short_pos.asset.quantity))

                    # update coverage
                    to_cover -= used_long_options

                    # update risk
                    risk += (longs_w_cov_score[0][0].asset.strike - stock.asset.bid) * used_long_options + \
                        longs_w_cov_score[0][0].asset.premium * np.ceil(used_long_options)  # todo inaccurate?

                    # delete long option if used up
                    if longs_w_cov_score[0][0].quantity == 0:
                        longs.remove(longs_w_cov_score[0][0])
                        longs_w_cov_score.remove(longs_w_cov_score[0])

                if to_cover > 0:
                    return float('inf')

        return risk * 100

    def get_cost(self):
        return sum([pos.cost for pos in self.pos_dict.values()])

    def add_asset(self, asset, quantity):
        if asset in self.pos_dict:
            self.pos_dict[asset].add_x(quantity)
        else:
            self.pos_dict[asset] = Position(asset, quantity, self.u_ask)

        self.cost = self.get_cost()
        self.greeks = self.get_greeks()

    def change_asset_quantity(self, asset, quantity):
        self.add_asset(asset, quantity)
        self.cost = self.get_cost()
        self.greeks = self.get_greeks()

    def remove_asset(self, asset):
        if asset not in self.pos_dict:
            return
        else:
            del self.pos_dict[asset]

        self.cost = self.get_cost()
        self.greeks = self.get_greeks()

    def get_greeks(self):
        greeks = DDict({"delta": 0, "gamma": 0, "theta": 0, "vega": 0, "rho": 0})
        for name, position in self.pos_dict.items():
            greeks.delta += position.greeks.delta
            greeks.gamma += position.greeks.gamma
            greeks.vega += position.greeks.vega
            greeks.theta += position.greeks.theta
            greeks.rho += position.greeks.rho
        return greeks


class OptionStrategy:
    """
    defines criteria for chosing options from a chain/stock to form a position
    """

    def __init__(self, name, requirements, timeframes, strikes, managing, hints):
        self.name = name
        self.requirements = requirements
        self.hints = hints
        self.timeframes = timeframes
        self.strikes = strikes
        self.managing = managing

        self.positions = list()

    def add_position(self, option):  # todo can be stock too
        self.positions.append(option)


class OptionChain:

    """
    methods return list of options that adhere to the given filter arguments
    """

    def __init__(self, chain=None, chain_dict=None, ):

        if chain is not None:
            self.options = chain
            if not chain.empty:
                self.expirations = sorted(chain.expiration.unique().tolist())
                self.ticker = chain.at[0, "name"][:6]
                self.ticker = ''.join([i for i in self.ticker if not i.isdigit()])
            else:
                self.expirations = []
                self.ticker = None

            print("After filtering:", self.expirations, "\n", self.options)
        else:
            self.expirations = list(chain_dict.keys())

            self.ticker = chain_dict[self.expirations[0]].puts.loc[0, "name"][:6]
            self.ticker = ''.join([i for i in self.ticker if not i.isdigit()])

            contract_list = list()

            for expiry in self.expirations:
                chain_dict[expiry].puts.loc[:, "contract"] = "p"
                chain_dict[expiry].puts.loc[:, "expiration"] = expiry
                chain_dict[expiry].puts.loc[:, "direction"] = "long"

                chain_dict[expiry].calls.loc[:, "contract"] = "c"
                chain_dict[expiry].calls.loc[:, "expiration"] = expiry
                chain_dict[expiry].calls.loc[:, "direction"] = "long"

                contract_list.extend((chain_dict[expiry].puts, chain_dict[expiry].calls))

            long_chain = pd.concat(contract_list, ignore_index=True)
            short_chain = long_chain.copy()

            for index, row in short_chain.iterrows():
                short_chain.loc[index, "direction"] = "short"
                short_chain.loc[index, "delta"] *= -1
                short_chain.loc[index, "gamma"] *= -1
                short_chain.loc[index, "vega"] *= -1
                short_chain.loc[index, "theta"] *= -1
                short_chain.loc[index, "rho"] *= -1

            self.options = pd.concat((long_chain, short_chain), ignore_index=True)

            self.save_as_file()

    def __repr__(self):
        return self.options.to_string()

    # <editor-fold desc="Type">

    def puts(self):
        return OptionChain(self.options.loc[self.options['contract'] == "P"])

    def calls(self):
        return OptionChain(self.options.loc[self.options['contract'] == "C"])

    def long(self):
        return OptionChain(self.options.loc[self.options['direction'] == "long"])

    def short(self):
        return OptionChain(self.options.loc[self.options['direction'] == "short"])

    # </editor-fold>

    # <editor-fold desc="Expiry">
    def expiry_range(self, lower_dte, upper_dte):
        dates = pd.date_range(start=datetime.now() + timedelta(days=lower_dte),
                              end=datetime.now() + timedelta(days=upper_dte),
                              normalize=True)
        # todo works?
        return OptionChain(self.options.loc[self.options['expiration'] in dates])

    def expiry_date(self, exp_date):
        print("Exp date:", exp_date)
        return OptionChain(self.options.loc[self.options['expiration'] == exp_date])

    def expiry_dte(self, dte):
        ...

    def expiry_next(self, i=0):
        """
        get i-th expiry's puts & calls
        :param i:
        :return:
        """
        print(self.expirations[i])
        return OptionChain(self.options.loc[self.options['expiration'] == self.expirations[i]])

    # </editor-fold>

    # <editor-fold desc="Greeks">

    def greek(self, g, lower=0, upper=1):
        """

        :param g:       type of greek
        :param lower:
        :param upper:
        :return:
        """
        ...

    def greek_close_to(self, d):
        return min_closest_index(...)

    def delta(self, lower=-1, upper=1):
        return self.greek("delta", lower, upper)

    def delta_close(self, d):
        ...

    def gamma(self, lower=-1, upper=1):
        return self.greek("gamma", lower, upper)

    def gamma_close(self, d):
        ...

    def theta(self, lower=-1, upper=1):
        return self.greek("theta", lower, upper)

    def theta_close(self, d):
        ...

    def vega(self, lower=-1, upper=1):
        return self.greek("vega", lower, upper)

    def vega_close(self, d):
        ...

    def rho(self, lower=-1, upper=1):
        return self.greek("rho", lower, upper)
    # </editor-fold>

    def iv(self, lower=0, upper=10000):
        ...

    # <editor-fold desc="File">
    def save_as_file(self):

        filename = self.ticker + "_chain"  # + get_timestamp().replace(".", "-").replace(":", "-")

        try:
            print(f'Creating file:\n'
                  f'\t{filename}.pickle ...')

            with open(filename + ".pickle", "wb") as file:
                pickle.dump(self, file)

            print("File created successfully!")

        except Exception as e:
            print("While creating the file", filename, "an exception occurred:", e)

    @staticmethod
    def from_file(filename):
        try:
            f = pickle.load(open(filename + ".pickle", "rb"))
            return f
        except Exception as e:
            print("While reading the file", filename, "an exception occurred:", e)
    # </editor-fold>


class LongCall(OptionStrategy):

    def test(self, params):
        # option chain, u price, short, mid, long outlook (-1 - 1),
        if params.short_outlook > 0.8:
            # which delta call to choose?
            ...


class CoveredCall(OptionStrategy):

    # df.loc[df['column_name'] == some_value]
    requirements = None
    timeframes = None
    strikes = None
    managing = None
    hints = None

    def __init__(self):
        super().__init__("Covered call", self.requirements, self.timeframes, self.strikes, self.managing, self.hints)

    @staticmethod
    def get_tasty_variation(chain):
        expiration = get_closest_date(chain.expirations, 45)
        strike = get_delta_option_strike(chain.expiry_date(expiration).calls(), 0.325)
        premium = float(chain
                        .expiry_date(expiration)
                        .calls()
                        .strike(strike)["ask"])

        return Option('c', expiration, strike, premium)


# </editor-fold>

propose_strategies("AMC")
