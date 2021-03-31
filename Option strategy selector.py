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
from IPython.display import display, HTML
import ipywidgets as widgets

pritn = print
online = False
debug = True


def _warn(msg):
    warnings.warn("\n\n\t" + msg + "\n", stacklevel=3)


if not online:
    _warn("OFFLINE!    "*17)


def _debug(*args):
    if debug:
        print()
        print(*args)


# <editor-fold desc="Pandas settings">
pd.set_option("display.max_rows", None, "display.max_columns", None)
pd.set_option('display.width', 1000)
pd.set_option('display.max_colwidth', 500)
pd.set_option('display.float_format', lambda x: '%.5f' % x)
pd.options.mode.chained_assignment = None  # default='warn'
# </editor-fold>

ACCOUNT_BALANCE = 1471


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


def get_sp500_tickers(exclude_sub_industries=('Pharmaceuticals',
                                              'Managed Health Care',
                                              'Health Care Services')):
    df = pd.read_html('https://en.wikipedia.org/wiki/List_of_S%26P_500_companies')[0]
    df = df[~df['GICS Sub-Industry'].isin(exclude_sub_industries)]
    return df['Symbol'].values.tolist()


libor_rates = get_libor_rates() if online else [0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01]
libor_expiries = (1, 7, 30, 61, 92, 182, 365)


def get_risk_free_rate(annualized_dte):
    """

    :param annualized_dte:
    :return:
    """
    return libor_rates[min_closest_index(libor_expiries, annualized_dte * 365)]


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


def get_rsi20(symbol):

    headers = {
        'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_11_5) AppleWebKit/537.36 (KHTML, like Gecko) '
                      'Chrome/50.0.2661.102 Safari/537.36'}
    req = urllib.request.Request('https://www.barchart.com/stocks/quotes/' + symbol + '/technical-analysis',
                                 headers=headers)
    with urllib.request.urlopen(req) as response:
        resp_text = response.read()
        soup = BeautifulSoup(resp_text, 'html.parser')

        tables = soup.findAll('table')
        for table in tables:
            table_rows = table.find_all('tr')
            index_names = [x.text for x in table.find_all('th')]
            if "Relative Strength" in index_names:

                l = []
                for tr in table_rows:
                    td = tr.find_all('td')
                    row = [tr.text for tr in td]
                    l.append(pd.DataFrame(row, index=index_names).T)

                df = pd.concat(l)
                raw = df.loc[df["Period"] == "14-Day", "Relative Strength"][0]
                rsi20 = float(raw.replace("%", "").strip())

                return rsi20

    return -1


def get_underlying_outlook(symbol):
    """
    TODO
    :param symbol:
    :return: a, b, c: short - mid - long term outlook, -1 = bearish, 1 = bullish
    """
    # barchart.com ? (seems shitty)
    # RSI, VWAP, Bollinger
    # use option chain data to get prob distribution of future price
    #   https://financial-hacker.com/the-mechanical-turk/#more-2974
    # yf.Ticker("AAPL").get_recommendations()

    headers = {
        'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_11_5) AppleWebKit/537.36 (KHTML, like Gecko) '
                      'Chrome/50.0.2661.102 Safari/537.36'}
    req = urllib.request.Request('https://www.barchart.com/stocks/quotes/' + symbol + '/opinion-legacy',
                                 headers=headers)

    with urllib.request.urlopen(req) as response:
        resp_text = response.read()
        soup = BeautifulSoup(resp_text, 'html.parser')

        outlooks = [0, 0, 0]

        texts = soup.findAll("td", attrs={'class': 'indicator-avg-signal'})[:3]
        for i, t in enumerate(texts):
            tmp = t.text
            if "Hold" in tmp:
                outlooks[i] = 0
                continue
            tmp = tmp.replace("Average:", "").replace("Buy", "").replace("%", "").strip()
            fac = 1
            if "Sell" in tmp:
                fac = -1
                tmp = tmp.replace("Sell", "")
            outlooks[i] = fac * float(tmp) / 100

    return outlooks[0], outlooks[1], outlooks[2]


class DDict(dict):
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__
    __hash__ = dict.__hash__


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
    relevant_fields = ["name", "strike", "bid", "ask", "mid", "last", "volume", "OI",
                       "IV", "P(ITM)", "delta", "gamma", "vega", "theta", "rho"]

    def delta_to_itm_prob(delta, contract_type):
        magic_number = 0.624
        a = sqrt(1 - (delta * 2 * magic_number - magic_number) ** 2) - sqrt(1-magic_number**2)
        r = 0
        if contract_type == "p":
            r = delta - a
        if contract_type == "c":
            r = delta + a
        return r * 100

    def populate_greeks(contracts, _dte, contract_type):
        contracts["delta"] = 0.0
        contracts["gamma"] = 0.0
        contracts["vega"] = 0.0
        contracts["theta"] = 0.0
        contracts["rho"] = 0.0
        contracts["mid"] = 0.0
        contracts["P(ITM)"] = 0.0

        risk_free_interest = get_risk_free_rate(_dte)

        for index, contract in contracts.iterrows():
            contracts.loc[index, "mid"] = (contract["bid"] + contract["ask"]) / 2
            contract_price = contract["last"]  # todo last or mid?
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

            contracts.loc[index, "P(ITM)"] = delta_to_itm_prob(delta, contract_type)

    for expiration in expirations:
        exp_string = expiration.strftime("%Y-%m-%d")

        print(f'Collecting option chain for {ticker_data.info["symbol"]} {exp_string} ...')

        chain = ticker_data.option_chain(exp_string)
        dte = float((expiration.date() - date.today()).days) / 365

        chain.puts.rename(columns={"impliedVolatility": "IV",
                                   "openInterest": "OI",
                                   "contractSymbol": "name",
                                   "lastPrice": "last"},
                          inplace=True)
        chain.calls.rename(columns={"impliedVolatility": "IV",
                                    "openInterest": "OI",
                                    "contractSymbol": "name",
                                    "lastPrice": "last"},
                           inplace=True)

        populate_greeks(chain.calls, dte, "c")
        populate_greeks(chain.puts, dte, "p")

        chain.puts.fillna(0, inplace=True)
        chain.calls.fillna(0, inplace=True)

        chain.puts.astype({"volume": int}, copy=False)
        chain.calls.astype({"volume": int}, copy=False)

        option_chain[exp_string] = DDict(
            {"puts": chain.puts[relevant_fields], "calls": chain.calls[relevant_fields]})

    print("Gathering options data took", (datetime.now() - start).total_seconds(), "seconds")

    return OptionChain(chain_dict=option_chain)


# todo
# type, positions, credit/debit, max risk, break even on exp, max prof, bpr, return on margin with 50% tp,
# close by date
# use monte carlo sim for P50,
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
        u_bid = yf_obj.info["bid"]
        u_ask = yf_obj.info["ask"]
        sector = yf_obj.info["sector"]
    else:
        class Dummy:
            def __init__(self):
                self.ticker = ticker
                self.info = DDict({"bid": -1, "ask": -1})

        yf_obj = Dummy()
        u_price = 10
        u_bid = 9
        u_ask = 10
        sector = "ABC"

    if u_price == 0:
        _warn("Data source reporting 0 price for underlying. Data is likely false, exiting ...")
        exit(-1)

    option_chain = get_option_chain(yf_obj)
    expirations = option_chain.expirations

    """
        import ipdb
        ipdb.set_trace()
        """

    if online:
        ivr, vol30, oi30, next_earnings = get_options_meta_data(ticker)
        rsi20 = get_rsi20(ticker)
    else:
        ivr, vol30, oi30, next_earnings, rsi20 = 0.5, 26000, 7000, "01/01/30", 50

    short_outlook, mid_outlook, long_outlook = get_underlying_outlook(ticker)

    # </editor-fold>

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
            return datetime.strptime(next_earnings, '%M/%d/%y').strftime("%Y-%m-%d")

        # todo check for anticipated news

    def is_liquid():

        r = True

        if vol30 < min_vol30:
            _warn(f'Warning! Average volume < {min_vol30}: {vol30}\n')
            if strict_mode:
                r = False

        if oi30 < min_oi30:
            _warn(f'Warning! Average open interest < {min_oi30}: {oi30}\n')
            if strict_mode:
                r = False

        put_spr_r, put_spr_abs = get_bid_ask_spread(next_puts)
        call_spr_r, call_spr_abs = get_bid_ask_spread(next_calls)

        if put_spr_r >= 10 and call_spr_r >= 10:
            _warn(
                f'Warning! Spread ratio is very wide: Puts = {put_spr_abs}, Calls = {call_spr_abs}\n')
            if strict_mode:
                r = False

        _debug(f'Liquidity:\n\n'
               f'                     Vol 30: {vol30:7d}\n'
               f'           Open interest 30: {oi30:7d}\n'
               f'     Put spread ratio (rel):    {put_spr_r:.2f}\n'
               f'     Put spread ratio (abs):    {put_spr_abs:.2f}\n'
               f'    Call spread ratio (rel):    {call_spr_r:.2f}\n'
               f'    Call spread ratio (abs):    {call_spr_abs:.2f}\n')

        return r

    def assingment_risk_tolerance_exceeded():

        if u_price * 100 > ACCOUNT_BALANCE / 2:
            s = f'Warning! Underlying asset price exceeds account risk tolerance! ' \
                f'{u_price * 100} > {ACCOUNT_BALANCE / 2}'
            _warn(s)
            if assignment_must_be_possible:
                return

    # -----------------------------------------------------------------------------------------------------------------

    # <editor-fold desc="1 Binary events">
    binary_event_date = binary_events_present()

    if type(binary_event_date) is bool and binary_event_date:
        _warn("Warning! Underlying may be subject to undefined binary events!")
        if strict_mode:
            return

    if binary_event_date and type(binary_event_date) is not bool:
        option_chain = option_chain.expiration_before(binary_event_date)

    # </editor-fold>

    # <editor-fold desc="2 liquidity">
    """
        only rough check, options to trade must be checked seperately when chosen
        """
    next_puts = option_chain.expiration_next().puts()
    next_calls = option_chain.expiration_next().calls()

    if not is_liquid():
        _warn(f'Warning! Underlying seems illiquid!')
        if strict_mode:
            return
    # </editor-fold>

    # <editor-fold desc="3 Price">
    assingment_risk_tolerance_exceeded()
    # </editor-fold>

    # print(option_chain.expiration_next().long().head(100))

    print("\nAll mandatory checks done! Starting strategy selection ...")

    # -----------------------------------------------------------------------------------------------------------------

    # todo compose environment for stock to give to strategies

    env = {
        "IV": ivr,
        "IV outlook": 0,

        "RSI20d": rsi20,
        "short term outlook": short_outlook,
        "mid term outlook": mid_outlook,
        "long term outlook": long_outlook,
    }

    env_con = EnvContainer(env, option_chain, u_bid, u_ask)

    print(CoveredCall(env_con))

    # <editor-fold desc="4 IV rank">

    """if high(ivr):

        # <editor-fold desc="defined risk">

        # covered call
        print(f'Covered call (tasty): {CoveredCall.get_tasty_variation(option_chain, u_bid, u_ask)}')

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

        # diagonals"""

    # </editor-fold>


# <editor-fold desc="Requirements">

"""
def high(x):  # x 0 to 100, high if > 50; returns scale from 0 to 1 indicating highness
    return 0 if x <= 50 else (x / 50 if x > 0 else 0) - 1


def low(x):  # x 0 to 100
    return 0 if x >= 50 else 1 - (x / 50 if x > 0 else 0)


def high_ivr(ivr):
    # todo adjust to overall market situation (give result in relation to VIX IV)
    return 0 if ivr <= 50 else (ivr / 50 if ivr > 0 else 0) - 1


def low_ivr(x):  # x 0 to 100
    # todo adjust to overall market situation (give result in relation to VIX IV)
    return 0 if x >= 50 else 1 - (x / 50 if x > 0 else 0)"""


def high_ivr(ivr):
    return (ivr - 50) / 50


def low_ivr(ivr):
    return 1 - high_ivr(ivr)


def low_rsi(rsi):

    return -rsi/50 + 1

    # todo which one is better?
    """if rsi <= 30:
        return rsi/30 - 1
    if rsi <= 70:
        return rsi
    else:
        return rsi/30 - (7/3)"""


def high_rsi(rsi):
    return -low_rsi(rsi)


def neutral(outlook):
    if outlook <= -0.5:
        return outlook
    if outlook <= 0:
        return 4*outlook+1
    if outlook <= 0.5:
        return -4*outlook+1
    else:
        return -outlook


def bullish(outlook):
    return outlook


def bearish(outlook):
    return outlook


def extreme(outlook):
    if outlook <= 0:
        return -2*outlook-1
    else:
        return 2*outlook-1


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
    """

    :param date_str: as string YYYY-MM-DD
    :return:
    """
    return abs((datetime.now().date() - exp_to_date(date_str)).days)


def dte_to_date(dte):
    return datetime.now() + timedelta(days=dte)


def exp_to_date(expiration):
    return datetime.strptime(expiration, "%Y-%m-%d").date()


def date_to_european_str(d):
    return datetime.strptime(d, "%Y-%m-%d").strftime("%m.%d.%Y")


def date_to_opt_format(d):
    return exp_to_date(d).strftime("%b %d")


def get_delta_option_strike(chain, delta):  # chain must contain deltas (obviously onii-chan <3)
    return chain.at["strike", min_closest_index(list(chain["delta"]), delta)]


def min_closest_index(a, v):
    """
    :param a: list of values
    :param v: target value
    :return: index of object in list that is closest to target value
    """
    return min(range(len(a)), key=lambda i: abs(abs(a[i]) - v))


def get_atm_strike(chain, _ask):
    return chain["strike"][min_closest_index(list(chain["strike"]), _ask)]


def get_bid_ask_spread(chain):
    """

    :param chain:
    :return: returns relative and absolute spread
    """

    atm_index = chain.options["gamma"].argmax()

    b = chain.options.loc[atm_index, "bid"]
    a = chain.options.loc[atm_index, "ask"]

    if a == b == 0:
        return 0, 0

    return 1 - b / a, a - b


# </editor-fold>

# <editor-fold desc="Option Framework">

# todo TEST!

class Option:

    def __init__(self, opt_type, expiration, strike, bid, ask,
                 delta, gamma, theta, vega, rho, iv):

        """
        :param opt_type:
        :param expiration: as string
        :param strike:

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

    @staticmethod
    def from_row(df):
        opt_type = df["contract"]
        expiration = df["expiration"]
        strike = df["strike"]
        bid = df["bid"]
        ask = df["ask"]
        delta = df["delta"]
        gamma = df["gamma"]
        vega = df["vega"]
        theta = df["theta"]
        rho = df["rho"]
        iv = df["iv"]
        return Option(opt_type, expiration, strike, bid, ask, delta, gamma, theta, vega, rho, iv)


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

    def get_risk(self):
        return self.bid


# todo TEST!

class Position:

    def __init__(self, asset, quantiy):

        self.asset = asset
        self.quantity = quantiy
        self.cost = self.get_cost()
        self.greeks = DDict({"delta": 0, "gamma": 0, "theta": 0, "vega": 0, "rho": 0})
        self.set_greeks()
        self.risk = self.get_risk()
        self.bpr = self.get_bpr()
        self.rom = self.get_rom()
        self.max_profit = self.get_max_profit()
        self.break_even = self.get_break_even()

    def __repr__(self):
        return self.repr()

    def repr(self, t=0):
        indent = "\t"*t
        return f'{indent}{self.quantity:+d} {self.asset} for a cost of {self.cost}' \
               f'\n' \
               f'{indent}\tGreeks:   ' \
               f'Δ = {self.greeks["delta"]}   ' \
               f'Γ = {self.greeks["gamma"]}   ' \
               f'ν = {self.greeks["vega"]}    ' \
               f'Θ = {self.greeks["theta"]}   ' \
               f'ρ = {self.greeks["rho"]}     ' \
               f'\n' \
               f'{indent}\tMax risk: {self.risk}   BPR: {self.bpr}' \
               f'\n' \
               f'\n' \
               f'{indent}       Break even on expiry: {self.break_even}' \
               f'\n' \
               f'{indent}                 Max profit: {self.max_profit}' \
               f'\n' \
               f'{indent}Return on margin (TP @ 50%): {self.rom}' \
               f'\n'

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

        return 0

    # <editor-fold desc="TODO">
    def get_bpr(self):
        """ TODO
        # http://tastytradenetwork.squarespace.com/tt/blog/buying-power

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
        ...

    def get_rom(self):
        pass

    def get_break_even(self):
        pass

    def get_max_profit(self):
        pass
    # </editor-fold>

    def get_cost(self):
        if self.quantity > 0:
            c = self.quantity * self.asset.ask
        else:
            c = -self.quantity * self.asset.bid
        if type(self.asset) is Option:
            return c * 100
        return c

    def change_quantity_to(self, x):
        self.quantity = x
        self.cost = self.get_cost()
        self.set_greeks()

    def add_x(self, x):
        self.change_quantity_to(self.quantity + x)

    @staticmethod
    def row_to_position(df, m=1):
        """

        :param df: dataframe with 1 row containing an option
        :param m: mulitplier for position size
        :return: Position from df with m*x size where x is -1 or 1
        """
        q = 1 if df["direction"] == "long" else -1
        return Position(Option.from_row(df), m * q)


# todo TEST!
class CombinedPosition:

    def __init__(self, pos_dict, u_bid, u_ask):
        """

        :param pos_dict:  (option symbol/stock ticker): position
        """
        self.pos_dict = pos_dict
        self.cost = self.get_cost()
        self.greeks = self.get_greeks()
        self.risk = self.get_risk()
        self.bpr = self.get_bpr()
        self.break_even = self.get_break_even()
        self.max_profit = self.get_max_profit()
        self.rom = self.get_rom()
        self.u_bid = u_bid
        self.u_ask = u_ask

        ticker = ''.join([i for i in list(self.pos_dict.values())[0].name[:6] if not i.isdigit()])

        self.underlying = ticker

        if not any(type(pos.asset) is Stock for pos in self.pos_dict.values()):
            self.stock = Stock(ticker, self.u_ask, self.u_ask)
            self.add_asset(self.stock, 0)
        else:
            self.stock = self.pos_dict[self.underlying]

    def __repr__(self):
        return self.repr()

    def repr(self, t=0):
        """
        :return: string representation
        """
        indent = "\t"*t
        s = ""
        for key, val in self.pos_dict:
            s += f'{indent}{val.repr(t=t+1)}'
        return f'{indent}{s}' \
               f'{indent}at a cost of {self.cost}' \
               f'\n' \
               f'{indent}\tCumulative Greeks:   ' \
               f'Δ = {self.greeks["delta"]}   ' \
               f'Γ = {self.greeks["gamma"]}   ' \
               f'ν = {self.greeks["vega"]}    ' \
               f'Θ = {self.greeks["theta"]}   ' \
               f'ρ = {self.greeks["rho"]}     ' \
               f'\n' \
               f'{indent}\tMax risk: {self.risk}   BPR: {self.bpr}' \
               f'\n' \
               f'\n' \
               f'{indent}       Break even on expiry: {self.break_even}' \
               f'\n' \
               f'{indent}                 Max profit: {self.max_profit}' \
               f'\n' \
               f'{indent}Return on margin (TP @ 50%): {self.rom}' \
               f'\n'

    # TODO TEST!!!
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
                        stock.add_x(-min(to_cover * 100, long_q))
                        to_cover -= min(to_cover, long_q / 100)

                        # todo update risk
                        # -short premium ...

                    # </editor-fold>

                if short_pos.asset.opt_type == "p":  # can only cover short puts with short stock

                    # <editor-fold desc="cover with short stock">

                    # is there any short stock?
                    if stock.quantity < 0:
                        short_stock_q = -stock.quantity

                        stock.add_x(min(to_cover * 100, short_stock_q))
                        stock_in_shorts = [s for s in shorts if type(s) is Stock][0]
                        stock_in_shorts.asset.add_x(min(to_cover * 100, short_stock_q))

                        to_cover -= min(to_cover, short_stock_q / 100)

                        # todo update risk
                        # -short premium...

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
                    risk += longs_w_cov_score[0][0].risk + abs(
                        short_pos.asset.strike - longs_w_cov_score[0][0].strike)

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
                    stock.add_x(min(to_cover * 100, -short_pos.asset.quantity))

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
        """
        :param asset: asset name, ticker for stock
        :param quantity:
        :return:
        """
        if asset in self.pos_dict:
            self.pos_dict[asset].add_x(quantity)
        else:
            self.pos_dict[asset] = Position(asset, quantity)

        self.cost = self.get_cost()
        self.greeks = self.get_greeks()

    def add_position(self, position):
        if position in self.pos_dict.values():
            self.pos_dict[position.asset.name].quantity += position.quantity
        else:
            self.pos_dict[position.asset.name] = position

    def add_positions(self, positions):
        for position in positions:
            self.add_position(position)

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

    @staticmethod
    def combined_pos_from_df(df, u_bid, u_ask):

        positions_to_add = []

        for index, row in df.iterrows():
            positions_to_add.append(Position.row_to_position(row))

        comb_pos = CombinedPosition(dict(), u_bid, u_ask)
        comb_pos.add_positions(positions_to_add)
        return comb_pos

    def get_bpr(self):
        return -1

    def get_break_even(self):
        return -1

    def get_max_profit(self):
        return -1

    def get_rom(self):
        return -1


class OptionChain:
    """
    methods return list of options that adhere to the given filter arguments
    """

    # <editor-fold desc="misc">
    cols = ["name", "strike", "bid", "ask", "mid", "last", "volume", "OI", "IV",
            "delta", "gamma", "vega", "theta", "rho", "contract", "expiration", "direction"]

    def __init__(self, chain=None, chain_dict=None, ):

        if chain is not None:
            chain.reset_index(inplace=True, drop=True)

            if type(chain) is pd.Series:
                self.options = chain.to_frame().T
                self.options.columns = OptionChain.cols
            if type(chain) is pd.DataFrame:
                self.options = chain

            if not chain.empty:
                self.expirations = sorted(self.options["expiration"].unique().tolist())
                self.ticker = self.options.at[0, "name"][:6]
                self.ticker = ''.join([i for i in self.ticker if not i.isdigit()])
            else:
                self.expirations = []
                self.ticker = None

            _debug(("-"*73) + "Chain after filtering" + ("-" * 73))
            _debug("Expirations:", self.expirations, "\nLength:", len(self.options), "\n", self.options.head(5),"\n...")

        else:
            self.expirations = list(chain_dict.keys())

            self.ticker = chain_dict[self.expirations[0]].puts.loc[0, "name"][:6]
            self.ticker = ''.join([i for i in self.ticker if not i.isdigit()])

            contract_list = list()

            for expiration in self.expirations:
                chain_dict[expiration].puts.loc[:, "contract"] = "p"
                chain_dict[expiration].puts.loc[:, "expiration"] = expiration
                chain_dict[expiration].puts.loc[:, "direction"] = "long"

                chain_dict[expiration].calls.loc[:, "contract"] = "c"
                chain_dict[expiration].calls.loc[:, "expiration"] = expiration
                chain_dict[expiration].calls.loc[:, "direction"] = "long"

                contract_list.extend((chain_dict[expiration].puts, chain_dict[expiration].calls))

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

            OptionChain.cols = list(self.options.columns.values)

            self.options.style.format({"OI": "{:7d}",
                                       "volume": "{:7d"})  # .background_gradient(cmap='Blues')

            # self.options.fillna(0, inplace=True)

            if online:
                self.save_as_file()

    def __repr__(self):
        return self.options.to_string()

    def head(self, n=5):
        return self.options.head(n)

    # </editor-fold>

    # <editor-fold desc="Type">

    def puts(self):
        _debug("Filter for puts")
        f = self.options.loc[self.options['contract'] == "p"]
        if type(f) is pd.Series:
            f = f.to_frame()
        return OptionChain(f)

    def calls(self):
        _debug("Filter for calls")
        f = self.options.loc[self.options['contract'] == "c"]
        if type(f) is pd.Series:
            f = f.to_frame()
        return OptionChain(f)

    def long(self):
        _debug("Filter for longs")
        f = self.options.loc[self.options['direction'] == "long"]
        if type(f) is pd.Series:
            f = f.to_frame()
        return OptionChain(f)

    def short(self):
        _debug("Filter for shorts")
        f = self.options.loc[self.options['direction'] == "short"]
        if type(f) is pd.Series:
            f = f.to_frame()
        return OptionChain(f)

    # </editor-fold>

    # <editor-fold desc="Expiry">
    def expiration_range(self, lower_dte, upper_dte):
        """

        :param lower_dte:
        :param upper_dte:
        :return: all options expiring between (inclusive) lower_dte and upper_dte
        """
        dates = pd.date_range(start=datetime.now() + timedelta(days=lower_dte),
                              end=datetime.now() + timedelta(days=upper_dte),
                              normalize=True)
        # todo works?
        f = self.options.loc[self.options['expiration'] in dates]
        if type(f) is pd.Series:
            f = f.to_frame()
        return OptionChain(f)

    def expiration_before(self, exp_date):
        """

        :param exp_date: date as string YYYY-MM-DD
        :return:
        """
        f = self.options.loc[self.options['expiration'] <= exp_date]  # works because dates are in ISO form
        if type(f) is pd.Series:
            f = f.to_frame()
        return OptionChain(f)

    def expiration_date(self, exp_date):
        """
        Return options having this exact expiration date
        :param exp_date: as string YYYY-MM-DD
        :return:
        """
        _debug("Exp date:", exp_date)
        f = self.options.loc[[self.options['expiration'] == exp_date]]
        if type(f) is pd.Series:
            f = f.to_frame()
        return OptionChain(f)

    def expiration_close_to_dte(self, dte):
        """

        :param dte:
        :return:
        """
        _debug("Filter for expiration close to", dte, "dte")
        f = self.options.loc[self.options['expiration'] == get_closest_date(self.expirations, dte)]
        if type(f) is pd.Series:
            f = f.to_frame()
        return OptionChain(f)

    def expiration_next(self, i=0):
        """
        get i-th expiration's puts & calls
        :param i:
        :return:
        """
        _debug("Filter for expiration: ", self.expirations[i])
        f = self.options.loc[self.options['expiration'] == self.expirations[i]]
        if type(f) is pd.Series:
            f = f.to_frame()
        return OptionChain(f)

    # </editor-fold>

    # <editor-fold desc="Greeks">

    def greek(self, g, lower=0, upper=1):
        f = self.options.loc[(lower <= self.options[g]) & (upper >= self.options[g])]
        if type(f) is pd.Series:
            f = f.to_frame()
        return OptionChain(f)

    def greek_close_to(self, greek, d):
        f = self.options.loc[min_closest_index(self.options[greek].to_list(), d)]
        if type(f) is pd.Series:
            f = f.to_frame().T  # TODO ???
        return OptionChain(f)

    # <editor-fold desc="sub greeks">
    def delta_range(self, lower=-1, upper=1):
        return self.greek("delta", lower, upper)

    def delta_close_to(self, d):
        return self.greek_close_to("delta", d)

    def gamma_range(self, lower=-1, upper=1):
        return self.greek("gamma", lower, upper)

    def gamma_close_to(self, d):
        return self.greek_close_to("gamma", d)

    def theta_range(self, lower=-1, upper=1):
        return self.greek("theta", lower, upper)

    def theta_close_to(self, d):
        return self.greek_close_to("theta", d)

    def vega_range(self, lower=-1, upper=1):
        return self.greek("vega", lower, upper)

    def vega_close_to(self, d):
        return self.greek_close_to("vega", d)

    def rho_range(self, lower=-1, upper=1):
        return self.greek("rho", lower, upper)

    # </editor-fold>
    # </editor-fold>

    # <editor-fold desc="Moneyness strike">
    def strike_n_itm_otm(self, n, moneyness, contract_default="c"):
        """
        :param contract_default:
        :param moneyness: OTM / ITM; string
        :param n:
        :return: put that is n strikes OTM
        """

        # puts or calls?
        contracts = self.options["contract"].unique().tolist()

        prefiltered = self.options

        if len(contracts) > 1:
            _warn("Filter for contract type before filtering on OTM strike!")

            # filter for contracts first
            prefiltered = self.puts() if contract_default == "p" else self.calls()
            contract_type = contract_default
        else:
            contract_type = contracts[0]

        # where is ATM? -> highest gamma
        atm_index = prefiltered["gamma"].argmax()
        if contract_type == "c":

            if moneyness == "ITM":
                f = prefiltered.iloc[max(atm_index - n, 0), :]
            if moneyness == "OTM":
                f = prefiltered.iloc[min(atm_index + n, len(prefiltered)), :]

        if contract_type == "p":

            if moneyness == "ITM":
                f = prefiltered.iloc[min(atm_index + n, len(prefiltered)), :]
            if moneyness == "OTM":
                f = prefiltered.iloc[max(atm_index - n, 0), :]

        if type(f) is pd.Series:
            f = f.to_frame()

        return OptionChain(f)

    def n_itm_strike_put(self, n):
        return self.strike_n_itm_otm(n, "ITM", contract_default="p")

    def n_otm_strike_put(self, n):
        return self.strike_n_itm_otm(n, "OTM", contract_default="p")

    def n_itm_strike_call(self, n):
        return self.strike_n_itm_otm(n, "ITM")

    def n_otm_strike_call(self, n):
        return self.strike_n_itm_otm(n, "OTM")

    # </editor-fold>

    # <editor-fold desc="IV">
    def iv_range(self, lower=0, upper=10000):
        f = self.options.loc[(lower <= self.options["IV"]) & (upper >= self.options["IV"])]
        if type(f) is pd.Series:
            f = f.to_frame()
        OptionChain(f)

    def iv_close_to(self, d):
        f = self.options.loc[min_closest_index(self.options["IV"].to_list(), d)]
        if type(f) is pd.Series:
            f = f.to_frame()
        return OptionChain(f)

    # </editor-fold>

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

    # </editor-fold>

    # <editor-fold desc="Strats">


# </editor-fold>

# <editor-fold desc="Strats">
class EnvContainer:

    def __init__(self, env, chain, u_bid, u_ask):
        self.env = env
        self.chain = chain
        self.u_bid = u_bid
        self.u_ask = u_ask


class OptionStrategy:
    """
    defines criteria for chosing options from a chain/stock to form a position

    Usage:

    longCall = LongCall(threshold=0.5)
    longCall.get_positions(env, chain, u_bid, u_ask)
    print(longCall)

    OR

    longCall = LongCall(env=env, chain=chain, u_bid=u_bid, u_ask=u_ask)
    print(longCall)

    """

    def __init__(self, name, position_builder, conditions, hints, tp_perc, sl_perc,
                 threshold=0.3, positions=None, env=None):
        self.name = name
        self.position_builder = position_builder  # name of the method that returns the position
        self.positions = positions
        self.conditions = conditions

        self.close_dte = 21
        self.close_perc = 50

        self.greek_exposure = None      # set after environment check
        self.close_date = None
        self.p50 = None

        self.hints = hints
        self.tp_percentage = tp_perc
        self.sl_percentage = sl_perc
        self.threshold = threshold

        if env is not None:
            self.get_positions(env)

    def __repr__(self):
        return self.name

    def __str__(self):
        return self.repr()

    def repr(self, t=0):

        indent = "\t"*t  # todo add greek exp in pretty

        if self.positions is None:
            return f'{indent}Empty {self.name}'

        return f'{indent}{self.name} for {self.positions.underlying}:' \
               f'\n' \
               f'{indent}{self.positions.repr(t=t+1)}' \
               f'\n' \
               f'{indent}Greek exposure:' \
               f'\n' \
               f'{indent}Δ = {self.greek_exposure["delta"]}' \
               f'\n' \
               f'{indent}Γ = {self.greek_exposure["gamma"]}' \
               f'\n' \
               f'{indent}ν = {self.greek_exposure["vega"]}' \
               f'\n' \
               f'{indent}Θ = {self.greek_exposure["theta"]}' \
               f'\n' \
               f'{indent}ρ = {self.greek_exposure["rho"]}' \
               f'\n' \
               f'\n' \
               f'{indent} Close by: {date_to_european_str(self.close_date)} ({date_to_dte(self.close_date)} DTE)' \
               f'\n' \
               f'{indent}      P50: {self.p50*100} %' \
               f'\n' \
               f'{indent}Stop loss: {self.sl_percentage/100 * self.positions.risk} $' \
               f'\n'

    def _test_environment(self, env):
        """

        :return: score from -1 to 1;
        1 if all conditions from self.conditions are fully fullfilled
        0 if application of strat is neutral
        -1 if application is unadived
        """
        try:
            score = 0
            for key, val in self.conditions.items():
                tmp = 0
                for entry in val:
                    tmp = max(entry(env[key]), tmp)
                    _debug(f'Testing {self.name} for: {key} = {entry.__name__}\n'
                           f'\t      {key}: {env[key]:.5f}\n'
                           f'\t{key} Score: {tmp:.5f}')
                score += tmp
            return score / len(self.conditions.keys())
        except KeyError as e:
            _warn(f'Key "{e}" was not found in environment "{env.keys()}" '
                          f'but was requested by {self.name}')
            return -1

    def _set_greek_exposure(self):
        if self.positions:
            # todo does this make sense???
            self.greek_exposure = {
                "delta": "long price" if self.positions.greeks["delta"] >= 0 else "short price",
                "gamma": "long" if self.positions.greeks["gamma"] >= 0 else "short",
                "vega": "long volatility" if self.positions.greeks["vega"] >= 0 else "short volatility",
                "theta": "short time decay" if self.positions.greeks["theta"] >= 0 else "long time decay (bad)",
                "rho": "long" if self.positions.greeks["rho"] >= 0 else "short"
            }
        else:
            _warn("Cannot set strategy greeks because no positions are existing yet")

    def get_positions(self, env):
        if type(env) is not EnvContainer:
            _debug(f'Wrong container type supplied for {self.name}: {type(env)}')
            return
        e = self._test_environment(env.env)
        if e >= self.threshold:
            _debug(f'Threshold of {self.threshold} for {self.name} was reached: {e} > {self.threshold}')
            self.positions = self.position_builder(env.chain, env.u_bid, env.u_ask)
            self._set_greek_exposure()
            self._set_close_date()
            # todo set p50
        else:
            _debug(f'Threshold of {self.threshold} for {self.name} was not reached: {e:.5f} < {self.threshold}')

    def _set_close_date(self):
        if self.positions:

            # get earliest exp date of position
            min_dte = 10000
            for pos in self.positions.values():
                if type(pos.asset) is Option:
                    if pos.asset.dte < min_dte:
                        min_dte = pos.asset.dte

            #  close after: close_dte days OR close perc of expiration, whichever is earlier
            self.close_date = dte_to_date(min(min_dte - self.close_dte, int(min_dte / (self.close_perc/100))))
        else:
            _warn("Cannot set strategy close date because no positions are existing yet")


class DummyStrat(OptionStrategy):

    def __init__(self, threshold=0.3):
        self.name = "dummy strategy"
        self.positions = CombinedPosition(...)
        self.conditions = {
            "IV":                   [high_ivr],  # high_ivr, put functions here that return values from 0 to 1 or from -1 to 1
            "IV outlook":           [extreme],

            "RSI20d":               [low_rsi],
            "short term outlook":   [bullish, neutral],
            "mid term outlook":     [neutral],
            "long term outlook":    [bearish],
        }
        self.greek_exposure = {
            "delta": ...,  # long / profit of rise // short / profit of decline
            "gamma": ...,  # long / profit of rise // short / profit of decline
            "vega": ...,  # long / profit of rise // short / profit of decline
            "theta": ...,  # long / profit of rise // short / profit of decline
            "rho": ...,  # long / profit of rise // short / profit of decline
        }
        self.close_date = ""
        self.p50 = 0
        self.hints = ["", ""]
        self.tp_percentage = 0
        self.sl_percentage = 0

        super().__init__(self.name,
                         self.position_builder,
                         self.conditions,
                         self.hints,
                         self.tp_percentage,
                         self.sl_percentage,
                         threshold=threshold)


class LongCall(OptionStrategy):
    ...


class LongPut(OptionStrategy):
    ...


class CoveredCall(OptionStrategy):

    name = "Covered Call"

    def __init__(self, env=None, threshold=0.3):
        """
        :param threshold: test_env must be greater than treshold in order to start building the position
        """
        self.threshold = threshold
        self.position_builder = self._get_tasty_variation
        self.conditions = {
            "IV": [high_ivr],  # high_ivr, put functions here that return values from 0 to 1 or from -1 to 1

            "RSI20d": [low_rsi],

            "short term outlook": [neutral, bullish],
            "mid term outlook": [neutral, bullish],
        }

        self.hints = ["Always set a take profit!", "It's just money :)",
                      "When do we close Covered Calls?\nWe close covered calls when the stock price has gone well past"
                      " our short call, as that usually yields close to max profit. We may also consider closing a "
                      "covered call if the stock price drops significantly and our assumption changes\nWhen do we "
                      "manage Covered Calls?\nWe roll a covered call when our assumption remains the same (that the "
                      "price of the stock will continue to rise). We look to roll the short call when there is little "
                      "to no extrinsic value left. For instance, if the stock price remains roughly the same as when "
                      "we executed the trade, we can roll the short call by buying back our short option, and selling "
                      "another call on the same strike in a further out expiration. We will also roll our call down if "
                      "the stock price drops. This allows us to collect more premium, and reduce our max loss & "
                      "breakeven point. We are always cognizant of our current breakeven point, and we do not roll our "
                      "call down further than that. Doing so can lock in a loss if the stock price actually comes back "
                      "up and leaves our call ITM."]

        self.tp_percentage = 50
        self.sl_percentage = 100

        self.close_dte = 21
        self.close_perc = 50

        super().__init__(self.name,
                         self.position_builder,
                         self.conditions,
                         self.hints,
                         self.tp_percentage,
                         self.sl_percentage,
                         threshold=threshold,
                         env=env)

    def _get_tasty_variation(self, chain, u_bid, u_ask):
        """
        :param u_ask:
        :param u_bid:
        :param chain:
        :return: combined position(s)
        """
        df = chain \
            .calls() \
            .expiration_close_to_dte(45) \
            .delta_close_to(0.3)

        cp = CombinedPosition.combined_pos_from_df(df, u_bid, u_ask)
        cp.add_asset(cp.underlying, 100)

        return cp


#todo
class VerticalDebitSpread(OptionStrategy):

    name = "Vertical Debit Spread"

    def __init__(self, threshold=0.3):
        """
        :param threshold: test_env must be greater than treshold in order to start building the position
        """
        self.threshold = threshold
        self.position_builder = self._get_tasty_variation
        self.conditions = {
            "IV": [high_ivr],  # high_ivr, put functions here that return values from 0 to 1 or from -1 to 1

            "RSI20d": [low_rsi],

            "short term outlook": [bullish],
            "mid term outlook": [neutral],
            "long term outlook": [bearish],
        }

        self.hints = ["Always set a take profit!", "It's just money :)"]

        self.tp_percentage = 50
        self.sl_percentage = 100

        self.close_dte = 21
        self.close_perc = 50

        super().__init__(self.name,
                         self.position_builder,
                         self.conditions,
                         self.hints,
                         self.tp_percentage,
                         self.sl_percentage,
                         threshold=threshold)

    def _get_tasty_variation(self, chain, u_bid, u_ask):
        """
        :param u_ask:
        :param u_bid:
        :param chain:
        :return: combined position(s)
        """
        df = chain \
            .calls() \
            .expiration_close_to_dte(45) \
            .delta_close_to(0.325)

        cp = CombinedPosition.combined_pos_from_df(df, u_bid, u_ask)
        cp.add_asset(cp.underlying, 100)

        return cp


#todo
class VerticalCreditSpread(OptionStrategy):

    name = "Vertical Credit Spread"

    def __init__(self, threshold=0.3):
        """
        :param threshold: test_env must be greater than treshold in order to start building the position
        """
        self.threshold = threshold
        self.position_builder = self._get_tasty_variation
        self.conditions = {
            "IV": [high_ivr],  # high_ivr, put functions here that return values from 0 to 1 or from -1 to 1

            "RSI20d": [low_rsi],

            "short term outlook": [bullish],
            "mid term outlook": [neutral],
            "long term outlook": [bearish],
        }

        self.hints = ["Always set a take profit!", "It's just money :)"]

        self.tp_percentage = 50
        self.sl_percentage = 100

        self.close_dte = 21
        self.close_perc = 50

        super().__init__(self.name,
                         self.position_builder,
                         self.conditions,
                         self.hints,
                         self.tp_percentage,
                         self.sl_percentage,
                         threshold=threshold)

    def _get_tasty_variation(self, chain, u_bid, u_ask):
        """
        :param u_ask:
        :param u_bid:
        :param chain:
        :return: combined position(s)
        """
        df = chain \
            .calls() \
            .expiration_close_to_dte(45) \
            .delta_close_to(0.325)

        cp = CombinedPosition.combined_pos_from_df(df, u_bid, u_ask)
        cp.add_asset(cp.underlying, 100)

        return cp


#todo
class CalendarSpread(OptionStrategy):

    name = "Calendar Spread"

    def __init__(self, threshold=0.3):
        """
        :param threshold: test_env must be greater than treshold in order to start building the position
        """
        self.threshold = threshold
        self.position_builder = self._get_tasty_variation
        self.conditions = {
            "IV": [high_ivr],  # high_ivr, put functions here that return values from 0 to 1 or from -1 to 1

            "RSI20d": [low_rsi],

            "short term outlook": [bullish],
            "mid term outlook": [neutral],
            "long term outlook": [bearish],
        }

        self.hints = ["Roll back month closer to front month when assignment seems likely (max. 5 DTE) "
                      "to save the extrinsic value of the long option and even make gain when assigned!"]

        self.tp_percentage = 50
        self.sl_percentage = 100

        self.close_dte = 21
        self.close_perc = 50

        super().__init__(self.name,
                         self.position_builder,
                         self.conditions,
                         self.hints,
                         self.tp_percentage,
                         self.sl_percentage,
                         threshold=threshold)

    def _get_tasty_variation(self, chain, u_bid, u_ask):
        """
        :param u_ask:
        :param u_bid:
        :param chain:
        :return: combined position(s)
        """
        df = chain \
            .calls() \
            .expiration_close_to_dte(45) \
            .delta_close_to(0.325)

        cp = CombinedPosition.combined_pos_from_df(df, u_bid, u_ask)
        cp.add_asset(cp.underlying, 100)

        return cp


class DiagonalSpread(OptionStrategy):
    ...


class Butterfly(OptionStrategy):
    ...


class Condor(OptionStrategy):
    ...


class ShortStrangle(OptionStrategy):
    ...

# </editor-fold>


propose_strategies("AMC")
