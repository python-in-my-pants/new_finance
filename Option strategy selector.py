import urllib.request
from bs4 import BeautifulSoup
import yfinance as yf
import pandas as pd
from datetime import datetime, date, timedelta
import numpy as np
from math import sqrt
from option_greeks import get_greeks
import warnings
import pickle
from copy import deepcopy
from eu_option import EuroOption


pritn = print
online = True
get_chain = True
debug = True

ACCOUNT_BALANCE = 1471
DEFAULT_THRESHOLD = 0.3  # todo 0.3


# <editor-fold desc="Miscellaneous">
def _warn(msg):
    warnings.warn("\n\n\t" + msg + "\n", stacklevel=3)


if not online:
    _warn("OFFLINE!    " * 17)


def _debug(*args):
    if debug:
        print()
        print(*args)


class DDict(dict):
    """dot.notation access to dictionary attributes"""

    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__

    def __deepcopy__(self, memo=None):
        return DDict(deepcopy(dict(self), memo=memo))


class RiskWarning(Warning):
    pass
# </editor-fold>


# <editor-fold desc="Pandas settings">
pd.set_option("display.max_rows", None, "display.max_columns", None)
pd.set_option('display.width', 1000)
pd.set_option('display.max_colwidth', 500)
pd.set_option('display.float_format', lambda x: '%.5f' % x)
pd.options.mode.chained_assignment = None  # default='warn'
# </editor-fold>


# <editor-fold desc="Libor">
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


def get_risk_free_rate(annualized_dte):
    """

    :param annualized_dte:
    :return:
    """
    return libor_rates[min_closest_index(libor_expiries, annualized_dte * 365)]


libor_rates = get_libor_rates() if online else [0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01]
libor_expiries = (1, 7, 30, 61, 92, 182, 365)
# </editor-fold>


def get_sp500_tickers(exclude_sub_industries=('Pharmaceuticals',
                                              'Managed Health Care',
                                              'Health Care Services')):
    df = pd.read_html('https://en.wikipedia.org/wiki/List_of_S%26P_500_companies')[0]
    df = df[~df['GICS Sub-Industry'].isin(exclude_sub_industries)]
    return df['Symbol'].values.tolist()


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

    if not online:
        return 0, 0, 0

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


def get_option_chain(yf_obj):
    # contractSymbol  lastTradeDate  strike  lastPrice    bid    ask  change  percentChange   volume  openInterest
    # ... impliedVolatility  inTheMoney contractSize currency

    start = datetime.now()

    if not get_chain:
        return OptionChain.from_file(filename=yf_obj.ticker + "_chain")

    option_chain = DDict()
    ticker_data = yf_obj
    underlying_ask = float(yf_obj.info["ask"])
    expirations = [datetime.strptime(t, "%Y-%m-%d") for t in ticker_data.options]
    relevant_fields = ["name", "strike", "bid", "ask", "mid", "last", "volume", "OI",
                       "IV", "gearing", "P(ITM)", "delta", "gamma", "vega", "theta", "rho"]

    def delta_to_itm_prob(delta, contract_type):
        magic_number = 0.624
        a = sqrt(1 - (delta * 2 * magic_number - magic_number) ** 2) - sqrt(1 - magic_number ** 2)
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
        contracts["gearing"] = 0.0

        risk_free_interest = get_risk_free_rate(_dte)

        for index, contract in contracts.iterrows():
            contracts.loc[index, "mid"] = (contract["bid"] + contract["ask"]) / 2
            contract_price = contract["mid"]  # todo last or mid?
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
            if contract_price > 0:
                contracts.loc[index, "gearing"] = contracts.loc[index, "delta"] * underlying_ask / contract_price
            else:
                contracts.loc[index, "gearing"] = float('inf')

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


class StrategyConstraints:

    def __init__(self, defined_risk_only=True, avoid_events=True,
                 min_vol30=25000, min_oi30=5000, min_sinle_opt_vol=50,
                 strict_mode=False, assignment_must_be_possible=False):
        self.assignment_must_be_possible = assignment_must_be_possible
        self.strict_mode = strict_mode
        self.min_sinle_opt_vol = min_sinle_opt_vol
        self.min_oi30 = min_oi30
        self.min_vol30 = min_vol30
        self.avoid_events = avoid_events
        self.defined_risk_only = defined_risk_only


# type, positions, credit/debit, max risk, break even on exp, max prof, bpr, return on margin with 50% tp,
# close by date
# use monte carlo sim for P50,
def propose_strategies(ticker, strat_cons=StrategyConstraints()):
    # todo get short term price outlook (technical analysis, resistances from barchart.com etc)

    # todo supply custom environment (like outlooks)

    # todo supply max risk limit, min gain, min pop, min p50, min expected return (0?) per position

    print(f'Settings:\n\n'
          f'          Defined risk only: {strat_cons.defined_risk_only}\n'
          f'               Avoid events: {strat_cons.avoid_events}\n'
          f'                 Min vol 30: {strat_cons.min_vol30}\n'
          f'   Min single option volume: {strat_cons.min_sinle_opt_vol}\n'
          f'       Min open interest 30: {strat_cons.min_oi30}\n'
          f'                Strict mode: {strat_cons.strict_mode}\n'
          f'Assignment must be possible: {strat_cons.assignment_must_be_possible}\n')

    # <editor-fold desc="Get info">
    if online:
        yf_obj = yf.Ticker(ticker)
        u_price = yf_obj.info["ask"]
        u_bid = yf_obj.info["bid"]
        u_ask = yf_obj.info["ask"]
        if "sector" in yf_obj.info.keys():
            sector = yf_obj.info["sector"]
        else:
            sector = ""
    else:
        class Dummy:
            def __init__(self):
                self.ticker = ticker
                self.info = DDict({"bid": -1, "ask": -1})

        yf_obj = Dummy()
        u_price = 10
        u_bid = 9
        u_ask = 10
        sector = ""

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
        ivr, vol30, oi30, next_earnings, rsi20 = 9.7, 26000, 7000, "01/01/30", 50

    short_outlook, mid_outlook, long_outlook = get_underlying_outlook(ticker)

    # </editor-fold>

    print(f'Stock info:\n\n'
          f'                  Stock bid: {u_bid}\n'
          f'                  Stock ask: {u_ask}\n'
          f'                     Sector: {sector if sector else "N/A"}\n'
          f'                        IVR: {ivr}\n'
          f'                     RSI 20: {rsi20}\n'
          f'                     Vol 30: {vol30}\n'
          f'           Open interest 30: {oi30}\n'
          f'              Next earnings: {next_earnings}\n'
          f'  Short term outlook (20 d): {short_outlook}\n'
          f'    Mid term outlook (50 d): {mid_outlook}\n'
          f'  Long term outlook (100 d): {long_outlook}\n')

    def binary_events_present():

        """
            returns true if undefined binary events can occur,
            returns the date of the earliest event as datetime object if any otherwise
            else returns false
            """

        # don't trade healthcare
        if sector == "Healthcare":
            return True

        if next_earnings and next_earnings != "N/A" and strat_cons.avoid_events:
            ne = datetime.strptime(next_earnings, '%M/%d/%y').strftime("%Y-%m-%d")
            if ne >= datetime.today().strftime("%Y-%m-%d"):
                return ne

        # todo check for anticipated news

    def is_liquid():

        r = True

        if vol30 < strat_cons.min_vol30:
            _warn(f'Warning! Average daily options volume < {strat_cons.min_vol30}: {vol30}\n')
            if strat_cons.strict_mode:
                r = False

        if oi30 < strat_cons.min_oi30:
            _warn(f'Warning! Average open interest < {strat_cons.min_oi30}: {oi30}\n')
            if strat_cons.strict_mode:
                r = False

        put_spr_r, put_spr_abs = get_bid_ask_spread(next_puts, u_ask)
        call_spr_r, call_spr_abs = get_bid_ask_spread(next_calls, u_ask)

        if put_spr_r >= 10 and call_spr_r >= 10:
            _warn(
                f'Warning! Spread ratio is very wide: Puts = {put_spr_abs}, Calls = {call_spr_abs}\n')
            if strat_cons.strict_mode:
                r = False

        _debug(f'Liquidity:\n\n'
               f'30 day average option volume: {vol30:7d}\n'
               f'30 day average open interest: {oi30:7d}\n'
               f'      Put spread ratio (rel):   {int(put_spr_r * 100):3d} %\n'
               f'      Put spread ratio (abs):    {put_spr_abs:.2f}\n'
               f'     Call spread ratio (rel):   {int(call_spr_r * 100):3d} %\n'
               f'     Call spread ratio (abs):    {call_spr_abs:.2f}\n')

        return r

    def assingment_risk_tolerance_exceeded():

        if u_price * 100 > ACCOUNT_BALANCE / 2:
            s = f'Warning! Underlying asset price exceeds account risk tolerance! ' \
                f'Assignment would result in margin call!' \
                f' {u_price * 100} > {ACCOUNT_BALANCE / 2}'
            _warn(s)
            if strat_cons.assignment_must_be_possible:
                return

    # -----------------------------------------------------------------------------------------------------------------

    # <editor-fold desc="1 Binary events">
    binary_event_date = binary_events_present()

    if type(binary_event_date) is bool and binary_event_date:
        _warn("Warning! Underlying may be subject to undefined binary events!")
        if strat_cons.strict_mode:
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
        if strat_cons.strict_mode:
            return
    # </editor-fold>

    # <editor-fold desc="3 Price">
    assingment_risk_tolerance_exceeded()
    # </editor-fold>

    print("\nAll mandatory checks done! Starting strategy selection ...")

    # -----------------------------------------------------------------------------------------------------------------

    env = {
        "IV": ivr,
        "IV outlook": 0,

        "RSI20d": rsi20,

        "short term outlook": short_outlook,
        "mid term outlook": mid_outlook,
        "long term outlook": long_outlook,
    }

    env_con = EnvContainer(env, option_chain, u_bid, u_ask, ticker, strat_cons.min_sinle_opt_vol)

    print(LongCall(env_con))
    print(LongPut(env_con))

    print(CoveredCall(env_con))

    print(VerticalDebitSpread(env_con, opt_type="c"))
    print(VerticalCreditSpread(env_con, opt_type="c"))

    print(VerticalDebitSpread(env_con, opt_type="p"))
    print(VerticalCreditSpread(env_con, opt_type="p"))


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


def neutral_ivr(ivr):
    return neutral((ivr / 50) - 1)


def low_ivr(ivr):
    return -high_ivr(ivr)


def low_rsi(rsi):
    # todo which one is better?
    """if rsi <= 30:
        return rsi/30 - 1
    if rsi <= 70:
        return rsi
    else:
        return rsi/30 - (7/3)"""
    return -rsi / 50 + 1


def high_rsi(rsi):
    return -low_rsi(rsi)


def neutral(outlook):
    if outlook <= -0.5:
        return outlook
    if outlook <= 0:
        return 4 * outlook + 1
    if outlook <= 0.5:
        return -4 * outlook + 1
    else:
        return -outlook


def bullish(outlook):
    return outlook


def bearish(outlook):
    return -outlook


def extreme(outlook):
    if outlook <= 0:
        return -2 * outlook - 1
    else:
        return 2 * outlook - 1


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
        diff = abs(dte - date_str_to_dte(expiration))
        if diff < best_diff:
            best_diff = abs(dte - date_str_to_dte(expiration))
            best_exp = expiration
    return best_exp


def date_str_to_dte(d: str):
    """
    :param d: as string YYYY-MM-DD
    :return:
    """
    return abs((datetime.now().date() - exp_to_date(d)).days)


def datetime_to_dte(d: datetime):
    """
    :param d: as string datetime obj
    :return:
    """
    return abs((datetime.now().date() - d.date()).days)


def date_to_dte(d: date):
    """
    :param d: as string datetime obj
    :return:
    """
    return abs((datetime.now().date() - d).days)


def dte_to_date(dte: int):
    return datetime.now() + timedelta(days=dte)


def exp_to_date(expiration: str):
    return datetime.strptime(expiration, "%Y-%m-%d").date()


def date_to_european_str(d: str):
    return datetime.strptime(d, "%Y-%m-%d").strftime("%m.%d.%Y")


def datetime_to_european_str(d: datetime):
    return d.strftime("%d.%m.%Y")


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


def get_bid_ask_spread(chain, u_ask):
    """

    :param chain:
    :return: returns relative and absolute spread
    """

    # atm_index = chain.options["gamma"].argmax()
    atm_index = min_closest_index(list(chain.options["strike"]), u_ask)

    b = chain.options.loc[atm_index, "bid"]
    a = chain.options.loc[atm_index, "ask"]

    if a == b == 0:
        return 0, 0

    return 1 - b / a, a - b


def get_extrinsic_at_dte(dte_now, extr_now, dte_then):
    # dte now -> extr now

    def decay_func(y):  # TODO add in piecewise functions
        return np.piecewise(y, [], [])

    # decay func(then) * factor
    # = decay func(then) * (extr now / f_dte_now)
    # = decay func(then) * (extr now / decay_func(dte_now)

    return decay_func(dte_then) * extr_now / decay_func(dte_now)


# </editor-fold>

# <editor-fold desc="Option Framework">

class Option:

    def __init__(self, name, opt_type, expiration, strike, bid, ask, vol,
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

        self.name = name

        self.opt_type, self.expiration, self.strike = opt_type, expiration, strike
        self.bid = bid
        self.ask = ask
        self.vol = vol

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
    def from_row(row):
        name = row["name"]
        opt_type = row["contract"]
        expiration = row["expiration"]
        strike = row["strike"]
        bid = row["bid"]
        ask = row["ask"]
        vol = row["volume"]

        delta = row["delta"] if row["direction"] == "long" else -row["delta"]
        gamma = row["gamma"] if row["direction"] == "long" else -row["gamma"]
        vega = row["vega"] if row["direction"] == "long" else -row["vega"]
        theta = row["theta"] if row["direction"] == "long" else -row["theta"]
        rho = row["rho"] if row["direction"] == "long" else -row["rho"]

        iv = row["IV"]
        return Option(name, opt_type, expiration, strike, bid, ask, vol, delta, gamma, theta, vega, rho, iv)

    @staticmethod
    def from_df(df: pd.DataFrame):

        df.reset_index(inplace=True)
        name = df.loc[0, "name"]
        opt_type = df.loc[0, "contract"]
        expiration = df.loc[0, "expiration"]
        strike = df.loc[0, "strike"]
        bid = df.loc[0, "bid"]
        ask = df.loc[0, "ask"]
        vol = df.loc[0, "volume"]

        delta = df.loc[0, "delta"]
        gamma = df.loc[0, "gamma"]
        vega = df.loc[0, "vega"]
        theta = df.loc[0, "theta"]
        rho = df.loc[0, "rho"]
        iv = df.loc[0, "IV"]
        return Option(name, opt_type, expiration, strike, bid, ask, vol, delta, gamma, theta, vega, rho, iv)


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


class Position:

    def __init__(self, asset, quantiy, underlying_price):

        self.asset = asset
        self.quantity = quantiy
        self.underlying_price = underlying_price

        self.greeks = DDict({"delta": 0, "gamma": 0, "theta": 0, "vega": 0, "rho": 0})
        self.set_greeks()

        self.cost = self.get_cost()
        self.risk = self.get_risk()
        self.max_profit = self.get_max_profit()
        self.bpr = self.get_bpr()
        self.rom = self.get_rom()
        self.break_even = self.get_break_even_at_exp()

        # todo
        self.p50 = 0

    def __repr__(self):
        return f'{self.quantity:+d} {self.asset} for a cost of {self.cost:.2f} $'

    def __str__(self):
        return self.repr()

    def short(self, t=0):
        indent = "\t" * t
        return f'{indent}{self.quantity:+d} {self.asset} = {self.cost:.2f} $'

    def repr(self, t=0):
        indent = "\t" * t
        return f'{indent}{self.quantity:+d} {self.asset} for a cost of {self.cost:.2f} $' \
            f'\n' \
            f'\n' \
            f'{indent}\tGreeks:   ' \
            f'Δ = {self.greeks["delta"]:+.5f}   ' \
            f'Γ = {self.greeks["gamma"]:+.5f}   ' \
            f'ν = {self.greeks["vega"]:+.5f}    ' \
            f'Θ = {self.greeks["theta"]:+.5f}   ' \
            f'ρ = {self.greeks["rho"]:+.5f}     ' \
            f'\n' \
            f'{indent}\tMax risk: {self.risk:.2f} $' \
            f'\n' \
            f'{indent}\tBPR:      {self.bpr:.2f} $' \
            f'\n' \
            f'\n' \
            f'{indent}       Break even on expiry: {self.break_even:.2f} $' \
            f'\n' \
            f'{indent}                 Max profit: {self.max_profit:.2f} $' \
            f'\n' \
            f'{indent}Return on margin (TP @ 50%): {self.rom:.2f}' \
            f'\n'

    def set_greeks(self):

        # long call -1
        self.greeks.delta = self.quantity * self.asset.greeks["delta"]
        self.greeks.gamma = self.quantity * self.asset.greeks["gamma"]
        self.greeks.vega = self.quantity * self.asset.greeks["vega"]
        self.greeks.theta = self.quantity * self.asset.greeks["theta"]
        self.greeks.rho = self.quantity * self.asset.greeks["rho"]

        if type(self.asset) is Option:
            self.greeks.delta *= 100
            self.greeks.gamma *= 100
            self.greeks.vega *= 100
            self.greeks.theta *= 100
            self.greeks.rho *= 100

        """if self.quantity < 0:
            self.greeks.delta *= -1
            self.greeks.gamma *= -1
            self.greeks.vega *= -1
            self.greeks.theta *= -1
            self.greeks.rho *= -1"""

    def get_risk(self):

        # long stock / option
        if self.quantity > 0:
            return self.asset.ask * self.quantity * (100 if type(self.asset) is Option else 1)

        # short options / stocks
        if self.quantity < 0:

            if type(self.asset) is Option and self.asset.opt_type == "p":
                return (-self.asset.bid + self.asset.strike) * -self.quantity

            return float('inf')

        return 0

    def get_bpr(self):

        # http://tastytradenetwork.squarespace.com/tt/blog/buying-power
        # https://support.tastyworks.com/support/solutions/articles/43000435243-short-stock-or-etfs-

        if self.quantity > 0:
            return self.cost
        elif self.quantity == 0:
            return 0
        elif self.quantity < 0:

            if type(self.asset) is Option:
                a = (0.2 * self.underlying_price - self.asset.strike + self.underlying_price + self.asset.bid) \
                    * (-self.quantity) * 100
                b = self.asset.strike * 10
                c = 50 * (-self.quantity) + self.asset.bid * 100
                return max(a, b, c)

            if type(self.asset) is Stock:

                if self.underlying_price > 5:
                    return max(-self.quantity * 5, -self.quantity * self.underlying_price)
                else:
                    return max(-self.quantity * self.underlying_price, -self.quantity * 2.5)

        _warn("BPR incomplete, value is wrong!")
        return -1

    def get_cost(self):
        if self.quantity > 0:
            c = self.quantity * self.asset.ask
        else:
            c = self.quantity * self.asset.bid  # was -
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
    def row_to_position(df: pd.DataFrame, u_ask: float, m=1):
        """

        :param u_ask:
        :param df: dataframe with 1 row containing an option
        :param m: mulitplier for position size
        :return: Position from df with m*x size where x is -1 or 1
        """
        q = 1 if df["direction"] == "long" else -1
        return Position(Option.from_row(df), m * q, u_ask)

    def get_max_profit(self):
        if self.quantity == 0:
            return 0
        if type(self.asset) is Option:
            if self.quantity < 0:
                return -self.cost  # can buy back for 0 at best
            if self.quantity > 0:
                if self.asset.opt_type == "c":
                    return float('inf')
                if self.asset.opt_type == "p":
                    return self.asset.strike * 100
        if type(self.asset) is Stock:
            if self.quantity > 0:
                return float('inf')
            if self.quantity < 0:
                return -self.cost  # buy back for 0 at best
        _warn(f"Something went wrong in getting max profit of single position {self.__repr__()}")
        return -1

    def get_profit_dist_at_exp(self, max_strike: int):

        """
        profit distribution in cent stes starting from 0.00
        can be used to plot P/L
        :param max_strike:
        :return:
        """

        gains = [-self.cost for _ in range(max_strike*100)]

        if type(self.asset) is Stock:

            if self.quantity > 0:

                for strike in range(max_strike*100):
                    gains[strike] += strike/100 * self.quantity

                return gains

        elif type(self.asset) is Option:

            if self.asset.opt_type == "c":

                if self.asset.direction == "long":

                    for strike in range(max_strike*100):

                        if strike * 100 > self.asset.strike:
                            gains[strike] += (self.asset.strike - strike/100) * 100*self.quantity

                    return [x/100 for x in gains]

                if self.asset.direction == "short":

                    for strike in range(max_strike*100):

                        if strike*100 > self.asset.strike:
                            gains[strike] -= (self.asset.strike - strike/100) * 100*self.quantity

                    return [x/100 for x in gains]

            if self.asset.opt_type == "p":

                if self.asset.direction == "long":

                    for strike in range(max_strike*100):

                        if strike * 100 < self.asset.strike:
                            gains[strike] += (self.asset.strike - strike/100) * 100*self.quantity

                    return [x / 100 for x in gains]

                if self.asset.direction == "short":

                    for strike in range(max_strike * 100):
                        if strike * 100 < self.asset.strike:
                            gains[strike] -= (self.asset.strike - strike/100) * 100*self.quantity

                    return [x / 100 for x in gains]

    def get_profit_dist_at_date(self, max_strike: int, d: str):
        exp_profit_dist = self.get_profit_dist_at_exp(max_strike=max_strike)

        if type(self.asset) is Stock:
            return exp_profit_dist

        if type(self.asset) is Option:
            # account for theta
            # how many DTE left till exp?
            # do i have to use option model with forecasted theta iterating over potential u_prices to predict option
            # price at x dte?

            # us_option = EuroOption(217.58, 215, 0.05, 0.1, 40,
            #                        {'is_call': True, 'eu_option': False, 'sigma': 1})
            """
            use extrinsic value now and extrinsic decay curve to find extrinsic value at date
            use dte_then and forecasted decayed option price to iterate over strikes and calc
            option price at different prices of u in the future using EuroOption
            """
            dte_now = date_str_to_dte(self.asset.expiration)
            dte_then = (exp_to_date(self.asset.expiration) - exp_to_date(d)).days

            # forecasted extrinsic value of long option when short option expires
            if self.asset.opt_type == "p":
                current_intr = self.asset.strike - self.underlying_price
            if self.asset.opt_type == "c":
                current_intr = self.underlying_price - self.asset.strike

            current_intr = max(0, current_intr)

            current_extr_per_opt = self.asset.ask - current_intr
            # we can only forecast this for the current option at the current underlying price
            # forecasted_extr_per_opt = get_extrinsic_at_dte(dte_now, current_extr_per_opt, dte_then)

            risk_free_rate = get_risk_free_rate(dte_then/365.0)

            is_call = self.asset.opt_type == "c"

            for strike in range(max_strike * 100):

                future_price = \
                    EuroOption(strike/100.0, self.asset.strike, risk_free_rate, dte_then/365.0,
                               40,
                               {'is_call': is_call,
                                'eu_option': False,
                                'sigma': self.asset.iv}).price()

                if self.asset.direction == "long":
                    profit_at_strike = self.quantity * future_price * 100 - self.cost
                if self.asset.direction == "short":
                    profit_at_strike = -self.cost + self.quantity * future_price * 100

                exp_profit_dist[strike] = profit_at_strike

            return exp_profit_dist

    def get_rom(self):
        if self.bpr == 0:
            return 0
        return self.max_profit / self.bpr

    def get_break_even_at_exp(self):

        # quantity is irrelevant for break even of single position

        if type(self.asset) is Option:
            if self.quantity < 0:
                if self.asset.opt_type == "c":
                    return self.asset.strike - self.cost / 100  # mind you: cost is negative for short options
                if self.asset.opt_type == "p":
                    return self.asset.strike + self.cost / 100
            if self.quantity > 0:
                if self.asset.opt_type == "c":
                    return self.asset.strike + self.cost / 100
                if self.asset.opt_type == "p":
                    return self.asset.strike - self.cost / 100

        if type(self.asset) is Stock:
            if self.quantity == 0:
                return 0
            if self.quantity > 0:
                return self.asset.ask
            if self.quantity < 0:
                return self.asset.bid

        _warn(f"Something went wrong while getting break even for single position {self.__repr__()}")


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

            _debug(("-" * 73) + "Chain after filtering" + ("-" * 73))
            _debug("Expirations:", self.expirations, "\nLength:", len(self.options))
            _debug(self.options.head(5))

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
            f = f.to_frame().T
        return OptionChain(f)

    def calls(self):
        _debug("Filter for calls")
        f = self.options.loc[self.options['contract'] == "c"]
        if type(f) is pd.Series:
            f = f.to_frame().T
        return OptionChain(f)

    def long(self):
        _debug("Filter for longs")
        f = self.options.loc[self.options['direction'] == "long"]
        if type(f) is pd.Series:
            f = f.to_frame().T
        return OptionChain(f)

    def short(self):
        _debug("Filter for shorts")
        f = self.options.loc[self.options['direction'] == "short"]
        if type(f) is pd.Series:
            f = f.to_frame().T
        return OptionChain(f)

    # </editor-fold>

    # <editor-fold desc="Expiry">
    def expiration_range(self, lower_dte, upper_dte):
        """
        :param lower_dte:
        :param upper_dte:
        :return: all options expiring between (inclusive) lower_dte and upper_dte
        """
        _debug(f'Filter for expiration in range {lower_dte}-{upper_dte} DTE')
        dates = pd.date_range(start=datetime.now() + timedelta(days=lower_dte),
                              end=datetime.now() + timedelta(days=upper_dte),
                              normalize=True)
        # todo works?
        f = self.options.loc[self.options['expiration'] in dates]
        if type(f) is pd.Series:
            f = f.to_frame().T
        return OptionChain(f)

    def expiration_before(self, exp_date):
        """

        :param exp_date: date as string YYYY-MM-DD
        :return:
        """
        _debug(f'Filter for expiration before {exp_date}')

        f = self.options.loc[self.options['expiration'] <= exp_date]  # works because dates are in ISO form
        if type(f) is pd.Series:
            f = f.to_frame().T
        return OptionChain(f)

    def expiration_date(self, exp_date):
        """
        Return options having this exact expiration date
        :param exp_date: as string YYYY-MM-DD
        :return:
        """
        _debug("Filter for exp date:", exp_date)
        f = self.options.loc[[self.options['expiration'] == exp_date]]
        if type(f) is pd.Series:
            f = f.to_frame().T
        return OptionChain(f)

    def expiration_close_to_dte(self, dte):
        """

        :param dte:
        :return:
        """
        _debug("Filter for expiration close to", dte, "dte")
        f = self.options.loc[self.options['expiration'] == get_closest_date(self.expirations, dte)]
        if type(f) is pd.Series:
            f = f.to_frame().T
        return OptionChain(f)

    def expiration_next(self, i=0):
        """
        get i-th expiration's puts & calls
        :param i:
        :return:
        """
        _debug(f"Filter for {i}th expiration: ", self.expirations[i])
        f = self.options.loc[self.options['expiration'] == self.expirations[i]]
        if type(f) is pd.Series:
            f = f.to_frame().T
        return OptionChain(f)

    # </editor-fold>

    # <editor-fold desc="Greeks">

    def greek(self, g, lower=0, upper=1):
        _debug(f'Filter for {g} in range {lower}-{upper}')
        f = self.options.loc[(lower <= self.options[g]) & (upper >= self.options[g])]
        if type(f) is pd.Series:
            f = f.to_frame().T
        return OptionChain(f)

    def greek_close_to(self, greek, d):
        _debug(f'Filter for {greek} close to {d}')
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
    def _strike_n_itm_otm(self, n, u_ask, moneyness, contract_default="c"):
        """
        TODO when filtering for short before this, gamma gets wrong and atm gets wrong
        :param contract_default:
        :param moneyness: OTM / ITM; string
        :param n:
        :return: put that is n strikes OTM
        """

        # puts or calls?
        contracts = self.options["contract"].unique().tolist()

        prefiltered = self.options

        if len(contracts) > 1:
            # _warn("Filter for contract type before filtering on OTM strike!")

            # filter for contracts first
            prefiltered = self.puts() if contract_default == "p" else self.calls()
            contract_type = contract_default
        else:
            contract_type = contracts[0]

        _debug(f'Filter for {n} {moneyness} strike on {contract_type}')

        # where is ATM? -> delta closest to 0.5
        # atm_index = np.asarray([abs(abs(g)-0.5) for g in prefiltered.options["delta"]]).argmax()
        # atm_index = min_closest_index(list(prefiltered.options["delta"]), 0.5)
        atm_index = min_closest_index(list(prefiltered.options["strike"]), u_ask)

        _debug(f'Detected ATM strike at {prefiltered.options.loc[atm_index, "strike"]}')

        f = None
        if contract_type == "c":  # 5

            if moneyness == "ITM":
                f = prefiltered.options.iloc[max(atm_index - n - 1, 0), :]
            if moneyness == "OTM":
                f = prefiltered.options.iloc[min(atm_index + n + 1, len(prefiltered.options)), :]

        if contract_type == "p":  # 3

            if moneyness == "ITM":
                f = prefiltered.options.iloc[min(atm_index + n + 1, len(prefiltered.options)), :]
            if moneyness == "OTM":
                f = prefiltered.options.iloc[max(atm_index - n - 1, 0), :]

        if type(f) is pd.Series:
            f = f.to_frame().T

        if True:
            _debug(f'Filtering for moneyness strike returned this for {n} {moneyness} on {contract_type}:')
            _debug(f.head(5))

        return OptionChain(f)

    def n_itm_strike_put(self, n, u_ask):
        return self._strike_n_itm_otm(n, u_ask, "ITM", contract_default="p")

    def n_otm_strike_put(self, n, u_ask):
        return self._strike_n_itm_otm(n, u_ask, "OTM", contract_default="p")

    def n_itm_strike_call(self, n, u_ask):
        return self._strike_n_itm_otm(n, u_ask, "ITM")

    def n_otm_strike_call(self, n, u_ask):
        return self._strike_n_itm_otm(n, u_ask, "OTM")

    # </editor-fold>

    # <editor-fold desc="IV">
    def iv_range(self, lower=0, upper=10000):
        f = self.options.loc[(lower <= self.options["IV"]) & (upper >= self.options["IV"])]
        if type(f) is pd.Series:
            f = f.to_frame().T
        OptionChain(f)

    def iv_close_to(self, d):
        f = self.options.loc[min_closest_index(self.options["IV"].to_list(), d)]
        if type(f) is pd.Series:
            f = f.to_frame().T
        return OptionChain(f)

    # </editor-fold>

    # <editor-fold desc="p(ITM)">
    def itm_prob_close_to(self, p):
        _debug(f'Filter for P(ITM) close to {p}')
        f = self.options.loc[min_closest_index(self.options["P(ITM)"].to_list(), p)]
        if type(f) is pd.Series:
            f = f.to_frame().T  # TODO ???
        return OptionChain(f)

    def itm_prob_range(self, g, lower=0, upper=1):
        _debug(f'Filter for {g} in range {lower}-{upper}')
        f = self.options.loc[(lower <= self.options["P(ITM)"]) & (upper >= self.options["P(ITM)"])]
        if type(f) is pd.Series:
            f = f.to_frame().T
        return OptionChain(f)

    # </editor-fold>

    def volume(self, min_vol):
        _debug(f"Filter for volume > {min_vol}")
        f = self.options.loc[self.options['volume'] > min_vol]
        if type(f) is pd.Series:
            f = f.to_frame().T
        return OptionChain(f)

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

    def remove_by_name(self, used_chain):
        """
        Removes all elements from chain that share their name with the first element of used_chain
        :param used_chain:
        :return:
        """
        name = used_chain.options.loc[list(used_chain.options.index.values)[0], "name"]
        tmp_chain = self.options.drop(self.options[self.options["name"] == name].index)
        return OptionChain(chain=tmp_chain)


# todo TEST!
class CombinedPosition:

    def __init__(self, pos_dict, u_bid, u_ask, ticker):
        """
        :param pos_dict:  (option symbol/stock ticker): position
        """
        self.pos_dict = pos_dict
        self.coverage_dict = dict()
        """
        holds dict[short.asset.name] = {"long1name": "covering quantity of long1", 
                                        "long2name": "covering ..."}
        """

        self.u_bid = u_bid
        self.u_ask = u_ask
        self.underlying = ticker

        self.cost = -1
        self.risk = -1
        self.bpr = -1
        self.break_even = -1
        self.max_profit = -1
        self.rom = -1
        self.p50 = -1

        if self.pos_dict.values():
            self.update_status()

        if not any(type(pos.asset) is Stock for pos in self.pos_dict.values()):
            self.stock = Stock(ticker, self.u_ask, self.u_ask)
            self.add_asset(self.stock, 0, no_update=True)
        else:
            self.stock = self.pos_dict[self.underlying]

        self.greeks = self.get_greeks()

    def __repr__(self):
        return self.repr()

    def repr(self, t=0, short=True):
        indent = "\t" * t
        s = ""
        for key, val in self.pos_dict.items():
            if val.quantity != 0:
                if short:
                    s += f'{indent}{val.short(t=t + 1)}\n'
                else:
                    s += f'\n' \
                        f'{indent}Position {key}:' \
                        f'\n\n' \
                        f'{indent}{val.short(t=t + 1)}' \
                        f'\n' \
                        f'{indent}{"." * 300}' \
                        f'\n'

        return f'\n\n' \
            f'{s}' \
            f'\n' \
            f'{indent}   Cumulative strategy cost: {self.cost:.2f} $' \
            f'\n' \
            f'\n' \
            f'{indent}          Cumulative Greeks: ' \
            f'Δ = {self.greeks["delta"]:+.5f}   ' \
            f'Γ = {self.greeks["gamma"]:+.5f}   ' \
            f'ν = {self.greeks["vega"]:+.5f}    ' \
            f'Θ = {self.greeks["theta"]:+.5f}   ' \
            f'ρ = {self.greeks["rho"]:+.5f}     ' \
            f'\n' \
            f'{indent}                Theta/Delta: {self.greeks["theta"]/max(abs(self.greeks["delta"]),0.00001):+.5f}' \
            f'\n' \
            f'{indent}                   Max risk: {self.risk:.2f} $' \
            f'\n' \
            f'{indent}                        BPR: {self.bpr:.2f} $' \
            f'\n' \
            f'\n' \
            f'{indent}       Break even on expiry: {self.break_even:.2f} $' \
            f'\n' \
            f'{indent}                 Max profit: {self.max_profit:.2f} $' \
            f'\n' \
            f'{indent}Return on margin (TP @ 50%): {self.rom:.2f}' \
            f'\n\n'

    def detail(self, t=0):
        """
        :return: string representation
        """
        indent = "\t" * t
        s = ""
        for key, val in self.pos_dict.items():
            if val.quantity != 0:
                s += f'\n' \
                    f'{indent}Position {key}:' \
                    f'\n\n' \
                    f'{indent}{val.repr(t=t + 1)}' \
                    f'\n' \
                    f'{indent}{"." * 300}' \
                    f'\n'
        return f'\n' \
            f'{indent}{s}' \
            f'\n' \
            f'{indent}   Cumulative strategy cost: {self.cost:.2f} $' \
            f'\n' \
            f'\n' \
            f'{indent}          Cumulative Greeks: ' \
            f'Δ = {self.greeks["delta"]:+.5f}   ' \
            f'Γ = {self.greeks["gamma"]:+.5f}   ' \
            f'ν = {self.greeks["vega"]:+.5f}    ' \
            f'Θ = {self.greeks["theta"]:+.5f}   ' \
            f'ρ = {self.greeks["rho"]:+.5f}     ' \
            f'\n' \
            f'{indent}                Theta/Delta: {self.greeks["theta"] / self.greeks["delta"]:+.5f}' \
            f'\n' \
            f'{indent}                   Max risk: {self.risk:.2f} $' \
            f'\n' \
            f'{indent}                        BPR: {self.bpr:.2f} $' \
            f'\n' \
            f'\n' \
            f'{indent}       Break even on expiry: {self.break_even:.2f} $' \
            f'\n' \
            f'{indent}                 Max profit: {self.max_profit:.2f} $' \
            f'\n' \
            f'{indent}Return on margin (TP @ 50%): {self.rom:.2f}' \
            f'\n' \
            f'{indent}{"-" * 300}' \
            f'\n' \
            f'\n'

    def update_status(self):
        self.cost = self.get_cost()
        self.greeks = self.get_greeks()
        self.risk = self.get_risk()
        self.bpr = self.get_bpr()
        self.break_even = self.get_break_even()
        self.max_profit = self.get_max_profit()
        self.rom = self.get_rom()

    def cover_short_with(self, short_pos_name: str, long_pos_name: str, covering_long_q):
        if short_pos_name in self.coverage_dict.keys():
            if long_pos_name in self.coverage_dict[short_pos_name]:
                self.coverage_dict[short_pos_name][long_pos_name] += covering_long_q
            else:
                self.coverage_dict[short_pos_name][long_pos_name] = covering_long_q
        else:
            self.coverage_dict[short_pos_name] = {long_pos_name: covering_long_q}

    # TODO TEST!!!
    def get_risk(self):

        # TODO build coverage pairs as dict:
        #  cov_pairs[short_pos_name1] = [covering_1=A, covering_2=B, ...]
        #  cov_pairs[short_pos_name2] = [covering_1=B, covering_2=C, ...]
        #  and update it when options expire

        positions = deepcopy(list(self.pos_dict.values()))

        shorts = []
        longs = []
        stock = None

        for pos in positions:
            if pos.quantity < 0:
                shorts.append(pos)
            if pos.quantity > 0:
                longs.append(pos)
            if type(pos.asset) is Stock:
                stock = pos
                """if pos.quantity > 0:
                    shorts.append(stock)  # shorts is proxy for positions to cover, long stock must be covered"""

        # BEWARE all you do to stock if quantity < 0 must be also done to stock in shorts

        shorts.sort(key=lambda x: x.risk, reverse=True)

        risk = 0  # all values based on 1 share, so in cents or small dollar values

        # TODO you have to cover long stock too ... although it was limited downside, but a substantial one
        """if stock.quantity > 0:  # long stock
            long_puts = [p for p in longs if p.asset.opt_type == "p"]

            # select highest strike put
            long_puts.sort(key=lambda x: x.asset.strike, reverse=True)
            high_put_pos = long_puts[0]

            to_cover = stock.quantity

            while to_cover > 0 and long_puts:
                ..."""

        if not shorts:
            return sum([long_pos.risk for long_pos in longs])

        for position_to_cover in shorts:

            to_cover = -position_to_cover.quantity
            self.coverage_dict[self.pos_dict[position_to_cover.asset.name]] = dict()

            def get_long_covering_score(lp):

                theta_decay_start_dte = 90  # assume no relevant change to theta before this period

                def extr_projection(x):  # todo use better approx for theta decay up to 200 dte
                    return sqrt(1 - x ** 2)  # circle upper right for extrinsic 90 dte to 0 dte

                def scale(x):
                    return -x / theta_decay_start_dte + 1  # maps 90...0 to 0...1

                strike_diff = abs(position_to_cover.asset.strike - lp.asset.strike)

                # V TODO this does not work and does not make sense, use extr_at_dte instead

                # forecasted extrinsic value of long option when short option expires
                if lp.asset.opt_type == "p":
                    current_intr = lp.asset.strike - stock.asset.bid
                if lp.asset.opt_type == "c":
                    current_intr = stock.asset.ask - lp.asset.strike

                current_intr = max(0, current_intr)

                current_extr = lp.asset.bid - current_intr

                l_dte = lp.asset.dte
                s_dte = position_to_cover.asset.dte
                dte_diff = l_dte - s_dte

                if dte_diff > theta_decay_start_dte:
                    given_up_extr_by_exe = current_extr + lp.asset.theta * dte_diff
                else:
                    given_up_extr_by_exe = extr_projection(scale(dte_diff)) * current_extr

                # ###################

                return strike_diff + given_up_extr_by_exe

            if type(position_to_cover.asset) is Option:

                if position_to_cover.asset.opt_type == "c":  # can only cover short calls with long stock

                    # <editor-fold desc="cover with long stock">

                    # is there any long stock?
                    if stock.quantity > 0:
                        long_q = stock.quantity
                        stock.add_x(-min(to_cover * 100, long_q))
                        to_cover -= min(to_cover, long_q / 100)

                        risk += position_to_cover.cost / 100

                        self.cover_short_with(position_to_cover.asset.name, stock.asset.name,
                                              min(to_cover * 100, long_q))

                    # </editor-fold>

                if position_to_cover.asset.opt_type == "p":  # can only cover short puts with short stock

                    # <editor-fold desc="cover with short stock">

                    # is there any short stock?
                    if stock.quantity < 0:
                        short_stock_q = -stock.quantity

                        stock.add_x(min(to_cover * 100, short_stock_q))
                        stock_in_shorts = [s for s in shorts if type(s) is Stock][0]
                        stock_in_shorts.asset.add_x(min(to_cover * 100, short_stock_q))

                        to_cover -= min(to_cover, short_stock_q / 100)

                        risk += position_to_cover.cost / 100

                        self.cover_short_with(position_to_cover.asset.name, stock.asset.name,
                                              -min(to_cover, short_stock_q / 100))

                    # </editor-fold>

                # <editor-fold desc="or with long option">

                same_type_longs = [p for p in longs if
                                   type(p.asset) is Option and p.asset.opt_type == position_to_cover.asset.opt_type]
                longs_w_cov_score = [(long_p, get_long_covering_score(long_p)) for long_p in same_type_longs]
                longs_w_cov_score.sort(key=lambda x: x[1])  # sort by covering score

                while to_cover > 0 and longs_w_cov_score:  # go until short position is fully covered or no longs remain

                    # update risk
                    risk += \
                        longs_w_cov_score[0][0].cost / 100 + \
                        position_to_cover.cost / 100

                    if self.cost < 0:  # position_to_cover.asset.opt_type == "c":
                        risk += abs(position_to_cover.asset.strike - longs_w_cov_score[0][0].asset.strike)

                    # if position_to_cover.asset.opt_type == "p":
                    #    risk += (position_to_cover.asset.strike - longs_w_cov_score[0][0].asset.strike)

                    # update long quantities
                    long_q = longs_w_cov_score[0][0].quantity
                    longs_w_cov_score[0][0].add_x(-min(to_cover, long_q))

                    # update coverage
                    to_cover -= min(to_cover, long_q)

                    # longs_w_cov_score[0][0] covers
                    self.cover_short_with(position_to_cover.asset.name, longs_w_cov_score[0][0], -min(to_cover, long_q))

                    if longs_w_cov_score[0][0].quantity == 0:
                        longs.remove(longs_w_cov_score[0][0])
                        longs_w_cov_score.remove(longs_w_cov_score[0])

                # </editor-fold>

                if to_cover > 0:

                    if position_to_cover.risk == float('inf'):
                        return float('inf')

                    risk += to_cover * position_to_cover.risk

            if type(position_to_cover.asset) is Stock:

                if position_to_cover.quantity < 0:  # short stock
                    to_cover /= 100

                    # receive stock bid upfront
                    risk -= position_to_cover.asset.bid * -position_to_cover.quantity

                    long_calls = [p for p in longs if p.asset.opt_type == "c"]
                    longs_w_cov_score = [(long_p, get_long_covering_score(long_p)) for long_p in long_calls]
                    longs_w_cov_score.sort(key=lambda x: x[1])  # sort by covering score

                    while to_cover > 0 and longs_w_cov_score:  # go until short position is fully covered or no longs remain

                        # adjust long position quantity
                        long_q = longs_w_cov_score[0][0].quantity
                        used_long_options = min(to_cover, long_q)
                        longs_w_cov_score[0][0].add_x(-used_long_options)

                        # adjust short position quantity
                        stock.add_x(min(to_cover * 100, -position_to_cover.asset.quantity))

                        stock_in_shorts = [s for s in shorts if type(s) is Stock][0]
                        stock_in_shorts.asset.add_x(min(to_cover * 100, -position_to_cover.asset.quantity))

                        # update coverage
                        to_cover -= used_long_options
                        # longs_w_cov_score[0][0] covers short stock
                        self.cover_short_with(position_to_cover.asset.name,
                                              longs_w_cov_score[0][0].asset.name,
                                              min(to_cover, long_q))

                        # update risk - cover short stock with long option
                        risk += \
                            (longs_w_cov_score[0][0].asset.strike - stock.asset.bid) * used_long_options + \
                            longs_w_cov_score[0][0].asset.premium * np.ceil(used_long_options)
                        # todo inaccurate?
                        #  TEST!!!
                        #  angefangene long position kann zB nur teil von ihren 100 deltas benutzen um
                        #  60 short stock zu covern, np.ceil nimmt aber an, dass sie ganz aufgebraucht wurde
                        #  jedoch dürfte dieser fall nur auftreten,

                        # delete long option if used up (it's just a copy)
                        if longs_w_cov_score[0][0].quantity == 0:
                            longs.remove(longs_w_cov_score[0][0])
                            longs_w_cov_score.remove(longs_w_cov_score[0])

                    if to_cover > 0:
                        return float('inf')

                if position_to_cover.quantity > 0:  # long stock
                    ...

        for long_pos in longs:
            risk += long_pos.risk / 100

        return risk * 100

    def get_cost(self):
        return sum([pos.cost for pos in self.pos_dict.values()])

    def add_asset(self, asset, quantity, no_update=False):
        """
        Go-to way to add stocks, for options use add_option_from_option_chain when possible
        :param no_update:
        :param asset: asset name, ticker for stock
        :param quantity:
        :return:
        """
        if asset.name in list(self.pos_dict.keys()):
            self.pos_dict[asset].add_x(quantity)
        else:
            self.pos_dict[asset] = Position(asset, quantity, self.u_ask)

        if not no_update:
            self.update_status()

    def _add_position(self, position: Position, no_update=False):
        if position in self.pos_dict.values():
            self.pos_dict[position.asset.name].quantity += position.quantity
        else:
            self.pos_dict[position.asset.name] = position

        if not no_update:
            self.update_status()

    """def add_position_from_df(self, df: pd.DataFrame):
        self.add_position(Position.row_to_position(df.loc[0, :], self.u_ask))"""

    def add_option_from_option_chain(self, oc: OptionChain):
        """
        takes the first entry of option chains options and adds the option as position
        :param oc:
        :return:
        """
        oc.options.reset_index(inplace=True)
        self._add_position(Position.row_to_position(oc.options.loc[0, :], self.u_ask))

    """def add_positions(self, *positions):
        if type(positions[0]) is list:
            positions = positions[0]
        for position in positions:
            self._add_position(position, no_update=True)

        self.update_status()"""

    def change_asset_quantity(self, asset, quantity):
        self.add_asset(asset, quantity)
        self.update_status()

    def remove_asset(self, asset):
        if asset not in self.pos_dict:
            return
        else:
            del self.pos_dict[asset]

        self.update_status()

    def get_greeks(self):
        greeks = DDict({"delta": 0, "gamma": 0, "theta": 0, "vega": 0, "rho": 0})
        for name, position in self.pos_dict.items():
            greeks.delta += position.greeks.delta
            greeks.gamma += position.greeks.gamma
            greeks.vega += position.greeks.vega
            greeks.theta += position.greeks.theta
            greeks.rho += position.greeks.rho
        return greeks

    """@staticmethod
    def combined_pos_from_df(df: pd.DataFrame, u_bid, u_ask, ticker):

        positions_to_add = []

        ind_list = list(df.columns.values)
        for index, row in df.iterrows():
            tmp = row.T.reindex(ind_list)  # todo need to update indices?
            positions_to_add.append(Position.row_to_position(tmp, u_ask))

        comb_pos = CombinedPosition(dict(), u_bid, u_ask, ticker)
        comb_pos.add_positions(positions_to_add)
        return comb_pos"""

    def get_bpr(self):
        if self.risk is not float('inf'):
            return self.risk
        else:
            return -1  # todo

    def get_break_even(self):
        return -1

    def get_profit_dist_at_exp(self):

        if len(self.pos_dict.values()) == 1:
            if self.pos_dict[self.underlying].quantity == 0:
                return list()
            return self.pos_dict[self.underlying].get_profit_dist(self.stock.ask * 2)

        # TODO
        """
        you can only do the following for options that have the same expiration
        """

        if len(set([p.asset.expiration for p in list(self.pos_dict.values()) if type(p) is Option])) <= 1:

            highest_strike = max([p.asset.strike for p in list(self.pos_dict.values()) if type(p.asset) is Option])
            profit_dist = np.zeros(range(int(highest_strike*100)))

            pos_profits = [np.asarray(pos.get_profit_dist(highest_strike*2)) for pos in list(self.pos_dict.values())]

            for profits in pos_profits:
                profit_dist += profits

            return list(profit_dist)

    def get_max_profit(self):
        """
        Add each positions profit curves
        Date of evaluation is first expiry
        Does a long pos expire first? what about the short that it was covering if any? risk rise to unlimited?
        :return:
        """
        return -1

    def get_max_profit_on_first_exp(self):
        return -1

    def get_rom(self):
        """
        cant use TP percentage bc we dont know it here
        return on margin based on: max profit / self.bpr
        :return:
        """
        return self.max_profit / self.bpr

    def _get_50(self):
        # todo
        """
        Use monte carlo sim...
        :return:
        """
        return 0

    def shift(self, n):
        """
        Shifts the strikes of all options in the position n strikes up (negative n shifts down)
        :param n:
        :return:
        """
        for position in self.pos_dict.values():
            if type(position.asset) is Option:
                position.asset.shift(n)

        self.update_status()

# </editor-fold>

# <editor-fold desc="Strats">


class EnvContainer:

    def __init__(self,
                 env: dict,
                 chain: OptionChain,
                 u_bid: float,
                 u_ask: float,
                 ticker: str,
                 min_per_option_vol: int):
        self.env = env
        self.chain = chain
        self.u_bid = u_bid
        self.u_ask = u_ask
        self.ticker = ticker
        self.min_per_option_vol = min_per_option_vol


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

    def __init__(self, name: str, position_builder, conditions: dict, hints: list, tp_perc: float, sl_perc: float,
                 env: EnvContainer, threshold=DEFAULT_THRESHOLD):

        self.name = name
        self.position_builder = position_builder  # name of the method that returns the position
        self.positions = None
        self.conditions = conditions
        self.env_container = env
        # self.env_container.chain = deepcopy(self.env_container.chain)
        self.recommendation = 0

        self.close_dte = 21
        self.close_perc = 50

        self.greek_exposure = None  # set after environment check
        self.close_date = None

        self.hints = hints
        self.tp_percentage = tp_perc
        self.sl_percentage = sl_perc
        self.threshold = threshold

        self.get_positions(self.env_container)

    def __repr__(self):
        return self.name

    def __str__(self):
        return self.short(t=0)

    def short(self, t=0):
        indent = ("\t" * t)  # todo add greek exp in pretty

        if self.positions is None:
            return f'{indent}Empty {self.name}'

        h = "\n"
        for hint in self.hints:
            h += f'{indent}           {hint}\n'

        return f'\n' \
            f'{indent}{"-" * 300}' \
            f'\n' \
            f'{indent}{self.name} for {self.positions.underlying} ({(self.recommendation * 100):+3.2f}%):' \
            f'\n' \
            f'{indent}{"-" * 300}' \
            f'{indent}{self.positions.repr(t=t + 1)}' \
            f'{indent}Greek exposure:' \
            f'\n' \
            f'{indent}\tΔ = {self.greek_exposure["delta"]}' \
            f'\n' \
            f'{indent}\tΓ = {self.greek_exposure["gamma"]}' \
            f'\n' \
            f'{indent}\tν = {self.greek_exposure["vega"]}' \
            f'\n' \
            f'{indent}\tΘ = {self.greek_exposure["theta"]}' \
            f'\n' \
            f'{indent}\tρ = {self.greek_exposure["rho"]}' \
            f'\n' \
            f'\n' \
            f'{indent} Close by: {datetime_to_european_str(self.close_date)} ({datetime_to_dte(self.close_date)} DTE)' \
            f'\n' \
            f'{indent}      P50: {self.positions.p50 * 100} %' \
            f'\n' \
            f'{indent}Stop loss: {self.sl_percentage / 100 * self.positions.risk:.2f} $' \
            f'\n' \
            f'{indent}    Hints: {h}'

    def repr(self, t=0):

        indent = ("\t" * t)  # todo add greek exp in pretty

        if self.positions is None:
            return f'{indent}Empty {self.name}'

        h = "\n"
        for hint in self.hints:
            h += f'{indent}           {hint}\n'

        return f'\n' \
            f'{indent}{"-" * 300}' \
            f'\n' \
            f'{indent}{self.name} for {self.positions.underlying}:' \
            f'\n' \
            f'{indent}{"-" * 300}' \
            f'{indent}{self.positions.repr(t=t + 1)}' \
            f'{indent}{"-" * 300}' \
            f'\n' \
            f'{indent}Greek exposure:' \
            f'\n' \
            f'{indent}\tΔ = {self.greek_exposure["delta"]}' \
            f'\n' \
            f'{indent}\tΓ = {self.greek_exposure["gamma"]}' \
            f'\n' \
            f'{indent}\tν = {self.greek_exposure["vega"]}' \
            f'\n' \
            f'{indent}\tΘ = {self.greek_exposure["theta"]}' \
            f'\n' \
            f'{indent}\tρ = {self.greek_exposure["rho"]}' \
            f'\n' \
            f'\n' \
            f'{indent} Close by: {datetime_to_european_str(self.close_date)} ({datetime_to_dte(self.close_date)} DTE)' \
            f'\n' \
            f'{indent}      P50: {self.positions.p50 * 100} %' \
            f'\n' \
            f'{indent}Stop loss: {self.sl_percentage / 100 * self.positions.risk:.2f} $' \
            f'\n' \
            f'{indent}    Hints: {h}' \
            f'\n'

    def _test_environment(self, env: dict):
        """

        :return: score from -1 to 1;
        1 if all conditions from self.conditions are fully fullfilled
        0 if application of strat is neutral
        -1 if application is unadived
        """
        try:
            score = 0
            for key, val in self.conditions.items():
                tmp = -2
                for entry in val:
                    tmp = max(entry(env[key]), tmp)
                    _debug(f'Testing {self.name} for: {key} = {entry.__name__}\n'
                           f'\t{key}:       {env[key]:+.5f}\n'
                           f'\t{key} Score: {tmp:+.5f}')
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
                "theta": "time in favor of position" if self.positions.greeks[
                                                            "theta"] >= 0 else "time against position",
                "rho": "long" if self.positions.greeks["rho"] >= 0 else "short"
            }
        else:
            _warn("Cannot set strategy greeks because no positions are existing yet")

    def get_positions(self, env: EnvContainer):
        self.recommendation = self._test_environment(env.env)

        if self.recommendation >= self.threshold:

            _debug(f'Threshold of {self.threshold} for {self.name} was reached: '
                   f'{self.recommendation} > {self.threshold}')
            _debug(f'Start building {self.name}')

            self.positions = self.position_builder()

            if not self._check_liquidity():
                self.positions = None
                return False

            self._set_greek_exposure()
            self._set_close_date()
            self._set_p50()
        else:
            print(f'\nThreshold of {self.threshold} for {self.name} was not reached: '
                  f'{self.recommendation:+.5f} < {self.threshold}\n')

    def _set_close_date(self):
        if self.positions:

            # get earliest exp date of position
            min_dte = 10000
            for pos in self.positions.pos_dict.values():
                if type(pos.asset) is Option:
                    if pos.asset.dte < min_dte:
                        min_dte = pos.asset.dte

            #  close after: close_dte days OR close perc of expiration, whichever is earlier
            self.close_date = dte_to_date(min(min_dte - self.close_dte, int(min_dte / (self.close_perc / 100))))
        else:
            _warn("Cannot set strategy close date because no positions are existing yet")

    def _set_p50(self):
        self.p50 = -1

    def _check_liquidity(self):
        return self._check_spread() and self._check_vol()

    def _check_spread(self):
        # TODO check spread
        return True

    def _check_vol(self):

        for name, pos in self.positions.pos_dict.items():
            if type(pos.asset) is Option and pos.asset.vol < self.env_container.min_per_option_vol:
                print(f'\t{self.name} on {self.env_container.ticker} failed to meet individual volume requirements: '
                      f'{pos.asset.vol} < {self.env_container.min_per_option_vol}\n')
                return False
        return True


class DummyStrat(OptionStrategy):

    def __init__(self, threshold=DEFAULT_THRESHOLD):
        self.name = "dummy strategy"
        self.positions = CombinedPosition(...)
        self.conditions = {
            "IV": [high_ivr],  # high_ivr, put functions here that return values from 0 to 1 or from -1 to 1
            "IV outlook": [extreme],

            "RSI20d": [low_rsi],
            "short term outlook": [bullish, neutral],
            "mid term outlook": [neutral],
            "long term outlook": [bearish],
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
    name = "Long Call"

    def __init__(self, env=None, threshold=0.5, short_term=False):
        """
        :param threshold: test_env must be greater than treshold in order to start building the position
        """
        self.short_term = short_term

        self.threshold = threshold
        self.position_builder = self._get_tasty_variation
        self.conditions = {
            "short term outlook": [bullish],
            "mid term outlook": [bullish] if not short_term else [],
        }

        self.hints = ["Always set a take profit!", "It's just money :)"]

        self.tp_percentage = 100
        self.sl_percentage = 50

        self.close_dte = 10000
        self.close_perc = 100

        super().__init__(self.name,
                         self.position_builder,
                         self.conditions,
                         self.hints,
                         self.tp_percentage,
                         self.sl_percentage,
                         threshold=threshold,
                         env=env)

    def _get_tasty_variation(self):
        """
        :return: combined position(s)
        """
        if self.short_term:
            df = self.env_container.chain \
                .calls() \
                .expiration_close_to_dte(7) \
                .delta_close_to(0.45)
        else:
            df = self.env_container.chain \
                .calls() \
                .expiration_close_to_dte(60) \
                .delta_close_to(0.45)

        cp = CombinedPosition(dict(), self.env_container.u_bid, self.env_container.u_ask, self.env_container.ticker)
        cp.add_option_from_option_chain(df)

        return cp


class LongPut(OptionStrategy):
    name = "Long Put"

    def __init__(self, env=None, threshold=0.5, short_term=False):
        """
        :param threshold: test_env must be greater than treshold in order to start building the position
        """
        self.short_term = short_term

        self.threshold = threshold
        self.position_builder = self._get_tasty_variation
        self.conditions = {
            "short term outlook": [bearish],
            "mid term outlook": [bearish] if not short_term else [],
        }

        self.hints = ["Always set a take profit!", "It's just money :)", ""]

        self.tp_percentage = 100
        self.sl_percentage = 50

        self.close_dte = 10000
        self.close_perc = 100

        super().__init__(self.name,
                         self.position_builder,
                         self.conditions,
                         self.hints,
                         self.tp_percentage,
                         self.sl_percentage,
                         threshold=threshold,
                         env=env)

    def _get_tasty_variation(self):
        """
        :return: combined position(s)
        """
        if self.short_term:
            df = self.env_container.chain \
                .puts() \
                .expiration_close_to_dte(7) \
                .delta_close_to(0.45)
        else:
            df = self.env_container.chain \
                .puts() \
                .expiration_close_to_dte(60) \
                .delta_close_to(0.45)

        cp = CombinedPosition(dict(), self.env_container.u_bid, self.env_container.u_ask, self.env_container.ticker)
        cp.add_option_from_option_chain(df)

        return cp


class CoveredCall(OptionStrategy):
    name = "Covered Call"

    def __init__(self, env=None, threshold=DEFAULT_THRESHOLD):
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
                         threshold=threshold,
                         env=env)

    def _get_tasty_variation(self):
        """
        :param u_ask:
        :param u_bid:
        :param chain:
        :return: combined position(s)
        """
        oc = self.env_container.chain \
            .calls() \
            .short() \
            .expiration_close_to_dte(45) \
            .delta_close_to(0.3)  # shouldnt this be -0.3???

        cp = CombinedPosition(dict(), self.env_container.u_bid, self.env_container.u_ask, self.env_container.ticker)

        cp.add_option_from_option_chain(oc)
        cp.add_asset(Stock(self.env_container.ticker, self.env_container.u_bid, self.env_container.u_ask), 100)

        return cp


class VerticalDebitSpread(OptionStrategy):
    name = "Vertical Debit Spread"

    def __init__(self, env: EnvContainer = None, threshold=DEFAULT_THRESHOLD, opt_type="p"):
        """
        :param threshold: test_env must be greater than treshold in order to start building the position
        """
        self.opt_type = opt_type
        self.name = f'Vertical {"Call" if opt_type == "c" else "Put"} Debit Spread'

        self.threshold = threshold
        self.position_builder = self._get_tasty_variation
        self.conditions = {
            "IV": [low_ivr, neutral_ivr],

            "short term outlook": [bullish] if opt_type == "c" else [bearish],
            "mid term outlook": [neutral, bullish] if opt_type == "c" else [neutral, bearish],
        }

        self.hints = ["Always set a take profit!", "It's just money :)"]

        self.tp_percentage = 100  # 1/2 strike width, how to do this?
        self.sl_percentage = 100

        self.close_dte = 10000
        self.close_perc = 100

        super().__init__(self.name,
                         self.position_builder,
                         self.conditions,
                         self.hints,
                         self.tp_percentage,
                         self.sl_percentage,
                         threshold=threshold,
                         env=env)

    def _get_tasty_variation(self):
        """
        TODO completely wrong strike
        :return: combined position(s)
        """
        # todo not working bc atm through gamma gets wrong bc of -gamma of shorts

        chain = deepcopy(self.env_container.chain)

        # buy 1 ITM, sell 1-2 OTM
        if self.opt_type == "c":
            long_leg = chain.expiration_close_to_dte(45).long().n_itm_strike_call(1, self.env_container.u_ask)
            chain = chain.remove_by_name(long_leg)
            short_leg = chain.expiration_close_to_dte(45).short().n_otm_strike_call(1, self.env_container.u_ask)
        else:
            long_leg = chain.expiration_close_to_dte(45).long().n_itm_strike_put(1, self.env_container.u_ask)
            chain = chain.remove_by_name(long_leg)
            short_leg = chain.expiration_close_to_dte(45).short().n_otm_strike_put(1, self.env_container.u_ask)

        # combined pos
        cp = CombinedPosition(dict(), self.env_container.u_bid, self.env_container.u_ask, self.env_container.ticker)
        cp.add_option_from_option_chain(long_leg)
        cp.add_option_from_option_chain(short_leg)

        # 1/2 strike width
        self.tp_percentage = \
            0.5 * 100 * abs(long_leg.options.loc[0, "strike"] - short_leg.options.loc[0, "strike"]) / cp.max_profit

        return cp


class VerticalCreditSpread(OptionStrategy):
    name = "Vertical Credit Spread"

    # bullish = put spread
    # bearish = call spread

    def __init__(self, env: EnvContainer = None, threshold=DEFAULT_THRESHOLD, opt_type="c", variation="tasty"):
        """
        :param threshold: test_env must be greater than treshold in order to start building the position
        """
        self.opt_type = opt_type
        self.name = f'Vertical {"Call" if opt_type == "c" else "Put"} Credit Spread'

        self.threshold = threshold
        if variation == "tasty":
            self.position_builder = self._get_tasty_variation
        if variation == "personal":
            self.position_builder = self._get_personal_variation
        self.conditions = {
            "IV": [high_ivr],

            "short term outlook": [bullish] if opt_type == "p" else [bearish],
            "mid term outlook": [neutral, bullish] if opt_type == "p" else [neutral, bearish],
        }

        self.hints = ["Call spreads are more liquid and tighter",
                      "Put spread offer higher premium and higher theta",
                      "You can sell a put credit spread against a call credit spread to reduce delta without additional"
                      " capital",
                      "Prefer bearish credit spreads; short credit = short delta"]

        self.tp_percentage = 100
        self.sl_percentage = 100

        self.close_dte = 21
        self.close_perc = 100

        super().__init__(self.name,
                         self.position_builder,
                         self.conditions,
                         self.hints,
                         self.tp_percentage,
                         self.sl_percentage,
                         threshold=threshold,
                         env=env)

    def _get_tasty_variation(self):  # todo prevent both legs beeing identical
        """
        TODO both legs are long, dunno y
        :return: combined position(s)
        """

        chain = deepcopy(self.env_container.chain)

        if self.opt_type == "c":
            # buy 20-25 delta
            long_leg = chain.expiration_close_to_dte(45).calls().long().delta_close_to(0.25)

            # remove long leg
            chain = chain.remove_by_name(long_leg)

            # sell 30-35 delta
            short_leg = chain.expiration_close_to_dte(45).calls().short().delta_close_to(0.3)

        if self.opt_type == "p":
            # buy 20-25 delta
            long_leg = chain.expiration_close_to_dte(45).puts().long().delta_close_to(0.25)

            # remove long leg
            chain = chain.remove_by_name(long_leg)

            # sell 30-35 delta
            short_leg = chain.expiration_close_to_dte(45).puts().short().delta_close_to(0.3)

        # combined pos
        cp = CombinedPosition(dict(), self.env_container.u_bid, self.env_container.u_ask, self.env_container.ticker)
        cp.add_option_from_option_chain(long_leg)
        cp.add_option_from_option_chain(short_leg)

        # 1/2 strike width
        self.tp_percentage = \
            0.5 * 100 * abs(long_leg.options.loc[0, "strike"] - short_leg.options.loc[0, "strike"]) / cp.max_profit

        return cp

    def _get_personal_variation(self):
        return None


# todo
class CalendarSpread(OptionStrategy):
    name = "Calendar Spread"

    def __init__(self, threshold=DEFAULT_THRESHOLD):
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


# todo
class DiagonalSpread(OptionStrategy):
    ...


# todo
class Butterfly(OptionStrategy):
    ...


# todo
class Condor(OptionStrategy):
    ...


# todo
class ShortStrangle(OptionStrategy):
    ...


# </editor-fold>CLO


propose_strategies("AMC")
