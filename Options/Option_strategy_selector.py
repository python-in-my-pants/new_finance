import io
import os
import urllib.request
from bs4 import BeautifulSoup
import yfinance as yf
import pandas as pd
from math import sqrt
from option_greeks import get_greeks
import warnings
import pickle
from copy import deepcopy
from eu_option import EuroOption
from matplotlib import pyplot as plt
from typing import Optional, Callable, List, Dict, Union, Any, Tuple, Iterable
from yahoo_fin import stock_info as si
from DDict import DDict
from MonteCarlo import MonteCarloSimulator, test_mc_accu
import pathlib
from Option import Option
from Option_utility import *
from pprint import pprint as pp, pformat as pf
from Utility import timeit, ppdict
from pandasgui import show
from collections import OrderedDict
from scipy.stats import norm
import traceback
from Libor import *


"""
import ipdb
ipdb.set_trace()
"""

pritn = print

online = False
force_chain_download = True and online
force_meta_download = True and online

debug = True
debug_level = 1

ACCOUNT_BALANCE = 1471
DEFAULT_THRESHOLD = .3

# np.seterr(all='raise')


# <editor-fold desc="Miscellaneous">

def _warn(msg, level=3):
    warnings.warn("\t" + msg, stacklevel=level)


if not online:
    _warn("OFFLINE!    " * 17)


def _debug(*args):
    # higher level = more detail
    _l = args[-1]
    if type(_l) == int:
        level = _l
        if debug and level <= debug_level:
            print()
            print(*args[:-1])
    else:
        # if no level is given, assume importance
        level = 1
        if debug and level <= debug_level:
            print()
            print(*args)


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

# <editor-fold desc="Pyplot settings">
plt.rc('figure', facecolor="#333333", edgecolor="#333333")
plt.rc('axes', facecolor="#353535", edgecolor="#000000")
plt.rc('lines', color="#393939")
plt.rc('grid', color="#121212")


# </editor-fold>

# <editor-fold desc="Data requests">

def get_price_sector_yf(ticker, yf_obj):
    # define fallback values
    u_price, u_bid, u_ask, sector = 0, 0, 0, ""

    if online:
        try:
            u_price = yf_obj.info["ask"]

            u_bid = yf_obj.info["bid"]
            u_ask = yf_obj.info["ask"]

            if "sector" in yf_obj.info.keys():
                sector = yf_obj.info["sector"]
            else:
                sector = ""

        except ValueError as e:
            _warn(f'No data found for ticker {ticker}: {e}')

        except KeyError as e:
            _warn(f'Key error when accessing data for {ticker}: {e}')

    return u_price, u_bid, u_ask, sector


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

        ivr, imp_vol, vol30, oi30 = 0, 0, 0, 0
        next_earnings = ""

        for i, _t in enumerate(text):
            try:
                if 'IV Rank' in _t and "%" in text[i + 2]:
                    ivr = float(text[i + 2].replace("%", ""))
                if 'Implied Volatility' in _t and "%" in text[i + 2] and not "Change" in text[i + 2]:
                    imp_vol = float(text[i + 2].replace("%", "")) / 100.0
                if 'Volume Avg (30-Day)' in _t and vol30 == 0:  # option volume
                    vol30 = int(text[i + 2].replace(",", ""))
                if 'Open Int (30-Day)' in _t and oi30 == 0:
                    oi30 = int(text[i + 2].replace(",", ""))
            except Exception as e:
                _warn(f"Getting options meta data failed for ticker {symbol} with exception: {e}")

            try:
                if "Next Earnings Date" in _t and not next_earnings:
                    # print(text[i+1], text[i+2], text[i+3])
                    next_earnings = text[i + 2].replace(" ", "")
            except Exception as e:
                _warn(f'Getting next earnigns date failed for ticker {symbol} with exception: {e}')

        if ivr == 0:
            _warn(f'IVR for ticker {symbol} is 0!')

        if imp_vol == 0:
            _warn(f'Implied volatility for symbol {symbol} is 0!')

        return ivr, imp_vol, vol30, oi30, next_earnings


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

    # -----------------------------------------------------------------------------------------------------------------

    req = urllib.request.Request('https://www.barchart.com/stocks/quotes/' + symbol + '/analyst-ratings',
                                 headers=headers)
    with urllib.request.urlopen(req) as response:
        resp_text = response.read()
        soup = BeautifulSoup(resp_text, 'html.parser')

        try:
            analyst_rating = soup.findAll("div", attrs={'class': 'block__average_value'})[3].find("span").text
            analyst_rating = (float(analyst_rating) - 3) / 2
        except IndexError:
            analyst_rating = 0

    return outlooks[0], outlooks[1], outlooks[2], analyst_rating


def get_option_chain(yf_obj, u_ask, save=True, min_vol=0):
    # contractSymbol  lastTradeDate  strike  lastPrice    bid    ask  change  percentChange   volume  openInterest
    # ... impliedVolatility  inTheMoney contractSize currency

    start = datetime.now()
    current_path = pathlib.Path().absolute().as_posix()

    if os.path.isfile(current_path + "\\chains\\" + yf_obj.ticker + "_chain.pickle") and not force_chain_download:
        return OptionChain.from_file(filename=yf_obj.ticker + "_chain").volume(min_vol)

    option_chain = DDict()
    ticker_data = yf_obj
    underlying_ask = u_ask
    expirations = [datetime.strptime(t, "%Y-%m-%d") for t in ticker_data.options]
    relevant_fields = ["name", "strike", "inTheMoney", "bid", "ask", "mid", "last", "volume", "OI",
                       "IV", "gearing", "P(ITM)", "P(OTM)", "delta", "gamma", "vega", "theta", "rho"]

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
        contracts["P(OTM)"] = 0.0
        # relative percent change of the value of the option for one percent change in the price of the underlying stock
        contracts["gearing"] = 0.0

        risk_free_interest = get_risk_free_rate(_dte)

        for index, contract in contracts.iterrows():
            contracts.loc[index, "mid"] = (contract["bid"] + contract["ask"]) / 2
            contract_price = (contract["bid"] + contract["ask"]) / 2
            iv = max(min(float(contract["IV"]), 100), 0.005)

            opt_price, iv, delta, gamma, theta, vega, rho = get_greeks(contract_type,
                                                                       underlying_ask,
                                                                       float(contract["strike"]),
                                                                       max(_dte, 0.001),
                                                                       risk_free_interest,
                                                                       iv)

            contracts.loc[index, "delta"] = -delta if contract_type == "p" else delta
            contracts.loc[index, "gamma"] = gamma
            contracts.loc[index, "vega"] = vega
            contracts.loc[index, "theta"] = theta / 100
            contracts.loc[index, "rho"] = rho
            contracts.loc[index, "IV"] = iv
            if contract_price > 0:
                contracts.loc[index, "gearing"] = 100 * contracts.loc[index, "delta"] * underlying_ask / contract_price
            else:
                contracts.loc[index, "gearing"] = float('inf')
            contracts.loc[index, "P(ITM)"] = delta_to_itm_prob(delta, contract_type)
            contracts.loc[index, "P(OTM)"] = 100 - contracts.loc[index, "P(ITM)"]

    for expiration in expirations:
        exp_string = expiration.strftime("%Y-%m-%d")

        print(f'Collecting option chain for {ticker_data.info["symbol"]} {exp_string} ...')

        res = ticker_data.option_chain(exp_string)

        chain = DDict({
            "puts": res.puts.loc[(res.puts["ask"] > 0)],
            "calls": res.calls.loc[(res.calls["ask"] > 0)]
        })
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

        chain.puts.fillna(0, inplace=True)
        chain.calls.fillna(0, inplace=True)

        populate_greeks(chain.calls, dte, "c")
        populate_greeks(chain.puts, dte, "p")

        chain.puts.reset_index(inplace=True, drop=True)
        chain.calls.reset_index(inplace=True, drop=True)

        chain.puts.astype({"volume": int}, copy=False)
        chain.calls.astype({"volume": int}, copy=False)

        chain.puts.astype({"inTheMoney": bool}, copy=False)
        chain.calls.astype({"inTheMoney": bool}, copy=False)

        option_chain[exp_string] = DDict({
            "puts": chain.puts[relevant_fields],
            "calls": chain.calls[relevant_fields]
        })

    print("Gathering options data took", (datetime.now() - start).total_seconds(), "seconds")

    return OptionChain(chain_dict=option_chain, save=save)


# </editor-fold>

# <editor-fold desc="Meta data routine">
class MetaData:

    def __init__(self, ticker, u_price, bid, ask, option_chain, ivr, imp_vol, sector, vol30, oi30, next_earnings,
                 rsi20, short_outlook, mid_outlook, long_outlook, analyst_rating, yf_obj: yf.Ticker = None):
        self.mid_outlook = mid_outlook
        self.ticker = ticker
        self.u_price = u_price
        self.u_bid = bid
        self.u_ask = ask
        self.option_chain = option_chain
        self.ivr = ivr
        self.imp_vol = imp_vol
        self.sector = sector
        self.vol30 = vol30
        self.oi30 = oi30
        self.next_earnings = next_earnings
        self.rsi20 = rsi20
        self.short_outlook = short_outlook
        self.mid_outlook = mid_outlook
        self.long_outlook = long_outlook
        self.analyst_rating = analyst_rating
        self.yf_obj = yf_obj

    def save_as_file(self):

        filename = self.ticker + "_meta"

        try:
            print(f'\nCreating file:\n'
                  f'\t{filename}.pickle ...')

            with open("meta/" + filename + ".pickle", "wb") as file:
                pickle.dump(self, file)

            print("File created successfully!")

        except Exception as e:
            print("While creating the file", filename, "an exception occurred:", e)

    @staticmethod
    def from_file(filename):
        try:
            _f = pickle.load(open(filename, "rb"))
            return _f
        except Exception as e:
            print("While reading the file", filename, "an exception occurred:", e)


def get_meta_data(ticker, min_single_opt_vol=0) -> Optional[MetaData]:
    current_path = pathlib.Path().absolute().as_posix()
    file_exists = os.path.isfile(current_path + '\\meta\\' + ticker + "_meta.pickle")

    if not online:
        if file_exists:  # offline, file exists, read from it
            return MetaData.from_file("meta/" + ticker + "_meta.pickle")
        else:  # offline and no file
            _warn(f'No meta data available for ticker {ticker} (offline and no file around)')
            return None
    else:
        if not force_meta_download and file_exists:  # online and no force, read file
            return MetaData.from_file("meta/" + ticker + "_meta.pickle")
        else:
            # online, force download (could still be junk data)
            meta = download_meta_data(ticker, min_single_opt_vol)
            if meta is not None:
                meta.save_as_file()
            return meta


def download_meta_data(ticker, min_single_opt_vol=0) -> Optional[MetaData]:
    if online:
        yf_obj = yf.Ticker(ticker)
    else:
        _warn(f'Unable to download yahoo finance object, are you offline and forcing download?')
        return None

    # get stock data (yahoo and barchart)
    # do this before requesting option chain beacause we need a underlying price for this

    u_price, u_bid, u_ask, sector = get_price_sector_yf(ticker, yf_obj)  # yahoo
    ivr, imp_vol, vol30, oi30, next_earnings = get_options_meta_data(ticker)  # barchart

    if u_price == 0:

        try:
            last_price = si.get_live_price(ticker)
            u_price = last_price
            u_bid = last_price
            u_ask = last_price
        except Exception as e:
            print(f'Exception in getting fallback ticker price: {e}')

        if u_price == 0:
            _warn(f"[{ticker}] Data sources reporting 0 price for underlying. Data is likely false, exiting ...")
            return None

    # most likely point of failure due to low option volume
    option_chain = get_option_chain(yf_obj, u_ask=u_price, save=True, min_vol=min_single_opt_vol)  # save was = not auto

    if option_chain.empty():
        _warn(f'No options with high enough volume found for ticker {ticker}')
        return None

    # request to barchart.com
    rsi20 = get_rsi20(ticker)

    # request to barchart.com
    short_outlook, mid_outlook, long_outlook, analyst_rating = get_underlying_outlook(ticker)

    return MetaData(ticker, u_price, u_bid, u_ask, option_chain, ivr, imp_vol, sector, vol30, oi30, next_earnings,
                    rsi20,
                    short_outlook, mid_outlook, long_outlook, analyst_rating, yf_obj=yf_obj)


# </editor-fold>


def get_monte_carlo(ticker):
    return MonteCarloSimulator(tickers_list=ticker)


class StrategyConstraints:

    def __init__(self, defined_risk_only=True, avoid_events=True,
                 min_vol30=25000, min_oi30=5000, min_sinle_opt_vol=100,
                 strict_mode=False, assignment_must_be_possible=False):
        """

        :param defined_risk_only:
        :param avoid_events:
        :param min_vol30:
        :param min_oi30:
        :param min_sinle_opt_vol:
        :param strict_mode:
        :param assignment_must_be_possible:
        """
        self.assignment_must_be_possible = assignment_must_be_possible
        self.strict_mode = strict_mode
        self.min_sinle_opt_vol = min_sinle_opt_vol
        self.min_oi30 = min_oi30
        self.min_vol30 = min_vol30
        self.avoid_events = avoid_events
        self.defined_risk_only = defined_risk_only


@timeit
def propose_strategies(ticker: str, strat_cons: StrategyConstraints, monte_carlo: MonteCarloSimulator,
                       pos_df: pd.DataFrame = None,
                       use_predefined_strats: bool = True,
                       single_strat_cons: tuple = None,
                       filters: List[List[Union[Callable, float]]] = None) -> pd.DataFrame:
    # todo get short term price outlook (technical analysis, resistances from barchart.com etc)

    # todo supply custom environment (like outlooks)

    # todo supply max risk limit, min gain, min pop, min p50, min expected return (0?) per position

    # todo substract spread of P/L graphs of positions, add it to cost

    # todo tkinter gui

    # todo watch out for dividends (yfinance actions?)

    # todo do sth if ask is 0 (stock or option)
    # should be done by filtering out options with 0 ask in get_option_chain

    # todo weiht start dtes with vix (mean ca 16) if vix is 14 for example, 1-(14/16) = x; sqrt(1/x) * dte = new dte
    #  45 dte is base

    if pos_df is None:
        pos_df = pd.DataFrame()

    ticker = ticker.upper()

    print(f'\nChecking ticker {ticker} ...\n')

    print(f'Settings:\n\n'
          f'           Defined risk only: {strat_cons.defined_risk_only}\n'
          f'                Avoid events: {strat_cons.avoid_events}\n'
          f'                  Min vol 30: {strat_cons.min_vol30}\n'
          f'    Min single option volume: {strat_cons.min_sinle_opt_vol}\n'
          f'        Min open interest 30: {strat_cons.min_oi30}\n'
          f'                 Strict mode: {strat_cons.strict_mode}\n'
          f' Assignment must be possible: {strat_cons.assignment_must_be_possible}\n')

    meta = get_meta_data(ticker, strat_cons.min_sinle_opt_vol)

    if meta is None:
        # _warn(f'Receiving empty data for ticker {ticker}!')
        return []

    print(f'\nStock info:\n\n'
          f'                   Stock bid: {meta.u_bid:.2f} $\n'
          f'                   Stock ask: {meta.u_ask:.2f} $\n'
          f'                      Sector: {meta.sector if meta.sector else "N/A"}\n'
          f'                         IVR: {meta.ivr}\n'
          f'                      RSI 20: {meta.rsi20}\n'
          f'                      Vol 30: {meta.vol30}\n'
          f'            Open interest 30: {meta.oi30}\n'
          f'               Next earnings: {meta.next_earnings}\n'
          f'   Short term outlook (20 d): {meta.short_outlook}\n'
          f'     Mid term outlook (50 d): {meta.mid_outlook}\n'
          f'   Long term outlook (100 d): {meta.long_outlook}\n'
          f'              Analyst rating: {meta.analyst_rating:.2f}')

    def binary_events_present():

        """
            returns true if undefined binary events can occur,
            returns the date of the earliest event as datetime object if any otherwise
            else returns false
            """

        # don't trade healthcare
        if meta.sector == "Healthcare":
            return True

        if meta.next_earnings and meta.next_earnings != "N/A" and strat_cons.avoid_events:
            ne = datetime.strptime(meta.next_earnings, '%M/%d/%y').strftime("%Y-%m-%d")
            if ne >= datetime.today().strftime("%Y-%m-%d"):
                return ne

        return False

        # todo check for anticipated news

        # todo check for dividends

    def is_liquid():

        r = True

        if meta.vol30 < strat_cons.min_vol30:
            _warn(f'[{ticker}] Warning! Average daily options volume < {strat_cons.min_vol30}: {meta.vol30}')
            if strat_cons.strict_mode:
                r = False

        if meta.oi30 < strat_cons.min_oi30:
            _warn(f'[{ticker}] Warning! Average open interest < {strat_cons.min_oi30}: {meta.oi30}')
            if strat_cons.strict_mode:
                r = False

        _debug(f'Liquidity:\n\n'
               f'30 day average option volume: {meta.vol30:7d}\n'
               f'30 day average open interest: {meta.oi30:7d}\n')

        if not next_puts.empty():
            put_spr_r, put_spr_abs = get_bid_ask_spread(next_puts, meta.u_ask)
            if put_spr_r >= 0.10:
                _warn(
                    f'[{ticker}] Warning! Bid/Ask-Spread  is very wide: '
                    f'Puts = {put_spr_abs:.2f} $ ({put_spr_r * 100:.2f} %)')
                if strat_cons.strict_mode:
                    r = False

            _debug(f'      Put spread ratio (rel):   {int(put_spr_r * 100):3d} %\n'
                   f'      Put spread ratio (abs):    {put_spr_abs:.2f}')

        if not next_calls.empty():
            call_spr_r, call_spr_abs = get_bid_ask_spread(next_calls, meta.u_ask)
            if call_spr_r >= 0.10:
                _warn(
                    f'[{ticker}] Warning! Spread ratio is very wide: '
                    f'Calls = {call_spr_abs:.2f} $ ({call_spr_r * 100:.2f} %)')
                if strat_cons.strict_mode:
                    r = False

            _debug(f'     Call spread ratio (rel):   {int(call_spr_r * 100):3d} %\n'
                   f'     Call spread ratio (abs):    {call_spr_abs:.2f}')

        return r

    def assingment_risk_tolerance_exceeded():

        if meta.u_price * 100 > ACCOUNT_BALANCE / 2:
            s = f'Warning! {ticker.upper()}s price exceeds account risk tolerance! ' \
                f'Assignment would result in margin call!' \
                f' {meta.u_price * 100:6.2f} $ > {ACCOUNT_BALANCE / 2:8.2f} $'
            _warn(s)
            if strat_cons.assignment_must_be_possible:
                return

    # <editor-fold desc="1 Binary events">
    binary_event_date = binary_events_present()

    if type(binary_event_date) is bool and binary_event_date:
        _warn("Warning! Underlying may be subject to undefined binary events!")
        if strat_cons.strict_mode:
            return

    if binary_event_date and type(binary_event_date) is not bool:
        meta.option_chain = meta.option_chain.expiration_before(binary_event_date)

    # </editor-fold>

    # <editor-fold desc="2 liquidity">
    """
        only rough check, options to trade must be checked seperately when chosen
        """
    next_puts = meta.option_chain.expiration_next().puts()
    next_calls = meta.option_chain.expiration_next().calls()

    """
    if next_puts.empty() or next_calls.empty():
        _warn(f'[{ticker}] Next puts or next calls empty')  
        # TODO then just use next expiration or sth, we filter
        #  for volume alread when getting option chain, beeing too strict here is unnecessarily restrictive
        return []
    """

    if not is_liquid():
        _warn(f'Warning! Underlying seems illiquid!')
        if strat_cons.strict_mode:
            return

    # </editor-fold>

    # <editor-fold desc="3 Price">
    assingment_risk_tolerance_exceeded()
    # </editor-fold>

    print()
    print("-" * 120)
    print("All mandatory checks done! Starting strategy selection ...")
    print("-" * 120)

    # -----------------------------------------------------------------------------------------------------------------

    """
    _debug("Expirations:", meta.option_chain.expirations, "\nLength:", len(meta.option_chain.options))
    show(meta.option_chain.options)
    #_debug(meta.option_chain.options.head(100))
    #"""

    # filter out low individual volume options after liquidity test
    meta.option_chain = meta.option_chain.volume(strat_cons.min_sinle_opt_vol)

    _debug(f'Option chain:\n {meta.option_chain.head(5000)}\n)', 3)

    env = DDict({
        "IVR": meta.ivr,
        "IV": meta.imp_vol,
        "IV outlook": 0,

        "RSI20d": meta.rsi20,

        "short term outlook": meta.short_outlook,
        "mid term outlook": meta.mid_outlook,
        "long term outlook": meta.long_outlook,

        "analyst rating": meta.analyst_rating
    })

    env_con = EnvContainer(env, monte_carlo, meta.option_chain, meta.u_bid, meta.u_ask, ticker,
                           strat_cons.min_sinle_opt_vol)

    if use_predefined_strats:
        proposed_strategies = [

            LongCall(env_con),
            LongPut(env_con),

            CoveredCall(env_con),

            VerticalDebitSpread(env_con, opt_type="c"),
            VerticalCreditSpread(env_con, opt_type="c"),

            VerticalDebitSpread(env_con, opt_type="p"),
            VerticalCreditSpread(env_con, opt_type="p"),
        ]
    else:
        proposed_strategies = CustomStratGenerator.get_strats(chain=meta.option_chain,
                                                              env=env_con,
                                                              target_param_tuple=single_strat_cons,
                                                              strat_type="vertical spread",
                                                              filters=filters)

    with io.open(f'StrategySummary-{get_timestamp().split()[0].replace(":", "-")}.txt',
                 "a", encoding="utf-8") as myfile:
        for pos in proposed_strategies:
            if pos.positions:
                print(pos)
                myfile.write(str(pos))
                pos_df = pd.concat([pos_df, pos.to_df()], copy=False, ignore_index=True)
                # pos.positions.plot_profit_dist_on_first_exp()

    return pos_df


def check_spread(strat_cons: StrategyConstraints):

    ticker = input("Ticker: ")

    ticker = ticker.upper()

    print(f'\nChecking ticker {ticker} ...\n')

    print(f'Settings:\n\n'
          f'           Defined risk only: {strat_cons.defined_risk_only}\n'
          f'                Avoid events: {strat_cons.avoid_events}\n'
          f'                  Min vol 30: {strat_cons.min_vol30}\n'
          f'    Min single option volume: {strat_cons.min_sinle_opt_vol}\n'
          f'        Min open interest 30: {strat_cons.min_oi30}\n'
          f'                 Strict mode: {strat_cons.strict_mode}\n'
          f' Assignment must be possible: {strat_cons.assignment_must_be_possible}\n')

    meta = get_meta_data(ticker, strat_cons.min_sinle_opt_vol)

    if meta is None:
        # _warn(f'Receiving empty data for ticker {ticker}!')
        return []

    print(f'\nStock info:\n\n'
          f'                   Stock bid: {meta.u_bid:.2f} $\n'
          f'                   Stock ask: {meta.u_ask:.2f} $\n'
          f'                      Sector: {meta.sector if meta.sector else "N/A"}\n'
          f'                         IVR: {meta.ivr}\n'
          f'                      RSI 20: {meta.rsi20}\n'
          f'                      Vol 30: {meta.vol30}\n'
          f'            Open interest 30: {meta.oi30}\n'
          f'               Next earnings: {meta.next_earnings}\n'
          f'   Short term outlook (20 d): {meta.short_outlook}\n'
          f'     Mid term outlook (50 d): {meta.mid_outlook}\n'
          f'   Long term outlook (100 d): {meta.long_outlook}\n'
          f'              Analyst rating: {meta.analyst_rating:.2f}')

    def perform_checks():

        def binary_events_present():

            """
                returns true if undefined binary events can occur,
                returns the date of the earliest event as datetime object if any otherwise
                else returns false
                """

            # don't trade healthcare
            if meta.sector == "Healthcare":
                return True

            if meta.next_earnings and meta.next_earnings != "N/A" and strat_cons.avoid_events:
                ne = datetime.strptime(meta.next_earnings, '%M/%d/%y').strftime("%Y-%m-%d")
                if ne >= datetime.today().strftime("%Y-%m-%d"):
                    return ne

            return False

            # todo check for anticipated news

            # todo check for dividends

        def is_liquid():

            r = True

            if meta.vol30 < strat_cons.min_vol30:
                _warn(f'[{ticker}] Warning! Average daily options volume < {strat_cons.min_vol30}: {meta.vol30}')
                if strat_cons.strict_mode:
                    r = False

            if meta.oi30 < strat_cons.min_oi30:
                _warn(f'[{ticker}] Warning! Average open interest < {strat_cons.min_oi30}: {meta.oi30}')
                if strat_cons.strict_mode:
                    r = False

            _debug(f'Liquidity:\n\n'
                   f'30 day average option volume: {meta.vol30:7d}\n'
                   f'30 day average open interest: {meta.oi30:7d}\n')

            if not next_puts.empty():
                put_spr_r, put_spr_abs = get_bid_ask_spread(next_puts, meta.u_ask)
                if put_spr_r >= 0.10:
                    _warn(
                        f'[{ticker}] Warning! Bid/Ask-Spread  is very wide: '
                        f'Puts = {put_spr_abs:.2f} $ ({put_spr_r * 100:.2f} %)')
                    if strat_cons.strict_mode:
                        r = False

                _debug(f'      Put spread ratio (rel):   {int(put_spr_r * 100):3d} %\n'
                       f'      Put spread ratio (abs):    {put_spr_abs:.2f}')

            if not next_calls.empty():
                call_spr_r, call_spr_abs = get_bid_ask_spread(next_calls, meta.u_ask)
                if call_spr_r >= 0.10:
                    _warn(
                        f'[{ticker}] Warning! Spread ratio is very wide: '
                        f'Calls = {call_spr_abs:.2f} $ ({call_spr_r * 100:.2f} %)')
                    if strat_cons.strict_mode:
                        r = False

                _debug(f'     Call spread ratio (rel):   {int(call_spr_r * 100):3d} %\n'
                       f'     Call spread ratio (abs):    {call_spr_abs:.2f}')

            return r

        def assingment_risk_tolerance_exceeded():

            if meta.u_price * 100 > ACCOUNT_BALANCE / 2:
                s = f'Warning! {ticker.upper()}s price exceeds account risk tolerance! ' \
                    f'Assignment would result in margin call!' \
                    f' {meta.u_price * 100:6.2f} $ > {ACCOUNT_BALANCE / 2:8.2f} $'
                _warn(s)
                if strat_cons.assignment_must_be_possible:
                    return

        # <editor-fold desc="1 Binary events">
        binary_event_date = binary_events_present()

        if type(binary_event_date) is bool and binary_event_date:
            _warn("Warning! Underlying may be subject to undefined binary events!")
            if strat_cons.strict_mode:
                return

        if binary_event_date and type(binary_event_date) is not bool:
            meta.option_chain = meta.option_chain.expiration_before(binary_event_date)

        # </editor-fold>

        # <editor-fold desc="2 liquidity">
        """
            only rough check, options to trade must be checked seperately when chosen
            """
        next_puts = meta.option_chain.expiration_next().puts()
        next_calls = meta.option_chain.expiration_next().calls()

        """
        if next_puts.empty() or next_calls.empty():
            _warn(f'[{ticker}] Next puts or next calls empty')  
            # TODO then just use next expiration or sth, we filter
            #  for volume alread when getting option chain, beeing too strict here is unnecessarily restrictive
            return []
        """

        if not is_liquid():
            _warn(f'Warning! Underlying seems illiquid!')
            if strat_cons.strict_mode:
                return

        # </editor-fold>

        # <editor-fold desc="3 Price">
        assingment_risk_tolerance_exceeded()
        # </editor-fold>

        return True

    if not perform_checks():
        return

    print()
    print("-" * 120)
    print("All mandatory checks done! Enter positions ...")
    print("-" * 120)

    # ----------------------------------------------------------------------------------

    print("\nExample: +1 Jul 14 21 P 10.5 @ 1.34/1.40")
    long_leg = Position.from_string(input("Long leg: "), ticker, meta.u_ask)
    print("\nExample: -1 Jul 14 21 P 12 @ 1.5/1.60")
    short_leg = Position.from_string(input("Short leg: "), ticker, meta.u_ask)

    if long_leg.asset.opt_type != short_leg.asset.opt_type:
        raise RuntimeError("Both legs must be of same option type (put/call)")

    opt_type = long_leg.asset.opt_type

    # ----------------------------------------------------------------------------------

    supply_custom_strat_params = input("Do you want to supply custom strategy parameters? (Y/N)\n")

    if supply_custom_strat_params != "Y":
        tp_perc = 50
        sl_perc = 100
        latest_close_dte = 21
    else:
        try:
            tp_perc = float(input("TP percentage: "))
        except:
            print("Exception occured getting TP percentage, falling back to default of 50%")
            tp_perc = 50

        try:
            sl_perc = float(input("SL percentage: "))
        except:
            sl_perc = 100
            print("Exception ocurred getting TP percentage, falling back to default of 100%")

        try:
            latest_close_dte = int(input("Close no later than N DTE: "))
        except:
            latest_close_dte = 21
            print("Exception ocurred getting latest close DTE, falling back to default of 21 days")

    mcs = MonteCarloSimulator([ticker])

    env = DDict({
        "IVR": meta.ivr,
        "IV": meta.imp_vol,
        "IV outlook": 0,

        "RSI20d": meta.rsi20,

        "short term outlook": meta.short_outlook,
        "mid term outlook": meta.mid_outlook,
        "long term outlook": meta.long_outlook,

        "analyst rating": meta.analyst_rating
    })

    env_con = EnvContainer(env, mcs, meta.option_chain, meta.u_bid, meta.u_ask, ticker, -1)

    spread = CombinedPosition({p.asset.name: p for p in (long_leg, short_leg)},
                              meta.u_bid, meta.u_ask, meta.ticker, mcs)

    debit_conditions = {
            "IVR": [low_ivr, neutral_ivr],

            "short term outlook": [bullish] if opt_type == "c" else [bearish],
            "mid term outlook": [neutral, bullish] if opt_type == "c" else [neutral, bearish],

            "analyst rating": [neutral, bullish] if opt_type == "c" else [neutral, bearish]
        }

    credit_conditions = {
            "IVR": [high_ivr],

            "short term outlook": [bullish] if opt_type == "p" else [bearish],
            "mid term outlook": [neutral, bullish] if opt_type == "p" else [neutral, bearish],

            "analyst rating": [neutral, bullish] if opt_type == "p" else [neutral, bearish]
        }

    opt_strat = OptionStrategy(name=f'User supplied Vertical {"Credit" if spread.cost < 0 else "Debit"} Spread',
                               position_builder=lambda: spread,
                               env=env_con,
                               conditions=debit_conditions if spread.cost > 0 else credit_conditions,
                               tp_perc=tp_perc,
                               sl_perc=sl_perc,
                               close_dte=latest_close_dte,
                               recommendation_threshold=-float('inf'))

    print(opt_strat)

    return [opt_strat]


# <editor-fold desc="Option Framework">

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

        self.cost = self.get_cost()
        self.risk = self.get_risk()
        self.bpr = self.get_bpr()

        self.max_profit = self.get_max_profit()

        self.rom = self.get_rom()

        self.break_even = self.get_break_even_at_exp()

        self.set_greeks()

    def __repr__(self):
        return f'{self.quantity} {self.asset} for a cost of {self.cost:.2f} $'

    def __str__(self):
        return self.repr()

    @staticmethod
    def from_string(s, ticker, u_ask):
        parts = s.split()
        if len(parts) == 4:  # it's a stock
            quantity, ticker, _, bid_ask = parts
            bid, ask = bid_ask.split("/")
            return Position(Stock(ticker, float(bid), float(ask)), float(quantity), float(ask))
        if len(parts) == 8:  # option
            quantity, month, day, year, opt_type, strike, _, bid_ask = parts
            bid, ask = bid_ask.split("/")
            d_str = f'{month} {day} {year}'
            return Position(Option.new_parse_option(d_str, opt_type, float(strike), float(bid), float(ask),
                                                    u_ask, ticker),
                            float(quantity), float(ask))

    def to_dict(self):

        return OrderedDict([
            ("l_name", self.asset.name),
            ("l_asset", self.asset),
            ("l_type", "Option" if type(self.asset) is Option else "Stock"),
            ("l_iv", self.asset.iv if type(self.asset) is Option else "N/A"),
            ("l_oi", self.asset.oi if type(self.asset) is Option else "N/A"),
            ("l_opt_type", self.asset.opt_type if type(self.asset) is Option else "N/A"),
            ("l_strike", self.asset.strike if type(self.asset) is Option else "N/A"),
            ("l_expiration", self.asset.expiration if type(self.asset) is Option else "N/A"),
            ("l_dte", self.asset.dte if type(self.asset) is Option else "N/A"),
            ("l_bid", self.asset.bid),
            ("l_ask", self.asset.ask),
            ("l_quantity", self.quantity),
            ("l_delta", self.greeks["delta"]),
            ("l_gamma", self.greeks["gamma"]),
            ("l_theta", self.greeks["theta"]),
            ("l_vega", self.greeks["vega"]),
            ("l_rho", self.greeks["rho"]),
            ("l_cost", self.cost),
            ("l_risk", self.risk),
            ("l_bpr", self.bpr),
            ("l_max_profit", self.max_profit),
            ("l_rom", self.rom),
            ("l_break_even", self.break_even)])

    def update(self):
        self.cost = self.get_cost()
        self.risk = self.get_risk()
        self.bpr = self.get_bpr()

        self.max_profit = self.get_max_profit()

        self.rom = self.get_rom()

        self.break_even = self.get_break_even_at_exp()

        self.set_greeks()

    def short(self, _t=0):
        indent = "\t" * _t
        voloi = ""
        if type(self.asset) is Option:
            voloi = f'(OI: {int(self.asset.oi):>8d},  ' \
                    f'Vol: {int(self.asset.vol):>8d}, ' \
                    f'ImpVol: {float(self.asset.iv) * 100: >3.2f} %)'
        return f'{indent}{self.quantity: >+8.2f} {self.asset} = {self.cost: >+8.2f} $\t\t{voloi}'

    def repr(self, _t=0):
        indent = "\t" * _t
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
               f'{indent}\tMax risk: {self.risk:.2f} $ {"(NO RISK FOUND)" if self.risk <= 0 else ""}' \
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
        self.update()

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
        return 0

    def get_profit_dist_at_exp(self, max_strike: int):

        """
        profit distribution in cent stes starting from 0.00
        can be used to plot P/L
        :param max_strike:
        :return:
        """

        gains = [-self.cost for _ in range(max_strike * 100)]

        if type(self.asset) is Stock:

            for strike in range(max_strike * 100):
                gains[strike] += strike / 100 * self.quantity

            return gains

        elif type(self.asset) is Option:

            if self.asset.opt_type == "c":

                if self.quantity > 0:

                    for strike in range(max_strike * 100):

                        if strike / 100 > self.asset.strike:
                            gains[strike] += (strike / 100 - self.asset.strike) * 100 * self.quantity

                    return gains

                if self.quantity < 0:

                    for strike in range(max_strike * 100):

                        if strike / 100 > self.asset.strike:
                            gains[strike] += (strike / 100 - self.asset.strike) * 100 * self.quantity

                    return gains

            if self.asset.opt_type == "p":

                if self.quantity > 0:

                    for strike in range(max_strike * 100):

                        if strike / 100 < self.asset.strike:
                            gains[strike] -= (strike / 100 - self.asset.strike) * 100 * self.quantity

                    return gains

                if self.quantity < 0:

                    for strike in range(max_strike * 100):
                        if strike / 100 < self.asset.strike:
                            gains[strike] -= (strike / 100 - self.asset.strike) * 100 * self.quantity

                    return gains

    def get_profit_dist_at_date(self, max_strike: int, d: str):
        exp_profit_dist = self.get_profit_dist_at_exp(max_strike=max_strike)

        if type(self.asset) is Stock:
            return exp_profit_dist

        if date_to_dte(str_to_date(d)) >= self.asset.dte:
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

            binomial_iterations = 10

            # dte_now = date_str_to_dte(self.asset.expiration)
            dte_then = (str_to_date(self.asset.expiration) - str_to_date(d)).days

            """
            # forecasted extrinsic value of long option when short option expires
            if self.asset.opt_type == "p":
                current_intr = self.asset.strike - self.underlying_price
            elif self.asset.opt_type == "c":
                current_intr = self.underlying_price - self.asset.strike
            else:
                raise RuntimeError("Option type must be 'c' or 'p'")

            current_intr = max(0, current_intr)
            """

            # current_extr_per_opt = self.asset.ask - current_intr
            # we can only forecast this for the current option at the current underlying price
            # forecasted_extr_per_opt = get_extrinsic_at_dte(dte_now, current_extr_per_opt, dte_then)

            risk_free_rate = get_risk_free_rate(dte_then / 365.0)

            is_call = self.asset.opt_type == "c"

            for u_price in range(max_strike * 100):

                # divide by 100 bc u_price is in cents
                future_price = \
                    EuroOption(u_price / 100.0, self.asset.strike, risk_free_rate, dte_then / 365.0,
                               binomial_iterations,
                               {'is_call': is_call,
                                'eu_option': False,
                                'sigma': self.asset.iv}).price()

                if self.quantity > 0:
                    profit_at_strike = self.quantity * future_price * 100 - self.cost
                elif self.quantity < 0:
                    profit_at_strike = -self.cost + self.quantity * future_price * 100
                else:
                    profit_at_strike = 0

                exp_profit_dist[u_price] = profit_at_strike

            return exp_profit_dist

    def plot_profit_dist_at_exp(self, max_strike: int):
        x = [i / 100 for i in range(max_strike * 100)]

        fig, ax1 = plt.subplots()

        ax1.plot(x, np.zeros(len(x)), 'k--')
        line, = ax1.plot(x, self.get_profit_dist_at_exp(max_strike), color='#0a7800', linestyle='-')

        ax1.set_title(self.__repr__().strip())

        snap_cursor = SnappingCursor(ax1, line)
        fig.canvas.mpl_connect('motion_notify_event', snap_cursor.on_mouse_move)

        # ax1.xaxis.set_major_locator(plticker.MultipleLocator(base=1.0))
        # ax1.yaxis.set_major_locator(plticker.MultipleLocator(base=50.0))

        ax1.plot(self.underlying_price, 0, 'r^')

        # plt.grid(True)
        plt.tight_layout()

        plt.show()

    def plot_profit_dist_at_date(self, max_strike: int, d: str):
        x = [i / 100 for i in range(max_strike * 100)]
        fig, ax1 = plt.subplots()

        ax1.plot(x, np.zeros(len(x)), 'k--', x, self.get_profit_dist_at_exp(max_strike), 'b-')
        line, = ax1.plot(x, self.get_profit_dist_at_date(max_strike, d), color='#0a7800', linestyle='-')

        snap_cursor = SnappingCursor(ax1, line)
        fig.canvas.mpl_connect('motion_notify_event', snap_cursor.on_mouse_move)

        # ax1.xaxis.set_major_locator(plticker.MultipleLocator(base=1.0))
        # ax1.yaxis.set_major_locator(plticker.MultipleLocator(base=50.0))

        ax1.plot(self.underlying_price, 0, 'r^')

        # plt.grid(True)
        plt.tight_layout()

        plt.show()

    def get_rom(self):
        if self.bpr == 0:
            return 0
        return self.max_profit / self.bpr

    def get_break_even_at_exp(self):

        # quantity is irrelevant for break even of single position

        if type(self.asset) is Option:
            if self.quantity == 0:
                return -1
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
                return -1
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

    def __init__(self, chain=None, chain_dict=None, save=True):

        if chain is not None:  # create from another OptionChain by filtering
            # chain.reset_index(inplace=True, drop=True)

            if type(chain) is pd.Series:
                self.options = chain.to_frame().T
                self.options = self.non_zero_ask(internal=True)
                self.options.columns = OptionChain.cols

            if type(chain) is pd.DataFrame:
                self.options = chain
                self.options = self.non_zero_ask(internal=True)

            if not self.options.empty:
                self.expirations = sorted(self.options["expiration"].unique().tolist())
                self.ticker = self.options.loc[:, "name"][0][:6]
                self.ticker = ''.join([i for i in self.ticker if not i.isdigit()])
            else:
                self.expirations = []
                self.ticker = None

            _debug(("-" * 73) + "Chain after filtering" + ("-" * 73), 3)
            _debug("Expirations:", self.expirations, "\nLength:", len(self.options), 3)
            _debug(self.options.head(100), 3)

        elif chain_dict is not None:  # create from yahoo data frame
            self.expirations = \
                [key for key in chain_dict.keys() if chain_dict[key].puts.size > 0 or chain_dict[key].calls.size]

            name_df = None
            for exp in self.expirations:
                if chain_dict[exp].puts.size > 0:
                    name_df = chain_dict[exp].puts
                    break
                if chain_dict[exp].calls.size > 0:
                    name_df = chain_dict[exp].calls
                    break

            if name_df is None:
                _warn("Emtpy yahoo data frame given to construct option chain")
                return

            self.ticker = name_df.loc[:, "name"][0][:6]
            self.ticker = ''.join([i for i in self.ticker if not i.isdigit()])

            contract_list = list()

            for expiration in self.expirations:
                chain_dict[expiration].puts["contract"] = "p"
                chain_dict[expiration].puts["expiration"] = expiration
                chain_dict[expiration].puts["direction"] = "long"

                chain_dict[expiration].calls["contract"] = "c"
                chain_dict[expiration].calls["expiration"] = expiration
                chain_dict[expiration].calls["direction"] = "long"

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
            self.options = self.non_zero_ask(internal=True)

            OptionChain.cols = list(self.options.columns.values)

            self.options.style.format({"OI": "{:7d}",
                                       "volume": "{:7d"})  # .background_gradient(cmap='Blues')

            # self.options.fillna(0, inplace=True)

            if online and save:
                self.save_as_file()
        else:
            self.options = pd.DataFrame()

    def empty(self):
        return self.options.empty

    def __bool__(self):
        return not self.empty()

    def __repr__(self):
        return self.options.to_string()

    def head(self, n=5):
        return self.options.head(n)

    # </editor-fold>

    # <editor-fold desc="Type">

    def puts(self):
        _debug("Filter for puts", 2)
        if self.options.empty:
            return self
        _f = self.options.loc[self.options['contract'] == "p"]
        if type(_f) is pd.Series:
            _f = _f.to_frame().T
        return OptionChain(_f)

    def calls(self):
        _debug("Filter for calls", 2)
        if self.options.empty:
            return self
        f = self.options.loc[self.options['contract'] == "c"]
        if type(f) is pd.Series:
            f = f.to_frame().T
        return OptionChain(f)

    def long(self):
        _debug("Filter for longs", 2)
        if self.options.empty:
            return self
        f = self.options.loc[self.options['direction'] == "long"]
        if type(f) is pd.Series:
            f = f.to_frame().T
        return OptionChain(f)

    def short(self):
        _debug("Filter for shorts", 2)
        if self.options.empty:
            return self
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
        _debug(f'Filter for expiration in range {lower_dte}-{upper_dte} DTE', 2)
        if self.options.empty:
            return self
        start_day = datetime_to_str(datetime.now() + timedelta(days=lower_dte))
        end_day = datetime_to_str(datetime.now() + timedelta(days=upper_dte))
        """        
        dates = list(pd.date_range(start=datetime.now() + timedelta(days=lower_dte),
                                   end=datetime.now() + timedelta(days=upper_dte),
                                   normalize=True))
        """
        # todo works?
        _f = self.options.loc[(self.options['expiration'] >= start_day) & (self.options['expiration'] <= end_day)]
        if type(_f) is pd.Series:
            _f = _f.to_frame().T
        return OptionChain(_f)

    def expiration_before(self, exp_date):
        """

        :param exp_date: date as string YYYY-MM-DD
        :return:
        """
        _debug(f'Filter for expiration before {exp_date}', 2)
        if self.options.empty:
            return self

        _f = self.options.loc[self.options['expiration'] <= exp_date]  # works because dates are in ISO form
        if type(_f) is pd.Series:
            _f = _f.to_frame().T
        return OptionChain(_f)

    def expiration_before_dte(self, dte):
        """
        :param dte:
        :return:
        """
        _debug(f'Filter for expiration before {dte} DTE', 2)
        if self.options.empty:
            return self

        f = self.options.loc[self.options['expiration'] < datetime_to_str(date_to_dte(dte))]
        if type(f) is pd.Series:
            f = f.to_frame().T
        return OptionChain(f)

    def expiration_after_dte(self, dte):
        """
        :param dte:
        :return:
        """
        _debug(f'Filter for expiration after {dte} DTE', 2)
        if self.options.empty:
            return self

        _f = self.options.loc[self.options['expiration'] > datetime_to_str(date_to_dte(dte))]
        if type(_f) is pd.Series:
            _f = _f.to_frame().T
        return OptionChain(_f)

    def expiration_after(self, exp_date):
        """

        :param exp_date: date as string YYYY-MM-DD
        :return:
        """
        _debug(f'Filter for expiration after {exp_date}', 2)
        if self.options.empty:
            return self

        _f = self.options.loc[self.options['expiration'] > exp_date]  # works because dates are in ISO form
        if type(_f) is pd.Series:
            _f = _f.to_frame().T
        return OptionChain(_f)

    def expiration_date(self, exp_date):
        """
        Return options having this exact expiration date
        :param exp_date: as string YYYY-MM-DD
        :return:
        """
        _debug("Filter for exp date:", exp_date, 2)
        if self.options.empty:
            return self
        _f = self.options.loc[self.options['expiration'] == exp_date]
        if type(_f) is pd.Series:
            _f = _f.to_frame().T
        return OptionChain(_f)

    def expiration_close_to_dte(self, dte):
        """

        :param dte:
        :return:
        """
        _debug("Filter for expiration close to", dte, "dte", 2)
        if self.options.empty:
            return self
        _f = self.options.loc[self.options['expiration'] == get_closest_date(self.expirations, dte)]
        if type(_f) is pd.Series:
            _f = _f.to_frame().T
        return OptionChain(_f)

    def expiration_next(self, i=0):
        """
        get i-th expiration's puts & calls
        :param i:
        :return:
        """
        _debug(f"Filter for {i}th expiration: ", self.expirations[i], 2)
        if self.options.empty:
            return self
        _f = self.options.loc[self.options['expiration'] == self.expirations[i]]
        if type(_f) is pd.Series:
            _f = _f.to_frame().T
        return OptionChain(_f)

    # </editor-fold>

    # <editor-fold desc="Greeks">

    def greek(self, g, lower=0, upper=1):
        _debug(f'Filter for {g} in range {lower}-{upper}', 2)
        if self.options.empty:
            return self
        _f = self.options.loc[(lower <= self.options[g]) & (upper >= self.options[g])]
        if type(_f) is pd.Series:
            _f = _f.to_frame().T
        return OptionChain(_f)

    def greek_close_to(self, greek, d):
        _debug(f'Filter for {greek} close to {d}', 2)
        if self.options.empty:
            return self
        f = self.options.loc[abs_min_closest_index(self.options[greek].to_list(), d)]
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

        if self.options.empty:
            return self

        # puts or calls?
        contracts = self.options["contract"].unique().tolist()

        if len(contracts) > 1:
            # _warn("Filter for contract type before filtering on OTM strike!")

            # filter for contracts first
            prefiltered = self.puts() if contract_default == "p" else self.calls()
            contract_type = contract_default
        else:
            contract_type = contract_default  # contracts[0]
            prefiltered = self

        _debug(f'Filter for {n} {moneyness} strike on {contract_type}', 2)

        if contract_type not in contracts:
            _debug(f'OptionChain does not contain any {"puts" if contract_type == "p" else "calls"}, returning empty '
                   f'OptionChain')
            return OptionChain()

        # where is ATM? -> delta closest to 0.5
        # atm_index = np.asarray([abs(abs(g)-0.5) for g in prefiltered.options["delta"]]).argmax()
        # atm_index = min_closest_index(list(prefiltered.options["delta"]), 0.5)
        atm_index = abs_min_closest_index(list(prefiltered.options["strike"]), u_ask)

        _debug(f'Detected ATM strike at {prefiltered.options.loc[atm_index, "strike"]}', 2)

        f = None
        use_atm = 0
        # TODO set bounds
        strikes = prefiltered.options["strike"].values.tolist()
        min_strike, max_strike = abs_min_closest_index(strikes, min(strikes)), abs_min_closest_index(strikes,
                                                                                                     max(strikes))

        def cap(x):
            return min(max(min_strike, x), max_strike)

        if contract_type == "c":  # 5

            if moneyness == "ITM":
                f = prefiltered.options.iloc[cap(max(atm_index - n - use_atm, 0)), :]
            if moneyness == "OTM":
                f = prefiltered.options.iloc[cap(min(atm_index + n + use_atm, len(prefiltered.options))), :]

        if contract_type == "p":  # 3

            if moneyness == "ITM":
                f = prefiltered.options.iloc[cap(min(atm_index + n + use_atm, len(prefiltered.options))), :]
            if moneyness == "OTM":
                f = prefiltered.options.iloc[cap(max(atm_index - n - use_atm, 0)), :]

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

    # <editor-fold desc="ITM/OTM">
    def itm(self):
        _debug("Filter for ITM", 2)
        if self.options.empty:
            return self
        f = self.options.loc[self.options['inTheMoney'] is True]
        if type(f) is pd.Series:
            f = f.to_frame().T
        return OptionChain(f)

    def otm(self):
        _debug("Filter for OTM", 2)
        if self.options.empty:
            return self
        f = self.options.loc[self.options['inTheMoney'] == False]
        if type(f) is pd.Series:
            f = f.to_frame().T
        return OptionChain(f)

    # </editor-fold>

    # <editor-fold desc="IV">
    def iv_range(self, lower=0, upper=10000):
        if self.options.empty:
            return self
        f = self.options.loc[(lower <= self.options["IV"]) & (upper >= self.options["IV"])]
        if type(f) is pd.Series:
            f = f.to_frame().T
        OptionChain(f)

    def iv_close_to(self, d):
        if self.options.empty:
            return self
        f = self.options.loc[abs_min_closest_index(self.options["IV"].to_list(), d)]
        if type(f) is pd.Series:
            f = f.to_frame().T
        return OptionChain(f)

    # </editor-fold>

    # <editor-fold desc="p(ITM)">
    def itm_prob_close_to(self, p):
        _debug(f'Filter for P(ITM) close to {p}', 2)
        if self.options.empty:
            return self
        f = self.options.loc[abs_min_closest_index(self.options["P(ITM)"].to_list(), p)]
        if type(f) is pd.Series:
            f = f.to_frame().T  # TODO ???
        return OptionChain(f)

    def itm_prob_range(self, g, lower=0, upper=1):
        _debug(f'Filter for {g} in range {lower}-{upper}', 2)
        if self.options.empty:
            return self
        f = self.options.loc[(lower <= self.options["P(ITM)"]) & (upper >= self.options["P(ITM)"])]
        if type(f) is pd.Series:
            f = f.to_frame().T
        return OptionChain(f)

    # </editor-fold>

    # <editor-fold desc="Strike ><">
    def strike_higher_or_eq(self, n):
        _debug("Filter for strike higher:", n, 2)
        if self.options.empty:
            return self
        f = self.options.loc[self.options['strike'] >= n]
        if type(f) is pd.Series:
            f = f.to_frame().T
        return OptionChain(f)

    def strike_lower_or_eq(self, n):
        _debug("Filter for strike lower:", n, 2)
        if self.options.empty:
            return self
        f = self.options.loc[self.options['strike'] <= n]
        if type(f) is pd.Series:
            f = f.to_frame().T
        return OptionChain(f)

    def strike_eq(self, n):
        _debug("Filter for strike equal to:", n, 2)
        if self.options.empty:
            return self
        f = self.options.loc[self.options['strike'] == n]
        if type(f) is pd.Series:
            f = f.to_frame().T
        return OptionChain(f)

    # </editor-fold>

    def premium_to_strike_diff(self, ratio, other_leg_cost, other_leg_strike, direction):
        """

        :param ratio: e.g. 1/3
        :param other_leg_cost:
        :param other_leg_strike:
        :param direction: is this leg long or short?
        :return:
        """
        _debug(f"Filter for premium/strike ratio close to: {ratio:.2f} "
               f"(other leg cost: {other_leg_cost}; other leg strike: {other_leg_strike})", 2)
        if self.options.empty:
            return self

        best_index = 0
        best_diff = float('inf')
        for row in self.options.itertuples():
            strike_diff = abs(other_leg_strike - row.strike)
            this_leg_cost = row.bid if direction == "short" else row.ask
            cost = abs(abs(other_leg_cost) - abs(this_leg_cost))

            if abs(ratio - cost / strike_diff) < best_diff:
                best_diff = abs(ratio - cost / strike_diff)
                best_index = row.Index

        f = self.options.loc[best_index]
        f = f.to_frame().T

        return OptionChain(f)

    # <editor-fold desc="Liquidity">
    def volume(self, min_vol):
        _debug(f"Filter for volume > {min_vol}", 2)
        if self.options.empty:
            return self
        f = self.options.loc[self.options['volume'] > min_vol]
        if type(f) is pd.Series:
            f = f.to_frame().T
        return OptionChain(f)

    def open_interest(self, min_oi):
        _debug(f"Filter for single option open interest > {min_oi}", 2)
        if self.options.empty:
            return self
        f = self.options.loc[self.options['OI'] > min_oi]
        if type(f) is pd.Series:
            f = f.to_frame().T
        return OptionChain(f)

    # </editor-fold>

    def non_zero_ask(self, internal=False):
        if self.options.empty:
            if internal:
                return self.options
            return self
        f = self.options.loc[self.options['ask'] > 0.0]
        if type(f) is pd.Series:
            f = f.to_frame().T
        f.reset_index(inplace=True, drop=True)
        if not internal:
            return OptionChain(f)
        return f

    # <editor-fold desc="File">
    def save_as_file(self):

        filename = self.ticker + "_chain"  # + get_timestamp().replace(".", "-").replace(":", "-")

        try:
            print(f'\nCreating file:\n'
                  f'\t{filename}.pickle ...')

            with open("chains/" + filename + ".pickle", "wb") as file:
                pickle.dump(self, file)

            print("File created successfully!")

        except Exception as e:
            print("While creating the file", filename, "an exception occurred:", e)

    @staticmethod
    def from_file(filename):
        try:
            f = pickle.load(open("chains/" + filename + ".pickle", "rb"))
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
        if self.options.empty or used_chain.empty():
            return self
        name = used_chain.options.loc[list(used_chain.options.index.values)[0], "name"]
        tmp_chain = self.options.drop(self.options[self.options["name"] == name].index)
        return OptionChain(chain=tmp_chain)


# todo TEST!
class CombinedPosition:

    def __init__(self, pos_dict: Dict[str, Position], u_bid: float, u_ask: float, ticker: str,
                 mcs: MonteCarloSimulator):
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
        self.u_mid = (self.u_bid + self.u_ask) / 2
        self.underlying = ticker.split()[0]
        self.expected_move = 0

        self.mcs = mcs

        self.cost = -1
        self.risk = -1
        self.bpr = -1
        self.break_even = -1  # todo multiple break evens possible!
        self.max_profit = -1
        # smallest underlying price at which max profit occurs todo mult possible (use enum and return list of indices)
        self.max_profit_point = -1
        self.rom = -1
        self.rom50 = -1
        self.prof_time = -1  # profitmax / dte
        self.prof_time50 = -1  # profitmax / dte * 0.5

        self.p50 = -1  # prob of reaching 50 % of max rofit until first expiration
        self.prob_of_profit = -1
        self.profit_dist_at_first_exp = None

        if not any(type(pos.asset) is Stock for pos in self.pos_dict.values()):
            stock = Stock(ticker, self.u_bid, self.u_ask)
            self.add_asset(stock, 0, no_update=True)
        self.stock = self.pos_dict[self.underlying]

        if any(pos.quantity != 0 for pos in self.pos_dict.values()):
            self.update_status()

        self.greeks = self.get_greeks()
        self.theta_to_delta = float(self.greeks["theta"] / float(max(abs(self.greeks["delta"]), 0.00001)))

    def get_dicts(self):
        d = [
            ("dtfe", self.dte_until_first_exp()),
            ("expected_move", self.expected_move),
            ("cost", self.cost),
            ("bpr", self.bpr),
            ("break_even", self.break_even),
            ("max_profit", self.max_profit),
            ("max_profit_point", self.max_profit_point),
            ("rom", self.rom),
            ("rom50", self.rom50),
            ("prof_per_day", self.prof_time),
            ("prof_per_day50", self.prof_time50),
            ("delta", self.greeks["delta"]),
            ("gamma", self.greeks["gamma"]),
            ("theta", self.greeks["theta"]),
            ("vega", self.greeks["vega"]),
            ("rho", self.greeks["rho"]),
            ("theta_delta", self.theta_to_delta)
        ]
        position_dict = [pos.to_dict() for pos in list(self.pos_dict.values())]
        return d, position_dict

    def __repr__(self):
        return self.repr()

    def __bool__(self):
        return not self.empty()

    def repr(self, _t=0, use_short=True):
        indent = "\t" * _t
        s = ""
        for key, val in self.pos_dict.items():
            if val.quantity != 0:
                if use_short:
                    s += f'{indent}{val.short(_t=_t + 1)}\n'
                else:
                    s += f'\n' \
                         f'{indent}Position {key}:' \
                         f'\n\n' \
                         f'{indent}{val.short(_t=_t + 1)}' \
                         f'\n' \
                         f'{indent}{"." * 300}' \
                         f'\n'

        first_exp, dte = self.first_exp_and_dte()

        return f'\n\n' \
               f'{s}' \
               f'\n' \
               f'{indent}   Cumulative strategy cost: {self.cost: >8.2f} $' \
               f'\n' \
               f'{indent}                 Expiration: {first_exp} ({dte} DTE)' \
               f'\n' \
               f'\n' \
               f'{indent}          Cumulative Greeks: ' \
               f'Δ = {self.greeks["delta"]:+.5f}   ' \
               f'Γ = {self.greeks["gamma"]:+.5f}   ' \
               f'ν = {self.greeks["vega"]:+.5f}    ' \
               f'Θ = {self.greeks["theta"]:+.5f}   ' \
               f'ρ = {self.greeks["rho"]:+.5f}     ' \
               f'\n' \
               f'{indent}                Theta/Delta: {self.theta_to_delta:.2f}' \
               f'\n' \
               f'\n' \
               f'{indent}                 Max profit: {self.max_profit:>+6.2f} $' \
               f'\n' \
               f'{indent}           Max profit point: {self.max_profit_point:>+6.2f} $' \
               f'\n' \
               f'{indent}                   Max risk: {self.risk:>+6.2f} $' \
               f'\n' \
               f'{indent}                        BPR: {self.bpr:>+6.2f} $' \
               f'\n' \
               f'\n' \
               f'{indent}       Break even on expiry: ' \
               f'{">" if self.max_profit_point > self.break_even else "<"}' \
               f'{self.break_even:4.2f} $' \
               f'\n' \
               f'{indent}     Implied 1 STD DEV move: ±{self.expected_move:4.2f} $ ' \
               f'({max(self.u_mid - self.expected_move, 0):.2f} $/{self.u_mid + self.expected_move:.2f} $)' \
               f'\n' \
               f'{indent}     Implied 2 STD DEV move: ±{2 * self.expected_move:4.2f} $ ' \
               f'({max(self.u_mid - 2 * self.expected_move, 0):.2f} $/{self.u_mid + 2 * self.expected_move:.2f} $)' \
               f'\n' \
               f'{indent}     Implied 3 STD DEV move: ±{3 * self.expected_move:4.2f} $ ' \
               f'({max(self.u_mid - 3 * self.expected_move, 0):.2f} $/{self.u_mid + 3 * self.expected_move:.2f} $)' \
               f'\n' \
               f'\n' \
               f'{indent}                Half profit: {self.max_profit / 2:+6.2f} $' \
               f'\n' \
               f'{indent}           Return on margin: {self.rom * 100:.2f}%' \
               f'\n' \
               f'{indent}Return on margin (TP @ 50%): {self.rom50 * 100:.2f}%' \
               f'\n' \
               f'{indent}           Max profit / DTE: {self.prof_time:.2f} $/d' \
               f'\n' \
               f'{indent}          Half profit / DTE: {self.prof_time50:.2f} $/d' \
               f'\n\n'  # todo dte or dtc?

    def detail(self, _t=0):
        """
        :return: string representation
        """
        indent = "\t" * _t
        s = ""
        for key, val in self.pos_dict.items():
            if val.quantity != 0:
                s += f'\n' \
                     f'{indent}Position {key}:' \
                     f'\n\n' \
                     f'{indent}{val.repr(_t=_t + 1)}' \
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

    def empty(self):
        return not any([pos.quantity for pos in list(self.pos_dict.values())])

    def update_status(self):
        self.profit_dist_at_first_exp = self.get_profit_dist_on_first_exp()
        self.cost = self.get_cost()
        self.greeks = self.get_greeks()
        self.theta_to_delta = float(self.greeks["theta"] / float(max(abs(self.greeks["delta"]), 0.00001)))
        self.risk = self.get_risk()
        self.bpr = self.get_bpr()
        self.break_even = self.get_break_even()
        self.max_profit, self.max_profit_point = self.get_max_profit()
        self.rom = self.get_rom()
        self.rom50 = self.rom / 2
        self.prof_time = self.max_profit / max(1, self.dte_until_first_exp())
        self.prof_time50 = self.max_profit / (2*max(21, self.dte_until_first_exp()))  # todo does not make sense yet

        # pop ptp psl
        # todo mcs gives only option value? what about naked shorts for example? stock in comb pos? ->
        #  should be ok tho, greeks are covering this - NO: theta does not affect stock, neither does gamma (CC)
        # split mcs between stock part and option part

    def set_expected_move(self, iv):
        # TODO make dte a float with day ending at market close
        std_dev = iv * ((self.dte_until_first_exp() + .1) / 365) ** 0.5
        self.expected_move = std_dev * self.u_mid

    def dte_until_first_exp(self):
        return date_str_to_dte(min([p.asset.expiration for p in self.pos_dict.values() if type(p.asset) is Option]))

    def first_exp_and_dte(self):
        min_exp = min([p.asset.expiration for p in self.pos_dict.values() if type(p.asset) is Option])
        first_exp = date_to_european_str(min_exp)
        dte = date_str_to_dte(min_exp)
        return first_exp, dte

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

        # todo even if long stock is covering sth, it still has to be covered itself by a long put

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

                def scale(x):  # todo min is new
                    return -min(x, theta_decay_start_dte) / theta_decay_start_dte + 1  # maps 90...0 to 0...1

                strike_diff = abs(position_to_cover.asset.strike - lp.asset.strike)

                # V TODO this does not work and does not make sense, use extr_at_dte instead

                # forecasted extrinsic value of long option when short option expires
                if lp.asset.opt_type == "p":
                    current_intr = lp.asset.strike - stock.asset.bid
                elif lp.asset.opt_type == "c":
                    current_intr = stock.asset.ask - lp.asset.strike
                else:
                    raise RuntimeError("Option type must be 'c' or 'p'")

                current_intr = max(0, current_intr)

                current_extr = lp.asset.bid - current_intr

                l_dte = lp.asset.dte
                s_dte = position_to_cover.asset.dte
                dte_diff = l_dte - s_dte  # what if this is negative??

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
                        # stock.add_x(-min(to_cover * 100, long_q))
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
                    self.cover_short_with(position_to_cover.asset.name,
                                          longs_w_cov_score[0][0].asset.name,
                                          -min(to_cover, long_q))

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

                    # go until short position is fully covered or no longs remain
                    while to_cover > 0 and longs_w_cov_score:

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

    def get_profit_n_dte(self, dte, stock_price, risk_free_rate, binomial_iterations=5, mode="bjerksund"):
        """

        :param dte: 0 = now, 1 = todays EOD, 2 = tomorrows EOD, etc.
        :param stock_price:
        :param risk_free_rate:
        :param binomial_iterations:
        :param mode:
        :return:
        """

        legs = [pos for pos in list(self.pos_dict.values()) if type(pos.asset) is Option]

        leg_gains = 0
        for leg in legs:
            if dte > 0:
                if mode == "bjerksund":
                    future_leg_price, _, delta, gamma, theta, vega, rho = get_greeks(leg.asset.opt_type,
                                                                                     max(0.01, stock_price),
                                                                                     leg.asset.strike,
                                                                                     dte / 365.0,
                                                                                     risk_free_rate,
                                                                                     leg.asset.iv)
                elif mode == "binomial":
                    future_leg_price = EuroOption(stock_price,
                                                  leg.asset.strike,
                                                  risk_free_rate,
                                                  dte / 365.0,  # dte then
                                                  binomial_iterations,
                                                  {'is_call': leg.asset.opt_type == "c",
                                                   'eu_option': False,
                                                   'sigma': leg.asset.iv}).price()
                else:
                    raise NotImplementedError("Allowed modes are 'binomial' and 'bjerksund'")

                if leg.cost >= 0:
                    leg_gain = future_leg_price * 100 - leg.cost
                    long_gain = leg_gain
                else:  # short leg
                    leg_gain = -(future_leg_price * 100 + leg.cost)
                    short_gain = leg_gain

            elif dte == 0:
                profit_dist = self.pos_dict[leg.asset.name].get_profit_dist_at_exp(max_strike=int(stock_price) + 1)

                leg_gain = profit_dist[int(stock_price * 100)]
            else:
                raise RuntimeError("Negative DTE encountered in get_profit_n_dte")

            leg_gains += leg_gain

        leg_gains += (stock_price - self.u_ask) * self.stock.quantity
        return round(leg_gains, 2)

    def add_asset(self, asset, quantity, no_update=False):
        """
        Go-to way to add stocks, for options use add_option_from_option_chain when possible
        :param no_update:
        :param asset: asset name, ticker for stock
        :param quantity:
        :return:
        """
        if asset.name in list(self.pos_dict.keys()):
            self.pos_dict[asset.name].add_x(quantity)
        else:
            self.pos_dict[asset.name] = Position(asset, quantity, self.u_ask)

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

    def add_option_from_option_chain(self, oc: OptionChain, no_update: bool = False):
        """
        takes the first entry of option chains options and adds the option as position
        :param no_update: can be set True when adding first legs of a spread for example, must be false for last leg
        :param oc: option chain to add from
        :return:
        """
        if oc.empty():
            return
        oc.options.reset_index(inplace=True)
        self._add_position(Position.row_to_position(oc.options.loc[0, :], self.u_ask), no_update=no_update)

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
        greeks = DDict({
            "delta": 0.0,
            "gamma": 0.0,
            "theta": 0.0,
            "vega": 0.0,
            "rho": 0.0
        })

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

    # todo
    def get_bpr(self):
        if self.risk is not float('inf'):
            return self.risk
        else:
            return -1  # todo

    def get_break_even(self):
        if self.empty():
            return 0
        dist = self.profit_dist_at_first_exp
        return abs_min_closest_index(dist, 0.0) / 100

    def get_profit_dist_at_exp(self):

        if len(self.pos_dict.values()) == 1:
            if self.pos_dict[self.underlying].quantity == 0:
                return list()
            # was get_profit_dist and self.stock.ask
            return self.pos_dict[self.underlying].get_profit_dist_at_exp(self.stock.asset.ask * 2)

        # TODO
        """
        you can only do the following for options that have the same expiration
        """

        # if there is max 1 unique expiration in the combined position
        if len(set([p.asset.expiration for p in list(self.pos_dict.values()) if type(p) is Option])) <= 1:

            highest_strike = max([p.asset.strike for p in list(self.pos_dict.values()) if type(p.asset) is Option])
            profit_dist = [0 for _ in range(int(highest_strike * 100))]
            # was get_profit_dist
            pos_profits = [np.asarray(pos.get_profit_dist_at_exp(highest_strike * 2)) for pos in list(self.pos_dict.values())]

            for profits in pos_profits:
                profit_dist += profits

            return list(profit_dist)

        _warn("Profit dist at exp returned None as there are multiple different expiries")
        raise RuntimeError("Profit dist at exp returned None as there are multiple different expiries")

    def get_max_profit(self):
        """
        Add each positions profit curves
        Date of evaluation is first expiry
        Does a long pos expire first? what about the short that it was covering if any? risk rise to unlimited?
        :return:
        """
        return self.get_max_profit_on_first_exp()  # todo put this together with the other call to this method to save time

    def get_max_profit_on_first_exp(self):
        d = self.profit_dist_at_first_exp
        m = max(d)
        return m, d.index(m) / 100

    def get_profit_dist_on_first_exp(self):
        options = [p for p in list(self.pos_dict.values()) if type(p.asset) is Option]
        first_exp = min([datetime_to_str(pos.asset.expiration_dt) for pos in options])
        # first_exp = datetime_to_str(str_to_date(first_exp) - timedelta(days=1))
        return self.get_profit_dist_at_date(first_exp)

    def get_profit_dist_at_date(self, d):

        options = [p for p in list(self.pos_dict.values()) if type(p.asset) is Option]

        # TODO this currently prevents you from getting P/L after first expiration
        if d > min([datetime_to_str(pos.asset.expiration_dt) for pos in options]):
            _warn("Expiration date exceeds first expiration date of options in CombinedPosition, falling back to first"
                  "expiration date P/L data")
            return self.get_profit_dist_at_exp()

        max_strike = int(max([pos.asset.strike for pos in options])) + 1

        option_profit_dists = [pos.get_profit_dist_at_date(max_strike * 2, d) for pos in options]
        stock_profit_dist = self.pos_dict[self.underlying].get_profit_dist_at_date(max_strike * 2, d)
        option_profit_dists.append(stock_profit_dist)

        combined_profits = [0 for _ in range(max_strike * 2 * 100)]

        for strike in range(max_strike * 2 * 100):
            combined_profits[strike] = sum([profit_dist[strike] for profit_dist in option_profit_dists])

        return combined_profits

    def plot_profit_dist_on_first_exp(self):
        options = [p for p in list(self.pos_dict.values()) if type(p.asset) is Option]
        first_exp = min([datetime_to_str(pos.asset.expiration_dt) for pos in options])
        return self.plot_profit_dist_at_date(first_exp)

    def plot_profit_dist_at_date(self, d):

        max_profits = self.profit_dist_at_first_exp

        x = [i / 100 for i in range(len(max_profits))]
        fig, ax1 = plt.subplots()

        ax1.plot(x, np.zeros(len(x)), 'k--', x, max_profits, '-')

        if self.expected_move != 0:
            m = min(max_profits)
            ma = max(max_profits)

            # x1 x2 y1 y2

            def plot_n_std_dev(n):
                ax1.plot([max(self.u_mid - n * self.expected_move, 0), max(self.u_mid - n * self.expected_move, 0)],
                         [m, ma], f"#{str(999999 - n * 222222)}")
                ax1.plot([self.u_mid + n * self.expected_move, self.u_mid + n * self.expected_move],
                         [m, ma], f"#{str(999999 - n * 222222)}")

            plot_n_std_dev(1)
            plot_n_std_dev(2)
            plot_n_std_dev(3)

        line, = ax1.plot(x, self.get_profit_dist_at_date(d), color='#0a7800', linestyle='-')

        snap_cursor = SnappingCursor(ax1, line)
        fig.canvas.mpl_connect('motion_notify_event', snap_cursor.on_mouse_move)

        s = f'{self.underlying.upper()} @ {self.u_bid:.2f}/{self.u_ask:.2f}\n'
        for pos in list(self.pos_dict.values()):
            if pos.quantity != 0:
                s += pos.__repr__() + "\n"
        ax1.set_title(s.strip())

        ax1.plot(self.u_ask, 0, 'r^')
        ax1.plot(self.u_bid, 0, 'rv')

        # ax1.xaxis.set_major_locator(plticker.MultipleLocator(base=1.0))
        # ax1.yaxis.set_major_locator(plticker.MultipleLocator(base=50.0))

        # plt.grid(True)
        plt.tight_layout()

        plt.show()

    def get_rom(self):
        """
        cant use TP percentage bc we dont know it here
        return on margin based on: max profit / self.bpr
        :return:
        """
        if self.bpr == 0:
            return float('inf')
        rom = self.max_profit / self.bpr
        return rom


class VerticalSpread(CombinedPosition):

    def __init__(self, quantity: int, long_leg: Option, short_leg: Option,
                 u_bid: float, u_ask: float, ticker: str):
        self.long_pos = Position(long_leg, quantity, u_ask)
        self.short_pos = Position(short_leg, quantity, u_ask)
        pos_dict = {
            long_leg.name: self.long_pos,
            short_leg.name: self.short_pos
        }
        super().__init__(pos_dict, u_bid, u_ask, ticker)

    def get_risk(self):  # todo test
        return self.long_pos.cost + self.short_pos.cost

    def get_max_profit_on_first_exp(self):  # todo test
        return -self.short_pos.cost

    @staticmethod
    def get_spread(chain, u_bid, u_ask, conditions: dict):
        """
        give me a vertical spread of some kind so that ...

        :param chain:
        :param u_bid:
        :param u_ask:
        :param conditions:
        :return:
        """

        """
        example conditions:
        
        conditions = {
        
            "DTE": 45,
            
            "max risk": 50,
            "min gain": 20,
            "min p50": 50
        }
        """

        # can scan option chain and filter out unfitting options (depend on single option values)
        scannable_keys = ["DTE", "min vol", "min oi"]  # for both legs uniform?

        # can only be checked after spread is created (depend on more than one options values)
        checkable_keys = ["max risk", "min gain", "min p50", "min pop", "min rom", "min theta/delta"]

        scannable_key2func = {
            "DTE": OptionChain.expiration_close_to_dte,
            "delta": OptionChain.delta_close_to,
            "gamma": OptionChain.gamma_close_to,
            "theta": OptionChain.theta_close_to,
            "vega": OptionChain.vega_close_to,
            "min vol": OptionChain.volume,
            "min oi": OptionChain.open_interest,
        }

        scannable_dict = {key: value for key, value in conditions.items() if key in scannable_keys}
        checkable_dict = {key: value for key, value in conditions.items() if key in checkable_keys}

        filtered_chain = deepcopy(chain)

        for condition, value in scannable_dict.items():  # todo works?
            filtered_chain = filtered_chain.condition(value)

        # build all possible spread positions from filtered chain


# </editor-fold>

# <editor-fold desc="Strats">


class EnvContainer:

    def __init__(self,
                 env: dict,
                 mc: MonteCarloSimulator,
                 chain: OptionChain,
                 u_bid: float,
                 u_ask: float,
                 ticker: str,
                 min_per_option_vol: int):
        self.env = env
        self.mc = mc
        self.chain = chain
        self.u_bid = u_bid
        self.u_ask = u_ask
        self.u_mid = (u_bid + u_ask) / 2
        self.ticker = ticker
        self.min_per_option_vol = min_per_option_vol

    def to_dict(self):
        d = {
            "env_ticker": self.ticker,
            "u_bid": self.u_bid,
            "u_ask": self.u_ask,
            "min_per_option_vol": self.min_per_option_vol,
        }
        d.update(self.env)
        return d


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

    def __init__(self, name: str, position_builder: Callable[[Any], Optional[CombinedPosition]], env: EnvContainer,
                 conditions: dict = None, hints: list = list(), tp_perc: float = 100, sl_perc: float = 100,
                 close_dte: int = 21, close_perc: float = 50, recommendation_threshold=DEFAULT_THRESHOLD):

        self.name = name
        self.position_builder = position_builder  # name of the method that returns the positios
        self.env_container = env

        # optionals
        self.conditions = conditions
        self.close_dte = close_dte
        self.close_perc = close_perc
        self.hints = ["Always set a take profit!", "It's just money :)"] + hints
        self.tp_percentage = tp_perc
        self.sl_percentage = sl_perc
        self.recommendation_threshold = recommendation_threshold

        # set when done building
        self.recommendation = 0
        self.test_results = "No tests deducted yet"
        self.test_results_dict = OrderedDict()
        self.positions: Optional[CombinedPosition] = None
        self.ptp = -1
        self.psl = -1
        self.exp_gains_on_day = None
        self.prob_of_profit = -1
        self.tp_med_d = -1
        self.tp_avg_d = -1
        self.close_pop = -1
        self.implied_pop = -1
        self.implied_pop_at_close = -1
        self.close_pn = -1
        self.close_be = -1
        self.e_close = None
        self.e_expiry = None
        self.greek_exposure = None  # set after environment check
        self.close_date = None

        self.get_positions(self.env_container)

    def __repr__(self):
        return self.name

    def __str__(self):
        return self.short(_t=0)

    def get_test_results_str(self, t=0):
        to_ret = ""
        indent = "\t" * t
        parts = self.test_results
        if isinstance(parts, str):
            parts = parts.split("\n")
            for part in parts:
                to_ret += f'{indent}{part}\n'
            to_ret = to_ret.replace("Testing", "Tested")[:-1]
            return to_ret
        if isinstance(parts, dict):
            return ppdict(parts, i=t)

    def short(self, _t=0):
        indent = ("\t" * _t)  # todo add greek exposure in pretty

        if self.positions is None:
            return f'{indent}Empty {self.name}'

        h = "\n"
        for hint in self.hints:
            h += f'{indent}           {hint}\n'

        inner_sl = "None" if self.sl_percentage >= 100 else f'{self.sl_percentage / 100 * self.positions.risk:.2f} $'

        greek_exp = f'{indent}Greek exposure:' \
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
                    f'\n'

        return f'\n' \
               f'{indent}{"-" * 5} {get_timestamp()} {"-" * 94}' \
               f'\n' \
               f'{indent}{self.name} for {self.positions.underlying} ' \
               f'({self.env_container.u_bid:.2f} $/{self.env_container.u_ask:.2f} $) ' \
               f'===> {(self.recommendation * 100):+3.2f} % (min. {self.recommendation_threshold * 100:.2f} %):' \
               f'\n' \
               f'{indent}{"-" * 120}' \
               f'\n' \
               f'{self.get_test_results_str(_t + 1)}' \
               f'\n' \
               f'{indent}{" ." * 60}' \
               f'{indent}{self.positions.repr(_t=_t + 1)}' \
               f'{greek_exp if False else ""}' \
               f'{indent}          --- CLOSE ---' \
               f'\n' \
               f'\n' \
               f'{indent}     Close by: {datetime_to_european_str(self.close_date)} ({datetime_to_dte(self.close_date)} days)' \
               f'\n' \
               f'{indent} MC PoP@close: {self.close_pop * 100: >5.2f} %' \
               f'\n' \
               f'{indent} ImpPoP@close: {self.implied_pop_at_close * 100: >5.2f} % WRONG' \
               f'\n' \
               f'{indent} P{int(self.tp_percentage)} at close: {self.close_pn * 100: >3.2f} %' \
               f'\n' \
               f'\n' \
               f'{indent}  BE at close: {self.close_be: >5.2f} $ WRONG' \
               f'\n' \
               f'{indent} Expect@close: {self.e_close: >+5.2f} $' \
               f'\n' \
               f'{indent}    Avg to TP: {int(self.tp_avg_d + 0.5)} days\t\t\tMed to TP: {int(self.tp_med_d)} days' \
               f'\n' \
               f'\n' \
               f'{indent}          --- EXPIRATION ---' \
               f'\n' \
               f'\n' \
               f'{indent} MC PoP @ exp: {self.prob_of_profit * 100: >5.2f} %' \
               f'\n' \
               f'{indent}  Implied PoP: {self.implied_pop * 100: >5.2f} %' \
               f'\n' \
               f'{indent}   P{int(self.tp_percentage)} at exp: {self.ptp * 100: >5.2f} %' \
               f'\n' \
               f'\n' \
               f'{indent}   Expect@exp: {self.e_expiry: >+5.2f} $' \
               f'\n' \
               f'{indent}\n    Stop loss: {inner_sl}' \
               f'\n' \
               f'\n' \
               f'{indent}        Hints: {h}'

    def repr(self, _t=0):

        indent = ("\t" * _t)  # todo add greek exp in pretty

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
               f'{indent}{self.positions.repr(_t=_t + 1)}' \
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

    def to_df(self):
        """
        self.name = name
        self.position_builder = position_builder  # name of the method that returns the positios
        self.env_container = env

        # optionals
        self.conditions = conditions
        self.close_dte = close_dte
        self.close_perc = close_perc
        self.hints = ["Always set a take profit!", "It's just money :)"] + hints
        self.tp_percentage = tp_perc
        self.sl_percentage = sl_perc
        self.recommendation_threshold = recommendation_threshold

        # set when done building
        self.recommendation = 0
        self.test_results = "No tests deducted yet"
        self.positions: Optional[CombinedPosition] = None
        self.ptp = -1
        self.psl = -1
        self.prob_of_profit = -1
        self.tp_med_d = -1
        self.tp_avg_d = -1
        self.close_pop = -1
        self.close_pn = -1
        self.e_tp_close = -1  # close_pn * tp_percentaga/100 * max_gain
        self.greek_exposure = None  # set after environment check
        self.close_date = None

        self.get_positions(self.env_container)
        """

        main_dict = [
            ("underlying", self.positions.underlying),
            ("name", self.name),
            ("creation_time", get_timestamp()),
            ("close_dte", self.close_dte),
            ("close_perc", self.close_perc),
            ("tp_percentage", self.tp_percentage),
            ("sl_percentage", self.sl_percentage),
            ("recommendation_threshold", self.recommendation_threshold),
            ("recommendation", self.recommendation),
            ("ptp", self.ptp),
            ("psl", self.psl),
            ("pop", self.prob_of_profit),
            ("tp_med_d", self.tp_med_d),
            ("tp_avg_d", self.tp_avg_d),
            ("close_pop", self.close_pop),
            ("close_pn", self.close_pn),
            ("e_close", self.e_close),
            ("e_exp", self.e_expiry),
            ("close_date", self.close_date.date()),
            ("stop_loss", "N/A" if self.sl_percentage >= 100 else self.sl_percentage / 100 * self.positions.risk),
        ]

        """("delta_exp", self.greek_exposure["delta"]),
        ("gamma_exp", self.greek_exposure["gamma"]),
        ("theta_exp", self.greek_exposure["theta"]),
        ("vega_exp", self.greek_exposure["vega"]),
        ("rho_exp", self.greek_exposure["rho"]),"""

        env_dict = self.env_container.to_dict()

        if self.conditions:
            condition_dict = OrderedDict(self.conditions)
        else:
            condition_dict = OrderedDict()

        test_res_dict = self.test_results_dict

        cpos_dict, leg_dicts = self.positions.get_dicts()

        full_dict = OrderedDict(main_dict +
                                list(env_dict.items()) +
                                list(condition_dict.items()) +
                                list(test_res_dict.items()) +
                                cpos_dict)
        i = 0
        for leg_dict in leg_dicts:
            if leg_dict["l_quantity"] != 0:
                i += 1
                full_dict.update({"leg " + str(i) + " " + key.split("_")[1]: value for key, value in leg_dict.items()})

        df = pd.DataFrame(full_dict, index=[0])
        return df

    def _test_environment(self, env: dict):
        """
        :return: score from -1 to 1;
        1 if all conditions from self.conditions are fully fullfilled
        0 if application of strat is neutral
        -1 if application is unadvised
        """
        try:
            score = 0
            if self.conditions:
                self.test_results = ""
                self.test_results_dict = OrderedDict()
                for key, val in self.conditions.items():
                    tmp = -2
                    for entry in val:
                        if entry(env[key]) > tmp:
                            self.test_results += f'\nTesting {self.name} for: {key} = {entry.__name__}\n' \
                                                 f'\t{key}:       {env[key]:+.2f}\n' \
                                                 f'\t{key} Score: {max(entry(env[key]), tmp):+.2f}\n'
                        tmp = max(entry(env[key]), tmp)
                        _debug(f'Testing {self.name} for: {key} = {entry.__name__}\n'
                               f'\t{key}:       {env[key]:+.2f}\n'
                               f'\t{key} Score: {tmp:+.2f}', 3)
                    self.test_results_dict[key] = tmp
                    score += tmp
                self.test_results_dict["score"] = score / len(self.conditions.keys())
                return score / len(self.conditions.keys())
            else:
                self.test_results = env
                self.test_results_dict = dict()
                return 1
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

        if self.recommendation >= self.recommendation_threshold:

            _debug(f'Threshold of {self.recommendation_threshold} for {self.name} was reached: '
                   f'{self.recommendation:.5f} > {self.recommendation_threshold}', 2)
            _debug(f'Start building {self.name}', 1)

            self.positions = self.position_builder()

            if self.positions is None:
                _debug(f'Failed to build position for {self.name}', 1)
                return False

            if not self._check_liquidity():
                self.positions = None
                _debug(f'Liquidity check failed for {self.name}', 1)
                return False

            self._set_greek_exposure()
            self._set_close_date()
            self._set_probs()
            self.positions.set_expected_move(self.env_container.env["IV"])

            _debug(f'Finished building position for {self.name}', 1)

        else:
            _debug(f'\nThreshold of {self.recommendation_threshold} for {self.name} was not reached: '
                   f'{self.recommendation:+.5f} < {self.recommendation_threshold}\n', 2)

    def _set_close_date(self):
        if self.positions:

            # get earliest exp date of position
            min_dte = 10000
            for pos in self.positions.pos_dict.values():
                if type(pos.asset) is Option:
                    if pos.asset.dte < min_dte:
                        min_dte = pos.asset.dte

            #  close after: close_dte days OR close perc of expiration, whichever is earlier
            self.close_date = dte_to_date(max(0, min(min_dte - self.close_dte,
                                                     int(min_dte / (self.close_perc / 100)))))
        else:
            _warn("Cannot set strategy close date because no positions are existing yet")

    """
    def _set_probs_in_com_pos(self):
        self.positions.set_probs(tp=self.tp_percentage / 100 * self.positions.max_profit,
                                 sl=self.sl_percentage / 100 * self.positions.risk)

        self.p50 = self.positions.p50
        self.prob_of_profit = self.positions.prob_of_profit
    """

    def _set_probs(self):
        """
        test_mc_accu(self, get_risk_free_rate(self.positions.dte_until_first_exp()), sim_size=10 ** 2)
        test_mc_accu(self, get_risk_free_rate(self.positions.dte_until_first_exp()), sim_size=10 ** 3)
        test_mc_accu(self, get_risk_free_rate(self.positions.dte_until_first_exp()))
        """

        prob_dict = self.positions.mcs.get_pop_pn_sl(self,
                                                     get_risk_free_rate(self.positions.dte_until_first_exp()/365))
        """clean, sel = self.positions.mcs.get_pop_dist(self.env_container.ticker,
                                                     self.positions.dte_until_first_exp(),
                                                     self.positions.break_even,
                                                     self.positions.max_profit_point,
                                                     self.positions.greeks["delta"])"""
        # tmp = pd.DataFrame({"close": clean, "be": sel}).sort_values(by=["close"]).reset_index()
        # show(tmp)
        self.ptp = prob_dict.p_tp
        self.prob_of_profit = prob_dict.prob_of_profit
        self.psl = prob_dict.p_sl
        self.tp_med_d = prob_dict.tp_med
        self.tp_avg_d = prob_dict.tp_avg
        self.close_pop = prob_dict.close_pop
        self.close_pn = prob_dict.close_pn
        self.close_be = prob_dict.close_be  # todo wrong??

        self.exp_gains_on_day = prob_dict.exp_gains
        self.e_close = self.exp_gains_on_day[datetime_to_dte(self.close_date) + 1]
        self.e_expiry = self.exp_gains_on_day[self.positions.dte_until_first_exp() + 1]

        # implied by option IVs

        opt_ivs = [opt.asset.iv for opt in self.positions.pos_dict.values() if type(opt.asset) is Option]
        avg_imp_vol = sum(opt_ivs) / len(opt_ivs)
        dtfe = self.positions.dte_until_first_exp()
        # divide by root of time to get std dev over other timeframes != 1
        std_dev = self.env_container.u_mid * avg_imp_vol * ((dtfe + 0.1) / 365.0) ** 0.5
        # loc=mean, scale=std dev
        implied_pop = norm.cdf(self.positions.break_even, loc=self.env_container.u_mid, scale=std_dev)

        # imp pop at close
        std_dev = self.env_container.u_mid * avg_imp_vol * ((datetime_to_dte(self.close_date) + .1) / 365.0) ** 0.5
        implied_pop_at_close = norm.cdf(self.close_be, loc=self.env_container.u_mid, scale=std_dev)

        if self.positions.break_even > self.positions.max_profit_point or \
                (self.positions.break_even == self.positions.max_profit_point and self.positions.greeks["delta"] <= 0):
            self.implied_pop = implied_pop
            self.implied_pop_at_close = implied_pop_at_close
        if self.positions.break_even < self.positions.max_profit_point or \
                (self.positions.break_even == self.positions.max_profit_point and self.positions.greeks["delta"] > 0):
            self.implied_pop = 1 - implied_pop
            self.implied_pop_at_close = 1 - implied_pop_at_close

    def _check_liquidity(self):
        return self._check_bid_ask_spread() and self._check_vol()

    def _check_vol(self):

        for name, pos in self.positions.pos_dict.items():
            if type(pos.asset) is Option and pos.asset.vol < self.env_container.min_per_option_vol:
                print(
                    f'\n\n----{self.name} on {self.env_container.ticker} failed to meet individual volume requirements: '
                    f'{pos.asset.vol} < {self.env_container.min_per_option_vol}\n')
                return False
        return True

    # todo
    def _check_bid_ask_spread(self):
        return True

    @staticmethod
    def build_comb_pos(env_container, *legs):
        # todo why not supply them all at once in a dict?
        if all(legs):  # no legs are empty, have to supply double legs multiple times, e.g. for ratio spreads
            cp = CombinedPosition(dict(), env_container.u_bid, env_container.u_ask, env_container.ticker,
                                  env_container.mc)
            for leg in legs:
                cp.add_option_from_option_chain(leg)
            return cp
        else:
            return None


class DummyStrat(OptionStrategy):

    def __init__(self, recommendation_threshold=DEFAULT_THRESHOLD):
        self.name = "dummy strategy"
        self.positions = CombinedPosition(...)
        self.conditions = {
            "IVR": [high_ivr],  # high_ivr, put functions here that return values from 0 to 1 or from -1 to 1
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
                         recommendation_threshold=recommendation_threshold)


class LongCall(OptionStrategy):

    def __init__(self, env: EnvContainer, recommendation_threshold=0.5, short_term=False):
        """
        :param recommendation_threshold: test_env must be greater than treshold in order to start building the position
        """
        self.short_term = short_term

        conditions = {
            "short term outlook": [bullish],
            "mid term outlook": [bullish] if not short_term else [],

            "analyst rating": [bullish]
        }

        super().__init__(name="Long Call",
                         position_builder=self._get_tasty_variation,
                         env=env,
                         conditions=conditions,
                         hints=[],
                         tp_perc=50,
                         sl_perc=100,
                         close_dte=21 if not short_term else 0,
                         close_perc=100,
                         recommendation_threshold=recommendation_threshold)

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

        cp = self.build_comb_pos(self.env_container, df)

        return cp if cp else None


class LongPut(OptionStrategy):

    def __init__(self, env: EnvContainer, recommendation_threshold=0.5, short_term=False):
        """
        :param recommendation_threshold: test_env must be greater than treshold in order to start building the position
        """
        self.short_term = short_term

        conditions = {
            "short term outlook": [bearish],
            "mid term outlook": [bearish] if not short_term else [],

            "analyst rating": [bearish]
        }

        super().__init__(name="Long Put",
                         position_builder=self._get_tasty_variation,
                         env=env,
                         conditions=conditions,
                         hints=[],
                         tp_perc=50,
                         sl_perc=100,
                         close_dte=21 if not short_term else 0,
                         close_perc=100,
                         recommendation_threshold=recommendation_threshold)

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

        cp = self.build_comb_pos(self.env_container, df)

        return cp if not cp.empty() else None


class VerticalDebitSpread(OptionStrategy):

    def __init__(self, env: EnvContainer, opt_type: str, recommendation_threshold=DEFAULT_THRESHOLD):
        """
        :param recommendation_threshold: test_env must be greater than treshold in order to start building the position
        """
        self.opt_type = opt_type

        conditions = {
            "IVR": [low_ivr, neutral_ivr],

            "short term outlook": [bullish] if opt_type == "c" else [bearish],
            "mid term outlook": [neutral, bullish] if opt_type == "c" else [neutral, bearish],

            "analyst rating": [neutral, bullish] if opt_type == "c" else [neutral, bearish]
        }

        super().__init__(name=f'Vertical {"Call" if opt_type == "c" else "Put"} Debit Spread',
                         position_builder=self._get_tasty_variation2,
                         env=env,
                         conditions=conditions,
                         hints=[],
                         tp_perc=50,
                         sl_perc=100,
                         close_dte=21,
                         close_perc=100,
                         recommendation_threshold=recommendation_threshold)

    def _get_tasty_variation(self):
        """
        TODO completely wrong strike
        :return: combined position(s)
        """

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

        if not long_leg or not short_leg:
            _debug(f'Long leg is missing in {self.name}, aborting ...', 1)
            return None

        if long_leg.options.loc[0, "expiration"] != short_leg.options.loc[0, "expiration"]:
            _debug(f'Expirations are not equal in {self.name}, aborting ...', 1)
            return None

        cp = self.build_comb_pos(self.env_container, long_leg, short_leg)

        if cp.empty() or cp.max_profit <= 0:
            _debug(f'CombinedPosition is empty or max profit is < 0 in {self.name}, aborting ...', 1)
            return None

        """
        # half strike width in relation to max profit for percentage
        self.tp_percentage = \
        5000 * abs(long_leg.options.loc[0, "strike"] - short_leg.options.loc[0, "strike"]) / cp.max_profit
        """

        return cp

    def _get_tasty_variation2(self):

        chain = deepcopy(self.env_container.chain)

        # collect closest to 1/3 width of spread as premium
        if self.opt_type == "c":

            short_leg = chain.calls().short().expiration_close_to_dte(45).delta_close_to(0.3)
            if short_leg:
                chain = chain.remove_by_name(short_leg)

                short_strike = short_leg.options.loc[0, "strike"]
                short_cost = short_leg.options.loc[0, "bid"]

                long_leg = chain \
                    .calls() \
                    .expiration_close_to_dte(45) \
                    .long() \
                    .strike_higher_or_eq(short_strike) \
                    .premium_to_strike_diff(1 / 3, short_cost, short_strike, "long")
            else:
                _debug(f'Short leg is missing in {self.name}, aborting ...', 1)
                return None
        else:
            short_leg = chain.puts().short().expiration_close_to_dte(45).delta_close_to(0.3)

            if short_leg:
                chain = chain.remove_by_name(short_leg)

                short_strike = short_leg.options.loc[0, "strike"]
                short_cost = short_leg.options.loc[0, "bid"]

                # collect closest to 1/3 width of spread as premium
                # long has higher strike with puts, lower with calls
                # short_price - long_price = 1/3 strike diff * 100
                long_leg = chain \
                    .puts() \
                    .expiration_close_to_dte(45) \
                    .long() \
                    .strike_lower_or_eq(short_strike) \
                    .premium_to_strike_diff(1 / 3, short_cost, short_strike, "long")
            else:
                _debug(f'Short leg is missing in {self.name}, aborting ...', 1)
                return None

        if not long_leg or not short_leg:
            _debug(f'Long leg is missing in {self.name}, aborting ...', 1)
            return None

        if long_leg.options.loc[0, "expiration"] != short_leg.options.loc[0, "expiration"]:
            _debug(f'Expirations are not equal in {self.name}, aborting ...', 1)
            return None

        cp = self.build_comb_pos(self.env_container, long_leg, short_leg)

        if cp.empty() or cp.max_profit <= 0:
            _debug(f'CombinedPosition is empty or max profit is < 0 in {self.name}, aborting ...', 1)
            return None

        """
        # half strike width in relation to max profit for percentage
        self.tp_percentage = \
        5000 * abs(long_leg.options.loc[0, "strike"] - short_leg.options.loc[0, "strike"]) / cp.max_profit
        """

        return cp


class VerticalCreditSpread(OptionStrategy):

    # bullish = put spread
    # bearish = call spread

    def __init__(self, env: EnvContainer, opt_type: str, recommendation_threshold=DEFAULT_THRESHOLD, variation="tasty"):
        """
        :param recommendation_threshold: test_env must be greater than treshold in order to start building the position
        """
        self.opt_type = opt_type

        if variation == "tasty":
            position_builder = self._get_tasty_variation2
        elif variation == "personal":
            position_builder = self._get_personal_variation
        else:
            position_builder = self._get_tasty_variation

        conditions = {
            "IVR": [high_ivr],

            "short term outlook": [bullish] if opt_type == "p" else [bearish],
            "mid term outlook": [neutral, bullish] if opt_type == "p" else [neutral, bearish],

            "analyst rating": [neutral, bullish] if opt_type == "p" else [neutral, bearish]
        }

        super().__init__(name=f'Vertical {"Call" if opt_type == "c" else "Put"} Credit Spread',
                         position_builder=position_builder,
                         env=env,
                         conditions=conditions,
                         hints=["Call spreads are more liquid and tighter",
                                "Put spread offer higher premium and higher theta",
                                "You can sell a put credit spread against a call credit spread to reduce delta without additional"
                                " capital",
                                "Prefer bearish credit spreads; short credit = short delta"],
                         tp_perc=50,
                         sl_perc=100,
                         close_dte=21,
                         close_perc=100,
                         recommendation_threshold=recommendation_threshold)

    def _get_tasty_variation(self):
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

        else:
            # buy 20-25 delta
            long_leg = chain.expiration_close_to_dte(45).puts().long().delta_close_to(0.25)

            # remove long leg
            chain = chain.remove_by_name(long_leg)

            # sell 30-35 delta
            short_leg = chain.expiration_close_to_dte(45).puts().short().delta_close_to(0.3)

        if not long_leg or not short_leg:
            _debug(f'Long leg is missing in {self.name}, aborting ...', 1)
            return None

        if long_leg.options.loc[0, "expiration"] != short_leg.options.loc[0, "expiration"]:
            _debug(f'Expirations are not equal in {self.name}, aborting ...', 1)
            return None

        cp = self.build_comb_pos(self.env_container, long_leg, short_leg)

        if cp.empty() or cp.max_profit <= 0:
            _debug(f'CombinedPosition is empty or max profit is < 0 in {self.name}, aborting ...', 1)
            return None

        """
        # half strike width in relation to max profit for percentage
        self.tp_percentage = \
        5000 * abs(long_leg.options.loc[0, "strike"] - short_leg.options.loc[0, "strike"]) / cp.max_profit
        """

        return cp

    def _get_tasty_variation2(self):

        chain = deepcopy(self.env_container.chain)

        # collect closest to 1/3 width of spread as premium
        if self.opt_type == "c":

            short_leg = chain.calls().short().expiration_close_to_dte(45).delta_close_to(0.3)

            if short_leg:
                chain = chain.remove_by_name(short_leg)

                short_strike = short_leg.options.loc[0, "strike"]
                short_cost = short_leg.options.loc[0, "bid"]

                long_leg = chain \
                    .calls() \
                    .expiration_close_to_dte(45) \
                    .long() \
                    .strike_higher_or_eq(short_strike) \
                    .premium_to_strike_diff(1 / 3, short_cost, short_strike, "long")
            else:
                _debug(f'Short leg is missing in {self.name}, aborting ...', 1)
                return None

        else:
            short_leg = chain.puts().short().expiration_close_to_dte(45).delta_close_to(0.3)

            if short_leg:
                chain = chain.remove_by_name(short_leg)

                short_strike = short_leg.options.loc[0, "strike"]
                short_cost = short_leg.options.loc[0, "bid"]

                # collect closest to 1/3 width of spread as premium
                # long has higher strike with puts, lower with calls
                # short_price - long_price = 1/3 strike diff * 100
                long_leg = chain \
                    .puts() \
                    .expiration_close_to_dte(45) \
                    .long() \
                    .strike_lower_or_eq(short_strike) \
                    .premium_to_strike_diff(1 / 3, short_cost, short_strike, "long")
            else:
                _debug(f'Short leg is missing in {self.name}, aborting ...', 1)
                return None

        if not long_leg or not short_leg:
            _debug(f'Long leg is missing in {self.name}, aborting ...', 1)
            return None

        if long_leg.options.loc[0, "expiration"] != short_leg.options.loc[0, "expiration"]:
            _debug(f'Expirations are not equal in {self.name}, aborting ...', 1)
            return None

        cp = self.build_comb_pos(self.env_container, long_leg, short_leg)

        if cp.empty() or cp.max_profit <= 0:
            _debug(f'CombinedPosition is empty or max profit is < 0 in {self.name}, aborting ...', 1)
            return None

        """
        # half strike width in relation to max profit for percentage
        self.tp_percentage = \
        5000 * abs(long_leg.options.loc[0, "strike"] - short_leg.options.loc[0, "strike"]) / cp.max_profit
        """

        return cp

    def _get_personal_variation(self):
        return None


# todo
class CalendarSpread(OptionStrategy):
    name = "Calendar Spread"

    def __init__(self, recommendation_threshold=DEFAULT_THRESHOLD):
        """
        :param recommendation_threshold: test_env must be greater than treshold in order to start building the position
        """
        self.threshold = recommendation_threshold
        self.position_builder = self._get_tasty_variation
        self.conditions = {
            "IVR": [high_ivr],  # high_ivr, put functions here that return values from 0 to 1 or from -1 to 1

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
                         recommendation_threshold=recommendation_threshold)

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


# todo
class CashSecuredPut(OptionStrategy):  # todo -> wheel
    ...
    """
    low price
    high options oi & vol
    positive earnings per share
    +dividend
    positive mid/long term outlook
    positive analyst rating
    at least neutral short term outlook
    sell 30% P(ITM) put option
    roll if stock gets close to strike
    sell at half credit
    sell at 21 dte
    buy close to 45 dte
    """


class CoveredCall(OptionStrategy):

    def __init__(self, env: EnvContainer, recommendation_threshold=DEFAULT_THRESHOLD):
        conditions = {
            "IVR": [high_ivr],

            "RSI20d": [low_rsi],

            "short term outlook": [neutral, bullish],
            "mid term outlook": [neutral, bullish],

            "analyst rating": [neutral, bullish]
        }

        super().__init__(name="Covered Call",
                         position_builder=self._get_tasty_variation,
                         env=env,
                         conditions=conditions,
                         hints=[],
                         tp_perc=50,
                         sl_perc=100,
                         close_dte=21,
                         close_perc=50,
                         recommendation_threshold=recommendation_threshold)

    def _get_tasty_variation(self):
        oc = self.env_container.chain \
            .calls() \
            .short() \
            .otm() \
            .expiration_close_to_dte(45) \
            .delta_close_to(0.3)

        if oc:
            cp = self.build_comb_pos(self.env_container, oc)
            cp.add_asset(Stock(self.env_container.ticker, self.env_container.u_bid, self.env_container.u_ask), 100)

            return cp

        return None


# todo
class RatioSpread(OptionStrategy):
    ...


class CustomStratGenerator:

    @staticmethod
    def get_strats(chain: OptionChain,
                   env: EnvContainer,
                   target_param_tuple: tuple,
                   strat_type: str,
                   filters: List[List[Union[Callable, float]]] = None) -> Optional[List[OptionStrategy]]:
        """
        :param env:
        :param chain:
        :param filters:
            methods of option chain to apply to it before iterating, e.g. expiration_close_to(45)
            example:
                [[exp_close_to, 45], ...]
        :param strat_type
            Spreads
                vertical
                horizontal
                diagonal = vert+hori
            Single leg
                long/short put/call
                csp
                cc
            Flies
            Condors
            Custom

        :param target_param_tuple:
            max_gain: ("g", 0),
            risk: ("le", 100),
            delta_exp: ("eq", True),               -> True= ">=" 0, else False
            theta/delta: ("ge", 0.5),
            some_attribute_of_option_strat: (comparator, value)

            simple_attributes = {"risk", "tp", "sl", "close_dte", "bpr", "max_gain",
                                 "delta", "gamma", "vega", "theta", "rho",
                                 "delta_exp", "gamma_exp", "vega_exp", "theta_exp", "rho_exp"}
            complex_attributes = {"ptp", "prob_of_profit", "psl", "tp_med_d", "tp_avg_d", "close_pop", "close_pn"}

        """

        # todo maybe put the vertical spread part to the vertical spread class

        # can be computed without explicitly creating the combined position
        simple_attributes = {"risk", "tp", "sl", "close_dte", "bpr", "max_gain", "rom", "rom50", "prof_time",
                             "prof_time50",
                             "delta", "gamma", "vega", "theta", "rho",
                             "delta_exp", "gamma_exp", "vega_exp", "theta_exp", "rho_exp"}
        complex_attributes = {"ptp", "prob_of_profit", "psl", "tp_med_d", "tp_avg_d", "close_pop", "close_pn",
                              "e_tp_close"}

        attribute_keys = [_t[0] for _t in target_param_tuple]

        if any([True for key in attribute_keys if key not in simple_attributes and key not in complex_attributes]):
            raise RuntimeError("Unsupported key provided!")

        private_chain = deepcopy(chain)
        if filters:
            for filt in filters:
                arg_string = ""
                for arg in filt[1:]:
                    if isinstance(arg, str):
                        arg_string += f'"{arg}",'
                    else:
                        arg_string += f'{arg},'

                private_chain = eval(f'private_chain.{filt[0].__name__}({arg_string[:-1]})')

        # <editor-fold desc="Vertical Spread">

        _debug(f"[{env.ticker}] Start checking all vertical spreads for ticker {env.ticker} ...")

        strat_check_counter = 0
        strat_name_counter = 1

        # set tp perc & sl perc
        tp_perc = 50
        sl_perc = 100
        latest_close_dte = 21

        # if no probs or complicated parameters are given in target_param_dict, compute simple params manually

        # difference set is empty = no complicated attributes present
        # if not set(target_param_dict.keys()) - simple_attributes:

        for opt_type in ["p", "c"]:

            for expiration_date in chain.expirations:

                tmp_chain = private_chain.expiration_date(expiration_date)
                df = tmp_chain.options

                strikes = list(set(df["strike"].tolist()))
                strikes.sort()

                #print(f'{len(strikes)} strikes for expiration: {expiration_date}')

                close_dte = date_str_to_dte(expiration_date) - latest_close_dte

                for long_strike in strikes:

                    for short_strike in strikes:

                        if long_strike != short_strike:

                            strat_check_counter += 1

                            # <editor-fold desc="setting parameters">

                            long = df.loc[(df["strike"] == long_strike) &
                                          (df["direction"] == "long") &
                                          (df["contract"] == opt_type)]
                            short = df.loc[(df["strike"] == short_strike) &
                                           (df["direction"] == "short") &
                                           (df["contract"] == opt_type)]

                            if long.empty or short.empty:
                                continue

                            if short["bid"].iloc[0] <= 0:
                                continue

                            # compute risk
                            # abs(long strike - short strike) + long ask - short bid
                            long_ask = long["ask"].iloc[0]
                            short_bid = short["bid"].iloc[0]
                            trade_cost = long_ask - short_bid
                            strike_diff = abs(long_strike - short_strike)

                            risk = (strike_diff + trade_cost) * 100  # todo wrong?

                            # for calls: (for puts inverted <>)
                            # if long strike < short strike => debit => max gain = strike_diff - debit paid
                            # if long strike > short strike => credit => max gain = credit received

                            if opt_type == "c":
                                # compute max gain
                                if long_strike > short_strike:
                                    risk = (strike_diff - trade_cost)
                                    max_gain = -trade_cost
                                else:  # short strike > long strike
                                    risk = (trade_cost + strike_diff)
                                    max_gain = (strike_diff - trade_cost)
                            else:
                                # compute max gain
                                if long_strike < short_strike:
                                    risk = (trade_cost + strike_diff)
                                    max_gain = -trade_cost
                                else:  # short strike < long strike
                                    risk = (trade_cost - strike_diff)
                                    max_gain = (strike_diff - trade_cost)

                            max_gain *= 100
                            risk *= 100
                            bpr = risk

                            tp = tp_perc / 100 * max_gain
                            sl = -1 if sl_perc == 100 else sl_perc / 100 * risk
                            rom = float('inf') if bpr == 0 else max_gain / bpr
                            rom50 = rom / 2

                            delta = long["delta"].iloc[0] + short["delta"].iloc[0]
                            gamma = long["gamma"].iloc[0] + short["gamma"].iloc[0]
                            theta = long["theta"].iloc[0] + short["theta"].iloc[0]
                            vega = long["vega"].iloc[0] + short["vega"].iloc[0]
                            rho = long["rho"].iloc[0] + short["rho"].iloc[0]

                            gamma_exp = gamma >= 0
                            delta_exp = delta >= 0
                            vega_exp = vega >= 0
                            theta_exp = theta >= 0
                            rho_exp = rho >= 0

                            # </editor-fold>

                            unfitting = False
                            simple_to_test = [tup for tup in target_param_tuple if tup[0] in simple_attributes]
                            for var, comp, val in simple_to_test:
                                # calls the function given by val[0] on local correspondant of param from dict
                                if not eval(f'{var}.{"__" + comp + "__"}({val})'):
                                    unfitting = True
                                    break
                                else:
                                    _debug(f'{var}: {var}.{"__" + comp + "__"}({val}) = True', 3)

                            if unfitting:
                                continue

                            _debug(f'Exp: {expiration_date}, OptType: {opt_type}, '
                                   f'LongS: {long_strike}, ShortS: {short_strike} '
                                   f'Risk: {risk}, Tp: {tp}', 3)

                            long_leg = tmp_chain \
                                .long() \
                                .expiration_date(expiration_date) \
                                .strike_eq(long_strike)
                            short_leg = tmp_chain \
                                .short() \
                                .expiration_date(expiration_date) \
                                .strike_eq(short_strike)

                            if opt_type == "c":
                                long_leg = long_leg.calls()
                                short_leg = short_leg.calls()
                            else:
                                long_leg = long_leg.puts()
                                short_leg = short_leg.puts()

                            # print(pd.concat((long_leg.options, short_leg.options)).head())
                            comb_pos = OptionStrategy.build_comb_pos(env, long_leg, short_leg)

                            def position_builder_function() -> CombinedPosition:
                                return comb_pos

                            opt_strat = OptionStrategy(name=env.ticker.upper() + " Custom Vertical Spread #" +
                                                            str(strat_name_counter),
                                                       position_builder=position_builder_function,
                                                       env=env,
                                                       tp_perc=tp_perc,
                                                       sl_perc=sl_perc,
                                                       close_dte=latest_close_dte,
                                                       recommendation_threshold=-float('inf'))

                            #print(opt_strat)

                            strat_name_counter += 1

                            # check other specific conditions here

                            hard_to_test = [tup for tup in target_param_tuple if tup[0] in complex_attributes]
                            for var, comp, val in hard_to_test:
                                # calls the function given by val[0] on local correspondant of param from dict
                                if not eval(f'opt_strat.{var}.{"__" + comp + "__"}({val})'):
                                    # print(opt_strat)
                                    """x = f'opt_strat.{param}'
                                    print(f'{param}: opt_strat.{param} = {eval(x)} not {val[0]}({val[1]})')"""
                                    unfitting = True
                                    break
                                else:
                                    _debug(f'{var}: opt_strat.{var}.{"__" + comp + "__"}({val}) = True', 3)

                            if unfitting:
                                continue

                            yield opt_strat

        _debug(f'Checked {strat_check_counter} strategies!', 1)

        # </editor-fold>


# </editor-fold>

# <editor-fold desc="Ticker functions">
def get_sp500_tickers(exclude_sub_industries=('Pharmaceuticals',
                                              'Managed Health Care',
                                              'Health Care Services')):
    df = pd.read_html('https://en.wikipedia.org/wiki/List_of_S%26P_500_companies')[0]
    df = df[~df['GICS Sub-Industry'].isin(exclude_sub_industries)]
    _l = df['Symbol'].values.tolist()

    # no tickers can be found for this by monte carlo
    _l.remove("BRK.B")
    _l.remove("BF.B")
    return _l


def get_high_option_vol_tickers(exclude_sub_industries=('Pharmaceuticals',
                                                        'Managed Health Care',
                                                        'Health Care Services')):
    return ["gnus", "crbp", "idex", "ivr", "mitt", "nok", "bbd", "uvxy", "ino", "kgc", "et", "cron", "srne", "pbr",
            "acb", "bb", "ung", "apt", "sqqq", "fcel"]


def get_trending_theta_strat_tickers():
    return ["LAZR", "SNDL", "CCIV", "T", "RBLX", "AMC", "WKHS", "AMZN", "RKT", "MSFT", "NIO", "GME", "PINS",
            "AAPL", "SPY", "TSLA", "AMD", "PLTR", "MVIS"]


# </editor-fold>


@timeit
def get_market_recommendations(ticker_f, start_from=None, use_predef=False, _ssc=tuple(), _filters=list()):
    print(f"[{get_timestamp()}] Start getting market recommendations")
    constraints = StrategyConstraints(strict_mode=False, min_oi30=100, min_vol30=1000, min_sinle_opt_vol=0)
    tickers = ticker_f()
    #tickers = ["nio"]
    mcs = MonteCarloSimulator(tickers)
    pos_df = pd.DataFrame()
    if start_from:
        tickers = tickers[tickers.index(start_from):]
    for _t in tickers:
        try:
            tmp_df = propose_strategies(_t, constraints, mcs,
                                        use_predefined_strats=use_predef,
                                        single_strat_cons=_ssc,
                                        filters=_filters)
            pos_df = pd.concat([pos_df, tmp_df])
        except Exception as e:
            _warn(f'Exception occured when getting strategies for ticker {_t}: {e}\n'
                  f'{traceback.print_tb(e.__traceback__)}', level=2)
            continue

    if not pos_df.empty:
        with open(f'StrategySummary-{get_timestamp().split()[0].replace(":", "-")}.json', "w") as file:
            pos_df.reset_index(inplace=True)
            pos_df.to_json(file)
        show(pos_df)


def read_and_show_dataframe(path: str = None):
    if path is None:
        path = f'StrategySummary-{get_timestamp().split()[0].replace(":", "-")}.json'
    try:
        with open(path, "r") as json_file:
            show(pd.read_json(json_file))
    except Exception as e:
        print(f'Exception occurred while trying to read file "{path}": {e}')


def binom_test():
    # EuroOption.price() is best with 5-10 iterations, 1s for 10k iterations

    opt_type = "c"
    u_price = 11.67
    strike = 8.5
    dte_then = 40
    risk_free_rate = 0.02
    is_call = opt_type == "c"
    iv = 1.3

    binomial_iterations = 50

    def a():
        return EuroOption(u_price,
                          strike,
                          risk_free_rate,
                          dte_then / 365.0,

                          binomial_iterations,
                          {'is_call': is_call,
                           'eu_option': False,
                           'sigma': iv}) \
            .price()

    def b():
        opt_price, _, delta, gamma, theta, vega, rho = get_greeks(opt_type,
                                                                  u_price,
                                                                  strike,
                                                                  max(dte_then / 365.0, 0.001),
                                                                  risk_free_rate,
                                                                  iv)
        return opt_price

    prices = []
    prices_b = []
    initial_dte = dte_then
    for i in range(initial_dte - 1):
        dte_then -= 1
        prices.append(a())
        prices_b.append(b())

    plt.plot(range(len(prices)), prices)
    plt.plot(range(len(prices_b)), prices_b)
    plt.show()


def model_test():
    # EuroOption.price() is best with 5-10 iterations, 1s for 10k iterations

    opt_type = "p"
    u_price = 11.67
    strike = 8.5
    dte_then = 38
    risk_free_rate = 0.02
    is_call = opt_type == "c"
    iv = 1.3

    start = time.time()

    binomial_iterations = 10

    def a():
        return EuroOption(u_price,
                          strike,
                          risk_free_rate,
                          dte_then / 365.0,

                          binomial_iterations,
                          {'is_call': is_call,
                           'eu_option': False,
                           'sigma': iv}) \
            .price()

    binom_price = sum([a() for _ in range(10000)]) / 10000

    print(f'            Binomial took {time.time() - start:.8f} seconds: {binom_price}')

    start2 = time.time()

    def b():
        opt_price, _, delta, gamma, theta, vega, rho = get_greeks(opt_type,
                                                                  u_price,
                                                                  strike,
                                                                  max(dte_then / 365.0, 0.001),
                                                                  risk_free_rate,
                                                                  iv)
        return opt_price

    opt_price = sum([b() for _ in range(10000)]) / 10000

    print(f'Bjerksund-Strensland took {time.time() - start2:.8f} seconds: {opt_price}')


def test_greeks():
    opt_type = "p"
    u_price = 11.45
    strike = 8.5
    dte_then = 38
    risk_free_rate = 0.02
    is_call = opt_type == "c"
    iv = 1.3

    # they use call iv for put calculations instead of put iv
    # prove: calc call delta -> correct, their put delta is 1-call delta (what you would get using call IV for put calc)
    # you cant do this bc put & call may have different IVs, their deltas only sum to 1 iff their IVs are identical

    print(get_greeks(opt_type, u_price, strike, dte_then / 365, risk_free_rate, iv))


if __name__ == "__main__":

    # TODO why do we only get positive delta trades???
    # TODO pretty representation of option chain, yk

    # todo pops differ for 0 dte
    # todo break even on exit day

    # todo print warnings to strat sum
    # todo add todays pl graph to non-auto

    # todo is monte carlo assuming too much volatility?
    # todo timestamp stock_data_pickle and redownload every day once
    # todo collect 1/3 of width of spreads as tasty variation without any other limitations (have to iterate)

    # todo expected value for spreads:
    #  p(<=max prof u)*max prof + p(>= max loss u) +
    #  (1 - p(<=max prof u) - p(>= max loss u)) * (max prof + max loss) / 2 ... not exact!!!

    # ERRORS
    # todo Exception occured when getting strategies for ticker ivr: math domain error
    # todo Exception occured when getting strategies for ticker et: math domain error
    # todo avg/med days until tp hit
    # todo pins +1 Jun 18 C 37.0 @ 30.0/34.55 = 3455.00 $   negative max loss
    # 			-1 Jun 18 C 25.0 @ 50.95/54.6 = -5095.00 $
    # todo negative max profit on long call msft +1 Jun 18 C 95.0 @ 120.3/120.85
    # todo Exception occured when getting strategies for ticker AFL: single positional indexer is out-of-bounds
    # todo Getting options meta data failed for ticker BLK with exception: invalid literal for int() with base 10: '\n'
    # todo Exception occured when getting strategies for ticker BIO: 'OptionChain' object has no attribute 'options'
    # todo Exception occured when getting strategies for ticker AZO: single positional indexer is out-of-bounds

    # LONG TERM
    # todo prob forecasts use historic variance, plot uses implied vol ... maybe use this too for forecasts?

    read_and_show_dataframe()

    t = "amc"
    ssc = (
        ("risk", "le", 100),
        ("rom", "le", 2),  # just to prefilter and make things faster
        ("max_gain", "ge", 20),
        ("rom50", "ge", 0.2),
        ("rom50", "le", 5),
        ("close_pn", "ge", 0.6),
        #("delta", "le", 0),
    )
    f = [
        [OptionChain.volume, 10],
        [OptionChain.expiration_range, 30, 90]
    ]

    """
    propose_strategies(t,
                       StrategyConstraints(strict_mode=False,
                                           min_oi30=1000,
                                           min_vol30=1000,
                                           min_sinle_opt_vol=10),  # used to prefilter chain
                       MonteCarloSimulator(tickers=[t]),
                       use_predefined_strats=False,
                       single_strat_cons=ssc,
                       filters=f,
                       mode="manual")
    # """

    """
    check_spread(StrategyConstraints(strict_mode=False,
                                     min_oi30=1000,
                                     min_vol30=1000,
                                     min_sinle_opt_vol=10))
    #"""

    get_market_recommendations(get_trending_theta_strat_tickers,
                               use_predef=False, _ssc=ssc, _filters=f)
