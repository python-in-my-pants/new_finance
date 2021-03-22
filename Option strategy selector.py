import urllib.request
from bs4 import BeautifulSoup
import yfinance as yf
import pandas as pd
from pprint import pprint as pp
import mibian

pd.set_option("display.max_rows", None, "display.max_columns", None)
pd.set_option('display.width', 1000)
pd.set_option('display.max_colwidth', 500)
pd.set_option('display.float_format', lambda x: '%.2f' % x)

ACCOUNT_BALANCE = 2000


def get_options_data(symbol):
    headers = {
        'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_11_5) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/50.0.2661.102 Safari/537.36'}
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
    return 1, 1, 1


# type, positions, credit/debit, max risk, break even on exp, max prof, bpr, return on margin with 50% tp, close by date
# use monto carlo sim for P50
def propose_strategies(yf_obj, defined_risk_only=True, avoid_events=True,
                       min_vol30=25000, min_oi30=5000, strict_mode=False):
    # todo get short term price outlook (technical analysis, resistances from barchart.com etc)

    # <editor-fold desc="Get info">
    ticker = yf_obj.info.symbol
    u_price = yf_obj.info.ask

    option_chain = get_option_chain(yf_obj)
    puts = option_chain.puts
    calls = option_chain.calls

    ivr, vol30, oi30, next_earnings = get_options_data(ticker)
    short, mid, long = get_underlying_outlook(ticker)
    # </editor-fold>

    # 1 # liquidity

    if vol30 < min_vol30:
        print(f'Warning! Average volume is < {min_vol30}: {vol30}')
        if strict_mode:
            return

    if oi30 < min_oi30:
        print(f'Warning! Average open interest is < {min_oi30}: {oi30}')
        if strict_mode:
            return

    # 2 # Price

    if u_price * 100 > ACCOUNT_BALANCE / 2:
        print(
            f'Warning! Underlying asset price exceeds account risk tolerance! {u_price * 100} > {ACCOUNT_BALANCE / 2}')
        if strict_mode:
            return

    # 3 # upcoming events (dividends, earnings, splits) -> todo need to check for news still

    if ...:  # upcoming event
        if avoid_events:
            # restrict expirations to those before earnings and resume
            ...
        else:
            # use special earnings plays (eg IV crush)
            ...

    # 4 # IV rank

    if high(ivr):
        ...

        # defined risk

        # Credit spreads (Bear/Bull)
        # Short condors

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


def get_option_chain(yf_obj):
    # contractSymbol  lastTradeDate  strike  lastPrice    bid    ask  change  percentChange   volume  openInterest
    # ... impliedVolatility  inTheMoney contractSize currency
    option_chain = DDict()
    ticker_data = yf_obj
    expirations = ticker_data.options

    risk_free_interest = 1

    def populate_call_greeks(contracts, dte):
        contracts["delta"] = 0
        contracts["gamma"] = 0
        contracts["vega"] = 0
        contracts["rho"] = 0
        contracts["theta"] = 0

        for contract in contracts:
            call_price = (contract["Bid"] + contract["Ask"]) / 2
            bs_obj = mibian.BS([ticker_data.info.ask,
                                contract["Strike Price"],
                                risk_free_interest,
                                dte],
                               volatility=contract["Implied volatility"],
                               callPrice=call_price)

            contract["delta"] = bs_obj.callDelta
            contract["gamma"] = bs_obj.gamma
            contract["vega"] = bs_obj.vega
            contract["theta"] = bs_obj.callTheta
            contract["rho"] = bs_obj.callRho

    def populate_put_greeks(contracts, dte):
        contracts["delta"] = 0
        contracts["gamma"] = 0
        contracts["vega"] = 0
        contracts["rho"] = 0
        contracts["theta"] = 0

        for contract in contracts:
            put_price = (contract["Bid"] + contract["Ask"]) / 2
            bs_obj = mibian.BS([ticker_data.info.ask,
                                contract["Strike Price"],
                                risk_free_interest,
                                dte],
                               volatility=contract["Implied volatility"],
                               putPrice=put_price)

            contract["delta"] = bs_obj.putDelta
            contract["gamma"] = bs_obj.gamma
            contract["vega"] = bs_obj.vega
            contract["theta"] = bs_obj.putTheta
            contract["rho"] = bs_obj.putRho

    for expiration in expirations:
        chain = ticker_data.option_chain(expiration)
        calls = chain.calls
        puts = chain.puts


    return option_chain


class DDict(dict):
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__
    __hash__ = dict.__hash__


# <editor-fold desc="Requirements">

def high(x):  # x 0 to 100, high if > 50; returns scale from 0 to 1 indicating highness
    return 0 if x <= 50 else (x / 50 if x > 0 else 0) - 1


def low(x):
    return 0 if x >= 50 else 1 - (x / 50 if x > 0 else 0)


def calendar_check(front_m_vol, back_m_vol):
    return front_m_vol >= back_m_vol * 1.2


def best_theta_dte(dtes):
    return min([45 - dte for dte in dtes])


# </editor-fold>


class OptionStrategy:

    def __init__(self, name, requirements, timeframes, strikes, managing, hints=None):
        self.name = name
        self.requirements = requirements
        self.hints = hints
        self.timeframes = timeframes
        self.strikes = strikes
        self.managing = managing


class LongCall(OptionStrategy):

    def test(self, params):
        # option chain, u price, short, mid, long outlook (-1 - 1),
        if params.short_outlook > 0.8:
            # which delta call to choose?
            ...


df = yf.Ticker("FREQ").option_chain(date="2021-05-21").puts
pp(df.loc[df['strike'] == 30])
