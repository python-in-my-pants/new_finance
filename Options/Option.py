import pandas as pd
from Option_utility import *
from DDict import DDict
from option_greeks import get_greeks
from Libor import get_risk_free_rate


class Option:

    def __init__(self, name, opt_type, expiration, strike, bid, ask, vol,
                 delta, gamma, theta, vega, rho, iv, oi):

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

        self.expiration_dt = str_to_date(self.expiration)
        self.dte = date_to_dte(self.expiration_dt)

        self.greeks = DDict({"delta": delta, "gamma": gamma, "theta": theta, "vega": vega, "rho": rho})

        self.iv = iv
        self.oi = oi

    @staticmethod
    def new_parse_option(date_str: str, opt_type: str, strike: float, bid: float, ask: float, u_ask: float, ticker: str):

        exp_date = opt_date_str_to_date(date_str)
        exp_date_str = datetime_to_str(exp_date)
        dte = date_to_dte(exp_date)
        opt_price, iv, delta, gamma, theta, vega, rho = get_greeks(opt_type.lower(),
                                                                   u_ask,
                                                                   strike,
                                                                   max(dte/365.0, 0.001),
                                                                   get_risk_free_rate(dte / 365.0),
                                                                   None,
                                                                   0,
                                                                   ask)

        # name, opt_type, expiration, strike, bid, ask, vol, delta, gamma, theta, vega, rho, iv, oi
        return Option(f'{ticker.upper()}{date_to_opt_name_format(exp_date)}{opt_type.upper()}{int(strike*1000):08}',
                      opt_type.lower(), exp_date_str, strike, bid, ask, -1, delta, gamma, theta, vega, rho, iv, -1)

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
        return f'{date_to_opt_format(self.expiration)} {self.opt_type.upper()} {self.strike: >4.2f} @ {self.bid: >3.2f}/{self.ask: >3.2f}'

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
        oi = row["OI"]
        return Option(name, opt_type, expiration, strike, bid, ask, vol, delta, gamma, theta, vega, rho, iv, oi)

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