import pandas as pd
from Option_utility import *
from DDict import DDict


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