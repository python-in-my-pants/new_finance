"""
Use on morning after (early) assignment of short put option to calculate best exit!

Script to compute whether to exercise a long leg or to use a covered stock order after
assignment of long shares (short put).
"""
import yfinance as yf
from datetime import datetime

EXERCISE_FEE = 5
manual = False


def parse_option(opt_string):

    lot, month, day, strike, opt_type, _, premium = opt_string.split()
    exp = datetime.strptime(month + " " + day + " 2021", '%b %d %Y').strftime("%Y-%m-%d")

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

    return lot, exp, strike, opt_type, premium


def check_best_exit(ticker=None, long_leg=None, short_leg=None):

    if not ticker:
        ticker = input("Underlying ticker: ")
    if not long_leg:
        long_leg = input("Long leg: ")
    if not short_leg:
        short_leg = input("Short leg: ")

    _, long_exp, long_strike, long_type, long_leg_buying_price = parse_option(long_leg)
    _, short_exp, short_strike, short_type, short_leg_sell_premium = parse_option(short_leg)

    if long_type != short_type:
        raise NotImplementedError
    if long_type != "put":
        raise NotImplementedError

    option_type = long_type

    credit = (short_leg_sell_premium - long_leg_buying_price) * 100

    if not manual:
        yf_obj = yf.Ticker(ticker)

        current_stock_bid = float(yf_obj.info["bid"])
        current_stock_ask = float(yf_obj.info["ask"])

        if option_type == "put":
            df = yf_obj.option_chain(date=long_exp).puts
            long_leg_bid = float(df.loc[df['strike'] == long_strike]["bid"])
            long_leg_ask = float(df.loc[df['strike'] == long_strike]["ask"])
        elif option_type == "call":
            df = yf_obj.option_chain(date=long_exp).calls
            long_leg_bid = float(df.loc[df['strike'] == long_strike]["bid"])
            long_leg_ask = float(df.loc[df['strike'] == long_strike]["ask"])

        if option_type == "put":
            df = yf_obj.option_chain(date=short_exp).puts
            short_leg_bid = float(df.loc[df['strike'] == short_strike]["bid"])
            short_leg_ask = float(df.loc[df['strike'] == short_strike]["ask"])
        elif option_type == "call":
            df = yf_obj.option_chain(date=short_exp).calls
            short_leg_bid = float(df.loc[df['strike'] == short_strike]["bid"])
            short_leg_ask = float(df.loc[df['strike'] == short_strike]["ask"])

    else:
        current_stock_bid = 23.45
        current_stock_ask = current_stock_bid
        short_leg_bid = short_leg_ask = 0

        long_leg_bid = (817 + max(long_strike - current_stock_bid, 0)*100 - 250) / 100
        long_leg_ask = long_leg_bid + 5

    print(
        "\n Ticker:", ticker, "\n",
        "Bid:   ", current_stock_bid, "\n",
        "Ask:   ", current_stock_ask, "\n",
        "Short option prices:\n",
        "    Bid:", short_leg_bid, "\n",
        "    Ask:", short_leg_ask, "\n",
        "Long option prices:\n",
        "    Bid:", long_leg_bid, "\n",
        "    Ask:", long_leg_ask, "\n",
    )

    print("-"*50)

    covered_stock_mid = credit \
                        + (current_stock_bid*100) \
                        - short_strike * 100 \
                        - (100*(long_leg_ask-long_leg_bid)/2) \
                        + long_leg_bid * 100
    print("Covered stock (mid):\t", int(covered_stock_mid))

    covered_stock_nat = credit \
                        + (current_stock_bid * 100) \
                        - short_strike * 100 \
                        - 100*(long_leg_ask-long_leg_bid) \
                        + long_leg_bid * 100
    print("Covered stock (nat):\t", int(covered_stock_nat))

    exercise = credit - EXERCISE_FEE + (long_strike-short_strike) * 100
    print("\nLegging out (exercise):\t", int(exercise))

    break_even_bid = (short_strike * 100
                     + (100*(long_leg_ask-long_leg_bid)/2)
                     - long_leg_bid * 100
                     - EXERCISE_FEE
                     + (long_strike-short_strike) * 100) / 100
    print(f"\nUse covered stock when stock bid is {break_even_bid:4.2f} or higher, else exercise the long leg.")

    """
    covered_stock_mid_break_even = (-credit + (100*(long_leg_ask-long_leg_bid)/2))/100 + short_strike
    print(f'\nBreak even with covered stock mid (stock bid): {covered_stock_mid_break_even:4.2f}')
    covered_stock_nat_break_even = (-credit + (100*(long_leg_ask-long_leg_bid)))/100 + short_strike
    print(f'Break even with covered stock nat (stock bid): {covered_stock_nat_break_even:4.2f}')
    """

    print("-"*50, "\n")

    if covered_stock_mid > exercise:
        print("Use covered stock with mid fill:", int(covered_stock_mid),
              "( nat fill:", int(covered_stock_nat), ")")
    elif exercise >= covered_stock_mid:
        print("Use exercise:", int(exercise))

t = "FREQ"
a = "+1 May 21 30 P @ 11.55"
b = "-1 Apr 16 35 P @ 13.20"
check_best_exit()

