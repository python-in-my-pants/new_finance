from selenium import webdriver
from bs4 import BeautifulSoup
import urllib.request
from selenium.webdriver.firefox.options import Options
from selenium.webdriver import ActionChains
import time
import pandas as pd

options = Options()
options.headless = True

pd.set_option("display.max_rows", None, "display.max_columns", None)
pd.set_option('display.width', 1000)
pd.set_option('display.max_colwidth', 500)
pd.set_option('display.float_format', lambda x: '%.5f' % x)

ltv = {
    "MATIC": 0.5,
    "DAI": 0.75,
    "USDC": 0.8,
    "USDT": 0,
    "WETH": 0.8,
    "WBTC": 0.7,
    "AAVE": 0.5,
}

stable_coins = ["DAI", "USDT", "USDC"]


def get_rates():
    browser = webdriver.Firefox(options=options)

    try:
        print("Getting website ...")
        browser.get('https://app.aave.com/markets')

        # wait for page to load
        time.sleep(2)

        # click the 3rd button (should lead to polygon)
        browser.find_elements_by_class_name("SelectMarketPanel__select-button")[2].click()

        print("Waiting for JS to finish ...")
        time.sleep(5)

        source = browser.find_element_by_xpath("//*").get_attribute("outerHTML")
        soup = BeautifulSoup(source, 'html.parser')

        info = []

        for row in soup.select('div[class*="TableItemWrapper"]'):

            ticker = "NONE"

            for token_name_div in row.select('p[class*="TokenIcon__name"]'):
                #print(token_name_div.get_text())
                _t = token_name_div.get_text()
                if "(" in _t:
                    ticker = _t.split("(")[1][:-1]
                else:
                    ticker = _t

            rates = [0, 0, -1, 0]

            for i, p in enumerate(row.select('p[class*="ValuePercent__value"]')):
                rates[i] = float(p.get_text().replace("%", "")) / 100

            # print(ticker, *rates)
            info.append([ticker, *rates[:4]])

        df = pd.DataFrame(info, columns=["Ticker", "Lending rate", "Lending rate (M)", "Borrowing rate", "Borrowing rate (M)"])
        df["Lending (comb)"] = df["Lending rate"] + df["Lending rate (M)"]
        df["Borrowing (comb)"] = -df["Borrowing rate"] + df["Borrowing rate (M)"]
        df["Sum"] = df["Borrowing (comb)"] + df["Lending (comb)"]
        df["LTV"] = df["Ticker"].map(lambda t: ltv[t])
        # df["Profit factor"] = 1 + df["Lending (comb)"] + df["LTV"] * (1 - df["Borrowing (comb)"])

        return df

    finally:
        browser.quit()
        print("Quitting selenium ...")


if input("Request fresh data? ") not in ("y", "Y"):
    try:
        rates = pd.read_json("aave_rates.json")
        print("Read from file!")
    except:
        rates = get_rates()
        rates.to_json("aave_rates.json")
        print("saved")
else:
    rates = get_rates()
    rates.to_json("aave_rates.json")
    print("saved")

print(rates)

pair_comp = dict()
for ticker in rates["Ticker"].values:
    for ticker_inner in rates["Ticker"].values:
        outer = rates.loc[rates["Ticker"] == ticker]
        inner = rates.loc[rates["Ticker"] == ticker_inner]
        a = float(outer["Lending (comb)"])
        b = float(outer["LTV"])
        c = float(inner["Borrowing (comb)"])
        # print(f'{ticker} - {ticker_inner}: {a} {b} {c}')
        pair_comp[f'Col: {ticker} - Loan: {ticker_inner}'] = a + b * (1 + c)

print("\nBest loan oppertunity on AAVE (Polygon):")

with_unstable = pd.DataFrame({
    "Pair": list(pair_comp.keys()),
    "Value": list(pair_comp.values())
})

unstable_coins = set(ltv.keys()) - set(stable_coins)
d = {k: v for k, v in pair_comp.items() if not any([u_coin in k for u_coin in unstable_coins])}
without_unstable = pd.DataFrame({
    "Pair": list(d.keys()),
    "Value": list(d.values())
})

print(f'\tGeneral: {with_unstable.sort_values("Value", ascending=False).iloc[0].values}\n'
      f'\t Stable: {without_unstable.sort_values("Value", ascending=False).iloc[0].values}')

