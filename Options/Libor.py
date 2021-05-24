import urllib.request
from bs4 import BeautifulSoup
from Option_utility import abs_min_closest_index


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
    return libor_rates[abs_min_closest_index(libor_expiries, annualized_dte * 365)]


try:
    libor_rates = get_libor_rates()
except:
    libor_rates = [0.0629, 0.0708, 0.0993, 0.122, 0.1553, 0.1838, 0.2628]

libor_expiries = (1, 7, 30, 61, 92, 182, 365)

# </editor-fold>