from eu_option import EuroOption

# option_eu = EuroOption(217.58, 215, 0.05, 0.1, 40, {'tk': 'AAPL', 'is_calc': True, 'start': '2017-08-18', 'end': '2018-08-18', 'eu_option': False})

us_option = EuroOption(217.58, 215, 0.05, 0.1, 40,
                       {'is_call': True, 'eu_option': False, 'sigma': 1})

print(us_option.price())
