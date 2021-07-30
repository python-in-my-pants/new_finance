from matplotlib import pyplot as plt
import numpy as np

asset1_prop = .5 #float(input("Asset 1 proportion in %: ")) / 100
asset2_prop = 1-asset1_prop

asset1_current_price = 5.3#float(input("Current price asset 1: "))
asset2_current_price = 53#float(input("Current price asset 2: "))

ratio = asset1_current_price / asset2_current_price


def get_il(new_asset1_price):
    print("Pool: ", asset1_prop, "of asset 1 @", asset1_current_price, "=", asset1_prop*asset1_current_price)
    print("      ", asset2_prop*ratio, "of asset 2 @", asset2_current_price, "=", asset2_prop*asset2_current_price*ratio)

    old_pool_val = asset1_current_price*asset1_prop + asset2_current_price*asset2_prop*ratio
    new_pool_val = asset1_prop * new_asset1_price + asset2_prop*asset2_current_price*ratio
    new_ratio = new_asset1_price / asset2_current_price

    new_asset1_prop = new_pool_val * asset1_prop / new_asset1_price
    new_asset2_prop = new_pool_val - new_asset1_prop

    print("Pool: ", new_asset1_prop, "of asset 1 @", new_asset1_price, "=", asset1_prop * new_asset1_price)
    print("      ", new_asset2_prop * new_ratio, "of asset 2 @", asset2_current_price, "=",
          asset2_prop * asset2_current_price * new_ratio)

    print(old_pool_val, new_pool_val, new_asset1_prop, new_asset2_prop)
    return 0


get_il(2*asset1_current_price)
"""impermanent_losses = [
    get_il(future_asset1_price)
    for future_asset1_price in np.linspace(0.001, 10*asset1_current_price, 1000)
]"""