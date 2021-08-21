from matplotlib import pyplot as plt

ranked_reward = 6
pve_reward = 75
slp_earnings = list(range(pve_reward, 20*ranked_reward))
seventyfive = [0.75 * x for x in slp_earnings]
pve_plus_quarter = [pve_reward+0.25*(x-pve_reward) for x in slp_earnings]
pve_plus_halve = [pve_reward+0.5*(x-pve_reward) for x in slp_earnings]
pve_plus_3 = [pve_reward+0.33*(x-pve_reward) for x in slp_earnings]

plt.plot(slp_earnings, seventyfive, label="75 %")
plt.plot(slp_earnings, pve_plus_quarter, label="PvE + 25 %")
plt.plot(slp_earnings, pve_plus_3, label="PvE + 33 %")

plt.legend()
plt.show()