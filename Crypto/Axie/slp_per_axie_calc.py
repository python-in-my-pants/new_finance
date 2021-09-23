if __name__ == "__main__":

    from matplotlib import pyplot as plt

    slp_per_own_match = 18
    pve_slp = 75
    pvp_slp_per_scholar = 9
    own_winrate = 0.55
    scholar_winrate = 0.5
    max_axies = 100
    avg_scholar_axie_price = 120
    slp_price = 0.06

    def get_energy(n):
        if n < 3:
            return 0
        if 3 <= n <= 9:
            return 20
        if 10 <= n <= 19:
            return 40
        if n >= 20:
            return 60

    def get_own_slp_per_day(axies):
        if axies < 3:
            return 0
        return pve_slp + slp_per_own_match * get_energy(axies) * own_winrate

    def get_slp_per_day_with_scholars(axies, axies_per_scholar, scholar_cut=0.5):

        if axies < 3:
            return 0
        axies -= 3
        own_slp = get_own_slp_per_day(3)

        scholar_cum_slp = sum([pve_slp + scholar_winrate * get_energy(axies_per_scholar) * pvp_slp_per_scholar
                               for _ in range(axies // axies_per_scholar)])

        if axies % axies_per_scholar >= 3:
            scholar_cum_slp += pve_slp + scholar_winrate * get_energy(axies%axies_per_scholar) * pvp_slp_per_scholar

        return own_slp + scholar_cum_slp * scholar_cut

    keep_all_axies = [get_own_slp_per_day(num_axies) / num_axies for num_axies in range(3, max_axies)]

    def slp_with_n_axies_per_scholar(n):
        return [get_slp_per_day_with_scholars(num_axies, n) / num_axies for num_axies in range(3, max_axies)]

    def plot_slp_per_axie():
        plt.plot(range(3, max_axies), slp_with_n_axies_per_scholar(3),  label="Scholars 3")
        plt.plot(range(3, max_axies), slp_with_n_axies_per_scholar(10),  label="Scholars 10")
        plt.plot(range(3, max_axies), keep_all_axies, label="Keep axies")

        plt.legend()
        plt.grid(True)
        plt.show()

    def get_risk(axies, axies_per_scholar):
        axies -= 3
        return max(0.01, (axies // axies_per_scholar) * avg_scholar_axie_price)

    def plot_earnings():

        with_scholars_3 = [slp_price * slp * n for slp, n in zip(slp_with_n_axies_per_scholar(3),
                                                                 range(3, max_axies))]
        with_scholars_10 = [slp_price * slp * n for slp, n in zip(slp_with_n_axies_per_scholar(10),
                                                                  range(3, max_axies))]

        no_scholars = [slp_price * get_own_slp_per_day(n) for n in range(3, max_axies)]

        fig, (ax1, ax2) = plt.subplots(1, 2, sharex="all")

        ax1.plot(range(3, max_axies), with_scholars_3, label="Scholars 3")
        #ax1.plot(range(3, max_axies), with_scholars_10, label="Scholars 10")
        ax1.plot(range(3, max_axies), no_scholars, label="no scholars")
        ax1.plot(range(6, max_axies), [get_risk(axies, 3) / (w-no) for w, no, axies in
                                       list(zip(with_scholars_3, no_scholars, range(3, max_axies)))[3:]],
                 label="Days to ammortise scholar risk")

        ax1.legend()
        ax1.grid(True)
        ax1.set_title("Earnings in dollar")

        ra_scholars_3 = [x / get_risk(axies, 3) for x, axies in zip(with_scholars_3, range(3, max_axies))]
        ra_scholars_10 = [x / get_risk(axies, 10) for x, axies in zip(with_scholars_10, range(3, max_axies))]

        ax2.plot(range(6, max_axies), ra_scholars_3[3:], label="Risk adjusted earnings 3")
        ax2.plot(range(13, max_axies), ra_scholars_10[10:], label="Risk adjusted earnings 10")
        ax2.grid(True)
        ax2.legend()
        ax2.set_title("dollar earned per dollar at risk")


        plt.show()

    plot_earnings()
