from Analyzer import Analyzer, Backtester
from History import AssetData
import random


class ArtificialHistory:
    # todo add knowledge about which kind of trend follows which kind ... like len, height etc

    def __init__(self, real_hist):

        self.real_hist = real_hist

        # use spread as min_threshold
        self.anal = Analyzer(self.real_hist, min_trend_h=vars(AssetData)[self.real_hist.asset_name][0]*9, realistic=True)

        self.asset_name = self.real_hist.asset_name
        self.timeframe = self.real_hist.timeframe
        self.times = list(range(len(self.real_hist.times)))

        trends = self.anal.trend_list
        artif_trend_series = []

        up_trends = list(filter(lambda t: t.type == "Up  ", trends))
        down_trends = list(filter(lambda t: t.type == "Down", trends))
        random.shuffle(up_trends)
        random.shuffle(down_trends)

        counter = 0

        up_longer = len(up_trends) >= len(down_trends)
        while len(up_trends)+len(down_trends) > 0:
            if up_longer:
                artif_trend_series.append(up_trends.pop(0))
                artif_trend_series.append(down_trends.pop(0))
            else:
                artif_trend_series.append(down_trends.pop(0))
                artif_trend_series.append(up_trends.pop(0))
            counter += 2

        lol = [t.prices for t in artif_trend_series]
        lel = []
        for x in lol:
            for elem in x:
                lel.append(elem)

        lul = Backtester.derive(lel)
        print("b4 sum:", min(lul), max(lul))

        artif_prices = [trends[0].prices[0]]
        for trend in artif_trend_series:
            miau = [elem + artif_prices[-1] for elem in Backtester.derive(trend.prices)]
            for m in miau:
                print(m)
            print()
            artif_prices += miau

        l = Backtester.derive(artif_prices)
        print("after sum:", min(l), max(l))

        artif_hist = []
        for i, p in enumerate(artif_prices):
            artif_hist.append({"price": p, "time": str(i)})

        self.hist = artif_hist
        self.prices = [elem["price"] for elem in artif_hist]
