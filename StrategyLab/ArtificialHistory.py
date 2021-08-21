import sys
from Analyzer import Analyzer, Trend
from History import AssetData
import random
from Utility import *


class ArtificialHistory:
    # todo add knowledge about which kind of trend follows which kind ... like len, height etc

    """
    def __init__(self, real_hist):

        self.real_hist = real_hist

        asset_spread = vars(AssetData)[self.real_hist.asset_name][0]

        # use spread as min_threshold
        self.anal = Analyzer(self.real_hist,
                             min_trend_h=asset_spread,
                             realistic=False)

        self.asset_name = self.real_hist.asset_name
        self.timeframe = self.real_hist.timeframe
        self.times = list(range(len(self.real_hist.times)))

        trends = self.anal.trend_list
        for i in range(len(trends)-1):
            trends[i].next_trend = trends[i+1]
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

        lul = Analyzer.derive_same_len(lel)
        print("b4 sum:", min(lul), max(lul))

        artif_prices = [trends[0].prices[0]]
        for trend in artif_trend_series:
            miau = [elem + artif_prices[-1] for elem in Analyzer.derive_same_len(trend.prices)]
            for m in miau:
                print(m)
            print()
            artif_prices += miau

        l = Analyzer.derive_same_len(artif_prices)
        print("after sum:", min(l), max(l))

        artif_hist = []
        for i, p in enumerate(artif_prices):
            artif_hist.append({"price": p, "time": str(i)})

        self.hist = artif_hist
        self.prices = [elem["price"] for elem in artif_hist]
    """

    @timeit
    def __init__(self, real_hist, out_len=None, min_trend_h=None, supress_print=False):

        self.real_hist = real_hist
        self.asset_name = real_hist.asset_name
        self.sim_table = None

        if not out_len:
            self.timeframe = real_hist.timeframe + " (artificial)"
            self.times = real_hist.times
        else:
            self.timeframe = "0-" + str(out_len-1)
            self.times = list(range(out_len))

        spread = vars(AssetData)[self.real_hist.asset_name][0]
        if not min_trend_h:
            min_trend_h = spread

        analyer = Analyzer(real_hist, min_trend_h=min_trend_h, realistic=False)
        self.real_trends = analyer.trend_list
        for i in range(len(self.real_trends) - 1):
            self.real_trends[i].next_trend = self.real_trends[i + 1]

        """sim_table_path = "TrendTableObjects/" + self.real_hist.asset_name + "_" + \
                         self.real_hist.timeframe + "_" + str(h)
        sim_table_path = sim_table_path.replace(" ", "_").replace(".", "-").replace(":", "")
        try:
            with open(sim_table_path + ".pickle", "rb") as file:
                self.sim_table = pickle.load(file)
                if self.sim_table is None:
                    raise TypeError("sim table is none after reading from file")
        except FileNotFoundError:
            print("Creating sim table at", sim_table_path + ".pickle with", len(self.real_trends), "^2 entries")
            self.sim_table = self.make_sim_table()

            if not os.path.exists(sim_table_path + ".pickle"):
                with open(sim_table_path + ".pickle", 'wb') as f:
                    pickle.dump(self.sim_table, f)"""

        self.trend_seq = [self.real_trends[random.randint(0, len(self.real_trends)-2)]]
        trend_seq_data_len = self.trend_seq[0].len

        i = 1
        times = []
        start = time.time()
        avg_samples_per_loop = avg([t.len for t in self.real_trends])
        while trend_seq_data_len < len(self.times):
            if not supress_print and i % 100 == 0:
                end = time.time()
                times.append(end-start)
                avg_run_time = avg(times)/100
                remaining_data_samples = (len(self.times)-trend_seq_data_len)
                print("\ntrend seq data len:", trend_seq_data_len,
                      " Time remaining: ca.",
                      seconds_to_timestamp(avg_run_time*remaining_data_samples/avg_samples_per_loop))
                start = time.time()
                # todo why old here?
            self.trend_seq.append(self.get_likely_next_trend_old(self.trend_seq[-1]))
            trend_seq_data_len += self.trend_seq[-1].len
            i += 1

        tmp = [[price for price in trend.prices] for trend in self.trend_seq]
        self.prices = flatten(tmp)[:len(self.times)]

        self.hist = []
        for i in range(len(self.times)):
            self.hist.append({"time": self.times[i], "price": self.prices[i]})

    def make_sim_table(self):

        sim_table = {}
        i = 1
        p = 0
        for t1 in self.real_trends:

            if i % (len(self.real_trends) // 100) == 0:
                p += 1
                print(p, "% done")
            i += 1

            sim_table[t1.trend_id] = {}
            for t2 in self.real_trends:
                if t1.trend_id != t2.trend_id:
                    sim_table[t1.trend_id][t2.trend_id] = t1.get_similarity(t2)
                else:
                    sim_table[t1.trend_id][t2.trend_id] = 0

        if sim_table is not None:
            print("Sim table is full")
        return sim_table

    def get_likely_next_trend(self, base_trend):

        """
        TODO check if this works, was late when coding
        gets a trend that is likely to follow the given one based upon the real_hist trend sequence data

        :param base_trend:
        :return:
        """

        if base_trend is None:
            raise TypeError("Base trend is none")

        if self.sim_table is None:
            raise TypeError("Sim table is None")

        class Container:
            def __init__(self, prob, trendd):
                self.prob = prob
                self.summed_prob = None
                self.trend = trendd

        containers = []
        summed_sim = 0
        for trend in self.real_trends[:-1]:
            if trend is None:
                raise TypeError("Trend is None")
            try:
                sim = self.sim_table[base_trend.trend_id][trend.trend_id]
            except KeyError as e:
                print("Key error:", e)
                print(base_trend)
                print(trend)
                sys.exit(-1)

            if sim > 0:
                containers.append(Container(sim, trend))
                summed_sim += sim

        for con in containers:
            con.prob /= summed_sim

        containers.sort(key=lambda x: x.prob)
        prob_sum = 0
        for con in containers:
            con.summed_prob = con.prob + prob_sum
            prob_sum += con.prob

        r = random.random()

        for con in containers:
            if r <= con.summed_prob:
                t = con.trend.next_trend

                price_diff = base_trend.prices[-1] - t.prices[0]
                new_prices = [t.prices[i] + price_diff for i in range(len(t.prices))]
                new_data = [{"time": i, "price": new_prices[i]} for i in range(len(t.prices))]

                to_return = Trend(new_data)
                return to_return

        return base_trend.next_trend

    def get_likely_next_trend_old(self, base_trend):

        """
        TODO check if this works, was late when coding
        gets a trend that is likely to follow the given one based upon the real_hist trend sequence data

        :param base_trend:
        :return:
        """

        class Container:
            def __init__(self, prob, trendd):
                self.prob = prob
                self.summed_prob = None
                self.trend = trendd

        containers = []
        for trend in self.real_trends[:-1]:

            sim = trend.get_similarity(base_trend)
            if sim >= 0:
                containers.append(Container(sim, trend))

        summed_sim = sum([container.prob for container in containers])
        for con in containers:
            con.prob /= summed_sim

        containers.sort(key=lambda x: x.prob)
        prob_sum = 0
        for con in containers:
            con.summed_prob = con.prob + prob_sum
            prob_sum += con.prob

        r = random.random()

        for con in containers:
            if r <= con.summed_prob:
                t = con.trend.next_trend

                price_diff = base_trend.prices[-1] - t.prices[0]
                new_prices = [t.prices[i] + price_diff for i in range(len(t.prices))]
                new_data = [{"time": i, "price": new_prices[i]} for i in range(len(t.prices))]

                to_return = Trend(new_data)
                return to_return
