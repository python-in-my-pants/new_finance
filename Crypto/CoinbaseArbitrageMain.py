import sys

from cbpro.public_client import PublicClient
from cbpro.websocket_client import WebsocketClient
import cbpro
import pprint
import json
import os
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import itertools
import time
import threading
import numpy as np
from datetime import datetime
from string import Template
import socket

# <editor-fold desc="Settings">
pd.set_option("display.max_rows", None, "display.max_columns", None)
pd.set_option('display.width', 1000)
pd.set_option('display.max_colwidth', 500)
pd.set_option('display.float_format', lambda x: '%.2f' % x)

np.seterr("raise")

maker_fee = 0.0025
taker_fee = 0.0025

debug = True


# </editor-fold>


# <editor-fold desc="Misc">
class DDict(dict):
    """dot.notation access to dictionary attributes"""
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__
    __hash__ = dict.__hash__


def p(x):
    pprint.pprint(x)


def time_it(func):
    def f(*args, **kwargs):
        print("Calling", func.__name__, "...")
        start = time.time()
        tmp = func(*args, **kwargs)
        print("Call to", func.__name__, "took", time.time() - start, "seconds!")
        return tmp

    return f


def flatten(l):
    return [item for sublist in l for item in sublist]


def rotate_list(l):
    return l[-1] + l[:-1]


def p_dict(d):
    s = ""
    for key, val in d.items():
        s += f'\n                 {key}:\t{float(val): 3.8f}'
    return s


def get_adj_pairs(_list):
    ret = []

    for i in range(1, len(_list)):
        base_currency = _list[i - 1]
        quote_currency = _list[i]

        ret.append((base_currency, quote_currency))

    ret.append((_list[-1], _list[0]))

    return ret


def get_timestamp():
    return datetime.now().strftime("%H:%M:%S.%f")


class DeltaTemplate(Template):
    delimiter = "%"


def strfdelta(tdelta, fmt):
    d = {"D": tdelta.days}
    hours, rem = divmod(tdelta.seconds, 3600)
    minutes, seconds = divmod(rem, 60)
    d["H"] = '{:02d}'.format(hours)
    d["M"] = '{:02d}'.format(minutes)
    d["S"] = '{:02d}'.format(seconds)
    t = DeltaTemplate(fmt)
    return t.substitute(**d)


##############################################################################################
# </editor-fold>

# <editor-fold desc="File handling">


def create_json_file(filename, file_contents):
    try:

        filename = filename.replace(".", "-").replace(":", "-")

        print(f'Creating file:\n'
              f'\t{filename}.json ...')

        with open(filename + ".json", "w") as file:
            json.dump(file_contents, file, indent=4)

        print("File created successfully!")

    except Exception as e:
        print("While creating the file", filename, "an exception occurred:", e)


def read_products_from_file(filename):
    f = json.load(open(filename, "r"))
    return [DDict(r) for r in f]


def get_csv_file(filename, mode):
    try:

        filename = filename.replace(".", "-").replace(":", "-")

        print(f'Creating file:\n'
              f'\t{filename}.json ...')

        return open("Arbitrage/" + filename + ".csv", mode)
        # file.writelines(file_contents.split("\n"))

    except Exception as e:
        print("While creating the file", filename, "an exception occurred:", e)


# </editor-fold>

# <editor-fold desc="Graph functions">
@time_it
def get_cyles_with_len_n(gr, n):
    _cycles = []
    for comb in itertools.combinations(gr.nodes(), n):

        """nodes = list(comb)

        for node in nodes:

            path_existent = False

            for other_node in nodes:

                if not path_existent and \
                    node != other_node and \
                        path_exists(gr, node, other_node, allowed_nodes=comb):

                    path_existent = True
                    nodes.remove(node)
                    nodes.remove(other_node)

            if not path_existent:
                continue"""

        ordered_cycle = is_cycle(gr, *comb)
        if ordered_cycle:
            _cycles.append(ordered_cycle)

    return sorted(_cycles)


def path_exists(g, start, target, allowed_nodes=None, deepness=0, path_nodes=None):
    if deepness >= len(g.nodes()) - 1:
        return []

    if allowed_nodes is None:
        allowed_nodes = g.nodes()

    if start is target:
        return [start]

    if path_nodes is None:
        path_nodes = [start]

    start_neighbors = list(set(g.neighbors(start)) & set(allowed_nodes))

    if target in start_neighbors:
        return [start, target]
    else:
        for neighbor in start_neighbors:
            next_allowed_nodes = set(allowed_nodes)
            next_allowed_nodes.remove(neighbor)
            potential_path = path_exists(g, neighbor, target, next_allowed_nodes,
                                         deepness=deepness + 1, path_nodes=path_nodes)

            if potential_path:
                return [start] + potential_path

        return False


def get_shortest_path(g, start, target, allowed_nodes=None):
    gtmp = g.copy()

    if allowed_nodes is None:
        return nx.shortest_path(g, start, target=target)

    if start not in allowed_nodes or target not in allowed_nodes:
        return []

    gtmp.remove_nodes_from(list(set(gtmp.nodes()) - set(allowed_nodes)))

    try:
        return nx.shortest_path(gtmp, start, target=target)
    except nx.exception.NetworkXNoPath as e:
        return []


def is_cycle(g, *args):
    """
    :param g:
    :param args:
    :return: True if each node is reachable from each other node given, False otherwise
    """
    cycle_elements = list()
    nodes = list(args)

    for node in nodes:

        other_nodes = nodes.copy()
        other_nodes.remove(node)

        for other_node in other_nodes:
            _p = path_exists(g, node, other_node, allowed_nodes=nodes)
            if not _p:
                return False
            # node and other node are connected

            # to be a cycle, they must still be connected when we remove one of the nodes connecting them as there must
            # be at least 2 different paths between them
            if len(_p) > 2:
                tmp_nodes = other_nodes.copy()
                tmp_nodes.remove(_p[1])
                if not path_exists(g, node, other_node, allowed_nodes=tmp_nodes):
                    return False

        cycle_elements.append(node)

    return cycle_elements


def get_cycles_with_currency(curr):
    _c = [cycle for cycle in cycles if curr in cycle]
    if not _c:
        print("No cycles containing", curr, "were found!")
    return _c


# </editor-fold>

def set_fees():
    fees = client.get_fees()  # TODO use authenticated client for request
    maker_fee = fees["maker_fee_rate"]
    taker_fee = fees["taker_fee_rate"]


def round_max_acc(amount, currency):
    f = float(currency_data[currency].factor)
    return np.floor(amount * f) / f


cycle_length = 3  # int(input("Cycle length: "))  TODO test best cycle_lengths

client = cbpro.PublicClient()

if not os.path.exists("/products.json"):
    currency_pairs = [DDict(p) for p in client.get_products()]
    create_json_file("products", currency_pairs)
else:
    currency_pairs = read_products_from_file("products.json")

if not os.path.exists("/currencies.json"):
    tmp = [p for p in client.get_currencies()]
    for dic in tmp:
        dic["factor"] = 10 ** (-int(np.log10(float(dic["max_precision"]))))
    currency_data = DDict({p["id"]: DDict(p) for p in tmp})

    create_json_file("currencies", currency_data)
else:
    currency_data = read_products_from_file("currencies.json")

product_ids = [x.id for x in currency_pairs]
currencies = set()

for product in currency_pairs:
    currencies.add(product.base_currency)
    currencies.add(product.quote_currency)

print(f'Found {len(currencies)} distinct currencies used in products!')

df = pd.DataFrame(data=" ", index=sorted(list(currencies)), columns=sorted(list(currencies)))

for product in currency_pairs:
    df[product.base_currency][product.quote_currency] = product.display_name
    # df[product.quote_currency][product.base_currency] = product.quote_currency + "/" + product.base_currency

print("Built dataframe:\n", df)

edges = [(p.base_currency, p.quote_currency) for p in currency_pairs] + \
        [(p.quote_currency, p.base_currency) for p in currency_pairs]

graph = nx.Graph()
graph.add_edges_from(edges)
print("Built graph!")

"""
nx.draw(graph, with_labels=True)
plt.show()
"""

prod_names = [prod.display_name for prod in currency_pairs]

cycles = get_cyles_with_len_n(graph, cycle_length)

print(f"\nFound {len(cycles)} cycles with lenght {cycle_length}!")
# p(cycles)

structural_arbitragable_currencies = sorted(list(set(flatten(cycles))))

print("\nFound", len(structural_arbitragable_currencies), "arbitragable currencies!")
# p(arbitragable_currencies)

structural_arbitragable_currency_pairs = [cp.id for cp in currency_pairs if
                                          cp.base_currency in structural_arbitragable_currencies or
                                          cp.quote_currency in structural_arbitragable_currencies]

currency_pair_feed_data = DDict({
    pair.id: DDict({
        "bid": -1,
        "ask": -1,
        "last_updated": "never",

        "base_max_prec": float(currency_data[pair.base_currency].max_precision),
        "base_min_size": float(pair.base_min_size),  # buy min this many A
        "base_increment": float(pair.base_increment),  # increment bought A by this quantity
        "base_max_size": float(pair.base_max_size),  # buy max this many A

        "quote_min_size": float(currency_data[pair.quote_currency].min_size),  # pay min this many B
        "quote_increment": float(pair.quote_increment)  # increment paid B amount by this
    })
    for pair in currency_pairs
})
currency_pair_feed_data = DDict({
    **currency_pair_feed_data,
    **DDict({pair.quote_currency + "-" + pair.base_currency:
        DDict({
            "bid": -1,
            "ask": -1,
            "last_updated": "never",

            "base_max_prec": float(currency_data[pair.base_currency].max_precision),
            "base_min_size": float(pair.base_min_size),
            "base_increment": float(pair.base_increment),
            "base_max_size": float(pair.base_max_size),

            "quote_min_size": float(currency_data[pair.quote_currency].min_size),
            "quote_increment": float(pair.quote_increment)
        })
        for pair in currency_pairs})
})

used_order_ids = set()
seen_reqs = dict()


# p(get_cycles_with_currency("UNI"))

# p(is_cycle(graph, "UNI", "BTC", "USD"))

# print(is_cycle(graph, *('XTZ', 'XLM', 'EUR', 'USD')))

# TODO tickers only represent price changes, though arbitrage oppertunities can occur just from changes to the order
#  book alone, without the price acutually changing, so we need the order book data for high freq arbitrage ...
#  .
#  although less data makes the timeframe for using the oppertunities larger and orders above the peaks of the order
#  book only  b e n e f i t  possible arbitrage gains, so using market orders, it should be possible with tickers.


class TickerWebSocketClient(cbpro.WebsocketClient):

    def __init__(self, products):
        super().__init__(products=products, channels=["ticker"])
        self.msgs_received = 0

    def on_open(self):
        print("Opening WebSocket ticker connection for products:\n\t", self.products, "\n")

    def on_message(self, msg):

        if msg["type"] == "error":
            p(msg)
        if msg["type"] == "ticker":
            currency_pair_feed_data[msg["product_id"]]["bid"] = float(msg["best_bid"])
            currency_pair_feed_data[msg["product_id"]]["ask"] = float(msg["best_ask"])
            currency_pair_feed_data[msg["product_id"]]["last_updated"] = msg["time"]

            split_id = msg["product_id"].split("-")
            rev_id = split_id[1] + "-" + split_id[0]

            currency_pair_feed_data[rev_id]["bid"] = 1 / float(msg["best_ask"])
            currency_pair_feed_data[rev_id]["ask"] = 1 / float(msg["best_bid"])
            currency_pair_feed_data[rev_id]["last_updated"] = msg["time"]

        if self.msgs_received % 1000 == 0:
            print(f"[{get_timestamp()}][   STATUS  ]  {self.msgs_received:7d} messages received")

        self.msgs_received += 1

    def on_close(self):
        print("-- WebSocket closed! --")


def start_receive(prod_ids):

    def threaded_receive():
        wsClient = TickerWebSocketClient(prod_ids)
        wsClient.start()

        def go():
            try:
                while True:
                    time.sleep(5)
            except KeyboardInterrupt:
                print(f"[ EXCEPTION ] Received KeyboardInterrupt, restarting order book ...")
                wsClient.close()
                go()
            except socket.timeout:
                print(f"[ EXCEPTION ] Socket timed out! Restarting ...")
                wsClient.close()
                go()
            except Exception as ex:
                print(f"[ EXCEPTION ] {ex}\n"
                      f"              Restarting WebSocket ...")
                wsClient.close()
                go()

        go()  # was commented out?!
        time.sleep(5)
        return

    try:
        threading.Thread(target=threaded_receive).start()
    except socket.timeout:
        print("[ EXCEPTION ] Socket timed out! Thread is being restarted ...")
        start_receive(prod_ids)
    except Exception as e:
        print("[ EXCEPTION ] An exception occured in the WebSocket thread:\n", e,
              "              Thread is being restarted ...")
        start_receive(prod_ids)


def print_price_data_for_cycle(_cycle):
    print()
    for i in range(1, len(_cycle)):
        print(f'{_cycle[i - 1]}-{_cycle[i]}: '
              f'{currency_pair_feed_data[f"{_cycle[i - 1]}-{_cycle[i]}"]["bid"]:6.8f}, '
              f'{currency_pair_feed_data[f"{_cycle[i - 1]}-{_cycle[i]}"]["ask"]:6.8f}')
    print(f'{_cycle[-1]}-{_cycle[0]}: '
          f'{currency_pair_feed_data[f"{_cycle[-1]}-{_cycle[0]}"]["bid"]: 6.8f}, '
          f'{currency_pair_feed_data[f"{_cycle[-1]}-{_cycle[0]}"]["ask"]: 6.8f}\n')


def derive_initial_vol_req_divisor(_i, _curs, fee):
    """
    :param fee: 1-percentual fee paid for 1 transaction
    :param _i: 'steps already taken in conversion when calling this function' + 1
    :param _curs: list of currencies
    :return: 1 if _i is 1, else bid(A/B) * bid(B/C) * bid(C/D) * ... for i-1 factors
    """
    ret = 1
    for k in range(_i - 1):
        ret *= (currency_pair_feed_data[f'{_curs[k]}-{_curs[k + 1]}']["bid"] / fee)
    return ret


def get_structural_vol_constraint(curs):
    """
    Calculate the minimal amount of curs[0] to have for this cycle to be arbitraged at current exchange rates
    :param curs: ordered list of currencies to check
    :return: minimal amount of curs[0] to have for this cycle to be arbitraged at current exchange rates
    """

    # todo s must be multiple of quote_increment

    # todo max constraint for s needed

    fee = (1 - taker_fee)

    vol_cons = []

    # <editor-fold desc="calculate volume constraint for each conversion step">
    for i in range(1, len(curs)):
        payment_currency = curs[i - 1]  # A
        target_currency = curs[i]  # B
        data = currency_pair_feed_data[f'{payment_currency}-{target_currency}']

        div = data.bid * fee if f'{target_currency}-{payment_currency}' in product_ids else 1

        vol_cons.append(data.base_min_size / div)
        if debug and False:
            print(f'[{get_timestamp()}]{curs} Struct vol req: {payment_currency}-{target_currency}: '
                  f'{data.base_min_size:3.10f} / {div:3.10f} = {data.base_min_size / div:3.10f} {payment_currency}')

    # last step
    payment_currency = curs[-1]  # A
    target_currency = curs[0]  # B
    data = currency_pair_feed_data[f'{payment_currency}-{target_currency}']

    div = data.bid * fee if f'{target_currency}-{payment_currency}' in product_ids else 1
    vol_cons.append(data.base_min_size / div)
    if debug and False:
        print(f'[{get_timestamp()}][{curs}] Struct vol req: {payment_currency}-{target_currency}: '
              f'{data.base_min_size:3.10f} / {div:3.10f} = {data.base_min_size / div:3.10f} {payment_currency}')
    # </editor-fold>

    # convert each steps volume constraint to constraint for initial currency volume
    derived_vol_cons = []
    for i in range(len(vol_cons)):
        derived_vol_cons.append(vol_cons[i] / derive_initial_vol_req_divisor(_i=i + 1, _curs=curs, fee=fee))
        if debug and False:
            print(f'[{get_timestamp()}][{curs}] Struct vol req: {i + 1} ({curs[0]}): '
                  f'{derived_vol_cons[-1]:3.10f} {curs[0]}')

    if debug and False:
        print(f'[{get_timestamp()}][{curs}] Final struct vol req: {max(derived_vol_cons):3.10f} {curs[0]}')
    return max(derived_vol_cons)


def can_cap_arbitrage_curs(curs, start_cap, _p=False):
    """
    Tells whether the given capital is sufficient for arbitraging the given currencies
    :param curs:
    :param start_cap:
    :param _p:
    :return: (True, earned gain) if capital is sufficient for arbitrage, else (False, additionally_needed_capital)
    """
    gain, init_s = calc_gain(curs, _p=_p)
    if start_cap >= init_s:
        return True, gain
    else:
        return False, init_s - start_cap


def get_effective_volume_constraint(curs, fee):
    # arbitrage profit is made, but with current initial capital, the gain is smaller than curs[0]s max precision,
    # rendering it virtually non existent, so we need to adjust the initially required capital to make effective profit

    # effective volume constraint = min initial capital needed to make effective profit

    # gain must be greater than max precision of curs[0]
    # multiply init_s by this factor to have an effective arbitrage
    # factor is only rounded because reversing intermediate rounding from calc_gain is unneccessarily expensive
    # just add 1% to compensate for rounding, result may be slightly bigger than actually needed init_s but we
    # save on computation time todo maybe adjust the 1% later

    init_s = round_max_acc(get_structural_vol_constraint(curs), curs[0])
    s = init_s
    for i in range(1, len(curs)):
        s *= currency_pair_feed_data[f'{curs[i - 1]}-{curs[i]}']["bid"] * fee
    s *= currency_pair_feed_data[f'{curs[-1]}-{curs[0]}']["bid"]

    # s is now bigger than 1 but the difference to init_s is smaller than max precision of curs[0]
    if 0 < s - init_s < float(currency_data[curs[0]].max_precision):

        # add 1 % additional capital to compensate for rounding in real calculation
        # todo maybe adjust to avoid conflicts with order book volume (if the additional 1% make a otherwise possible
        #  trade impossible)
        eff_vol_req = round_max_acc(
            init_s * float(currency_data[curs[0]].max_precision) / (s - init_s) +
            float(currency_data[curs[0]].max_precision),
            curs[0])

        if debug and \
                (tuple(curs) not in seen_reqs.keys() or
                 init_s * float(currency_data[curs[0]].max_precision) / (s - init_s) not in seen_reqs[tuple(curs)]):

            if tuple(curs) in seen_reqs.keys():
                seen_reqs[tuple(curs)].add(init_s * float(currency_data[curs[0]].max_precision) / (s - init_s))
            else:
                seen_reqs[tuple(curs)] = {init_s * float(currency_data[curs[0]].max_precision) / (s - init_s)}

            print()
            print(f'[{get_timestamp()}]{curs} {s - init_s:3.10f} < {float(currency_data[curs[0]].max_precision):3.10f}')
            print(f'[{get_timestamp()}]{curs} Effective vol req:\n\n\t'
                  f'init_s * float(currency_data[curs[0]].max_precision) / (s - init_s) =\n\t'
                  f'{init_s:3.10f} * {float(currency_data[curs[0]].max_precision):3.10f} / ({s:3.10f} - {init_s:3.10f}) = '
                  f'{init_s:3.10f} * {float(currency_data[curs[0]].max_precision) / (s - init_s):3.10f} = '
                  f'{eff_vol_req:3.10f} {curs[0]} '
                  f'vs. structural requirement: {init_s:3.10f} {curs[0]}\n')
        return eff_vol_req
    else:
        return init_s


def calc_gain(curs, init_s=None, _p=False):
    """
    Calculate arbitrage gain relative to the initial amount s of curs[0] that is supplied,
    e.g. s=1 for currencies [A, B, C] may result in 0.003 which means a gain of 0.3 % when arbitraging
    from A -> B -> C -> A

    :param curs: ordered list of currencies to calculate arbitrage gain for
    :param init_s: initial amount of curs[0] supplied
    :param _p: print additional information for each step
    :return: arbitrage gain relative to the initial amount s of curs[0] that is supplied, minimal start capital needed
    """
    # todo check cancel only etc

    # todo mind order volume from order book

    # todo s must be multiple of quote_increment

    fee = (1 - taker_fee)

    effective_volume_constraint = round_max_acc(get_effective_volume_constraint(curs, fee), curs[0])

    volumes = []

    if init_s is None:
        init_s = effective_volume_constraint
    elif init_s == 0:
        return 0, []
    else:
        if init_s < effective_volume_constraint:
            # arbitrage structually impossible
            return 0, []

    init_s = round_max_acc(init_s, curs[0])
    s = init_s
    volumes.append(init_s)

    for i in range(1, len(curs)):

        """
        sell base -> bid
        buy  base -> ask
        """

        # A/B, B -> A
        """
        having A -> B -> C -> A 
        one would usually need B/A, C/B, etc.
        we use A/B, B/C, etc. instead
        with B/A one would divide the amount of A by the ask price
        instead with A/B we can multiply by the bid of A/B because the bid of A/B = 1/ask of B/A
        """
        payment_currency = curs[i - 1]  # A
        target_currency = curs[i]  # B
        data = currency_pair_feed_data[f'{payment_currency}-{target_currency}']

        if _p:
            print(f'Paying '
                  f'{s} {payment_currency} for '
                  f'{s} * '
                  f'{data["bid"]} ({payment_currency}->{target_currency}) * '
                  f'{fee}'
                  f' = '
                  f'{s * fee * data["bid"]} '
                  f'{target_currency}')

        """# check volume constraints
        # todo buy at max x of target currency where x is the volume of the best entry in the order book for this transaction
        
        vol_constraint = max(data.base_min_size,
                             # ^ pay min smallest holdable amount of currency that you pay with PAYMENT CONSTRAINT
                             
                             data.quote_min_size / (data["bid"] * fee))
                             # ^ pay min enough to buy smallest holdable amount of target currency DERIVED PAYMENT CONSTRAINT
        
        if not quote_base_exists:
            vol_constraint = max( data.base_min_size,   # pay at most base_max
        else:
            vol_constraint = base_min_size / ( data.bid * fee)
            
        div = 1 if not quote_base_exists else data.bid * fee
        vol_constraint = data.base_min_size / div

        if s >= vol_constraint:
            s = round_max_acc(data["bid"] * fee, payment_currency if quote_base_exists else target_currency)
        else:
            # derive initial volume constraint from i-th step volume constraint and call recursively with new
            # initial constraint
            initial_vol_constraint = vol_constraint / derive_initial_vol_req_divisor(i, _curs=curs)
            return calc_gain(curs=curs, s=initial_vol_constraint)"""

        s = round_max_acc(s * data["bid"] * fee, target_currency)
        volumes.append(s)

        if _p:
            print(f'\tHolding {s} {target_currency} now')

    # <editor-fold desc="Way back to start currency">
    payment_currency = curs[-1]
    target_currency = curs[0]
    data = currency_pair_feed_data[f'{payment_currency}-{target_currency}']

    if _p:
        print(f'Paying '
              f'{s:6.8f} {payment_currency} for '
              f'{s:6.8f} * '
              f'{data["bid"]:6.8f} ({payment_currency}->{target_currency}) * '
              f'{fee}'
              f' = '
              f'{s * fee * data["bid"]:6.8f} '
              f'{target_currency}')

    """# check volume constraints
    # todo buy at max x of target currency where x is the volume of the best entry in the order book for this transaction
    vol_constraint = max(data.base_min_size,
                         # ^ pay min smallest holdable amount of currency that you pay with
                         data.quote_min_size / (data["bid"] * fee),
                         # ^ pay min enough to buy smallest holdable amount of target currency
                         data.quote_max_prec)
                         # ^ volume constraint is at least max prec of target currency
                         # (this should alread be assured by the exchange course and min quote size though)

    if s >= vol_constraint:
        s = round_max_acc(data["bid"] * fee, base_currency if quote_base_exists else quote_currency)
    else:
        # derive initial volume constraint from i-th step volume constraint and call recursively with new
        # initial constraint
        initial_vol_constraint = vol_constraint / derive_initial_vol_req_divisor(i, _curs=curs)
        return calc_gain(curs=curs, s=initial_vol_constraint)"""

    s = round_max_acc(s * data["bid"] * fee, target_currency)
    volumes.append(s)

    # </editor-fold>

    try:
        overall_percentual_gain = (s / round_max_acc(init_s, curs[0])) - 1
    except FloatingPointError as e:
        print(f'Error calculating overall percentual gain, are all currencies initialized? {e}')
        overall_percentual_gain = 0

    if _p:
        print(f'\tHolding {s:6.8f} {curs[0]} now')
        print(f'{"Gained" if overall_percentual_gain > 0 else "Lost"} '
              f'{overall_percentual_gain * 100 if overall_percentual_gain >= 0 else -overall_percentual_gain * 100:2.5f} '
              f'% on {init_s:2.8f} {target_currency} = {overall_percentual_gain * init_s:2.8f} {target_currency}\n')

    return overall_percentual_gain, volumes


def get_arbitrage(arb_currencies, _p=False):
    """
    Calculate if arbitrage in this cycle is possible considering:
    - minimum volume requirements
    - current exchange rates
    AND top order book entry volume

    :param arb_currencies:
    :param _p:
    :return:
    """

    """
    check A -> B -> ... -> Y (rotations don't matter)
    check Y -> X -> ... -> A
    TODO are other options possibly effective?

    :param _p:
    :param arb_currencies:
    :return: tuple (a, b) where a is the arbitrage factor and b is the list in order of the arbitrage
    """

    if debug and False:
        print("\n" + "-" * 120)

    forward_relative_arbitrage, forward_volumes = calc_gain(arb_currencies, _p=_p)
    backward_relative_arbitrage, backward_volumes = calc_gain(arb_currencies[::-1], _p=_p)

    '''
    If arbitrage is theoretically possible, we have to look at the top order book entries. Because we can only arbitrage
    them, if their volume is less or equal to the desired trading volume.
    
    API throttles to 3 requests per second, 6 per second in bursts (like here).
    '''

    forward_used_order_ids = []  # order ids used up if arbitrage will be done
    backward_used_order_ids = []

    # only send out order book requests if arbitrage seems possible to lower the risk of getting 429'd out
    if forward_relative_arbitrage > 0 or backward_relative_arbitrage > 0:

        arb_pairs = get_adj_pairs(arb_currencies)
        arb_pairs_rev = get_adj_pairs(arb_currencies[::-1])
        sell_vol_limits = []
        sell_order_ids = []
        buy_vol_limits = []
        buy_order_ids = []

        order_book_data = []
        for index, pair in enumerate(arb_pairs):

            # convert from pair[0] to pair[1]

            # pair[0]  A  paying_currency
            # pair[1]  B  target_currency

            # level 3 requests must be made to obtain order ids, these are neccessary to omit double spending on same order
            if f'{pair[0]}-{pair[1]}' in product_ids:
                order_book_data.append(client.get_product_order_book(product_id=f'{pair[0]}-{pair[1]}', level=3))

            else:  # weird structure to assure the api call is not made twice in case of faulty product_ids
                if f'{pair[1]}-{pair[0]}' in product_ids:
                    order_book_data.append(client.get_product_order_book(product_id=f'{pair[1]}-{pair[0]}', level=3))

            if index % 5 == 0 and index > 0:  # api rate limit is 6 requests per second in bursts
                time.sleep(1)  # to omit spamming the api server with requests for order book

            print(f'[{get_timestamp()}][    INFO   ] OrderBook API request was made!')

            # not 100% accurate, as LVL1 order book data is aggregated, but again, the only alternative here is to build
            # and manage the whole LVL3 order book for all symbols which would create a giant overhead in traffic as it
            # adds plenty of unneccessarry messages

            # In theory, we only need the top bid & ask of the order book (unaggregated) with the price and the volume

            # aggregate top entries

        # aggregate orders
        for pair_order_book_data in order_book_data:

            best_bid_price = float(pair_order_book_data["bids"][0][0])
            best_bid_order_ids = [pair_order_book_data["bids"][0][2]]
            agg_top_bid_vol = float(pair_order_book_data["bids"][0][1])
            for bid in pair_order_book_data["bids"][1:]:
                if float(bid[0]) == best_bid_price:
                    best_bid_order_ids.append(bid[2])
                    agg_top_bid_vol += float(bid[1])
                else:
                    break

            best_ask_price = float(pair_order_book_data["asks"][0][0])
            best_ask_order_ids = [pair_order_book_data["asks"][0][2]]
            agg_top_ask_vol = float(pair_order_book_data["asks"][0][1])
            for ask in pair_order_book_data["asks"][1:]:
                if float(ask[0]) == best_ask_price:
                    best_ask_order_ids.append(ask[2])
                    agg_top_ask_vol += float(ask[1])
                else:
                    break

            sell_vol_limits.append(agg_top_bid_vol * best_bid_price)
            sell_order_ids.append(best_bid_order_ids)

            buy_vol_limits.append(agg_top_ask_vol * best_ask_price)
            buy_order_ids.append(best_ask_order_ids)

        def _check_order_limit_exeeded(_paying_currency, _target_currency, _index, forward):
            _paying_currency = pair[0]
            _target_currency = pair[1]
            a = None

            if f'{_paying_currency}-{_target_currency}' in product_ids:
                _paying_currency_limit = sell_vol_limits[_index]
                if forward:
                    forward_used_order_ids.extend(sell_order_ids[_index])
                else:
                    backward_used_order_ids.extend(sell_order_ids[_index])
                a = f'{_paying_currency}-{_target_currency}'

            if f'{_target_currency}-{_paying_currency}' in product_ids:
                _paying_currency_limit = buy_vol_limits[_index]
                if forward:
                    forward_used_order_ids.extend(buy_order_ids[_index])
                else:
                    backward_used_order_ids.extend(buy_order_ids[_index])
                a = f'{_target_currency}-{_paying_currency}'

            if a is None:
                print(_paying_currency, _target_currency)
                p(product_ids)
                sys.exit()

            try:
                if forward_volumes[_index] > _paying_currency_limit:
                    print(f'[{get_timestamp()}][    INFO   ] '
                          f'{arb_currencies if forward else arb_currencies[::-1]} '
                          f'{" forward" if forward else "backward"} arbitrage stopped due to order book limits of {a},'
                          f' max payable {_paying_currency}: {_paying_currency_limit}, '
                          f'required payment: {forward_volumes[_index]} {_paying_currency}')
                    if forward:
                        global forward_relative_arbitrage
                        forward_relative_arbitrage = 0
                    else:
                        global backward_relative_arbitrage
                        backward_relative_arbitrage = 0
            except UnboundLocalError as e:
                print(f'[{get_timestamp()}][    INFO   ] '
                      f'{arb_currencies if forward else arb_currencies[::-1]} arbitrage stopped due to order book limits of {a},'
                      f' max payable {_paying_currency}: {_paying_currency_limit}, '
                      f'required payment: {forward_volumes[_index]} {_paying_currency}')
                print("_paying_currency_limit was not assigned!")
                print(e)

        # impose lower volume limits given by limit order volume
        # todo impose upper limits given by limit order volume

        # forward
        for index, pair in enumerate(arb_pairs):
            _check_order_limit_exeeded(pair[0], pair[1], index, forward=True)
        _check_order_limit_exeeded(arb_currencies[-1], arb_currencies[0], len(arb_currencies) - 1, forward=True)

        # backward
        for index, pair in enumerate(arb_pairs_rev):
            _check_order_limit_exeeded(pair[0], pair[1], index, forward=False)
        _check_order_limit_exeeded(arb_currencies[0], arb_currencies[-1], len(arb_currencies) - 1, forward=False)

        # todo find better solution to omit spamming
        time.sleep(1)

    arbitrage_info = DDict({
        "forward_relative_arbitrage": forward_relative_arbitrage,
        "forward_volumes": forward_volumes,
        "forward_currencies": arb_currencies,
        "forward_order_ids": set(forward_used_order_ids),
        "forward_max_start_cap": -1,  # todo

        "backward_relative_arbitrage": backward_relative_arbitrage,
        "backward_volumes": backward_volumes,
        "backward_currencies": arb_currencies[::-1],
        "backward_order_ids": set(backward_used_order_ids),
        "backward_max_start_cap": -1,
    })

    return arbitrage_info


def arb_info_to_csv(info):
    contents = ""

    # forward
    if len(info.forward_order_ids.intersection(used_order_ids)) == 0:

        contents += f'{info.forward_relative_arbitrage:5.9f} '

        # currencies
        for curr in info.forward_currencies:
            contents += curr + ","
        contents = contents[:-1] + " "

        # volumes
        for vol in info.forward_volumes:
            contents += f'{vol:5.8f},'
        contents = contents[:-1] + "\n"

    # backward
    if len(info.backward_order_ids.intersection(used_order_ids)) == 0:

        contents += f'{info.backward_relative_arbitrage:5.8f} '

        # currencies
        for curr in info.backward_currencies:
            contents += curr + ","
        contents = contents[:-1] + " "

        # volumes
        for vol in info.backward_volumes:
            contents += f'{vol:5.9f},'
        contents = contents[:-1] + "\n"

    return contents


def cur_to_eur(curr, amount):
    shortest_path = nx.shortest_path(graph, curr, "EUR")
    adj_pairs = get_adj_pairs(shortest_path)[:-1]

    val_in_eur = amount
    for pair in adj_pairs:
        # print(pair, currency_pair_feed_data[f'{pair[0]}-{pair[1]}']["ask"])
        val_in_eur *= currency_pair_feed_data[f'{pair[0]}-{pair[1]}']["ask"]

    return val_in_eur


def check_for_arbitrage():
    time.sleep(5)
    low_lim = 0
    k = 0
    arbitrage_gains = {currency: 0 for currency in currencies}
    req_start_cap = {currency: 0 for currency in currencies}
    loop_len_in_s = 0.01
    start_time = datetime.now()
    print_interval = 10
    outfile = get_csv_file(f'{get_timestamp()}-arbitrage_data', "a")

    try:
        with outfile as file:

            while True:

                time.sleep(loop_len_in_s)

                if k % (print_interval / loop_len_in_s) == 0:
                    uptime = (datetime.now() - start_time)
                    print(f'[{get_timestamp()}][   UPTIME  ] {strfdelta(uptime, "%H:%M:%S")}')

                    d = {key: f'{val:3.{currency_data[key].factor}}' for key, val in arbitrage_gains.items() if val > 0}
                    if d:
                        summed_gains_in_eur = sum([cur_to_eur(key, float(val)) for key, val in arbitrage_gains.items()])
                        print(f"[{get_timestamp()}][   GAINS   ] Arbitrage gains: {summed_gains_in_eur:5.2f}"
                              f"{p_dict(d)}")

                    d = {key: f'{val:3.{currency_data[key].factor}}' for key, val in req_start_cap.items() if val > 0}
                    if d:
                        summed_req_in_eur = sum([cur_to_eur(key, float(val)) for key, val in req_start_cap.items()])
                        print(f"[{get_timestamp()}][   GAINS   ] Required start capital: {summed_req_in_eur:5.2f}"
                              f"{p_dict(d)}")

                k += 1

                for cycle in cycles:

                    arb_info = get_arbitrage(cycle, _p=False)

                    file.writelines(arb_info_to_csv(arb_info))

                    forward_rel_arb = arb_info.forward_relative_arbitrage
                    forward_start_caps = arb_info.forward_volumes
                    forward_start_curr = arb_info.forward_currencies[0]
                    forward_order_ids = arb_info.forward_order_ids

                    n = arb_info.backward_relative_arbitrage
                    n_start_caps = arb_info.backward_volumes
                    n_currency = arb_info.backward_currencies[0]
                    backward_order_ids = arb_info.backward_order_ids

                    """print(f'{cycle}\t'
                          f'Forward: {m:2.8f} {cycle[0]}\t'
                          f'Backward: {n:2.8f} {cycle[-1]}')"""

                    if forward_rel_arb > low_lim or n > low_lim:

                        print()
                        print(cycle, f'{forward_rel_arb:3.9f}', cycle[0], ":::", f'{n:3.9f}', cycle[-1])
                        print_price_data_for_cycle(cycle)
                        get_arbitrage(cycle, _p=True)

                        if forward_rel_arb > n and len(forward_order_ids.intersection(used_order_ids)) == 0:
                            arbitrage_gains[forward_start_curr] += forward_start_caps[0] * forward_rel_arb
                            used_order_ids.update(forward_order_ids)
                            for i, c in enumerate(cycle):
                                req_start_cap[c] = max(forward_start_caps[i], req_start_cap[c])

                            print(f'Gained {forward_start_caps[0] * forward_rel_arb:3.8f} {forward_start_curr}\t'
                                  f'Req start cap: {forward_start_caps[0]} {forward_start_curr}\n')
                        elif n > forward_rel_arb and len(backward_order_ids.intersection(used_order_ids)) == 0:
                            arbitrage_gains[n_currency] += n_start_caps[0] * n
                            used_order_ids.update(backward_order_ids)
                            for i, c in enumerate(cycle):
                                req_start_cap[c] = max(n_start_caps[i], req_start_cap[c])

                            print(f'Gained {n_start_caps[0] * n:3.8f} {n_currency}\t'
                                  f'Req start cap: {n_start_caps[0]} {n_currency}\n')

                        print("Arbitrage gains:",
                              p_dict({key: f'{val:3.{currency_data[key].factor}}'
                                      for key, val in arbitrage_gains.items() if val > 0}))
                        print("  req_start_cap:",
                              p_dict({key: f'{val:3.{currency_data[key].factor}}\n'
                                      for key, val in req_start_cap.items() if val > 0}))

            # print("-" * 50, f"Profit: {(cap - start_cap) / start_cap} Capital: {cap}", "-" * 50, k)

    except KeyboardInterrupt:
        print("keyb")
    """except Exception as e:
        print(e)
    finally:
        print(f'Profit: {100 * ((cap / start_cap) - 1)} % gain in {k} seconds!')"""


def set_stuff():
    currency_pair_feed_data["BTC-EUR"]["bid"] = 40000
    currency_pair_feed_data["BTC-EUR"]["ask"] = 40100

    currency_pair_feed_data["EUR-BTC"]["bid"] = 1 / 40100
    currency_pair_feed_data["EUR-BTC"]["ask"] = 1 / 40000

    currency_pair_feed_data["ETH-EUR"]["bid"] = 1500
    currency_pair_feed_data["ETH-EUR"]["ask"] = 1550

    currency_pair_feed_data["EUR-ETH"]["bid"] = 1 / 1550
    currency_pair_feed_data["EUR-ETH"]["ask"] = 1 / 1500

    currency_pair_feed_data["BTC-ETH"]["bid"] = 27
    currency_pair_feed_data["BTC-ETH"]["ask"] = 28

    currency_pair_feed_data["ETH-BTC"]["bid"] = 1 / 28
    currency_pair_feed_data["ETH-BTC"]["ask"] = 1 / 27


start_receive(structural_arbitragable_currency_pairs)

check_for_arbitrage()

# todo choose currency to receive arbitrage gains in by rotating the currency list beforehand

# todo use best and second best price from order book and calc a weigted avg using respective volumes to check for
#  further arbitrage options

# todo extend for binance
