from selenium import webdriver
from bs4 import BeautifulSoup
import urllib.request
from selenium.webdriver.firefox.options import Options
from selenium.webdriver import ActionChains
import time
import pandas as pd
from pandasgui import show
from pprint import pprint as pp
import numpy as np
import networkx as nx
from pyvis.network import Network

np.set_printoptions(edgeitems=200, linewidth=1000)
pd.set_option("display.max_rows", None, "display.max_columns", None)
pd.set_option('display.width', 1000)
pd.set_option('display.max_colwidth', 500)
pd.set_option('display.float_format', lambda x: '%.5f' % x)


def scrape_dynamic(url):

    options = Options()
    options.headless = True
    browser = webdriver.Firefox(options=options)

    try:
        print(f'Getting website {url}')
        browser.get(url)

        # wait for page to load
        time.sleep(2)

        browser.execute_script("window.scrollTo(0, document.body.scrollHeight);")

        print("Waiting for JS to finish ...")
        time.sleep(3)

        source = browser.find_element_by_xpath("//*").get_attribute("outerHTML")
        return BeautifulSoup(source, 'html.parser')
    except Exception as e:
        print(e)


def read_axie_cards_df():
    try:
        with open("axie_card_data_auto.json") as file:
            return pd.read_json(file)
    except Exception as e:
        print(e)


def download_axie_cards():

    soup = scrape_dynamic("https://www.axieworld.com/en/tools/cards-explorer")
    cards = soup.find_all("div", {"class": "_15NnuVhmyNHKrZ0lLQ6AA_"})

    keys = ["Type", "Part name", "Part", "Range", "Card name", "Cost", "Attack", "Defense", "Effect", "Tags"]
    values = []

    for i,res in enumerate(cards):
        children = res.findChildren("div", recursive=False)

        tmp = [[c.text if not isinstance(c, str) else c for c in child.children] for child in children]
        tmp = [[e for e in l if e != ''] for l in tmp]
        n = []
        for e in tmp:
            if isinstance(e, str):
                n.append(e)
            else:
                for e2 in e:
                    n.append(e2)

        t = "Back"
        if i % 22 > 5:
            t = "Horn"
        if i % 22 > 11:
            t = "Mouth"
        if i % 22 > 15:
            t = "Tail"

        el = "Aqua"
        if int(i / 22) == 1:
            el = "Beast"
        elif int(i / 22) == 2:
            el = "Bird"
        elif int(i / 22) == 3:
            el = "Bug"
        elif int(i / 22) == 4:
            el = "Plant"
        elif int(i / 22) == 2:
            el = "Reptile"

        tmp = [el] + n[:1] + [t] + n[1:] + ["..."]

        values.append({k: v for k, v in zip(keys, tmp)})

    d = pd.DataFrame(values, columns=keys)
    d = d.astype(str)
    d = d.astype({"Cost": int, "Attack": int, "Defense": int})
    d.to_json("axie_card_data_auto.json")
    return d


def get_axie_df():
    r = read_axie_cards_df()
    if r is not None and not r.empty:
        _f = r.astype(str)
        return _f.astype({"Cost": int, "Attack": int, "Defense": int})
    else:
        return download_axie_cards()


# <editor-fold desc="Queries">

def get_by_keyword(df, word):
    return df[df["Effect"].str.contains(word)]


def get_by_keywords_and(df, words):
    s = ""
    for word in words:
        s += word + ".*"
    s = s[:-2]
    return df[df.Effect.str.contains(s)]


def get_by_keywords_or(df, words):
    s = ""
    for word in words:
        s += word + "|"
    s = s[:-1]
    return df[df.Effect.str.contains(s)]

# </editor-fold>


def word_count(s):
    counts = dict()
    words = s.split()

    for word in words:
        if word in counts:
            counts[word] += 1
        else:
            counts[word] = 1

    return counts


def get_stop_words():
    with open("stop_words.txt") as file:
        return [e.replace("\n", "").lower() for e in file.readlines()]


type_to_color = {"Aqua": "blue", "Plant": "green", "Bird": "#ff66ff", "Reptile": "#b366ff", "Beast": "yellow"}
part_to_shape = {"Horn": "dot", "Mouth": "triagle", "Back": "box", "Tail": "diamond"}

stop_words = get_stop_words() + \
             ["axie", "target", "card", "2", "round", "1", "rounds", "axie's", "cards", "targets", "card"] + \
             ["deal", "apply", "damage"]


f = get_axie_df()

print(f)


eff_words = word_count(' '.join([w.lower().replace(".", "") for w in f.Effect.values]))
#d_view = [(v, k) for k, v in eff_words.items() if k not in stop_words]
#d_view.sort(reverse=True)


def common_words(eff1, eff2):

    if eff1 == eff2:
        return set()

    eff1 = [w for w in eff1.replace(".", "").lower().split()]
    eff2 = [w for w in eff2.replace(".", "").lower().split()]

    w1 = [w for w in eff1 if w not in stop_words]
    w2 = [w for w in eff2 if w not in stop_words]

    return set(w1).intersection(set(w2))


def show_common_eff_graph():
    weights = np.empty((len(f), len(f)), dtype=set)

    for i in range(len(f)):
        for j in range(len(f)):
            weights[i, j] = common_words(f.loc[i, "Effect"], f.loc[j, "Effect"])

    graph_df = pd.DataFrame(columns=["S", "T", "W", "C", "E1", "E2"])
    graph_df.astype({"S": str,
                     "T": str,
                     "W": int,
                     "C": str,
                     "E1": str,
                     "E2": str})
    c = 0
    for i in range(len(f)):
        for j in range(len(f)):
            if i < j and len(weights[i, j]) > 0:
                graph_df.loc[c] = [f.loc[i, "Card name"],
                                   f.loc[j, "Card name"],
                                   len(weights[i, j]),
                                   ", ".join(weights[i, j]),
                                   f.loc[i, "Effect"],
                                   f.loc[j, "Effect"]]
                c += 1

    print(graph_df)

    graph = nx.from_pandas_edgelist(graph_df, source="S", target="T", edge_attr=True)

    net = Network(notebook=True, width="100%", height="100%")
    net.from_nx(graph)
    for node in net.nodes:
        try:
            color = type_to_color[f.loc[f["Card name"] == node["label"]]["Type"][0]]
            size = (f.loc[f["Card name"] == node["label"]]["Cost"][0]+1)*5
            shape = part_to_shape[f.loc[f["Card name"] == node["label"]]["Part"][0]]

            node.options.update({"color": color, "size": size, "shape": shape})
        except KeyError as e:
            print(node)

    net.show("example.html")

    import webbrowser
    webbrowser.open("example.html", new=2)

    show(graph_df)
