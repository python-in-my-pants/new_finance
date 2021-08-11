from bs4 import BeautifulSoup
import time
from dataclasses import dataclass
import pickle
from selenium import webdriver
from selenium.webdriver.firefox.options import Options
import pandas as pd
from pandasgui import show

pd.set_option("display.max_rows", None, "display.max_columns", None)
pd.set_option('display.width', 1000)
pd.set_option('display.max_colwidth', 500)
pd.set_option('display.float_format', lambda x: '%.2f' % x)

options = Options()
options.headless = True

browser = webdriver.Firefox(options=options)


def flatten(t):
    out = []
    for item in t:
        if isinstance(item, (list, tuple)):
            out.extend(flatten(item))
        else:
            out.append(item)
    return out


def selenium_scrape(url):

    try:
        print("Getting website", url)
        browser.get(url)

        # wait for page to load
        time.sleep(2)

        """
        # click the 3rd button (should lead to polygon)
        browser.find_elements_by_class_name("SelectMarketPanel__select-button")[2].click()
        """

        print("Waiting for JS to finish ...")
        time.sleep(5)

        source = browser.find_element_by_xpath("//*").get_attribute("outerHTML")
        return BeautifulSoup(source, 'html.parser')

    except Exception as e:
        print(e)


@dataclass
class SimplePart:

    part_name: str
    part_type: str

    def __repr__(self):
        return f'{self.part_name} ({self.part_type})'

    def __le__(self, other):
        return self.part_name.__le__(other.part_name)

    def __lt__(self, other):
        return self.part_name.__lt__(other.part_name)

    def __hash__(self):
        return self.part_name.__hash__()+self.part_type.__hash__()

    def toJSON(self):
        import json
        return json.dumps({"name": self.part_type, "type": self.part_type})


@dataclass
class SimpleAxieModel:

    element: str
    eyes: SimplePart
    ears: SimplePart
    back: SimplePart
    mouth: SimplePart
    horn: SimplePart
    tail: SimplePart

    def cards(self):
        return self.back, self.mouth, self.horn, self.tail

    def __repr__(self):
        return f'{self.element} axie:\n\t' \
            f'Back:\t{self.back}\n\t'\
            f'Mouth:\t{self.mouth}\n\t'\
            f'Horn:\t{self.horn}\n\t'\
            f'Tail:\t{self.tail}\n\t'\
            f'Eyes:\t{self.eyes}\n\t'\
            f'Ears:\t{self.ears}\n'

    def to_dict(self):
        return {
            "element": self.element,
            "back": self.back,
            "horn": self.horn,
            "mouth": self.mouth,
            "tail": self.tail
        }

    def __hash__(self):
        h = 0
        for c in self.cards():
            h += c.__hash__()
        return h


def get_top_100_decks(use_cached=True, limit=100):

    if use_cached:
        try:
            with open("leaderboard_teams.pickle", "rb") as file:
                obj = pickle.load(file)
                for i, team in enumerate(obj):
                    #print(f'~~~~~~~~ Team #{i+1} ~~~~~~~~\n')
                    for axie in team:
                        # print(axie)
                        ...
                return [o for o in obj if o]
        except FileNotFoundError as e:
            print(f'Exception occured while reading leaderboard file: {e}')

    player_teams = ["No teams present yet"]
    try:

        # get leaderboard site
        soup = selenium_scrape("https://axie.zone/leaderboard")
        #content = open("test_soup.html", encoding="utf-8").read()
        #soup = BeautifulSoup(content, 'html.parser')

        # extract player links from it
        player_urls = [x.attrs["href"] for x in soup.select('a[href*="/profile?ron_addr="]')][:limit]

        # request player decks one by one with sleeps to not trigger rate limits
        base_url = "http://axie.zone"
        player_teams = []
        counter = 0
        for url in player_urls:

            counter += 1

            print("Requesting player #", counter)

            player_team = []

            player_overview = selenium_scrape(base_url+url)

            cards = player_overview.find_all("div", {"class": "search_result_wrapper"})[:3]

            for card in cards:
                element = card.select('a[class*="search_result "]')[0].attrs["class"][1]

                parts = card.find_all("div", {"class": "purity_part"})

                b_parts = []
                for part in parts:
                    body_part, rest = part.contents[0].attrs["title"].split(":")
                    part_name, rest = rest.split("[")
                    part_element = rest.replace("]", "")

                    b_parts.append(SimplePart(part_name.strip(), part_element))

                player_team.append(SimpleAxieModel(element, *b_parts))

            player_teams.append(player_team)

            print()
            for axie in player_team:
                print(axie)

            time.sleep(60)

        with open("leaderboard_teams.pickle", "wb") as file:
            pickle.dump(player_teams, file)

        # beautify data & return
        return [o for o in player_teams if o]

    except Exception as e:
        from traceback import print_exc
        print(f'Exception occured while requesting: {e}')
        print_exc()
    finally:
        browser.quit()
        with open("leaderboard_teams.pickle", "wb") as file:
            pickle.dump(player_teams, file)


decks = get_top_100_decks(use_cached=True, limit=100)


def show_as_df(deck_list):
    try:
        with open("leaderboard_team_df.pickle", "rb") as file:
            df = pickle.load(file)
    except Exception:
        full_decks = [deck for deck in deck_list if deck]
        ind = [(f'Deck #{n+1}', k) for n in range(len(full_decks)) for k in range(1, 4)]
        multi_ind = pd.MultiIndex.from_tuples(ind, names=["Deck", "Axie"])
        df = pd.DataFrame.from_records([axie.to_dict() for axie in flatten(full_decks)], index=multi_ind)

        with open("leaderboard_team_df.pickle", "wb") as file:
            pickle.dump(df, file)

    print(df)
    show(df)


def simple_hist(word_list):

    from collections import Counter
    import numpy as np
    import matplotlib.pyplot as plt

    counts = Counter(word_list)

    labels, values = zip(*counts.items())

    # sort your values in descending order
    indSort = np.argsort(values)[::-1]

    # rearrange your data
    labels = np.array(labels)[indSort]
    values = np.array(values)[indSort]

    indexes = np.arange(len(labels))

    bar_width = 0.35

    plt.bar(indexes, values)

    # add labels
    plt.xticks(indexes + bar_width, labels, rotation="vertical")
    plt.tight_layout()
    plt.show()


def super_simple_hist(elements):

    d = {elem: elements.count(elem) for elem in elements}
    d = dict(sorted(d.items(), key=lambda item: item[1], reverse=True))

    mlen = max([len(str(k)) for k in d.keys()])
    s = sum(d.values())
    for k, v in d.items():
        print(f'{" "*(mlen-len(str(k)))}{k}: [{"+"*v}{" "*(s-v)}]')


def get_deck_combos():
    return sorted([tuple(sorted([card.element for card in deck])) for deck in decks], reverse=True)


def get_single_class_usage():
    return [card.element for card in flatten(decks)]


def get_decks_with_composition(comp):
    rel_decks = [deck for deck in decks if not {card.element for card in deck} - set(comp)]
    return rel_decks


def get_part_distribution(axies, i=17):
    """
    Reptile:
        Mouth:
            mouth1: 7 %
            mouth2: 3 %
            ...

    :param i: indent
    :param axies:
    :return:
    """
    indent = "\t"
    elem_to_part_list = {elem: {part: dict() for part in ("back", "mouth", "horn", "tail")} for elem
                         in set([axie.element for axie in axies])}
    for axie in axies:
        elem_to_part_list[axie.element]["back"].update(
            {axie.back: elem_to_part_list[axie.element]["back"].get(axie.back, 0)+1})
        elem_to_part_list[axie.element]["mouth"].update(
            {axie.mouth: elem_to_part_list[axie.element]["mouth"].get(axie.mouth, 0)+1})
        elem_to_part_list[axie.element]["horn"].update(
            {axie.horn: elem_to_part_list[axie.element]["horn"].get(axie.horn, 0)+1})
        elem_to_part_list[axie.element]["tail"].update(
            {axie.tail: elem_to_part_list[axie.element]["tail"].get(axie.tail, 0)+1})

    rel_dict = {elem: {part: dict() for part in ("back", "mouth", "horn", "tail")} for elem
                in set([axie.element for axie in axies])}

    for elem in elem_to_part_list.keys():
        print(f'{indent*i}{elem.upper()}:')
        for bpart, distr in elem_to_part_list[elem].items():
            print(f'{indent*i}\t\t{bpart.upper()}:')
            for part, occ in elem_to_part_list[elem][bpart].items():
                rel_dict[elem][bpart][part] = 100 * occ / sum(elem_to_part_list[elem][bpart].values())

            mlen = max([len(str(k)) for k in elem_to_part_list[elem][bpart].keys()])

            """
            s = dict(sorted(rel_dict[elem][bpart].items(), key=lambda item: item[1], reverse=True))
            for card, v in s.items():
                print(f'{indent*i}\t\t{card}:{" " * (mlen - len(str(card)))} {v: >.2f} %')
            """

            s = dict(sorted(elem_to_part_list[elem][bpart].items(), key=lambda item: item[1], reverse=True))
            v_max = sum(s.values())
            for card, v in s.items():
                print(f'{indent * i}\t\t\t\t{card}:{" " * (mlen - len(str(card)))} [{v*"+"}{" "*(v_max-v)}]')


def get_best_card_combs(axies):

    combs = dict()
    for axie in axies:
        for card in axie.cards():
            for card2 in axie.cards():
                if card != card2:
                    d = tuple(sorted((card, card2)))
                    combs[d] = combs.get(d, 0) + 1

    r = dict(sorted(combs.items(), key=lambda item: item[1], reverse=True))
    mlen = max([len(str(s)) for s in r.keys()])
    for k, v in r.items():
        print(f'{k}: {" "*(mlen-len(str(k)))}{v}')
    return r


super_simple_hist(get_single_class_usage())
super_simple_hist(get_deck_combos())

best_comb = get_deck_combos()[0]
print("\nBest comb:", best_comb, "\n")
get_part_distribution(flatten(get_decks_with_composition(best_comb)))

get_best_card_combs(flatten(get_decks_with_composition(best_comb)))


try:
    browser.close()
    browser.quit()
    import os
    os.system("taskkill /f /im geckodriver.exe /T")
    os.system("taskkill /f /im chromedriver.exe /T")
    os.system("taskkill /f /im IEDriverServer.exe /T")
except Exception as e:
    pass
