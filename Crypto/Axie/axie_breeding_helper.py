from axie_graphql import get_axie_price, get_genes_by_id, get_current_listings
from functools import reduce
from pprint import pprint as pp
import json
import pandas as pd
from pandasgui import show
from AxieBuild import AxieBuild
from time import time
from Axie_models import flatten
from copy import copy, deepcopy


"""

### Which children to get for breeding a certain build? ###

compute minBreedingGain (like in google sheets) = min price to sell children for to break even on the breeding cost

select meta build to breed (or build with variations)

select maxChildren per paren

check past sales (1 week?, 2 weekss?, 1 month?) of this build to get minBuildPrice, check for sufficient volume of sales
in axies, if too low volume, choose another build

for each body part:
    remove those parts from sales query and show only those axies which sale price was above current minBreedingGain
    -> add all seen parts to "ok parts" until you reach an axie which exactly fits the given meta build (no bad parts
    present), after first meta build hit, add all other seen parts to "hit parts" as they are alternatives to the
    given build with a higher valueation (improvement: value past sales less than recent ones)

with "hit parts" from meta build and "ok parts" from previous search: get selection of breeding partners from current
listings

for each 2 distinct offers, calc expected value of breeding 1, 2, 3, 4 times:
    query past sales with all parts that could result as dominant in the child
    for each sale in sales sorted by price and starting from first virgin (so non-virgins don't drag price down as we
    will be selling virgins):
        calc the prob of the offspring to equal those genes, if variations occur multiple times, average over them
        = weight sale prices by prob of outcome
    expected breeding profit = expected child value (sum of weighted prices) - breeding cost for this child (1-4?)

present potential parent pairs in descending profitability order

-----------------------------------
"""
"""

### Refined search for axie listings ### (this is not really breeding though)

like:

axie with x shield / dmg
    average / cumulative
    median
y beast dmg
n cumulative card cost
m zero cost cards
[a,b,c] tags
parts ([l, m, n] and [u, v, w]) OR ([a,b,c] and [q,r])
top 100 most played deck front liner (exact build with most popular parts / rough build with more uncommon parts 
included)
exclude certain parts

"""


def compute_min_breeding_gain(p1_children, p2_children, breed_times):
    ...
    return 0


def select_optimal_parents(target_build: AxieBuild, max_children: (0, 0)):

    min_breeding_gain = compute_min_breeding_gain(max_children[0], max_children[1], ...)

    with open("axie_sales.json", "r") as sales_file:
        sales_data = json.load(sales_file)

    min_build_price = 10*18  # min price that was paid for a virgin with this build
    max_time = 60*60*24*7  # 1 week
    current_timestamp = int(time())

    desireable_genes = {b_part: [] for b_part in ("mouth", "back", "horn", "tail")}
    acceptable_genes = {b_part: [] for b_part in ("mouth", "back", "horn", "tail")}

    # find desireable and acceptable genes based on past sales
    for axie_id, axie_data in sales_data.items():

        for to_omit in ( (), ["mouth"], ["back"], ["tail"], ["horn"] ):
            if target_build.compliant(axie_id, to_omit):

                if not to_omit:
                    if axie_data["transfers"][0]["usd"] < min_build_price and \
                            axie_data["breedCount"] == 0 and \
                            axie_data["transfers"][0]["timestamp"] >= current_timestamp - max_time:
                        min_build_price = axie_data["transfers"][0]["usd"]

                else:
                    ...

    # query market listings for all desirable and acceptable genes


def simple_parent_selection(target_build):
    """
    {
        "classes": ["dusk"],
        "parts": {
            "back": ["snail shell"],
            "mouth": ["tiny turtle"],
            "tail": ["thorny caterpillar"],
            "horn": ["lagging"]
        },
        "genes": {"printer": 1},
    }
    ignores:
        potential better builds achieved by cheaper parents,
        more cost effective parents with prior children where the price difference makes up for the difference in breeding cost
        acutal sales instead of pure listings

    :param target_build:
    :return:
    """
    orig_target_build = deepcopy(target_build)
    breeding_cost_for_4_children_from_2_virgins = 800

    target_build.update({"breedCount": 0})

    relevant_parts = set(list(target_build.get("genes", dict()).keys())
                         + list(target_build.get("parts", dict()).keys()))
    if "printer" in relevant_parts:
        relevant_parts.remove("printer")
    desireable_genes = flatten(list(target_build["parts"].values()))
    if target_build.get("genes", dict()).get("printer", 0) == 0 and len(target_build.get("genes", dict()).keys()) > 1:
        desireable_genes.extend(flatten([flatten(list(target_build["genes"][b_part].values()))
                                         for b_part in target_build["genes"].keys()]))

    printer_listings = get_current_listings(target_build)

    parent_evaluation = {
        listing["id"]: {
            "price": float(listing["auction"]["currentPriceUSD"]),
            "hit_prob": reduce(lambda x, y: x*y,
                               [2*x for x in get_part_percentages(listing["genes_refined"], desireable_genes, (),
                                                                  relevant_parts)[0].values()])
        }
        for listing in printer_listings
    }

    for axie_id, data in parent_evaluation.items():
        parent_evaluation[axie_id]["price/hit_prob"] = data["price"] / data["hit_prob"]

    parent_evaluation = sorted(parent_evaluation.items(), key=lambda item: item[1]["price/hit_prob"])

    pp(parent_evaluation)

    if len(parent_evaluation) >= 2:

        child_build = deepcopy(orig_target_build)
        if "genes" in child_build:
            del child_build["genes"]
        min_child_value = float(get_current_listings(child_build)[0]["auction"]["currentPriceUSD"])

        parent_after_build = deepcopy(orig_target_build)
        parent_after_build.update({"breedCount": 4})
        parent_val_after_breeding = float(get_current_listings(parent_after_build)[0]["auction"]["currentPriceUSD"])

        parent_investment = parent_evaluation[0][1]["price"] + parent_evaluation[1][1]["price"]
        cum_invest = parent_investment + breeding_cost_for_4_children_from_2_virgins
        parent_value_decrease = parent_evaluation[0][1]["price"] + parent_evaluation[1][1]["price"] - parent_val_after_breeding*2
        profit = min_child_value*4 - parent_value_decrease + parent_investment - breeding_cost_for_4_children_from_2_virgins

        print(f'\n'
              f'      Min child value: {min_child_value: >4.2f}\n'
              f'      Cum child value: {min_child_value*4: >4.2f}\n'
              f'    Parent investment: {parent_investment: >4.2f}\n'
              f'Parent after breeding: {parent_val_after_breeding: >4.2f}\n'
              f'Parent value decrease: {parent_value_decrease: >4.2f}\n'
              f'\n'
              f'       Cum investment: {cum_invest: >4.2f}\n'
              f'                 Gain: {min_child_value*4: >4.2f}\n'
              f'               Profit: {profit: >4.2f}\n')

    return parent_evaluation


def get_part_percentages(genes, desireable_genes, acceptable_genes, parts):

    probs = (0.375, 0.09375, 0.03125)
    # parts = ("back", "tail", "horn", "mouth", "eyes", "ears")

    axie_desi_percentages = {part: 0 for part in parts}
    axie_acce_percentages = {part: 0 for part in parts}

    for part, part_genes in genes.items():

        if part in parts:

            for dominance, gene in part_genes.items():

                if dominance == "element":
                    continue

                if gene.lower() in desireable_genes:

                    if axie_desi_percentages[part] == 0.5:
                        axie_desi_percentages[part] = 0

                    if dominance == "d":
                        axie_desi_percentages[part] += probs[0]
                    elif dominance == "r1":
                        axie_desi_percentages[part] += probs[1]
                    elif dominance == "r2":
                        axie_desi_percentages[part] += probs[2]

                elif gene.lower() in acceptable_genes:

                    if axie_acce_percentages[part] == 0.5:
                        axie_acce_percentages[part] = 0

                    if dominance == "d":
                        axie_acce_percentages[part] += probs[0]
                    elif dominance == "r1":
                        axie_acce_percentages[part] += probs[1]
                    elif dominance == "r2":
                        axie_acce_percentages[part] += probs[2]

    return [axie_desi_percentages, axie_acce_percentages]


def manual_parent_selection_helper():
    pd.set_option("display.max_rows", None, "display.max_columns", None)
    pd.set_option('display.width', 1000)
    pd.set_option('display.max_colwidth', 500)
    pd.set_option('display.float_format', lambda x: '%.2f' % x)

    # TODO maybe show market floor price for hit / miss / ok
    # TODO use card names instead of part names

    probs = (0.375, 0.09375, 0.03125)
    # parts = ("back", "tail", "horn", "mouth", "eyes", "ears")

    while 1:

        try:
            with open("breeding_helper_hist.json", "r+") as file:
                hist = json.load(file)

                c_imp = hist["c_imp"]
                c_desi = hist["c_desi"]
                c_acce = hist["c_acce"]

                c_parent = hist["c_parent"]
                c_candidates = hist["c_candidates"]

        except Exception as e:
            c_imp = list()
            c_desi = list()
            c_acce = list()

            c_parent = ""
            c_candidates = list()

        print("Input card names split by ';' or type '.' to use previous settings")

        imp, desi, acc = None, None, None

        while not all([imp, desi, acc]):
            imp = input("  Relevant parts: ")
            desi = input("Desireable genes: ")
            acc = input("Acceptable genes: ")

        parts = [s.strip().lower() for s in imp.split(",")] + c_imp if desi != "." else c_imp
        desireable_genes = [s.strip().lower() for s in desi.split(";")] + c_desi if desi != "." else c_desi
        acceptable_genes = [s.strip().lower() for s in acc.split(";")] + c_acce if acc != "." else c_acce

        parent = None
        while not parent:
            parent = input("       Parent ID: ")
            parent_id = parent if parent != "." else c_parent

        candidates = input("   Candidate IDs: ")
        candidate_ids = candidates.split() + c_candidates if candidates != "." else c_candidates

        print(f'     Parts: {parts}\n'
              f'Desireable: {desireable_genes}\n'
              f'Acceptable: {acceptable_genes}\n'
              f'    Parent: {parent_id}\n'
              f'Candidates: {candidate_ids}\n')

        with open("breeding_helper_hist.json", "w+") as file:
            json.dump({"c_imp": list(set(parts + c_imp)),
                       "c_desi": list(set(desireable_genes + c_desi)),
                       "c_acce": list(set(acceptable_genes + c_acce)),
                       "c_parent": parent_id,
                       "c_candidates": list(set(candidate_ids + c_candidates))}, file)

        parent_genes = get_genes_by_id(parent_id)
        candidate_data = [(get_genes_by_id(cid), get_axie_price(cid)) for cid in candidate_ids]
        for i, (g, p) in enumerate(candidate_data):
            if not p:
                del candidate_ids[i]
        candidate_data = [(g, p) for g, p in candidate_data if p]

        parent_probs = get_part_percentages(parent_genes, desireable_genes, acceptable_genes)

        results = {cid: {  # "genes": candidate_data[i][0],
            " hit prob": -1,
            "  ok prob": -1,
            "miss prob": -1,
            "price": -1,
            "hit p/price": -1,
            "ok p/price": -1} for i, cid in enumerate(candidate_ids)}
        part_probs = {cid: 0 for cid in candidate_ids}

        for cid, (c_genes, c_price) in zip(candidate_ids, candidate_data):
            cand_probs = get_part_percentages(c_genes)

            breed_probs = {t: {part: parent_probs[i][part] + cand_probs[i][part] for part in parts}
                           for i, t in enumerate(("hit", "ok"))}

            overall_hit_prob = reduce(lambda x, y: x * y, breed_probs["hit"].values())
            overall_acc_prob = reduce(lambda x, y: x * y,
                                      [min(1, breed_probs["hit"][part] + breed_probs["ok"][part]) for part in parts])
            overall_miss_prob = 1 - overall_acc_prob

            part_probs[cid] = breed_probs
            results[cid][" hit prob"] = int(overall_hit_prob * 100)
            results[cid]["  ok prob"] = int(overall_acc_prob * 100)
            results[cid]["miss prob"] = int(overall_miss_prob * 100)
            results[cid]["price"] = c_price
            results[cid]["hit p/price"] = int(100 * overall_hit_prob / c_price)
            results[cid]["ok p/price"] = int(100 * overall_acc_prob / c_price)
            results[cid]["no hit after 2"] = (1 - overall_hit_prob) ** 2
            results[cid]["no hit after 3"] = (1 - overall_hit_prob) ** 3
            results[cid]["no hit after 4"] = (1 - overall_hit_prob) ** 4

        """for k, v in dict(sorted(results.items(), key=lambda r: r[1]["hit p/price"], reverse=True)).items():
            print("Axie", k, f'Worth: {v["hit p/price"]}')
            pp(v, indent=4)"""

        df = pd.DataFrame(dict(sorted(results.items(), key=lambda r: r[1]["hit p/price"], reverse=True)))
        print(df)

        for i in range(len(candidate_ids)):
            print("\n", candidate_ids[i])
            pp(part_probs[candidate_ids[i]])
            for k, v in candidate_data[i][0].items():
                if k == "element":
                    print(k, v)
                else:
                    print("\t", k)
                    print("\t\t", v)

        show(df.transpose())

        action = input("\nType 'reset' to reset settings, 'exit' to exit: ")

        if action == "exit":
            print("Exiting ...")
            exit(0)

        if action == "reset":
            print("Deleting history file ...")
            import os
            os.remove("breeding_helper_hist.json")


if __name__ == "__main__":

    # TODO ear/ element gene support

    from AxieBuild import builds

    # manual_parent_selection_helper()
    simple_parent_selection(builds.backdoor_bird)
