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
from Utility import timeit


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


def select_optimal_parents(target_build: AxieBuild, max_children: (0, 0)):  # TODO WIP

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


def get_listings_of_build(target_build, gene_printer_lvl=2):
    orig_target_build = deepcopy(target_build)

    if "genes" in orig_target_build and "printer" in orig_target_build["genes"]:
        orig_target_build["genes"]["printer"] = gene_printer_lvl

    orig_target_build.update({"breedCount": 0})

    return get_current_listings(orig_target_build)


def get_stat_probs(p1genes, p2genes):
    from Axie_models import part_to_stats, type_to_stats

    part_name_to_class = json.load(open("part_name_to_class_type.json", "r"))

    dom2prob = {"d": 0.375, "r1": 0.09375, "r2": 0.03125}

    b_parts = ("horn", "tail", "eyes", "back", "mouth", "ears")

    child_gene_probs = {part: dict() for part in b_parts}
    child_gene_probs["element"] = dict()

    # element
    for elem in (p1genes["element"], p2genes["element"]):
        if elem in child_gene_probs["element"]:
            child_gene_probs["element"][elem] += 0.5
        else:
            child_gene_probs["element"][elem] = 0.5

    for bpart in b_parts:

        # parent 1 body parts
        for dominance, part_name in p1genes[bpart].items():

            if dominance == "element":
                continue

            if part_name in child_gene_probs[bpart]:
                child_gene_probs[bpart][part_name] += dom2prob[dominance]
            else:
                child_gene_probs[bpart][part_name] = dom2prob[dominance]

        # parent 2 body parts
        for dominance, part_name in p2genes[bpart].items():

            if dominance == "element":
                continue

            if part_name in child_gene_probs[bpart]:
                child_gene_probs[bpart][part_name] += dom2prob[dominance]
            else:
                child_gene_probs[bpart][part_name] = dom2prob[dominance]

    possible_elements = {p1genes["element"], p2genes["element"]}

    child_stat_probs = dict()

    pp(child_gene_probs)

    for elem in possible_elements:
        for horn, horn_prob in child_gene_probs["horn"].items():
            for mouth, mouth_prob in child_gene_probs["mouth"].items():
                for eyes, eyes_prob in child_gene_probs["eyes"].items():
                    for tail, tail_prob in child_gene_probs["tail"].items():
                        for ears, ears_prob in child_gene_probs["ears"].items():
                            for back, back_prob in child_gene_probs["back"].items():
                                p = horn_prob * mouth_prob * eyes_prob * tail_prob * ears_prob * back_prob
                                if len(possible_elements) > 1:
                                    p *= 0.5

                                # hp speed skill morale
                                child_stats = tuple([part_to_stats[part_name_to_class["horn"][horn]][i] +
                                                     part_to_stats[part_name_to_class["mouth"][mouth]][i] +
                                                     part_to_stats[part_name_to_class["tail"][tail]][i] +
                                                     part_to_stats[part_name_to_class["eyes"][eyes]][i] +
                                                     part_to_stats[part_name_to_class["ears"][ears]][i] +
                                                     part_to_stats[part_name_to_class["back"][back]][i] +
                                                     type_to_stats[elem][i] for i in range(4)])

                                if child_stats in child_stat_probs:
                                    child_stat_probs[child_stats] += p
                                else:
                                    child_stat_probs[child_stats] = p

    return child_stat_probs


def get_child_hit_prob(p1genes, p2genes, target_build):

    """
    :param p1genes:
    :param p2genes:
    :param target_build:
    :return: probability that the given target build is hit breeding the given parents (regarding parts and stats)
    """

    from Axie_models import part_to_stats, type_to_stats

    part_name_to_class = json.load(open("part_name_to_class_type.json", "r"))

    # constats
    dom2prob = {"d": 0.375, "r1": 0.09375, "r2": 0.03125}
    stat_names = ("hp", "speed", "skill", "morale")
    b_parts = ("horn", "tail", "eyes", "back", "mouth", "ears")

    child_gene_probs = {part: dict() for part in b_parts}
    child_gene_probs["element"] = dict()

    # element
    for elem in (p1genes["element"], p2genes["element"]):
        if elem in child_gene_probs["element"]:
            child_gene_probs["element"][elem] += 0.5
        else:
            child_gene_probs["element"][elem] = 0.5

    # parts
    for bpart in b_parts:

        # parent 1 body parts
        for dominance, part_name in p1genes[bpart].items():

            if dominance == "element":
                continue

            if part_name in child_gene_probs[bpart]:
                child_gene_probs[bpart][part_name] += dom2prob[dominance]
            else:
                child_gene_probs[bpart][part_name] = dom2prob[dominance]

        # parent 2 body parts
        for dominance, part_name in p2genes[bpart].items():

            if dominance == "element":
                continue

            if part_name in child_gene_probs[bpart]:
                child_gene_probs[bpart][part_name] += dom2prob[dominance]
            else:
                child_gene_probs[bpart][part_name] = dom2prob[dominance]

    possible_elements = {p1genes["element"], p2genes["element"]}
    child_build_hit = 0

    for elem in possible_elements:
        for horn, horn_prob in child_gene_probs["horn"].items():
            for mouth, mouth_prob in child_gene_probs["mouth"].items():
                for eyes, eyes_prob in child_gene_probs["eyes"].items():
                    for tail, tail_prob in child_gene_probs["tail"].items():
                        for ears, ears_prob in child_gene_probs["ears"].items():
                            for back, back_prob in child_gene_probs["back"].items():

                                p = horn_prob * mouth_prob * eyes_prob * tail_prob * ears_prob * back_prob
                                if len(possible_elements) > 1:
                                    p *= 0.5

                                # hp speed skill morale
                                child_stats = tuple([part_to_stats[part_name_to_class["horn"][horn]][i] +
                                                     part_to_stats[part_name_to_class["mouth"][mouth]][i] +
                                                     part_to_stats[part_name_to_class["tail"][tail]][i] +
                                                     part_to_stats[part_name_to_class["eyes"][eyes]][i] +
                                                     part_to_stats[part_name_to_class["ears"][ears]][i] +
                                                     part_to_stats[part_name_to_class["back"][back]][i] +
                                                     type_to_stats[elem][i] for i in range(4)])

                                # --------------------------------------

                                stats_compliant = True
                                for i, stat in enumerate(stat_names):
                                    if stat in target_build and not (target_build[stat][0] <= child_stats[i] <= target_build[stat][1]):
                                        stats_compliant = False

                                parts_compliant = True
                                if stats_compliant and target_build.get("parts", None):

                                    # element is not wrong
                                    if not ("element" in target_build["parts"] and elem.lower() not in target_build["parts"]["element"]):

                                        if ("horn" in target_build["parts"] and horn.lower() not in target_build["parts"]["horn"]) or \
                                           ("tail" in target_build["parts"] and tail.lower() not in target_build["parts"]["tail"]) or \
                                           ("back" in target_build["parts"] and back.lower() not in target_build["parts"]["back"]) or \
                                           ("mouth" in target_build["parts"] and mouth.lower() not in target_build["parts"]["mouth"]):

                                            parts_compliant = False

                                if parts_compliant and stats_compliant:
                                    child_build_hit += p

    return child_build_hit


def get_stat_hit_prob(target_build, stat_probs):

    probs = [1 for _ in range(4)]
    stat_names = ("hp", "speed", "skill", "morale")

    for i, stat in enumerate(stat_names):
        if stat in target_build:
            probs[i] = \
                sum([prob for stats, prob in stat_probs if target_build[stat][0] <= stats[i] <= target_build[stat][1]])

    return {stat_name: probs[i] for i, stat_name in zip(probs, stat_names)}


def simple_parent_selection(target_build, slp_price=0.08, axs_price=67, daily_child_value_decrease=0.05,
                            verbose=False, axie_floor=125):
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

    TODO dont just select parent 1 & 2 but calc hit percentage for every pair to assure optimal results

    ignores:
        potential better builds achieved by cheaper parents,
        more cost effective parents with prior children where the price difference makes up for the difference in breeding cost
        acutal sales instead of pure listings
        parent value acutally might seem to increase after breeding due to the lack of differentiation of ok genes and
            usefull genes, so bred parents might carry only usefull genes while virgins carry ok genes, making the
            bred parents more expensive
        if no data avail, uses 25% value decrease from breeding = inaccurate

    :param axie_floor:
    :param verbose:
    :param daily_child_value_decrease:
    :param axs_price:
    :param slp_price:
    :param target_build:
    :return:
    """

    breeding_cost_for_4_children_from_2_virgins = 5400 * slp_price + 4 * axs_price

    orig_target_build = deepcopy(target_build)

    # set relevant parts
    relevant_parts = set(list(target_build.get("genes", dict()).keys())+list(target_build.get("parts", dict()).keys()))
    if "printer" in relevant_parts:
        relevant_parts.remove("printer")

    # set desirable genes
    desireable_genes = flatten(list(target_build["parts"].values()))
    if target_build.get("genes", dict()).get("printer", 0) == 0 and len(target_build.get("genes", dict()).keys()) > 1:
        desireable_genes.extend(
            flatten([flatten(list(target_build["genes"][b_part].values())) for b_part in target_build["genes"].keys()])
        )

    printer_listings = \
        get_listings_of_build(orig_target_build, 0) + \
        get_listings_of_build(orig_target_build, 1) + \
        get_listings_of_build(orig_target_build, 2)

    parent_evaluation = dict()
    for listing1 in printer_listings:
        for listing2 in printer_listings:
            if listing1["id"] != listing2["id"]:

                s = sorted((listing1["id"], listing2["id"]))
                if f'{s[0]} & {s[1]}' not in parent_evaluation:

                    parent_evaluation[f'{s[0]} & {s[1]}'] = {
                        "price": float(listing1["auction"]["currentPriceUSD"]) +
                                 float(listing2["auction"]["currentPriceUSD"]),
                        "hit_prob": get_child_hit_prob(listing1["genes_refined"],
                                                       listing2["genes_refined"],
                                                       target_build)
                    }

    for axie_ids, data in parent_evaluation.items():
        parent_evaluation[axie_ids]["price/hit_prob"] = data["price"] / data["hit_prob"]

    parent_evaluation = sorted([(ids, data) for ids, data in parent_evaluation.items()],
                               key=lambda item: item[1]["price/hit_prob"])

    parent_eval_df = pd.DataFrame(index=[elem[0] for elem in parent_evaluation],
                                  data=[elem[1] for elem in parent_evaluation]).transpose()

    if verbose:
        print(parent_eval_df)
        print()

    if len(parent_evaluation) >= 2:

        # <editor-fold desc="get min child value">
        child_build = deepcopy(orig_target_build)
        if "genes" in child_build:
            del child_build["genes"]
        min_child_value = (float(get_current_listings(child_build)[0]["auction"]["currentPriceUSD"]) * \
                          (1-daily_child_value_decrease)**5 * parent_evaluation[0][1]["hit_prob"]) + \
                          ((1 - parent_evaluation[0][1]["hit_prob"]) * axie_floor)

        # </editor-fold>

        # <editor-fold desc="parent value after breeding">
        parent_after_build = deepcopy(orig_target_build)
        parent_after_build.update({"breedCount": 4})
        parent_liststins_after_breeding = get_current_listings(parent_after_build)
        try:
            parent_val_after_breeding = float(parent_liststins_after_breeding[0]["auction"]["currentPriceUSD"])
        except IndexError:
            # use 25 % decrease in value as proxy if no data available
            parent_val_after_breeding = 0.75 * parent_evaluation[0][1]["price"]
        # </editor-fold>

        # <editor-fold desc="calculate investments">
        parent_investment = parent_evaluation[0][1]["price"]
        cum_invest = parent_investment + breeding_cost_for_4_children_from_2_virgins

        parent_value_decrease = parent_evaluation[0][1]["price"] - (parent_val_after_breeding*2)
        if parent_value_decrease <= 0:
            parent_value_decrease = parent_evaluation[0][1]["price"] * 0.25

        profit = min_child_value*4 - parent_value_decrease + parent_investment - breeding_cost_for_4_children_from_2_virgins
        # </editor-fold>

        if verbose:
            print(f'\n'
                  f'        Breeding cost: {breeding_cost_for_4_children_from_2_virgins}\n'
                  f'      Min child value: {min_child_value: >4.2f}\n'
                  f'      Cum child value: {min_child_value*4: >4.2f}\n'
                  f'    Parent investment: {parent_investment: >4.2f}\n'
                  f'Parent after breeding: {parent_val_after_breeding: >4.2f}\n'
                  f'Parent value decrease: {parent_value_decrease: >4.2f}\n'
                  f'\n'
                  f'       Cum investment: {cum_invest: >4.2f}\n'
                  f'                 Gain: {min_child_value*4: >4.2f}\n'
                  f'               Profit: {profit: >4.2f}\n'
                  f'                  ROI: {100 * profit / cum_invest:.2f} %')

        breeding_summary = {
            "min_child_value": min_child_value,
            "cum_child_value": min_child_value*4,
            "parent_investment": parent_investment,
            "parent_after_breeding": parent_val_after_breeding,
            "parent_value_decrease": parent_value_decrease,
            "cum_investment": cum_invest,
            "gain": min_child_value*4,
            "profit": profit,
            "roi in %": 100 * profit / cum_invest,
            "roi/investment": 10000 * (profit / cum_invest) / cum_invest,
            "best_parents": f'{parent_evaluation[0][0]}',
            "hit_prob in %": parent_evaluation[0][1]["hit_prob"] * 100
        }

        parent_eval_sum = {
            "parent_eval": parent_evaluation,
            "parent_eval_df": parent_eval_df,
        }

        return breeding_summary, parent_eval_sum

    return dict(), {
        "parent_eval": parent_evaluation,
        "parent_eval_df": parent_eval_df
    }


def get_part_percentages(genes, desireable_genes, acceptable_genes, parts):

    probs = (0.375, 0.09375, 0.03125)
    # parts = ("back", "tail", "horn", "mouth", "eyes", "ears")
    part_classes = ("bug", "beast", "aqua", "plant", "reptile")

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

                elif gene.lower() in part_classes:

                    ...

    return [axie_desi_percentages, axie_acce_percentages]


def all_build_parent_selection():

    from AxieBuild import builds

    all_build_summaries = \
        sorted([(build_name, simple_parent_selection(build)[0]) for build_name, build in builds.items()],
               key=lambda item: item[1].get("roi/investment", 0), reverse=True)

    build_summary_df = pd.DataFrame(index=[item[0] for item in all_build_summaries],
                                    data=[item[1] for item in all_build_summaries])

    print()
    pp(build_summary_df)


def manual_parent_selection_helper():
    pd.set_option("display.max_rows", None, "display.max_columns", None)
    pd.set_option('display.width', 1000)
    pd.set_option('display.max_colwidth', 500)
    pd.set_option('display.float_format', lambda x: '%.2f' % x)

    # TODO maybe show market floor price for hit / miss / ok
    # TODO use card names instead of part names

    probs = (0.375, 0.09375, 0.03125)

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

    pd.set_option("display.max_rows", None, "display.max_columns", None)
    pd.set_option('display.width', 250)
    pd.set_option('display.max_colwidth', 250)
    pd.set_option('display.float_format', lambda x: '%.2f' % x)

    from AxieBuild import builds

    all_build_parent_selection()

    #pp(simple_parent_selection(builds.max_speed_dusk_terminator)[0])

