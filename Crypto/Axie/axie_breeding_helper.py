from axie_graphql import get_axie_price, get_genes_by_id
from functools import reduce
from pprint import pprint as pp
import json
import pandas as pd
from pandasgui import show

if __name__ == "__main__":

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

        parts = [s.strip().lower() for s in imp.split(",")]+c_imp if desi != "." else c_imp
        desireable_genes = [s.strip().lower() for s in desi.split(";")]+c_desi if desi != "." else c_desi
        acceptable_genes = [s.strip().lower() for s in acc.split(";")]+c_acce if acc != "." else c_acce

        parent = None
        while not parent:
            parent = input("       Parent ID: ")
            parent_id = parent if parent != "." else c_parent

        candidates = input("   Candidate IDs: ")
        candidate_ids = candidates.split()+c_candidates if candidates != "." else c_candidates

        print(f'     Parts: {parts}\n'
              f'Desireable: {desireable_genes}\n'
              f'Acceptable: {acceptable_genes}\n'
              f'    Parent: {parent_id}\n'
              f'Candidates: {candidate_ids}\n')

        with open("breeding_helper_hist.json", "w+") as file:
            json.dump({"c_imp": list(set(parts+c_imp)),
                       "c_desi": list(set(desireable_genes+c_desi)),
                       "c_acce": list(set(acceptable_genes+c_acce)),
                       "c_parent": parent_id,
                       "c_candidates": list(set(candidate_ids+c_candidates))}, file)

        parent_genes = get_genes_by_id(parent_id)
        candidate_data = [(get_genes_by_id(cid), get_axie_price(cid)) for cid in candidate_ids]
        for i, (g, p) in enumerate(candidate_data):
            if not p:
                del candidate_ids[i]
        candidate_data = [(g, p) for g, p in candidate_data if p]

        def get_part_percentages(genes):

            axie_desi_percentages = {part: 0 for part in parts}
            axie_acce_percentages = {part: 0 for part in parts}

            for part, genes in genes.items():

                if part in parts:

                    for dominance, gene in genes.items():

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

        parent_probs = get_part_percentages(parent_genes)

        results = {cid: {#"genes": candidate_data[i][0],
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

            overall_hit_prob = reduce(lambda x, y: x*y, breed_probs["hit"].values())
            overall_acc_prob = reduce(lambda x, y: x*y, [min(1, breed_probs["hit"][part]+breed_probs["ok"][part]) for part in parts])
            overall_miss_prob = 1 - overall_acc_prob

            part_probs[cid] = breed_probs
            results[cid][" hit prob"] = int(overall_hit_prob*100)
            results[cid]["  ok prob"] = int(overall_acc_prob*100)
            results[cid]["miss prob"] = int(overall_miss_prob*100)
            results[cid]["price"] = c_price
            results[cid]["hit p/price"] = int(100 * overall_hit_prob / c_price)
            results[cid]["ok p/price"] = int(100 * overall_acc_prob / c_price)
            results[cid]["no hit after 2"] = (1-overall_hit_prob)**2
            results[cid]["no hit after 3"] = (1-overall_hit_prob)**3
            results[cid]["no hit after 4"] = (1-overall_hit_prob)**4

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
