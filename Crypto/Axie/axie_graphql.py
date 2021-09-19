import requests
import json
from pprint import pprint as pp
from DDict import DDict
from time import time, sleep
from datetime import datetime


def print_axie_db():

    with open("axie_gene_cache.json", "r") as cache_file:
        pp(json.load(cache_file))


def graphql_request(body):
    print(f"[{datetime.now()}] GraphQl request sent!")  # tODO remove
    headers = {'Content-type': 'application/json', 'Accept': 'text/plain'}
    r = requests.post('https://axieinfinity.com/graphql-server-v2/graphql',  headers=headers, json=body)
    if r.status_code is not 200 or True:
        print("Status code:", r.status_code)
    return r


def axie_technology_request(url):
    headers = {'Content-type': 'application/json', 'Accept': 'text/plain'}
    r = requests.post(url, headers=headers)
    if r.status_code is not 200 or True:
        print("Status code:", r.status_code)
    return r


def axie_tech_request_genes_batch(axie_ids):
    axie_ids = [str(_id) for _id in axie_ids]
    ids_string = ','.join(axie_ids)
    url = f'https://api.axie.technology/getgenes/{ids_string}'

    print("Batch request for genes:", ids_string)

    resp = axie_technology_request(url).json()
    if isinstance(resp, dict):
        resp = [resp]

    return DDict({
        axie_id: DDict({
            b_part: DDict({
                dom: f'{resp[i][b_part][dom]["name"]}'  # ({resp[i][b_part][dom]["class"]})'
                for dom in ("d", "r1", "r2")
            })
            for b_part in ("mouth", "tail", "horn", "back", "eyes", "ears")
        }).update({
            "element": resp[i]["cls"]
        })
        for i, axie_id in enumerate(axie_ids)
    })


def get_last_sales(min_age_seconds_before_new_request=0, length=100):

    hist_sales_file = dict()
    try:

        with open("axie_sales.json", "r") as file:
            hist_sales_file = json.load(file)
            time_delta = int(time()) - int(hist_sales_file["timestamp"])
            from_time = int(hist_sales_file["timestamp"])
            print("Seconds since last update", time_delta)
            print("Sales timestamp:", from_time)

    except FileNotFoundError:
        time_delta = 10**18
        from_time = 0

    if time_delta >= min_age_seconds_before_new_request:
    
        body = {
          "operationName": "GetRecentlyAxiesSold",
          "variables": {
            "from": 0,
            "size": length
          },
          "query": "query GetRecentlyAxiesSold($from: Int, $size: Int) {\n  settledAuctions {\n    axies(from: $from, "
                   "size: $size) {\n      total\n      results {\n        ...AxieSettledBrief\n        transferHistory "
                   "{\n          ...TransferHistoryInSettledAuction\n          __typename\n        }\n        __typename"
                   "\n      }\n      __typename\n    }\n    __typename\n  }\n}\n\nfragment AxieSettledBrief on Axie {\n  "
                   "id\n  name\n  image\n  class\n  breedCount\n  __typename\n}\n\nfragment "
                   "TransferHistoryInSettledAuction on TransferRecords {\n  total\n  results {\n    "
                   "...TransferRecordInSettledAuction\n    __typename\n  }\n  __typename\n}\n\nfragment "
                   "TransferRecordInSettledAuction on TransferRecord {\n  from\n  to\n  txHash\n  timestamp\n  withPrice"
                   "\n  withPriceUsd\n  fromProfile {\n    name\n    __typename\n  }\n  toProfile {\n    name\n    __"
                   "typename\n  }\n  __typename\n}\n"
        }

        try:
            resp = graphql_request(body).json()

            sale_times = [int(entry["transferHistory"]["results"][0]["timestamp"])
                          for entry in resp["data"]["settledAuctions"]["axies"]["results"] if
                          int(entry["transferHistory"]["results"][0]["timestamp"]) > from_time]
            print("Last sale:", max(sale_times) if sale_times else -1)

            resp = [entry for entry in resp["data"]["settledAuctions"]["axies"]["results"] if
                    int(entry["transferHistory"]["results"][0]["timestamp"]) >= from_time]

        except TypeError:
            print("Response is empty! Marketplace server seems down")
            return dict()
        except json.JSONDecodeError:
            print("Response is empty! Marketplace server seems down")
            return dict()

        if resp:

            try:

                gene_dict = get_axie_details_batch([entry["id"] for entry in resp])

                refined_resp = {
                    entry["id"]: {
                        "transfers": [{
                                "wei": int(entry["transferHistory"]["results"][i]["withPrice"]),
                                "usd": float(entry["transferHistory"]["results"][i]["withPriceUsd"]),
                                "time": int(entry["transferHistory"]["results"][i]["timestamp"])}
                            for i in range(len(entry["transferHistory"]["results"]))],
                        "genes": gene_dict[entry["id"]],
                        "breedCount": int(entry["breedCount"]),
                    } for entry in resp
                }

                if hist_sales_file:
                    refined_resp.update(hist_sales_file)
                refined_resp.update({"timestamp": int(time())})

                # save as json
                with open("axie_sales.json", "w") as file:
                    json.dump(refined_resp, file)

                return refined_resp

            except Exception as e:
                from traceback import print_exc
                print("Exception occured:", e)
                print_exc()
                return hist_sales_file
        else:
            return hist_sales_file

    else:
        return hist_sales_file


def get_current_listings(criteria, n=10000):

    """
    example usage:

    get_current_listings({
        "classes": ["aqua"],
        "parts": {
            "back": ["sponge"]
        },
        "genes": {
            "mouth": {
                "r1": ["risky fish"]
            }
        },
        "price": [160, 1000]
    })

    :param criteria:
    :param n:
    :return:
    """

    gene_criteria = criteria.get("genes", dict())
    if "genes" in criteria.keys():
        del criteria["genes"]

    if gene_criteria.get("printer", 0) == 2:
        gene_criteria = {part: {"r1": criteria["parts"][part], "r2": criteria["parts"][part]}
                         for part in criteria["parts"].keys()}
    elif gene_criteria.get("printer", 0) == 1:
        gene_criteria = {part: {"r1": criteria["parts"][part]} for part in criteria["parts"].keys()}
    elif "printer" in gene_criteria.keys() and gene_criteria.get("printer", 0) == 0:
        del gene_criteria["printer"]

    price = criteria.get("price", [0, 10**18])
    if "price" in criteria.keys():
        del criteria["price"]

    if "parts" in criteria.keys():
        converted_part_names = []
        for b_part, parts in criteria["parts"].items():
            for part in parts:
                converted_part_names.append(f'{b_part.lower()}-{part.lower().replace(" ", "-")}')
        criteria["parts"] = converted_part_names

    if "classes" in criteria.keys():
        if "aqua" in criteria["classes"]:
            criteria["classes"].remove("aqua")
            criteria["classes"].append("aquatic")
        criteria["classes"] = [c.title() for c in criteria["classes"]]

    body = {
      "operationName": "GetAxieLatest",
      "variables": {
        "from": 0,
        "size": n,
        "sort": "PriceAsc",
        "auctionType": "Sale",
        "criteria": criteria
      },
      "query": "query GetAxieLatest($auctionType: AuctionType, $criteria: AxieSearchCriteria, $from: Int, $sort: SortBy, $size: Int, $owner: String) {\n  axies(auctionType: $auctionType, criteria: $criteria, from: $from, sort: $sort, size: $size, owner: $owner) {\n    total\n    results {\n      ...AxieRowData\n      __typename\n    }\n    __typename\n  }\n}\n\nfragment AxieRowData on Axie {\n  id\n  image\n  class\n  name\n  genes\n  owner\n  class\n  stage\n  title\n  breedCount\n  level\n  parts {\n    ...AxiePart\n    __typename\n  }\n  stats {\n    ...AxieStats\n    __typename\n  }\n  auction {\n    ...AxieAuction\n    __typename\n  }\n  __typename\n}\n\nfragment AxiePart on AxiePart {\n  id\n  name\n  class\n  type\n  specialGenes\n  stage\n  abilities {\n    ...AxieCardAbility\n    __typename\n  }\n  __typename\n}\n\nfragment AxieCardAbility on AxieCardAbility {\n  id\n  name\n  attack\n  defense\n  energy\n  description\n  backgroundUrl\n  effectIconUrl\n  __typename\n}\n\nfragment AxieStats on AxieStats {\n  hp\n  speed\n  skill\n  morale\n  __typename\n}\n\nfragment AxieAuction on Auction {\n  startingPrice\n  endingPrice\n  startingTimestamp\n  endingTimestamp\n  duration\n  timeLeft\n  currentPrice\n  currentPriceUSD\n  suggestedPrice\n  seller\n  listingIndex\n  state\n  __typename\n}\n"
    }

    resp = graphql_request(body).json()["data"]["axies"]["results"]

    for entry in resp:
        entry["genes_refined"] = part_names_from_genes(hex_to_256bin_str(entry["genes"]))

    def genes_compliant(genes):
        if not gene_criteria:
            return True
        else:
            for _b_part, gene_dict in gene_criteria.items():
                for recessive_level, _parts in gene_dict.items():
                    if not genes[_b_part][recessive_level].lower() in [p.lower() for p in _parts]:
                        return False
            return True

    resp = [entry for entry in resp if
            genes_compliant(entry["genes_refined"]) and
            price[1] >= float(entry["auction"]["currentPriceUSD"]) >= price[0]]

    return resp


def get_axie_details(axie_id, force_request=False):

    try:
        with open("axie_gene_cache.json", "r") as cache_file:

            cache = json.load(cache_file)

            if not force_request and str(axie_id) in cache.keys():
                return cache[str(axie_id)]

    except json.decoder.JSONDecodeError:
        cache = dict()

    request_body = {
      "operation": "GetAxieDetail",
      "variables": {
        "axieId": str(axie_id)
      },
      "query": "query GetAxieDetail($axieId: ID!) {\n  axie(axieId: $axieId) {\n    ...AxieDetail\n    __typename\n  }\n}"
               "\n\nfragment AxieDetail on Axie {\n  id\n  image\n  class\n  chain\n  name\n  genes\n  owner\n  birthDate"
               "\n  bodyShape\n  class\n  sireId\n  sireClass\n  matronId\n  matronClass\n  stage\n  title\n  breedCount"
               "\n  level\n  figure {\n    atlas\n    model\n    image\n    __typename\n  }\n  parts {\n    ...AxiePart"
               "\n    __typename\n  }\n  stats {\n    ...AxieStats\n    __typename\n  }\n  auction {\n    ...AxieAuction"
               "\n    __typename\n  }\n  ownerProfile {\n    name\n    __typename\n  }\n  battleInfo {\n    "
               "...AxieBattleInfo\n    __typename\n  }\n  children {\n    id\n    name\n    class\n    image\n    "
               "title\n    stage\n    __typename\n  }\n  __typename\n}\n\nfragment AxieBattleInfo on AxieBattleInfo {\n  "
               "banned\n  banUntil\n  level\n  __typename\n}\n\nfragment AxiePart on AxiePart {\n  id\n  name\n  class\n  "
               "type\n  specialGenes\n  stage\n  abilities {\n    ...AxieCardAbility\n    __typename\n  }\n  __typename\n}"
               "\n\nfragment AxieCardAbility on AxieCardAbility {\n  id\n  name\n  attack\n  defense\n  energy\n  "
               "description\n  backgroundUrl\n  effectIconUrl\n  __typename\n}\n\nfragment AxieStats on AxieStats {\n  "
               "hp\n  speed\n  skill\n  morale\n  __typename\n}\n\nfragment AxieAuction on Auction {\n  startingPrice\n  "
               "endingPrice\n  startingTimestamp\n  endingTimestamp\n  duration\n  timeLeft\n  currentPrice\n  "
               "currentPriceUSD\n  suggestedPrice\n  seller\n  listingIndex\n  state\n  __typename\n}\n"
        }

    resp = graphql_request(request_body)

    cache[str(axie_id)] = resp.json()

    with open("axie_gene_cache.json", "w") as cache_file:
        json.dump(cache, cache_file)

    return resp.json()


def get_axie_details_batch(id_list, force_request=False):

    to_request = id_list
    in_cache = dict()
    try:
        with open("axie_gene_cache.json", "r") as cache_file:

            cache = json.load(cache_file)
            id_in_cache = [_id in cache for _id in id_list]

            if not force_request and all(id_in_cache):
                return {axie_id: cache[str(axie_id)] for axie_id in id_list}
            else:
                to_request = [_id for _id in id_list if _id not in cache]
                in_cache = {axie_id: cache[str(axie_id)] for i, axie_id in enumerate(id_list) if id_in_cache[i]}

    except json.decoder.JSONDecodeError:
        cache = dict()

    gene_data = axie_tech_request_genes_batch(to_request)
    cache.update(gene_data)

    with open("axie_gene_cache.json", "w") as cache_file:
        json.dump(cache, cache_file)

    gene_data.update(in_cache)
    return gene_data


def genes_from_axie_details(details):
    """
    :param details: axie details in json as returned by graphql api
    :return:
    """
    return details["data"]["axie"]["genes"]


def hex_to_256bin_str(h):
    binary_string = str(bin(int(h, 16))[2:])
    return ("0"*(256-len(binary_string))) + binary_string


def part_names_from_genes(genes: str):

    """
    expects 256 bit gene string in binary

    :param genes:
    :return: DDict with: "element: ..., Part: Dom, r1, r2"
    """

    part_gene_dict = json.load(open("traits.json", "r"))

    gene_dict = DDict({
        "Class": genes[0:4],
        "Region": genes[8:13],
        "Tag": genes[13:18],
        "BodySkin": genes[18:22],
        "Xmas": genes[22:34],
        "Pattern": genes[34:52],
        "Color": genes[52:64],
        "Eyes": genes[64:96],
        "Mouth": genes[96:128],
        "Ears": genes[128:160],
        "Horn": genes[160:192],
        "Back": genes[192:224],
        "Tail": genes[224:256],
    })

    bin_to_elem = {
        "0000": "Beast",
        "0001": "Bug",
        "0010": "Bird",
        "0011": "Plant",
        "0100": "Aquatic",
        "0101": "Reptile",
        "1000": "Mech",
        "1010": "Dusk",
        "1001": "Dawn",
    }

    # each parts genes info comes as 32-bit string: 2 skin, 10 dom, 10 r1, 10 r2

    # each genes info comes as 10-bit string: 4 element, 6 part

    def part_from_gene(class_gene, part_type, part_gene):
        try:
            return part_gene_dict[bin_to_elem[class_gene].lower()][part_type][part_gene]["global"]
        except KeyError as e:
            print("Key error:", e, class_gene, part_type, part_gene)
            return dict()
        except json.JSONDecodeError as e:
            print("JSON decode error:", e)
            return dict()

    def get_part_genes(part_type):

        bit_genes32 = gene_dict[part_type.title()]

        dom_genes = bit_genes32[2:12]
        dom_name = part_from_gene(dom_genes[0:4], part_type, dom_genes[4:])

        r1_genes = bit_genes32[12:22]
        r1_name = part_from_gene(r1_genes[0:4], part_type, r1_genes[4:])

        r2_genes = bit_genes32[22:32]
        r2_name = part_from_gene(r2_genes[0:4], part_type, r2_genes[4:])

        return DDict({"element": bin_to_elem[dom_genes[0:4]], "d": dom_name, "r1": r1_name, "r2": r2_name})

    return DDict({
        "element": bin_to_elem[gene_dict.Class].lower().replace("aquatic", "aqua"),
        "back": get_part_genes("back"),
        "mouth": get_part_genes("mouth"),
        "horn": get_part_genes("horn"),
        "tail": get_part_genes("tail"),
        "ears": get_part_genes("ears"),
        "eyes": get_part_genes("eyes")
    })


def get_genes_by_id(axie_id, force_request=False):
    """
    :param force_request:
    :param axie_id: axie id as int or string
    :return:
    """
    try:
        r = part_names_from_genes(hex_to_256bin_str(genes_from_axie_details(get_axie_details(
            axie_id, force_request=force_request))))
    except Exception as e:
        from traceback import print_exc
        print_exc()
        print(e)
        exit(-1)
    return r


def get_axie_price(axie_id, force_request=False):
    auction = get_axie_details(axie_id, force_request=force_request)["data"]["axie"]["auction"]
    if auction is not None:
        return int(auction["currentPrice"]) / 10**18
    else:
        print("Axie with id", axie_id, "is not for sale anymore!")
        return None


if __name__ == "__main__":

    while 0:

        print("\nGetting axie sales @", str(datetime.now()))
        print("Len last sales:", len(get_last_sales().keys()))
        with open("axie_gene_cache.json", "r") as _cache_file:
            _cache = json.load(_cache_file)
            print("Len gene cache:", len(_cache.keys()))

        sleep(20)

    termis = get_current_listings({
        "classes": ["dusk"],
        "parts": {
            "back": ["snail shell"],
            "mouth": ["tiny turtle"],
            "tail": ["thorny caterpillar"],
            "horn": ["lagging"]
        },
        #"genes": {"printer": 1},
        "breedCount": 4,
    })

    pp([(res["id"], res["auction"]["currentPriceUSD"]) for res in termis])
