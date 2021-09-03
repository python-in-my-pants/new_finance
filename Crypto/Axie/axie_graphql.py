import requests
import json
from pprint import pprint as pp
from DDict import DDict


def print_axie_db():

    with open("axie_gene_cache.json", "r") as cache_file:
        pp(json.load(cache_file))


def get_axie_details(axie_id, force_request=False):

    try:
        with open("axie_gene_cache.json", "r") as cache_file:

            cache = json.load(cache_file)

            if not force_request and str(axie_id) in cache.keys():
                return cache[str(axie_id)]

    except json.decoder.JSONDecodeError:
        cache = dict()
    
    headers = {'Content-type': 'application/json', 'Accept': 'text/plain'}
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

    resp = requests.post('https://axieinfinity.com/graphql-server-v2/graphql',
                         headers=headers, json=request_body)

    cache[str(axie_id)] = resp.json()

    with open("axie_gene_cache.json", "w") as cache_file:
        json.dump(cache, cache_file)

    print("Requesting axie data returned code:", resp.status_code)

    return resp.json()


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
        return part_gene_dict[bin_to_elem[class_gene].lower()][part_type][part_gene]["global"]

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
