from typing import Tuple
import pickle
import pandas as pd
from copy import copy
from Axie.Axie_models import Card, flatten

debuffs = ["aroma", "stench", "attack down", "morale down", "speed down",
           "fear", "chill", "stun", "poison", "jinx", "fragile", "lethal", "sleep"]

# TODO check if on_attack calls should not acutally be on_dmg_inflicted calls !!!

# TODO quantify difference between 2 tuples of tags

card_dataframe = pd.read_json("axie_card_data_w_tags.json")


class TagGroup:
    
    def __init__(self, members, parent=None):
        self.parent = parent
        self.members = members
        
        
tag_groups = (
    {"bird", "aqua", "dawn"},
    {"bird", "aqua", "dawn"},
    {"plant", "reptile", "dusk"},
    {"sleep", "fragile", "stun"},
    {"lethal", "crit"},
    {"stun", "fear", "disable"},
    {"backdoor", "stench"},
    {"chill", "last stand"},
    {"energy gain", "energy"},
    {"energy", "energy control"},
    {"speed+", "speed-"},
    {"increase shield", "heal", "on:defense", "on:shield break", "reduce damage", "attack-"},
    {"attack+", "increase damage"}
)


def set_tags_in_df():

    effects = [row["Effect"].lower() for _, row in card_dataframe.iterrows()]
    tags = [list() for _ in range(len(effects))]
    elems = ("beast", "bird", "bug", "plant", "aqua", "reptile", "dusk", "dawn", "mech")

    for i, eff in enumerate(effects):

        if "apply" in eff:
            if "+" in eff:
                tags[i].append("buff")
                if "speed" in eff:
                    tags[i].append("speed+")
                if "attack" in eff:
                    tags[i].append("attack+")
                if "morale" in eff:
                    tags[i].append("morale+")
            elif "-" in eff:
                if "speed" in eff:
                    tags[i].append("speed-")
                if "attack" in eff:
                    tags[i].append("attack-")
                if "morale" in eff:
                    tags[i].append("morale-")
            else:
                tags[i].append("debuff")

        if "energy" in eff:
            tags[i].append("energy")

        if "steal" in eff or "destroy" in eff:
            tags[i].append("energy control")

        if "energy" in eff and ("gain" in eff or "generate" in eff):
            tags[i].append("energy gain")

        if ("remove" in eff and "buff" in eff) or ("transfer" in eff and "debuff" in eff):
            tags[i].append("cleanse")

        if "heal" in eff or "restore" in eff:
            tags[i].append("heal")

        for debuff in debuffs:
            if debuff in eff:
                tags[i].append(debuff)

        if "attacker" in eff or "attacked" in eff or "struck" in eff or "hit" in eff:
            tags[i].append("on:defense")

        if "combo" in eff:
            tags[i].append("combo")

        if "chain" in eff:
            tags[i].append("chain")

        if "draw" in eff:
            tags[i].append("draw")
        if "target injured" in eff or \
            "target a" in eff or \
            "prioritize" in eff or \
            "skip" in eff or \
            "stench" in eff or \
            ("target" in eff and "enem" in eff):
            tags[i].append("backdoor")

        for elem in elems:
            if elem in eff:
                tags[i].append(elem)

        if "last stand" in eff:
            tags[i].append("last stand")

        if "deal" in eff and "damage" in eff and "%" in eff:
            tags[i].append("increase damage")

        if "shield breaks" in eff:
            tags[i].append("on:shield break")

        if ("add " in eff or "recover" in eff) and "shield" in eff:
            tags[i].append("increase shield")

        if "hp is below" in eff:
            tags[i].append("on:low hp")

        if "crit" in eff or "morale+" in eff:
            tags[i].append("crit")

        if "disable" in eff:
            tags[i].append("disable")

        if "twice" in eff or ("strike" in eff and "times" in eff):
            tags[i].append("multi attack")

        if "cannot do additional damage" in eff or "reduce damage" in eff:
            tags[i].append("reduce damage")

        if "attacks first" in eff or "attacks at the beginning" in eff:
            tags[i].append("on:first attack")

    with open("tags.txt", "w") as file:
        tag_list = sorted(list(set(flatten(tags))))
        for line in tag_list:
            file.write(f'{line}\n')
        file.close()

    for i, _ in card_dataframe.iterrows():
        card_dataframe.at[i, "Tags"] = tags[i]

    for _, row in card_dataframe.iterrows():
        print(row["Card name"], row["Effect"], row["Tags"])

    card_dataframe.to_json("axie_card_data_w_tags.json")


def get_cards_with_tags(tags, selection_type="and"):

    tags = set(tags)
    to_ret = list()

    for name in card_dataframe["Card name"].values:

        if selection_type == "and":
            # all occur
            if not tags - set(card_dataframe[card_dataframe["Card name"] == name]["Tags"].values[0]):
                to_ret.append(name)

        if selection_type == "or":
            if set(card_dataframe[card_dataframe["Card name"] == name]["Tags"].values[0]).intersection(tags):
                to_ret.append(name)

    return card_dataframe[card_dataframe["Card name"].isin(to_ret)]


def get_card_tags(card_name):
    return card_dataframe[card_dataframe["Card name"] == card_name]["Tags"].values[0]


def card_set_difference(card_set_1, card_set_2):
    """
    account for: dmg type, cost, shield, effects

    :param card_set_1:
    :param card_set_2:
    :return:
    """
    set1_data = get_card_set_data(card_set_1)
    set2_data = get_card_set_data(card_set_2)

    plant_dmg_differece = (set2_data["dmg"]["plant"] / set1_data["dmg"]["plant"]) - 1
    beast_dmg_differece = (set2_data["dmg"]["beast"] / set1_data["dmg"]["beast"]) - 1
    aqua_dmg_differece = (set2_data["dmg"]["aqua"] / set1_data["dmg"]["aqua"]) - 1

    shield_differece = (set2_data["shield"] / set1_data["shield"]) - 1

    tag_differecne = get_tag_difference(set1_data["tags"], set2_data["tags"])

    return {
        "plant": plant_dmg_differece,
        "beast": beast_dmg_differece,
        "aqua": aqua_dmg_differece,
        "shield": shield_differece,
        "tag": tag_differecne,
        "difference": (abs(plant_dmg_differece)+abs(beast_dmg_differece)+abs(aqua_dmg_differece)+
                       abs(shield_differece)+abs(tag_differecne)) / 5
    }


def get_tag_difference(t1, t2, not_found_penality=2):

    """
    quantify the difference between 2 lists of tags using tag groups as distance (same tag = 0, same group = 1, else 2)

    this is just a very crude implementation, the problem is actually not that trivial

    :param t1:
    :param t2:
    :return: value between 0 and 1 where 1 means no match from t1 in t2 found and 0 means an exact match plus potential
        additional elements in t2
    """

    def tag_groups_with_tag(t):
        """
        list of tag groups including the given tag with given tag removed respectively
        :param t:
        :return:
        """
        return [g-{t} for g in tag_groups if t in g]

    t1_copy = copy(t1)
    t2_copy = copy(t2)
    distance = 0

    # exact matches
    for tag1 in t1:
        if tag1 in t2_copy:
            t2_copy.remove(tag1)
            t1_copy.remove(tag1)

    # same tag group
    for tag1 in t1_copy:
        group = tag_groups_with_tag(tag1)
        matched = False
        for elem in group:
            if elem in t2_copy:
                matched = True
                distance += 1
                t2_copy.remove(elem)
                break
        if not matched:
            distance += not_found_penality

    return distance / (not_found_penality * len(t1))


def get_card_set_data(cards):
    """

    :param cards: damage per class type, cumulative shield, tags
    :return:
    """
    group1 = ("reptile", "plant", "dusk")
    group2 = ("aqua", "bird", "dawn")
    group3 = ("mech", "beast", "bug")

    g1, g2, g3 = 0, 0, 0
    for card in cards:
        if card.element in group1:
            g1 += card.attack
        if card.element in group2:
            g2 += card.attack
        if card.element in group3:
            g3 += card.attack

    return {
        "dmg": {
            "plant": g1, "aqua": g2, "beast": g3
        },
        "shield": sum([card.defense for card in cards]),
        "tags": [get_card_tags(c.card_name) for c in cards]
    }


def part_name_to_card_name(part_name, part):

    new_name_dict = {
        "Maki": "Fish Snack",
        "Origami": "Cupid",
        "Kawaii": "Cute Bunny",
        "Dango": "Tiny Turtle",
        "Hamaya": "Risky Beast",
        "Hasagi": "Ronin",
        "Yakitori": "Shiitake",
        "Geisha": "Piranha",
        "Kendama": "Imp",
        "Umaibo": "Pocky",
        "Yorishiro": "Beech",
        "Omatsuri": "Granma's Fan",
        "Koinobori": "Koi"
    }

    if new_name_dict.get(part_name, None):
        part_name = new_name_dict[part_name]

    row = card_dataframe.loc[(card_dataframe["Part name"] == part_name) &
                             (card_dataframe["Part"] == part.title())]
    try:
        return row["Card name"].iloc[0]
    except IndexError as e:
        from traceback import print_exc
        print(e)
        print_exc()
        exit()


def ctitle(s):
    return ' '.join([part.capitalize() for part in s.split()])


def attributes_from_df(card_name):

    try:
        row = card_dataframe.loc[card_dataframe["Card name"] == ctitle(card_name)]
        values = [row["Card name"], row["Part"], row["Part name"], row["Type"], row["Cost"], row["Range"], row["Attack"], row["Defense"], row["Effect"]]
        return [x.iloc[0] for x in values]
    except Exception as e:
        from traceback import print_exc
        print(e, card_name)
        print_exc()
        exit(-1)


def instantiate(s):
    return exec(f'{s}()')


# <editor-fold desc="Aqua">

class Shelter(Card):

    def __init__(self):
        # card_name, body_part, part_name, element, cost, r, attack, defense, effect
        super().__init__(*attributes_from_df("Shelter"))

    def pre_crit(self, match, cards_played_this_turn, attacker, defender) -> Tuple:
        return 0, True, 0


class ScaleDart(Card):

    def __init__(self):
        # card_name, body_part, part_name, element, cost, r, attack, defense, effect
        super().__init__(*attributes_from_df("Scale dart"))

    def on_attack(self, match, cards_played_this_turn, attacker, defender, atk_card) -> float:
        if defender.last_stand:
            self.owner.player.draw_cards(1)
        return 0


class SwiftEscape(Card):

    def __init__(self):
        # card_name, body_part, part_name, element, cost, r, attack, defense, effect
        super().__init__(*attributes_from_df("Swift escape"))

    def on_defense(self, match, cards_played_this_turn, attacker, defender, atk_card) -> float:
        if defender is self.owner:
            self.owner.apply_stat_eff("speed up", (3, 1))
        return 0


class Shipwreck(Card):

    def __init__(self):
        # card_name, body_part, part_name, element, cost, r, attack, defense, effect
        super().__init__(*attributes_from_df("Shipwreck"))

    def on_shield_break(self, match, cards_played_this_turn, attacker, defender):
        if defender is self.owner:
            self.owner.apply_stat_eff("attack up", 1)
        return False


class AquaVitality(Card):

    def __init__(self):
        # card_name, body_part, part_name, element, cost, r, attack, defense, effect
        super().__init__(*attributes_from_df("Aqua vitality"))

    def on_damage_inflicted(self, match, cards_played_this_turn, attacker, defender, amount, atk_card):
        if atk_card is self:
            anemone_parts = len([part for part in self.owner.cards if "Anemone" in part.part_name])
            self.owner.change_hp(anemone_parts * 50)


class Aquaponics(Card):

    def __init__(self):
        # card_name, body_part, part_name, element, cost, r, attack, defense, effect
        super().__init__(*attributes_from_df("Aquaponics"))

    def on_damage_inflicted(self, match, cards_played_this_turn, attacker, defender, amount, atk_card):
        if atk_card is self:
            anemone_parts = len([part for part in self.owner.cards if "Anemone" in part.part_name])
            self.owner.change_hp(anemone_parts * 50)


class SpinalTap(Card):

    def __init__(self):
        # card_name, body_part, part_name, element, cost, r, attack, defense, effect
        super().__init__(*attributes_from_df("Spinal tap"))

    def on_target_select(self, match, cards_played_this_turn, attacker, defender):
        if attacker is self.owner and self is cards_played_this_turn[self.owner][0]:
            if len(cards_played_this_turn[self.owner]) >= 3:
                to_focus = [axie for axie in defender.player.team if axie.alive() and not cards_played_this_turn[axie]]
                return set(), to_focus
        return set(), list()


class ShellJab(Card):

    def __init__(self):
        # card_name, body_part, part_name, element, cost, r, attack, defense, effect
        super().__init__(*attributes_from_df("Shell jab"))

    def on_attack(self, match, cards_played_this_turn, attacker, defender, atk_card) -> float:
        if not cards_played_this_turn.get(defender, []):
            return 0.3
        return 0


class DeepSeaGore(Card):

    def __init__(self):
        # card_name, body_part, part_name, element, cost, r, attack, defense, effect
        super().__init__(*attributes_from_df("Deep sea gore"))

    def on_attack(self, match, cards_played_this_turn, attacker, defender, atk_card) -> float:
        self.owner.shield += self.defense * 0.3
        return 0


class ClamSlash(Card):

    def __init__(self):
        # card_name, body_part, part_name, element, cost, r, attack, defense, effect
        super().__init__(*attributes_from_df("Clam slash"))

    def on_attack(self, match, cards_played_this_turn, attacker, defender, atk_card) -> float:
        if defender.element in ("beat", "bug", "mech"):
            self.owner.apply_stat_eff("attack up", 1)
        return 0


class HerosBane(Card):

    def __init__(self):
        # card_name, body_part, part_name, element, cost, r, attack, defense, effect
        super().__init__(*attributes_from_df("Hero's Bane"))

    def on_last_stand(self, match, cards_played_this_turn, attacker, defender):
        return self.owner is attacker


class StarShuriken(Card):

    def __init__(self):
        # card_name, body_part, part_name, element, cost, r, attack, defense, effect
        super().__init__(*attributes_from_df("Star shuriken"))

    def on_pre_last_stand(self, match, cards_played_this_turn, attacker, defender):
        return self.owner is attacker


class AngryLam(Card):

    def __init__(self):
        # card_name, body_part, part_name, element, cost, r, attack, defense, effect
        super().__init__(*attributes_from_df("Angry lam"))

    def on_attack(self, match, cards_played_this_turn, attacker, defender, atk_card) -> float:
        if self.owner.hp <= self.owner.base_hp * 0.5:
            return 0.5
        return 0


class Swallow(Card):

    def __init__(self):
        # card_name, body_part, part_name, element, cost, r, attack, defense, effect
        super().__init__(*attributes_from_df("Swallow"))

    def on_damage_inflicted(self, match, cards_played_this_turn, attacker, defender, amount, atk_card):
        if atk_card is self:
            self.owner.change_hp(amount)


class FishHook(Card):

    def __init__(self):
        # card_name, body_part, part_name, element, cost, r, attack, defense, effect
        super().__init__(*attributes_from_df("Fish hook"))

    def on_attack(self, match, cards_played_this_turn, attacker, defender, atk_card) -> float:
        if defender.element in ("plant", "reptile", "dusk"):
            self.owner.apply_stat_eff("attack up", 1)
        return 0


class CrimsonWater(Card):

    def __init__(self):
        # card_name, body_part, part_name, element, cost, r, attack, defense, effect
        super().__init__(*attributes_from_df("Crimson water"))

    def on_target_select(self, match, cards_played_this_turn, attacker, defender):
        if self.owner is attacker and self is cards_played_this_turn[self.owner][0] and self.owner.hp <= self.owner.base_hp * 0.5:
            to_focus = [axie for axie in defender.player.team if axie.alive() and axie.hp < axie.base_hp]
            return set(), to_focus
        return set(), list()


class UpstreamSwim(Card):

    def __init__(self):
        # card_name, body_part, part_name, element, cost, r, attack, defense, effect
        super().__init__(*attributes_from_df("Upstream swim"))

    def on_combo(self, match, cards_played_this_turn, combo_cards):
        if len([card for card in combo_cards if card.element == "aqua"]) >= 2:
            self.owner.apply_stat_eff("speed up", (3, 1))


class TailSlap(Card):

    def __init__(self):
        # card_name, body_part, part_name, element, cost, r, attack, defense, effect
        super().__init__(*attributes_from_df("Tail slap"))

    def on_damage_inflicted(self, match, cards_played_this_turn, attacker, defender, amount, atk_card):
        # TODO this is still executed twice sometimes
        if atk_card is self and len(cards_played_this_turn[attacker]) >= 2:
            self.owner.player.gain_energy(1)


class BlackBubble(Card):

    def __init__(self):
        # card_name, body_part, part_name, element, cost, r, attack, defense, effect
        super().__init__(*attributes_from_df("Black bubble"))

    def on_damage_inflicted(self, match, cards_played_this_turn, attacker, defender, amount, atk_card):
        if atk_card is self:
            defender.apply_stat_eff("jinx", 2)
        return 0


class WaterSphere(Card):

    def __init__(self):
        super().__init__(*attributes_from_df("Water sphere"))

    def on_damage_inflicted(self, match, cards_played_this_turn, attacker, defender, amount, atk_card):
        if atk_card is self:
            defender.apply_stat_eff("chill", 2)
        return 0


class FlankingSmack(Card):

    def __init__(self):
        super().__init__(*attributes_from_df("Flanking smack"))

    def on_attack(self, match, cards_played_this_turn, attacker, defender, atk_card) -> float:
        if self is match.axies_in_attack_order[0]:
            return 0.2
        return 0


class ChitinJump(Card):

    def __init__(self):
        super().__init__(*attributes_from_df("Chitin jump"))

    def on_target_select(self, match, cards_played_this_turn, attacker, defender):
        if self.owner is attacker and self is cards_played_this_turn[self.owner][0]:
            return set(), [axie for axie in defender.player.team[::-1] if axie.alive()]
        return set(), list()

# </editor-fold>

# <editor-fold desc="Beasts">


class SingleCombat(Card):

    def __init__(self):
        super().__init__(*attributes_from_df("Single combat"))

    def pre_crit(self, match, cards_played_this_turn, attacker, defender) -> Tuple:
        if self.owner is attacker and len(cards_played_this_turn[attacker]) >= 3:
            return 1, False, 1
        return 1, False, 0


class HeroicReward(Card):

    def __init__(self):
        super().__init__(*attributes_from_df("Heroic reward"))

    def on_attack(self, match, cards_played_this_turn, attacker, defender, atk_card) -> float:
        if defender.element in ("aqua", "bird", "dawn"):
            self.owner.player.draw_cards(1)
        return 0


class NitroLeap(Card):

    def __init__(self):
        super().__init__(*attributes_from_df("Nitro leap"))

    def on_determine_order(self):
        if self.owner.last_stand:
            return True
        return False


class RevengeArrow(Card):

    def __init__(self):
        super().__init__("Revenge arrow", "back", "Risky beast", "beast", 1, "ranged", 125, 25,
                         "Deal 150% damage if this Axie is in Last Stand.")

    def on_attack(self, match, cards_played_this_turn, attacker, defender, atk_card) -> float:
        if self.owner.last_stand:
            return 0.5
        return 0


class WoodmanPower(Card):

    def __init__(self):
        super().__init__(*attributes_from_df("Woodman power"))

    def on_damage_inflicted(self, match, cards_played_this_turn, attacker, defender, amount, atk_card):
        if atk_card is self:
            if defender.element == "plant":
                self.owner.shield += amount


class JugglingBalls(Card):

    def __init__(self):
        super().__init__(*attributes_from_df("Juggling balls"))

    def attack_times(self, match, cards_played_this_turn, attacker, defender):
        return 3


class BranchCharge(Card):

    def __init__(self):
        super().__init__(*attributes_from_df("Branch charge"))

    def pre_crit(self, match, cards_played_this_turn, attacker, defender) -> Tuple:
        if self.owner is attacker:
            if "plant" in [card.element for card in flatten(cards_played_this_turn.values())]:
                return 1, False, 0.2
        return 1, False, 0


class IvoryStab(Card):

    def __init__(self):
        super().__init__(*attributes_from_df("Ivory stab"))

    def on_crit(self, match, cards_played_this_turn, attacker, defender):
        if self.owner.player is attacker.player:
            self.owner.player.gain_energy(1)


class MerryLegion(Card):

    def __init__(self):
        super().__init__(*attributes_from_df("Merry legion"))

    def on_start_turn(self, match, cards_played_this_turn):
        if set(flatten(cards_played_this_turn.values())) - set(cards_played_this_turn[self.owner]):
            self.owner.shield += int(self.defense * 0.2)


class SugarRush(Card):

    def __init__(self):
        super().__init__(*attributes_from_df("Sugar rush"))

    def on_attack(self, match, cards_played_this_turn, attacker, defender, atk_card) -> float:
        return 0.1 * len([axie for axie in self.owner.player.team if axie.element == "bug" and axie.alive()])


class SinisterStrike(Card):

    def __init__(self):
        super().__init__(*attributes_from_df("Sinister strike"))

    def pre_crit(self, match, cards_played_this_turn, attacker, defender) -> Tuple:
        if self.owner is attacker:
            return 2.5, False, 0
        return 1, False, 0


class Acrobatic(Card):

    def __init__(self):
        super().__init__(*attributes_from_df("Acrobatic"))

    def on_attack(self, match, cards_played_this_turn, attacker, defender, card) -> float:
        self.owner.apply_stat_eff("speed up", (3, 1))
        return 0


class NutCrack(Card):

    def __init__(self):
        super().__init__(*attributes_from_df("Nut crack"))

    def on_attack(self, match, cards_played_this_turn, attacker, defender, atk_card) -> float:
        if len([card for card in cards_played_this_turn[self.owner] if card.part_name == "Nut cracker"]):
            return 0.2
        return 0


class PiercingSound(Card):

    def __init__(self):
        super().__init__(*attributes_from_df("Piercing sound"))

    def on_damage_inflicted(self, match, cards_played_this_turn, attacker, defender, amount, atk_card):
        if atk_card is self:
            defender.player.gain_energy(-1)
        return 0


class DeathMark(Card):

    def __init__(self):
        super().__init__(*attributes_from_df("Death mark"))

    def on_damage_inflicted(self, match, cards_played_this_turn, attacker, defender, amount, atk_card):
        if atk_card is self:
            if self.owner.hp < self.owner.base_hp * 0.3:
                defender.apply_stat_eff("lethal", True)


class SelfRally(Card):

    def __init__(self):
        super().__init__(*attributes_from_df("Self rally"))

    def on_play(self, match, cards_played_this_turn):
        self.owner.apply_stat_eff("morale up", (3, 2))


class LunaAbsorb(Card):

    def __init__(self):
        super().__init__(*attributes_from_df("Luna absorb"))

    def on_play(self, match, cards_played_this_turn):
        self.owner.player.gain_energy(1)


class NightSteal(Card):

    def __init__(self):
        super().__init__(*attributes_from_df("Night steal"))

    def on_damage_inflicted(self, match, cards_played_this_turn, attacker, defender, amount, atk_card):
        if atk_card is self and len(cards_played_this_turn[self.owner]) > 1:
            if defender.player.energy >= 1:
                defender.player.gain_energy(-1)
                self.owner.player.gain_energy(1)
        return 0


class RampantHowl(Card):

    def __init__(self):
        super().__init__(*attributes_from_df("Rampant howl"))

    def on_attack(self, match, cards_played_this_turn, attacker, defender, atk_card) -> float:
        if self.owner is attacker and self.owner.last_stand:
            for teammember in self.owner.player.team:
                if teammember.alive():
                    teammember.apply_stat_eff("morale up", (3, 1))
        return 0


class HareDagger(Card):

    def __init__(self):
        super().__init__(*attributes_from_df("Hare dagger"))

    def on_attack(self, match, cards_played_this_turn, attacker, defender, atk_card) -> float:
        if self.owner is attacker:
            if self is list(cards_played_this_turn.keys())[0]:
                self.owner.player.draw_cards(1)
        return 0


class NutThrow(Card):

    def __init__(self):
        super().__init__(*attributes_from_df("Nut throw"))

    def on_attack(self, match, cards_played_this_turn, attacker, defender, atk_card) -> float:  # TODO not working
        if len([card for card in cards_played_this_turn[self.owner] if card.part_name == "Nut cracker"]) >= 2:
            return 0.2
        return 0


class GerbilJump(Card):

    def __init__(self):
        super().__init__(*attributes_from_df("Gerbil jump"))

    def on_target_select(self, match, cards_played_this_turn, attacker, defender):
        if self.owner is attacker and self is cards_played_this_turn[self.owner][0]:
            tmp = [axie for axie in defender.player.team if axie.alive()]
            if len(tmp) >= 2:
                return set(), [tmp[1]]
        return set(), list()


# </editor-fold>

# <editor-fold desc="Birds">

class BalloonPop(Card):

    def __init__(self):
        super().__init__(*attributes_from_df("Balloon pop"))

    def on_attack(self, match, cards_played_this_turn, attacker, defender, atk_card) -> float:
        defender.apply_stat_eff("fear", 2)
        return 0

    def on_defense(self, match, cards_played_this_turn, attacker, defender, card) -> float:
        if self.owner is defender:
            self.owner.apply_stat_eff("fear", 1)  # TODO check this works as intended
        return 0


class HeartBreak(Card):

    def __init__(self):
        super().__init__(*attributes_from_df("Heart break"))

    def on_attack(self, match, cards_played_this_turn, attacker, defender, atk_card) -> float:
        if self.owner is attacker:
            defender.apply_stat_eff("morale down", (2, 2))
        return 0


class IllOmened(Card):

    def __init__(self):
        super().__init__(*attributes_from_df("Ill-omened"))

    def on_attack(self, match, cards_played_this_turn, attacker, defender, atk_card) -> float:
        defender.apply_stat_eff("jinx", 2)
        return 0


class PatientHunter(Card):

    def __init__(self):
        super().__init__(*attributes_from_df("Patient hunter"))

    def on_target_select(self, match, cards_played_this_turn, attacker, defender):
        if self is cards_played_this_turn[self.owner][0] and self.owner is attacker and self.owner.hp < self.owner.base_hp * 0.5:
            return set(), [axie for axie in defender.player.team if axie.alive() and axie.element == "aqua"]
        return set(), list()


class TripleThreat(Card):

    def __init__(self):
        super().__init__(*attributes_from_df("Triple threat"))

    def attack_times(self, match, cards_played_this_turn, attacker, defender):
        if self.owner is attacker and self.owner.buffs.is_debuffed():
            return 2
        return 1


class Cockadoodledoo(Card):

    def __init__(self):
        super().__init__(*attributes_from_df("Cockadoodledoo"))

    def on_attack(self, match, cards_played_this_turn, attacker, defender, atk_card) -> float:
        if self.owner is attacker:
            self.owner.apply_stat_eff("attack up", 1)
        return 0


class AirForceOne(Card):

    def __init__(self):
        super().__init__(*attributes_from_df("Air force one"))

    def on_attack(self, match, cards_played_this_turn, attacker, defender, atk_card) -> float:
        if "Trump" in [card.part_name for card in match.chain_cards]:
            return 0.2
        return 0


class Headshot(Card):

    def __init__(self):
        super().__init__(*attributes_from_df("Headshot"))

    def on_damage_inflicted(self, match, cards_played_this_turn, attacker, defender, amount, atk_card):
        if atk_card is self:
            defender.disable("horn", 1)


class SmartShot(Card):

    def __init__(self):
        super().__init__(*attributes_from_df("Smart shot"))

    def on_target_select(self, match, cards_played_this_turn, attacker, defender):
        if self is cards_played_this_turn[self.owner][0] and self.owner is attacker:
            living_opp = [axie for axie in defender.player.team if axie.alive()]
            if len(living_opp) >= 2:
                return {living_opp[0]}, list()
        return set(), list()


class FeatherLunge(Card):

    def __init__(self):
        super().__init__(*attributes_from_df("Feather lunge"))

    def on_attack(self, match, cards_played_this_turn, attacker, defender, atk_card) -> float:
        if self.owner is attacker:
            chain_cards = []
            for xe in self.owner.player.team:
                if xe.alive() and xe is not self:
                    other_cards = cards_played_this_turn.get(xe, None)
                    if other_cards and self.element in [c.element for c in other_cards]:
                        chain_cards += cards_played_this_turn[xe]

            if "Lunge" in [card.part_name for card in chain_cards]:
                return 0.2
        return 0


class SoothingSong(Card):

    def __init__(self):
        super().__init__(*attributes_from_df("Soothing song"))

    def on_attack(self, match, cards_played_this_turn, attacker, defender, atk_card) -> float:
        # this really applies sleep before the attack
        if self.owner is attacker:
            defender.apply_stat_eff("sleep", 1)
        return 0


class PeaceTreaty(Card):

    def __init__(self):
        super().__init__(*attributes_from_df("Peace treaty"))

    def on_attack(self, match, cards_played_this_turn, attacker, defender, atk_card) -> float:
        if self.owner is attacker:
            defender.apply_stat_eff("attack down", 1)
        return 0


class Insectivore(Card):

    def __init__(self):
        super().__init__(*attributes_from_df("Insectivore"))

    def on_target_select(self, match, cards_played_this_turn, attacker, defender):
        if self is cards_played_this_turn[self.owner][0] and self.owner is attacker:
            bugs = [axie for axie in defender.player.team if axie.element == "bug" and axie.alive()]
            if self.owner.hp <= self.owner.base_hp * 0.5 and bugs:
                return set(), [bugs[0]]
        return set(), list()


class EarlyBird(Card):

    def __init__(self):
        super().__init__(*attributes_from_df("Early bird"))

    def on_attack(self, match, cards_played_this_turn, attacker, defender, atk_card) -> float:
        if self.owner is attacker:
            if self is match.axies_in_attack_order[0]:
                return 0.2
        return 0


class SunderArmor(Card):

    def __init__(self):
        super().__init__(*attributes_from_df("Sunder armor"))

    def on_start_turn(self, match, cards_played_this_turn):
        self.owner.shield += self.defense * 0.2 * self.owner.buffs.count_debuffs()


class PuffySmack(Card):

    def __init__(self):
        super().__init__(*attributes_from_df("Puffy smack"))

    def on_target_select(self, match, cards_played_this_turn, attacker, defender):
        if self is cards_played_this_turn[self.owner][0] and self.owner is attacker:
            return {axie for axie in defender.player.team if axie.alive() and axie.last_stand}, list()
        return set(), list()


class CoolBreeze(Card):

    def __init__(self):
        super().__init__(*attributes_from_df("Cool breeze"))

    def on_attack(self, match, cards_played_this_turn, attacker, defender, atk_card) -> float:
        if self.owner is attacker:
            defender.apply_stat_eff("chill", 2)
        return 0


class AllOutShot(Card):

    def __init__(self):
        super().__init__(*attributes_from_df("All-out shot"))

    def on_damage_inflicted(self, match, cards_played_this_turn, attacker, defender, amount, atk_card):
        if atk_card is self and not self.triggered:
            self.owner.change_hp(-int(0.3*self.owner.base_hp))


class Eggbomb(Card):

    def __init__(self):
        # card_name, body_part, part_name, element, cost, r, attack, defense, effect
        super().__init__("Eggbomb", "horn", "Eggshell", "bird", 1, "ranged", 120, 0,
                         "Apply aroma on this axie until next round")

    def on_attack(self, match, cards_played_this_turn, attacker, defender, atk_card) -> float:
        self.owner.apply_stat_eff("aroma", 1)  # should disappear at start of next turn
        return 0


class DarkSwoop(Card):

    def __init__(self):
        super().__init__("Little owl", "mouth", "Dark swoop", "bird", 1, "melee", 25, 0,
                         "Target fastest enemy")

    def on_target_select(self, match, cards_played_this_turn, attacker, defender):
        """
         key=lambda axie: final_speeds[axie], reverse=True)[0]]
        IndexError: list index out of range
        :param match:
        :param cards_played_this_turn:
        :param attacker:
        :param defender:
        :return:
        """
        if self is cards_played_this_turn[self.owner][0] and self.owner is attacker:
            final_speeds = {axie: 0 for axie in attacker.player.team+defender.player.team}

            for axie, speed in [(axie, axie.speed) for axie in cards_played_this_turn.keys()]:
                s, m = axie.buffs.on_target_select()
                speed_mult = 1 + s

                final_speeds[axie] = (speed * speed_mult, axie.hp, axie.morale * (1 + m))

            # tODO only choose from opponents team

            return set(),\
                   [sorted([a for a in defender.player.team if a.alive()],
                           key=lambda axie: final_speeds[axie], reverse=True)[0]]
        return set(), list()


class Blackmail(Card):

    def __init__(self):
        super().__init__("Pigeon post", "back", "Blackmail", "bird", 1, "ranged", 120, 10,
                         "Transfer all debuffs on this axie to target")

    def on_attack(self, match, cards_played_this_turn, attacker, defender, atk_card) -> float:  # TODO

        self.owner.buffs.transfer(defender.buffs)

        return 0


class RiskyFeather(Card):

    def __init__(self):
        super().__init__("The last one", "tail", "Risky feather", "bird", 1, "ranged", 140, 10,
                         "Apply 2 Attack- to this axie")

    def on_attack(self, match, cards_played_this_turn, attacker, defender, atk_card) -> float:
        self.owner.apply_stat_eff("attack down", 2)  # goes away on attack
        return 0

# </editor-fold>

# <editor-fold desc="Bugs">


class StickyGoo(Card):

    def __init__(self):
        super().__init__(*attributes_from_df("Sticky goo"))

    def on_shield_break(self, match, cards_played_this_turn, attacker, defender):
        if self.owner is defender and not self.triggered:
            attacker.apply_stat_eff("stun", True)
            self.triggered = True
        return False


class BarbStrike(Card):

    def __init__(self):
        super().__init__(*attributes_from_df("Barb strike"))

    def on_attack(self, match, cards_played_this_turn, attacker, defender, atk_card) -> float:
        if self.owner is attacker and len(match.chain_cards) > 1:
            defender.apply_stat_eff("poison", 1)
        return 0


class BugNoise(Card):

    def __init__(self):
        super().__init__(*attributes_from_df("Bug noise"))

    def on_attack(self, match, cards_played_this_turn, attacker, defender, atk_card) -> float:
        if self.owner is attacker:
            defender.apply_stat_eff("attack down", 1)
        return 0


class BugSplat(Card):

    def __init__(self):
        super().__init__(*attributes_from_df("Bug splat"))

    def on_attack(self, match, cards_played_this_turn, attacker, defender, atk_card) -> float:
        if self.owner is attacker and defender.element == "bug":
            return 0.5
        return 0


class ScarabCurse(Card):

    def __init__(self):
        super().__init__(*attributes_from_df("Scarab curse"))

    def on_attack(self, match, cards_played_this_turn, attacker, defender, atk_card) -> float:
        if self.owner is attacker:
            defender.disable("heal", 0, asap=True)
        return 0


class BuzzingWind(Card):

    def __init__(self):
        super().__init__(*attributes_from_df("Buzzing wind"))

    def on_attack(self, match, cards_played_this_turn, attacker, defender, atk_card) -> float:
        if self.owner is attacker:
            defender.apply_stat_eff("fragile", True)
        return 0


class MysticRush(Card):

    def __init__(self):
        super().__init__(*attributes_from_df("Mystic rush"))

    def on_attack(self, match, cards_played_this_turn, attacker, defender, atk_card) -> float:
        if self.owner is attacker:
            defender.apply_stat_eff("speed down", (2, 1))
        return 0


class BugSignal(Card):

    def __init__(self):
        super().__init__(*attributes_from_df("Bug signal"))

    def on_attack(self, match, cards_played_this_turn, attacker, defender, atk_card) -> float:
        if self.owner is attacker and "Bug Signal" in [card.card_name for card in cards_played_this_turn[self.owner]]:
            if defender.player.energy >= 1:
                defender.player.gain_energy(-1)
                self.owner.player.gain_energy(1)
        return 0


class GrubSurprise(Card):

    def __init__(self):
        super().__init__(*attributes_from_df("Grub surprise"))

    def on_attack(self, match, cards_played_this_turn, attacker, defender, atk_card) -> float:
        if self.owner is attacker:
            if defender.shield:
                defender.apply_stat_eff("fear", 1)   # TODO is that correct? does it only apply for current turn?
        return 0


class DullGrip(Card):

    def __init__(self):
        super().__init__(*attributes_from_df("Dull grip"))

    def on_attack(self, match, cards_played_this_turn, attacker, defender, atk_card) -> float:
        if self.owner is attacker:
            if defender.shield:
                return 0.3
        return 0


class ThirdGlance(Card):

    def __init__(self):
        super().__init__(*attributes_from_df("Third glance"))

    def on_attack(self, match, cards_played_this_turn, attacker, defender, atk_card) -> float:
        if self.owner is attacker:
            defender.player.random_discard()
        return 0


class Disguise(Card):

    def __init__(self):
        super().__init__(*attributes_from_df("Disguise"))

    def on_attack(self, match, cards_played_this_turn, attacker, defender, atk_card) -> float:
        if self.owner is attacker and len([card for card in cards_played_this_turn[self.owner] if card.element == "plant"]) > 0:
            self.owner.player.gain_energy(1)
        return 0


class BloodTaste(Card):

    def __init__(self):
        super().__init__(*attributes_from_df("Blood Taste"))

    def on_damage_inflicted(self, match, cards_played_this_turn, attacker, defender, amount, atk_card):
        if atk_card is self:
            self.owner.change_hp(amount)


class SunderClaw(Card):

    def __init__(self):
        super().__init__(*attributes_from_df("Sunder claw"))

    def on_attack(self, match, cards_played_this_turn, attacker, defender, atk_card) -> float:
        if self.owner is attacker:
            defender.player.random_discard()
        return 0


class TerrorChomp(Card):

    def __init__(self):
        super().__init__(*attributes_from_df("Terror chomp"))

    def on_attack(self, match, cards_played_this_turn, attacker, defender, atk_card) -> float:
        if self.owner is attacker:
            if len(match.chain_cards) > 0:
                defender.apply_stat_eff("fear", 3)
        return 0


class MiteBite(Card):

    def __init__(self):
        super().__init__(*attributes_from_df("Mite Bite"))

    def on_attack(self, match, cards_played_this_turn, attacker, defender, atk_card) -> float:
        if self.owner is attacker and len(cards_played_this_turn[self.owner]) > 1:
            return 1
        return 0


class ChemicalWarfare(Card):

    def __init__(self):
        super().__init__(*attributes_from_df("Chemical warfare"))

    def on_attack(self, match, cards_played_this_turn, attacker, defender, atk_card) -> float:
        if self.owner is attacker:
            defender.apply_stat_eff("stench", 1)
        return 0


class TwinNeedle(Card):

    def __init__(self):
        super().__init__(*attributes_from_df("Twin needle"))

    def attack_times(self, match, cards_played_this_turn, attacker, defender):
        if len(cards_played_this_turn[self.owner]) > 1:
            return 2
        return 1


class AnestheticBait(Card):

    def __init__(self):
        super().__init__(*attributes_from_df("Anesthetic bait"))

    def on_attack(self, match, cards_played_this_turn, attacker, defender, atk_card) -> float:
        if self.owner is defender:
            if attacker.element in ("aqua", "bird"):
                attacker.apply_stat_eff("stun", True)
        return 0


class NumbingLecretion(Card):

    def __init__(self):
        super().__init__(*attributes_from_df("Numbing lecretion"))

    def on_attack(self, match, cards_played_this_turn, attacker, defender, atk_card) -> float:
        if self.owner is attacker:
            defender.disable("melee", 1)
        return 0


class GrubExplode(Card):

    def __init__(self):
        super().__init__(*attributes_from_df("Grub explode"))

    def on_attack(self, match, cards_played_this_turn, attacker, defender, atk_card) -> float:
        if self.owner.last_stand:
            self.owner.last_stand = False
            return 1
        return 0


class AllergicReaction(Card):

    def __init__(self):
        super().__init__(*attributes_from_df("Allergic reaction"))

    def on_attack(self, match, cards_played_this_turn, attacker, defender, atk_card) -> float:
        if self.owner is attacker and defender.buffs.is_debuffed():
            return 0.3
        return 0

# </editor-fold>

# <editor-fold desc="Plants">


class TurnipRocket(Card):

    def __init__(self):
        super().__init__(*attributes_from_df("Turnip rocket"))

    def on_target_select(self, match, cards_played_this_turn, attacker, defender):
        if self is cards_played_this_turn[self.owner][0] and self.owner is attacker:
            if len(cards_played_this_turn[self.owner]) >= 3:
                return set(), [axie for axie in defender.player.team if axie.alive() and axie.element == "bird"]
        return set(), list()


class ShroomsGrace(Card):

    def __init__(self):
        super().__init__(*attributes_from_df("Shroom's grace"))

    def on_play(self, match, cards_played_this_turn):
        self.owner.change_hp(120)


class CleanseScent(Card):

    def __init__(self):
        super().__init__(*attributes_from_df("Cleanse scent"))

    def on_play(self, match, cards_played_this_turn):
        self.owner.buffs.remove_debuffs()


class AquaStock(Card):

    def __init__(self):
        super().__init__(*attributes_from_df("Aqua stock"))

    def on_damage_inflicted(self, match, cards_played_this_turn, attacker, defender, amount, atk_card):
        if self.owner is defender:
            if atk_card.element == "aqua":
                self.owner.player.gain_energy(1)


class Refresh(Card):

    def __init__(self):
        super().__init__(*attributes_from_df("Refresh"))

    def on_play(self, match, cards_played_this_turn):
        i = self.owner.player.team.index(self.owner)
        if i == 1:
            front_mate = self.owner.player.team[0]
        elif i == 2:
            front_mate = self.owner.player.team[1]
        else:
            front_mate = None

        if front_mate:
            if front_mate.alive() and not front_mate.last_stand:

                front_mate.buffs.remove_debuffs()


class OctoberTreat(Card):

    def __init__(self):
        super().__init__(*attributes_from_df("October treat"))

    def on_end_turn(self, match, cards_played_this_turn):
        if self.owner.shield > 0:
            self.owner.player.draw_cards(1)


class BambooClan(Card):

    def __init__(self):
        super().__init__(*attributes_from_df("Bamboo clan"))

    def on_attack(self, match, cards_played_this_turn, attacker, defender, atk_card) -> float:
        if self.owner is attacker:
            if sum(map(lambda c: c.element == "plant", match.chain_cards)) >= 1:
                return 0.2
        return 0


class WoodenStab(Card):

    def __init__(self):
        super().__init__(*attributes_from_df("Wooden stab"))

    def on_attack(self, match, cards_played_this_turn, attacker, defender, atk_card) -> float:
        if self.owner is attacker:
            if self.owner.shield_broke_this_turn:
                return 0.2
        return 0


class HealingAroma(Card):

    def __init__(self):
        super().__init__(*attributes_from_df("Healing aroma"))

    def on_play(self, match, cards_played_this_turn):
        self.owner.change_hp(120)


class SweetParty(Card):

    def __init__(self):
        super().__init__(*attributes_from_df("Sweet party"))

    def on_play(self, match, cards_played_this_turn):
        # if there are front teammates
        front_mates = [axie for axie in self.owner.player.team[:self.owner.player.team.index(self.owner)] if axie.alive()]
        if front_mates:
            front_mates[-1].change_hp(270)
        else:
            self.owner.change_hp(270)


class PricklyTrap(Card):

    def __init__(self):
        super().__init__(*attributes_from_df("Prickly trap"))

    def on_attack(self, match, cards_played_this_turn, attacker, defender, atk_card) -> float:
        if self.owner is match.axies_in_attack_order[-1]:
            return 0.2
        return 0


class DrainBite(Card):

    def __init__(self):
        super().__init__(*attributes_from_df("Drain bite"))

    def on_damage_inflicted(self, match, cards_played_this_turn, attacker, defender, amount, atk_card):
        if self is atk_card:
            self.owner.change_hp(amount)


class SeedBullet(Card):

    def __init__(self):
        super().__init__(*attributes_from_df("Seed bullet"))

    def on_target_select(self, match, cards_played_this_turn, attacker, defender):
        if self.owner is attacker and self is cards_played_this_turn[self.owner][0]:
            final_speeds = {axie: None for axie in defender.player.team if axie.alive()}

            for axie, speed in [(axie, axie.speed) for axie in final_speeds.keys()]:

                s, m = axie.buffs.on_target_select()
                speed_mult = 1 + s

                final_speeds[axie] = (speed * speed_mult, axie.hp, axie.morale * (1+m))

            return set(), [sorted(final_speeds.keys(), key=lambda axie: final_speeds[axie], reverse=True)[0]]
        return set(), list()


class VegetalBite(Card):

    def __init__(self):
        super().__init__(*attributes_from_df("Vegetal bite"))

    def on_damage_inflicted(self, match, cards_played_this_turn, attacker, defender, amount, atk_card):
        if atk_card is self:
            if len(cards_played_this_turn[self.owner]) > 1 and defender.player.energy > 0:
                defender.player.gain_energy(-1)
                self.owner.player.gain_energy(1)
            return 0


class VeganDiet(Card):

    def __init__(self):
        super().__init__(*attributes_from_df("Vegan diet"))

    def on_damage_inflicted(self, match, cards_played_this_turn, attacker, defender, amount, atk_card):
        if atk_card is self and defender.element == "plant":
            self.owner.change_hp(amount)


class ForestSpirit(Card):

    def __init__(self):
        super().__init__(*attributes_from_df("Forest spirit"))

    def on_play(self, match, cards_played_this_turn):
        front_temmates = [axie for axie in self.owner.player.team[:self.owner.player.team.index(self.owner)] if axie.alive()]
        if front_temmates:
            front_temmates[-1].change_hp(190)


class CarrotHammer(Card):

    def __init__(self):
        super().__init__(*attributes_from_df("Carrot hammer"))

    def on_shield_break(self, match, cards_played_this_turn, attacker, defender):
        if self.owner is defender:
            self.owner.player.gain_energy(1)
            self.triggered = True
        return False


class CattailSlap(Card):

    def __init__(self):
        super().__init__(*attributes_from_df("Cattail slap"))

    def on_damage_inflicted(self, match, cards_played_this_turn, attacker, defender, amount, atk_card):
        if self.owner is defender:
            if atk_card.element in ("beast", "bug", "mech"):
                self.owner.player.draw_cards(1)


class LeekLeak(Card):

    def __init__(self):
        super().__init__(*attributes_from_df("Leek leak"))

    def on_damage_inflicted(self, match, cards_played_this_turn, attacker, defender, amount, atk_card):
        if self.owner is defender:
            attacker.disable("ranged", 1)


class GasUnleash(Card):

    def __init__(self):
        super().__init__(*attributes_from_df("Gas unleash"))

    def on_damage_inflicted(self, match, cards_played_this_turn, attacker, defender, amount, atk_card):
        if atk_card is self:
            defender.apply_stat_eff("poison", 1)
        if self.owner is defender:
            attacker.apply_stat_eff("poison", 1)


class AquaDeflect(Card):

    def __init__(self):
        super().__init__(*attributes_from_df("Aqua deflect"))

    def on_target_select(self, match, cards_played_this_turn, attacker, defender):
        if self.owner is defender:
            teammates_remaining = len(self.owner.player.team) > 1
            aqua_card = cards_played_this_turn[attacker][0].element == "aqua"
            if teammates_remaining and aqua_card:
                return {self.owner}, list()
        return set(), list()


class SpicySurprise(Card):

    def __init__(self):
        super().__init__(*attributes_from_df("Spicy surprise"))

    def on_damage_inflicted(self, match, cards_played_this_turn, attacker, defender, amount, atk_card):
        if atk_card is self:
            defender.disable("mouth", 1)


# </editor-fold>

# <editor-fold desc="Reptiles">

class IvoryChop(Card):

    def __init__(self):
        super().__init__(*attributes_from_df("Ivory chop"))

    def on_shield_break(self, match, cards_played_this_turn, attacker, defender):
        if self.owner is defender:
            self.owner.player.draw_cards(1)
        return False


class SpikeThrow(Card):

    def __init__(self):
        super().__init__(*attributes_from_df("Spike Throw"))

    def on_target_select(self, match, cards_played_this_turn, attacker, defender):
        if self is cards_played_this_turn[self.owner][0] and self.owner is attacker and len(cards_played_this_turn[self.owner]) >= 3:
            return set(), sorted([axie for axie in defender.player.team if axie.alive()], key=lambda xe: xe.shield)
        return set(), list()


class VineDagger(Card):

    def __init__(self):
        super().__init__(*attributes_from_df("Vine Dagger"))

    def on_start_turn(self, match, cards_played_this_turn):
        if "plant" in [card.element for card in cards_played_this_turn[self.owner]]:
            self.owner.shield += self.defense


class Bulkwark(Card):

    def __init__(self):
        super().__init__(*attributes_from_df("Bulkwark"))

    def on_damage_inflicted(self, match, cards_played_this_turn, attacker, defender, amount, atk_card):
        if self.owner is defender and atk_card is not None:
            if atk_card.range == "melee":
                attacker.apply_damage(match, cards_played_this_turn, self.owner, int(0.4 * amount), False, False, None)


class SlipperyShield(Card):

    def __init__(self):
        super().__init__(*attributes_from_df("Slippery Shield"))

    def on_start_turn(self, match, cards_played_this_turn):

        if self.owner is self.owner.player.team[0] and self.owner.player.team[1].alive():
            self.owner.player.team[1].shield += int(self.owner.shield * 0.15)

        if self.owner is self.owner.player.team[1]:
            if self.owner.player.team[2].alive():
                self.owner.player.team[2].shield += int(self.owner.shield * 0.15)
            if self.owner.player.team[0].alive():
                self.owner.player.team[0].shield += int(self.owner.shield * 0.15)

        if self.owner is self.owner.player.team[2] and self.owner.player.team[1].alive():
            self.owner.player.team[2].shield += int(self.owner.shield * 0.15)


class NileStrike(Card):

    def __init__(self):
        super().__init__(*attributes_from_df("Nile Strike"))

    def on_damage_inflicted(self, match, cards_played_this_turn, attacker, defender, amount, atk_card):
        if atk_card is self:
            defender.apply_stat_eff("speed down", (3, 1))


class PooFling(Card):

    def __init__(self):
        super().__init__(*attributes_from_df("Poo Fling"))

    def on_damage_inflicted(self, match, cards_played_this_turn, attacker, defender, amount, atk_card):
        if atk_card is self:
            defender.apply_stat_eff("stench", 1)  # todo check this


class ScalyLunge(Card):

    def __init__(self):
        super().__init__(*attributes_from_df("Scaly Lunge"))

    def on_attack(self, match, cards_played_this_turn, attacker, defender, card) -> float:
        if "lunge" in [card.card_name.lower() for card in match.chain_cards]:
            return 0.2
        return 0


class SurpriseInvasion(Card):

    def __init__(self):
        super().__init__(*attributes_from_df("Surprise Invasion"))

    def on_attack(self, match, cards_played_this_turn, attacker, defender, card) -> float:
        if self.owner in match.axies_in_attack_order and defender in match.axies_in_attack_order:
            if match.axies_in_attack_order.index(self.owner) > match.axies_in_attack_order.index(defender):
                return 0.3
        else:
            axies = [self.owner, defender]

            init_speeds = [(axie, axie.speed) for axie in axies]
            final_speeds = {axie: 0 for axie in axies}

            for axie, speed in init_speeds:
                s, m = axie.buffs.on_target_select()
                speed_mult = 1 + s

                final_speeds[axie] = (speed * speed_mult, axie.hp, axie.morale * (1 + m))

            ranked_axies = sorted(axies, key=lambda axie: final_speeds[axie], reverse=True)

            if self.owner is ranked_axies[1]:
                return 0.3

        return 0


class TinyCatapult(Card):

    def __init__(self):
        super().__init__(*attributes_from_df("Tiny Catapult"))

    def on_damage_inflicted(self, match, cards_played_this_turn, attacker, defender, amount, atk_card):
        if self.owner is defender and atk_card is not None:
            if atk_card.range == "ranged":
                attacker.apply_damage(match, cards_played_this_turn, self.owner, int(0.5 * amount), False, False, None)


class Disarm(Card):

    def __init__(self):
        super().__init__(*attributes_from_df("Disarm"))

    def on_damage_inflicted(self, match, cards_played_this_turn, attacker, defender, amount, atk_card):
        if atk_card is self:
            defender.apply_stat_eff("speed down", (3, 1))


class OvergrowKeratin(Card):

    def __init__(self):
        super().__init__(*attributes_from_df("Overgrow Keratin"))

    def on_play(self, match, cards_played_this_turn):
        if self.owner.shield <= 0:
            self.owner.shield += 20


class SneakyRaid(Card):

    def __init__(self):
        super().__init__(*attributes_from_df("Sneaky Raid"))

    def on_target_select(self, match, cards_played_this_turn, attacker, defender):
        if self is cards_played_this_turn[self.owner][0] and self.owner is attacker:
            return set(), [axie for axie in defender.player.team[::-1] if axie.alive()]
        return set(), list()


class KotaroBite(Card):

    def __init__(self):
        super().__init__(*attributes_from_df("Kotaro bite"))

    def on_damage_inflicted(self, match, cards_played_this_turn, attacker, defender, amount, atk_card):
        if atk_card is self:
            if self.owner in match.axies_in_attack_order and defender in match.axies_in_attack_order:
                if match.axies_in_attack_order.index(self.owner) > match.axies_in_attack_order.index(defender):
                    self.owner.player.gain_energy(1)
            else:
                axies = [self.owner, defender]

                init_speeds = [(axie, axie.speed) for axie in axies]
                final_speeds = {axie: 0 for axie in axies}

                for axie, speed in init_speeds:
                    s, m = axie.buffs.on_target_select()
                    speed_mult = 1 + s

                    final_speeds[axie] = (speed * speed_mult, axie.hp, axie.morale * (1 + m))

                ranked_axies = sorted(axies, key=lambda axie: final_speeds[axie], reverse=True)
                if self.owner is ranked_axies[1]:
                    self.owner.player.gain_energy(1)


class WhySoSerious(Card):

    def __init__(self):
        super().__init__(*attributes_from_df("Why So Serious"))

    def on_damage_inflicted(self, match, cards_played_this_turn, attacker, defender, amount, atk_card):
        if atk_card is self and defender.element == "aqua":
            self.owner.change_hp(amount)


class Chomp(Card):

    def __init__(self):
        super().__init__(*attributes_from_df("Chomp"))

    def on_damage_inflicted(self, match, cards_played_this_turn, attacker, defender, amount, atk_card):
        if atk_card is self and len(cards_played_this_turn[self.owner]) >= 3:
            defender.apply_stat_eff("stun", True)


class CriticalEscape(Card):

    def __init__(self):
        super().__init__(*attributes_from_df("Critical Escape"))

    def on_defense(self, match, cards_played_this_turn, attacker, defender, card) -> float:
        if self.owner is defender:
            return 0.15
        return 0


class ScaleDartIguana(Card):

    def __init__(self):
        super().__init__(*attributes_from_df("Scale Dart"))

    def on_damage_inflicted(self, match, cards_played_this_turn, attacker, defender, amount, atk_card):
        if self is atk_card and defender.buffs.is_buffed():
            self.owner.player.gain_energy(1)


class TinySwing(Card):

    def __init__(self):
        super().__init__(*attributes_from_df("Tiny Swing"))

    def on_attack(self, match, cards_played_this_turn, attacker, defender, card) -> float:
        if match.round >= 5:
            return 0.5
        return 0


class JarBarrage(Card):

    def __init__(self):
        super().__init__(*attributes_from_df("Jar Barrage"))

    def on_shield_break(self, match, cards_played_this_turn, attacker, defender):
        if self.owner is defender:
            return True
        return False


class NeuroToxin(Card):

    def __init__(self):
        super().__init__(*attributes_from_df("Neuro Toxin"))

    def on_damage_inflicted(self, match, cards_played_this_turn, attacker, defender, amount, atk_card):
        if self is atk_card:
            if defender.buffs.entries["poison"] > 0:
                defender.apply_stat_eff("attack down", 1)


class VenomSpray(Card):

    def __init__(self):
        super().__init__(*attributes_from_df("Venom Spray"))

    def on_damage_inflicted(self, match, cards_played_this_turn, attacker, defender, amount, atk_card):
        if self is atk_card:
            defender.apply_stat_eff("poison", 2)

# </editor-fold>


def get_all_card_classes():
    import importlib, inspect
    all_cards = [cls for _, cls in inspect.getmembers(importlib.import_module("cards"), inspect.isclass)[:-1]
                 if issubclass(cls, Card) and cls is not Card]
    return {part: [card for card in all_cards if card().body_part == part] for part in ("mouth", "back", "horn", "tail")}


if __name__ == "__main__":
    pd.set_option("display.max_rows", None, "display.max_columns", None)
    pd.set_option('display.width', 1000)
    pd.set_option('display.max_colwidth', 500)
    pd.set_option('display.float_format', lambda x: '%.2f' % x)

    print(get_cards_with_tags(["backdoor"]))
