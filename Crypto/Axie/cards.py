from typing import Tuple
import pickle
import pandas as pd
from Axie.Axie_models import Card, flatten

debuffs = ["aroma", "stench", "attack down", "morale down", "speed down",
           "fear", "chill", "stun", "poison", "jinx", "fragile", "lethal", "sleep"]


card_dataframe = pd.read_json("axie_card_data_auto.json")


def attributes_from_df(card_name):

    row = card_dataframe.loc[card_dataframe["Card name"] == card_name]
    values = [row["Card name"], row["Part"], row["Part name"], row["Type"], row["Cost"], row["Range"], row["Attack"], \
              row["Defense"], row["Effect"]]
    return [x.iloc[0] for x in values]


# <editor-fold desc="Aqua">

class Shelter(Card):

    def __init__(self):
        # card_name, body_part, part_name, element, cost, r, attack, defense, effect
        super().__init__("Shelter", "back", "Hermit", "aqua", 1, "melee", 0, 115,
                         "Disable critical strikes on this axie during this round")

    def pre_crit(self, match, cards_played_this_turn, attacker, defender) -> Tuple:
        return 0, True, 0


class ScaleDart(Card):

    def __init__(self):
        # card_name, body_part, part_name, element, cost, r, attack, defense, effect
        super().__init__("Scale dart", "back", "Blue moon", "aqua", 1, "ranged", 120, 30,
                         "Draw a card if target is in Last Stand.")

    def on_attack(self, match, cards_played_this_turn, attacker, defender) -> float:
        if attacker is self.owner and defender.last_stand:
            self.owner.player.draw_cards(1)
        return 0


class SwiftEscape(Card):

    def __init__(self):
        # card_name, body_part, part_name, element, cost, r, attack, defense, effect
        super().__init__("Swift escape", "back", "Goldfish", "aqua", 1, "melee", 110, 20,
                         "Apply Speed+ to this Axie for 2 rounds when attacked.")

    def on_attack(self, match, cards_played_this_turn, attacker, defender) -> float:
        if defender is self.owner:
            self.owner.apply_stat_eff("speed up", 1)
        return 0


class Shipwreck(Card):

    def __init__(self):
        # card_name, body_part, part_name, element, cost, r, attack, defense, effect
        super().__init__("Shipwreck", "back", "Sponge", "aqua", 1, "melee", 60, 90,
                         "Apply Attack+ to this Axie if its shield breaks.")

    def on_shield_break(self, match, cards_played_this_turn, attacker, defender):
        if defender is self.owner:
            self.owner.apply_stat_eff("attack up", 1)
        return 0


class AquaVitality(Card):

    def __init__(self):
        # card_name, body_part, part_name, element, cost, r, attack, defense, effect
        super().__init__("Aqua vitality", "back", "Anemone", "aqua", 1, "melee", 80, 40,
                         "Successful attacks restore 50 HP for each Anemone part this Axie posseses.")

    def on_damage_inflicted(self, match, cards_played_this_turn, attacker, defender, amount):
        if attacker is self.owner:
            anemone_parts = len([part for part in self.owner.cards if "Anemone" in part.part_name])
            self.owner.change_hp(anemone_parts * 50)


class AquaPonics(Card):

    def __init__(self):
        # card_name, body_part, part_name, element, cost, r, attack, defense, effect
        super().__init__("Aquaponics", "horn", "Anemone", "aqua", 1, "ranged", 80, 40,
                         "Successful attacks restore 50 HP for each Anemone part this Axie posseses.")

    def on_damage_inflicted(self, match, cards_played_this_turn, attacker, defender, amount):
        if attacker is self.owner:
            anemone_parts = len([part for part in self.owner.cards if "Anemone" in part.part_name])
            self.owner.change_hp(anemone_parts * 50)


class SpinalTap(Card):

    def __init__(self):
        # card_name, body_part, part_name, element, cost, r, attack, defense, effect
        super().__init__("Spinal tap", "back", "Perch", "aqua", 1, "melee", 100, 20,
                         "Prioritize idle target when comboed with at least 2 additional cards.")

    def on_target_select(self, match, cards_played_this_turn, attacker, defender):
        if attacker is self.owner:
            if len(cards_played_this_turn[self.owner]) >= 3:
                to_focus = [axie for axie in defender.player.team if axie.alive() and not cards_played_this_turn[axie]]
                return set(), to_focus
        return set(), list()


class ShellJab(Card):

    def __init__(self):
        # card_name, body_part, part_name, element, cost, r, attack, defense, effect
        super().__init__("Shell jab", "horn", "Babylonia", "aqua", 1, "melee", 100, 50,
                         "Deal 130% damage when attacking an idle target.")

    def on_attack(self, match, cards_played_this_turn, attacker, defender) -> float:
        if attacker is self.owner and cards_played_this_turn[defender] == []:
            return 0.3
        return 0


class DeppSeaGore(Card):

    def __init__(self):
        # card_name, body_part, part_name, element, cost, r, attack, defense, effect
        super().__init__("Deep sea gore", "horn", "Teal shell", "aqua", 1, "melee", 50, 80,
                         "Add 30% to this Axie's shield when attacking.")

    def on_attack(self, match, cards_played_this_turn, attacker, defender) -> float:
        if attacker is self.owner:
            self.owner.shield += self.defense * 0.3
        return 0


class ClamSlash(Card):

    def __init__(self):
        # card_name, body_part, part_name, element, cost, r, attack, defense, effect
        super().__init__("Clam slash", "horn", "Clam shell", "aqua", 1, "melee", 110, 40,
                         "Apply Attack+ to this Axie when attacking Beast, Bug, or Mech targets.")

    def on_attack(self, match, cards_played_this_turn, attacker, defender) -> float:
        if attacker is self.owner:
            if defender.element in ("beat", "bug", "mech"):
                self.owner.apply_stat_eff("attack up", 1)
        return 0


class HerosBane(Card):

    def __init__(self):
        # card_name, body_part, part_name, element, cost, r, attack, defense, effect
        super().__init__("Heros Bane", "horn", "Oranda", "aqua", 1, "ranged", 120, 30,
                         "End target's Last Stand.")

    def on_last_stand(self, match, cards_played_this_turn, attacker, defender):
        return self.owner is attacker


class StarShuriken(Card):

    def __init__(self):
        # card_name, body_part, part_name, element, cost, r, attack, defense, effect
        super().__init__("Star shuriken", "horn", "Shoal star", "aqua", 1, "ranged", 115, 10,
                         "Target cannot enter Last Stand if this card brings its HP to zero.")

    def on_pre_last_stand(self, match, cards_played_this_turn, attacker, defender):
        return self.owner is attacker


class AngryLam(Card):

    def __init__(self):
        # card_name, body_part, part_name, element, cost, r, attack, defense, effect
        super().__init__("Angry lam", "mouth", "Lam", "aqua", 1, "melee", 110, 40,
                         "Deal 120% damage if this Axie's HP is below 50%.")

    def on_attack(self, match, cards_played_this_turn, attacker, defender) -> float:
        if self.owner is attacker and self.owner.hp <= self.owner.base_hp * 0.5:
            return 0.5
        return 0


class SwallowCatfish(Card):

    def __init__(self):
        # card_name, body_part, part_name, element, cost, r, attack, defense, effect
        super().__init__("Swallow", "mouth", "Catfish", "aqua", 1, "melee", 80, 30,
                         "Heal this Axie by the damage inflicted with this card.")

    def on_damage_inflicted(self, match, cards_played_this_turn, attacker, defender, amount):
        if self.owner is attacker:
            self.owner.change_hp(amount)


class FishHook(Card):

    def __init__(self):
        # card_name, body_part, part_name, element, cost, r, attack, defense, effect
        super().__init__("Fish hook", "mouth", "Risky fish", "aqua", 1, "melee", 110, 30,
                         "Apply Attack+ to this Axie when attacking Plant, Reptile, or Dusk targets.")

    def on_attack(self, match, cards_played_this_turn, attacker, defender) -> float:
        if self.owner is attacker and defender.element in ("plant", "reptile", "dusk"):
            self.owner.apply_stat_eff("attack up", 1)
        return 0


class CrimsonWater(Card):

    def __init__(self):
        # card_name, body_part, part_name, element, cost, r, attack, defense, effect
        super().__init__("Crimson water", "mouth", "Piranha", "aqua", 1, "melee", 130, 20,
                         "Target injured enemy if this Axie's HP is below 50%.")

    def on_target_select(self, match, cards_played_this_turn, attacker, defender):
        if self.owner is attacker and self.owner.hp <= self.owner.base_hp * 0.5:
            to_focus = [axie for axie in defender.player.team if axie.hp < axie.base_hp]
            return set(), to_focus
        return set(), list()


class UpstreamSwim(Card):

    def __init__(self):
        # card_name, body_part, part_name, element, cost, r, attack, defense, effect
        super().__init__("Upstream swim", "tail", "Koi", "aqua", 1, "melee", 110, 30,
                         "Apply Speed+ to this Axie for 2 rounds when comboed with another Aquatic class card.")

    def on_combo(self, match, cards_played_this_turn, combo_cards):
        if len([card for card in combo_cards if card.element == "aqua"]) >= 2:
            self.owner.apply_stat_eff("speed up", 1, 2)


class TailSlap(Card):

    def __init__(self):
        # card_name, body_part, part_name, element, cost, r, attack, defense, effect
        super().__init__("Tail slap", "tail", "Nimo", "aqua", 0, "melee", 30, 0,
                         "Gain 1 energy when comboed with another card.")

    def on_attack(self, match, cards_played_this_turn, attacker, defender) -> float:
        if self.owner is attacker and len(cards_played_this_turn[attacker]) >= 2:
            self.owner.player.gain_energy(1)
        return 0


class BlackBubble(Card):

    def __init__(self):
        # card_name, body_part, part_name, element, cost, r, attack, defense, effect
        super().__init__("Black bubble", "tail", "Tadpole", "aqua", 1, "ranged", 100, 40,
                         "Apply Jinx to target for 2 rounds.")

    def on_attack(self, match, cards_played_this_turn, attacker, defender) -> float:
        if self.owner is attacker:
            defender.apply_stat_eff("jinx", 1, 2)
        return 0


class WaterSphere(Card):

    def __init__(self):
        super().__init__("Water sphere", "tail", "Ranchu", "aqua", 1, "ranged", 110, 30,
                         "Apply Chill to target for 2 rounds.")

    def on_attack(self, match, cards_played_this_turn, attacker, defender) -> float:
        if self.owner is attacker:
            defender.apply_stat_eff("chill", 1, 2)
        return 0


class FlankingSmack(Card):

    def __init__(self):
        super().__init__("Flanking smack", "tail", "Navaga", "aqua", 1, "melee", 100, 40,
                         "Deal 120% damage if this Axie attacks first.")

    def on_attack(self, match, cards_played_this_turn, attacker, defender) -> float:
        if self.owner is attacker:
            if self is match.get_axies_in_attack_order(cards_played_this_turn)[0]:
                return 0.2
        return 0


class ChitinJump(Card):

    def __init__(self):
        super().__init__("Chitin jump", "tail", "Shrimp", "aqua", 1, "melee", 30, 30,
                         "Prioritizes furthest target")

    def on_target_select(self, match, cards_played_this_turn, attacker, defender):
        if self.owner is attacker:
            return set(), [axie for axie in defender.player.team[::-1]]
        return set(), list()

# </editor-fold>

# <editor-fold desc="Beasts">


class SingleCombat(Card):

    def __init__(self):
        super().__init__("Single combat", "back", "Ronin", "beast", 1, "melee", 75, 0,
                         "Guaranteed critical strike when comboed with at least 2 other cards.")

    def pre_crit(self, match, cards_played_this_turn, attacker, defender) -> Tuple:
        if self.owner is attacker and len(cards_played_this_turn[attacker]) >= 3:
            return 1, False, 1
        return 1, False, 0


class HeroicReward(Card):

    def __init__(self):
        super().__init__("Heroic reward", "back", "Hero", "beast", 0, "melee", 50, 0,
                         "Draw a card when attacking an Aquatic, Bird, or Dawn target.")

    def on_attack(self, match, cards_played_this_turn, attacker, defender) -> float:
        if self.owner is attacker:
            if defender.element in ("aqua", "bird", "dawn"):
                self.owner.player.draw_cards(1)
        return 0


class NitroLeap(Card):

    def __init__(self):
        super().__init__("Nitro leap", "back", "Jaguar", "beast", 0, "melee", 115, 35,
                         "Always strike first if this Axie is in Last Stand.")

    def on_determine_order(self):
        if self.owner.last_stand:
            return True
        return False


class RevengeArrow(Card):

    def __init__(self):
        super().__init__("Revenge arrow", "back", "Risky beast", "beast", 1, "ranged", 125, 25,
                         "Deal 150% damage if this Axie is in Last Stand.")

    def on_attack(self, match, cards_played_this_turn, attacker, defender) -> float:
        if self.owner is attacker and self.owner.last_stand:
            return 0.5
        return 0


class WoodmanPower(Card):

    def __init__(self):
        super().__init__("Woodman power", "back", "Timber", "beast", 1, "melee", 50, 100,
                         "Add Shield equal to the damage this cards deals to Plant targets.")

    def on_damage_inflicted(self, match, cards_played_this_turn, attacker, defender, amount):
        if self.owner is attacker:
            if defender.element == "plant":
                self.owner.shield += amount


class JugglingBalls(Card):

    def __init__(self):
        super().__init__("Juggling balls", "back", "Furball", "beast", 1, "ranged", 40, 30,
                         "Strike 3 times.")

    def attack_times(self, match, cards_played_this_turn, attacker, defender):
        return 3


class BranchCharge(Card):

    def __init__(self):
        super().__init__("Branch charge", "horn", "Little branch", "beast", 1, "melee", 125, 25,
                         "Increase crit chance by 20% if chained or comboed with a plant card.")

    def pre_crit(self, match, cards_played_this_turn, attacker, defender) -> Tuple:
        if self.owner is attacker:
            if "plant" in [card.element for card in flatten(cards_played_this_turn.values())]:
                return 1, False, 0.2
        return 1, False, 0


class IvoryStab(Card):

    def __init__(self):
        super().__init__("Ivory stab", "horn", "Imp", "beast", 1, "melee", 70, 20,
                         "Gain 1 energy per critical strike dealt by your team this round.")

    def on_crit(self, match, cards_played_this_turn, attacker, defender):
        if self.owner.player is attacker.player:
            self.owner.player.gain_energy(1)


class MerryLegion(Card):

    def __init__(self):
        super().__init__("Merry legion", "horn", "Merry", "beast", 1, "melee", 65, 85,
                         "Add 20% shield to this Axie when played in a chain.")

    def on_start_turn(self, match, cards_played_this_turn):
        if set(flatten(cards_played_this_turn.values())) - set(cards_played_this_turn[self.owner]):
            self.owner.shield += int(self.defense * 0.2)


class SugarRush(Card):

    def __init__(self):
        super().__init__("Sugar rush", "horn", "Pocky", "beast", 1, "melee", 120, 20,
                         "Deal 10% additional damage for each allied Bug Axie.")

    def on_attack(self, match, cards_played_this_turn, attacker, defender) -> float:
        if self.owner is attacker:
            return 0.1 * len([axie for axie in self.owner.player.team if axie.element == "bug" and axie.alive()])
        return 0


class SinisterStrike(Card):

    def __init__(self):
        super().__init__("Sinister strike", "horn", "Dual blade", "beast", 1, "melee", 130, 20,
                         "Deal 250% damage on critical strikes.")

    def pre_crit(self, match, cards_played_this_turn, attacker, defender) -> Tuple:
        if self.owner is attacker:
            return 2.5, False, 0
        return 1, False, 0


class Acrobatic(Card):

    def __init__(self):
        super().__init__("Acrobatic", "horn", "Arco", "beast", 1, "melee", 100, 50,
                         "Apply Speed + to this Axie for 2 rounds when attacked.")

    def on_attack(self, match, cards_played_this_turn, attacker, defender) -> float:
        if self.owner is defender:
            self.owner.apply_stat_eff("speed up", 1, turns=2)
        return 0


class NutCrack(Card):

    def __init__(self):
        super().__init__("Nut crack", "mouth", "Nut cracker", "beast", 1, "melee", 105, 30,
                         "Deal 120% damage when comboed with another 'Nut Cracker' card.")

    def on_attack(self, match, cards_played_this_turn, attacker, defender) -> float:
        if self.owner is attacker:
            if len([card for card in cards_played_this_turn[self.owner] if card.part_name == "Nut cracker"]):
                return 0.2
        return 0


class PiercingSound(Card):

    def __init__(self):
        super().__init__("Piercing sound", "mouth", "Goda", "beast", 1, "ranged", 80, 40,
                         "Destoy 1 of your opponent's energy.")

    def on_attack(self, match, cards_played_this_turn, attacker, defender) -> float:
        if self.owner is attacker:
            defender.player.gain_energy(-1)
        return 0


class DeathMark(Card):

    def __init__(self):
        super().__init__("Death mark", "mouth", "Axie kiss", "beast", 1, "ranged", 90, 30,
                         "Apply Lethal to target if this Axie's HP is below 30%.")

    def on_attack(self, match, cards_played_this_turn, attacker, defender) -> float:
        if self.owner is attacker:
            if self.owner.hp < self.owner.base_hp * 0.3:
                defender.apply_stat_eff("lethal", 1)
        return 0


class SelfRally(Card):

    def __init__(self):
        super().__init__("Self rally", "mouth", "Confident", "beast", 0, "support", 0, 30,
                         "Apply 2 Morale+ to this Axie for 2 rounds.")

    def on_play(self, match, cards_played_this_turn):
        self.owner.apply_stat_eff("morale up", 1, 2)


class LunarAbsorb(Card):

    def __init__(self):
        super().__init__("Lunar absorb", "tail", "Cottontail", "beast", 0, "ranged", 0, 0,
                         "Gain 1 energy.")

    def on_play(self, match, cards_played_this_turn):
        self.owner.player.gain_energy(1)


class NightSteal(Card):

    def __init__(self):
        super().__init__("Night steal", "tail", "Rice", "beast", 1, "melee", 80, 10,
                         "Steal 1 energy from your opponent when comboed with another card.")

    def on_attack(self, match, cards_played_this_turn, attacker, defender) -> float:
        if self.owner is attacker and len(cards_played_this_turn[self.owner]) > 1:
            if defender.player.energy >= 1:
                defender.player.gain_energy(-1)
                self.owner.player.gain_energy(1)
        return 0


class RampantHowl(Card):

    def __init__(self):
        super().__init__("Rampant howl", "tail", "Shiba", "beast", 1, "melee", 110, 40,
                         "Apply Morale+ to your team for 2 rounds if this Axie attacks while in Last Stand.")

    def on_attack(self, match, cards_played_this_turn, attacker, defender) -> float:
        if self.owner is attacker and self.owner.last_stand:
            self.owner.apply_stat_eff("morale up", 1, 2)
        return 0


class HareDagger(Card):

    def __init__(self):
        super().__init__("Hare dagger", "tail", "Hare", "beast", 1, "melee", 120, 30,
                         "Draw a card if this Axie attacks at the beginning of the round.")

    def on_attack(self, match, cards_played_this_turn, attacker, defender) -> float:
        if self.owner is attacker:
            if self is list(cards_played_this_turn.keys())[0]:
                self.owner.player.draw_cards(1)
        return 0


class NutThrow(Card):

    def __init__(self):
        super().__init__("Nut throw", "tail", "Nut cracker", "beast", 1, "ranged", 105, 30,
                         "Draw a card if this Axie attacks at the beginning of the round.")

    def on_attack(self, match, cards_played_this_turn, attacker, defender) -> float:
        if self.owner is attacker:
            if len([card for card in cards_played_this_turn[self.owner] if card.part_name == "Nut cracker"]) >= 2:
                return 0.2
        return 0


class GebrilJump(Card):

    def __init__(self):
        super().__init__("Gebril jump", "tail", "Gebril", "beast", 1, "melee", 40, 20,
                         "Skip the closest target if there are 2 or more enemies remaining.")

    def on_target_select(self, match, cards_played_this_turn, attacker, defender):
        if self.owner is attacker:
            tmp = [axie for axie in defender.player.team if axie.alive()]
            if len(tmp) >= 2:
                return set(), [tmp[1]]
        return set(), list()


# </editor-fold>


# <editor-fold desc="Birds">

class BalloonPop(Card):

    def __init__(self):
        super().__init__("Balloon pop", "back", "Balloon", "bird", 0, "ranged", 40, 0,
                         "Apply Fear to target for 1 turn. If defending, apply Fear to self until next round.")

    def on_attack(self, match, cards_played_this_turn, attacker, defender) -> float:
        if self.owner is attacker:
            defender.apply_stat_eff("fear", 1)
        elif self.owner is defender:
            self.owner.apply_stat_eff("fear", 0)  # TODO check this works as intended
        return 0


class HeartBreak(Card):

    def __init__(self):
        super().__init__("Heart break", "back", "Cupid", "bird", 1, "ranged", 120, 20,
                         "Apply Morale- to enemy for 2 rounds.")

    def on_attack(self, match, cards_played_this_turn, attacker, defender) -> float:
        if self.owner is attacker:
            defender.apply_stat_eff("morale down", 1, 2)
        return 0


class IllOmened(Card):

    def __init__(self):
        super().__init__("Ill omened", "back", "Raven", "bird", 1, "ranged", 110, 30,
                         "Apply Jinx to target for 2 rounds.")

    def on_attack(self, match, cards_played_this_turn, attacker, defender) -> float:
        if self.owner is attacker:
            defender.apply_stat_eff("jinx", 1, 2)
        return 0


class PatientHunter(Card):

    def __init__(self):
        super().__init__("Patient hunter", "back", "Kingfisher", "bird", 1, "melee", 130, 0,
                         "Target an Aquatic class enemy if this Axie's HP is below 50%")

    def on_target_select(self, match, cards_played_this_turn, attacker, defender):
        if self.owner is attacker and self.owner.hp < self.owner.base_hp * 0.5:
            return set(), [axie for axie in defender.player.team if axie.alive() and axie.element == "aqua"]
        return set(), list()


class TripleThreat(Card):

    def __init__(self):
        super().__init__("Triple threat", "back", "Tri feather", "bird", 0, "ranged", 30, 10,
                         "Attack twice if this Axie has any debuffs.")

    def attack_times(self, match, cards_played_this_turn, attacker, defender):
        if self.owner is attacker and self.owner.buffs.values():
            return 2
        return 1


class Cockadoodledoo(Card):

    def __init__(self):
        super().__init__("Cockadoodledoo", "horn", "Cuckoo", "bird", 0, "ranged", 0, 40,
                         "Apply Attack+ to this Axie.")

    def on_attack(self, match, cards_played_this_turn, attacker, defender) -> float:
        if self.owner is attacker:
            self.owner.apply_stat_eff("attack up", 1)
        return 0


class AirForceOne(Card):

    def __init__(self):
        super().__init__("Air force one", "horn", "Trump", "bird", 1, "melee", 120, 30,
                         'Deal 120% damage when chained with another "Trump" card.')

    def on_attack(self, match, cards_played_this_turn, attacker, defender) -> float:
        if self.owner is attacker:
            chain_cards = []
            for xe in self.owner.player.team:
                if xe.alive() and xe is not self:
                    other_cards = cards_played_this_turn.get(xe, None)
                    if other_cards and self.element in [c.element for c in other_cards]:
                        chain_cards += cards_played_this_turn[xe]

            if "Trump" in [card.part_name for card in chain_cards]:
                return 0.2

        return 0


class Headshot(Card):

    def __init__(self):
        super().__init__("Headshot", "horn", "Kestrel", "bird", 1, "ranged", 130, 0,
                         'Disable targets horn cards next round.')

    def on_attack(self, match, cards_played_this_turn, attacker, defender) -> float:
        if self.owner is attacker:
            defender.disable("horn", 1)
        return 0


class SmartShot(Card):

    def __init__(self):
        super().__init__("Smart shot", "horn", "Wing horn", "bird", 0, "ranged", 50, 10,
                         'Skip the closest target if there are 2 or more enemies remaining.')

    ...


class FeatherLunge(Card):

    def __init__(self):
        super().__init__(*attributes_from_df("Feather lunge"))

    ...



class Eggbomb(Card):

    def __init__(self):
        # card_name, body_part, part_name, element, cost, r, attack, defense, effect
        super().__init__("Eggbomb", "horn", "Eggshell", "bird", 1, "ranged", 120, 0,
                         "Apply aroma on this axie until next round")

    def on_attack(self, match, cards_played_this_turn, attacker, defender):
        self.owner.apply_stat_eff("aroma", turns=0)  # should disappear at start of next turn
        return 0


class DarkSwoop(Card):

    def __init__(self):
        super().__init__("Little owl", "mouth", "Dark swoop", "bird", 1, "melee", 25, 0,
                         "Target fastest enemy")

    def on_target_select(self, match, cards_played_this_turn, attacker, defender):
        final_speeds = {axie: 0 for axie in cards_played_this_turn.keys()}

        for axie, speed in [(axie, axie.speed) for axie in cards_played_this_turn.keys()]:

            speed_mult = 1

            # status effects
            for buff, (count, rounds) in axie.buffs.items():
                if buff == "speed up":
                    speed_mult += 0.2 * count
                if buff == "speed down":
                    speed_mult -= 0.2 * count

            final_speeds[axie] = speed * speed_mult

        # tODO only choose from opponents team

        return set(),\
               [sorted([c for c in flatten(cards_played_this_turn.keys()) if c.player != self.owner.player],
                       key=lambda axie: final_speeds[axie], reverse=True)[0]]


class Blackmail(Card):

    def __init__(self):
        super().__init__("Pigeon post", "back", "Blackmail", "bird", 1, "ranged", 120, 10,
                         "Transfer all debuffs on this axie to target")

    def on_attack(self, match, cards_played_this_turn, attacker, defender):

        keys_to_del = []
        for debuff, (stacks, duration) in self.owner.buffs.items():
            if debuff in debuffs:
                defender.apply_stat_eff(debuff, stacks, duration)
                keys_to_del.append(debuff)
        for key in keys_to_del:
            del self.owner.buffs[key]

        return 0


class RiskyFeather(Card):

    def __init__(self):
        super().__init__("The last one", "tail", "Risky feather", "bird", 1, "ranged", 140, 10,
                         "Apply 2 Attack- to this axie")

    def on_attack(self, match, cards_played_this_turn, attacker, defender):
        self.owner.apply_stat_eff("attack down", times=2, turns=10000)  # goes away on attack
        return 0

# </editor-fold>


def get_all_card_classes():
    import importlib, inspect
    all_cards = [cls for _, cls in inspect.getmembers(importlib.import_module("cards"), inspect.isclass)[:-1]
                 if issubclass(cls, Card) and cls is not Card]
    return {part: [card for card in all_cards if card().body_part == part] for part in ("mouth", "back", "horn", "tail")}
