from typing import Tuple

from Axie.Axie_models import Card, flatten

debuffs = ["aroma", "stench", "attack down", "morale down", "speed down",
           "fear", "chill", "stun", "poison", "jinx", "fragile", "lethal", "sleep"]


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
            self.owner.heal(anemone_parts * 50)


class AquaPonics(Card):

    def __init__(self):
        # card_name, body_part, part_name, element, cost, r, attack, defense, effect
        super().__init__("Aquaponics", "horn", "Anemone", "aqua", 1, "ranged", 80, 40,
                         "Successful attacks restore 50 HP for each Anemone part this Axie posseses.")

    def on_damage_inflicted(self, match, cards_played_this_turn, attacker, defender, amount):
        if attacker is self.owner:
            anemone_parts = len([part for part in self.owner.cards if "Anemone" in part.part_name])
            self.owner.heal(anemone_parts * 50)


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
            self.owner.shield *= 1.3
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
            self.owner.heal(amount)


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
            if self is match.determine_order(flatten(list(cards_played_this_turn.keys())))[0]:
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

# </editor-fold>


# <editor-fold desc="Birds">
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
