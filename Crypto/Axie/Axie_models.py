from abc import abstractmethod, ABC
from typing import List, Dict, Tuple
from random import shuffle, sample, random, choice, randint
from copy import deepcopy, copy as shallowcopy
from functools import reduce
from traceback import print_exc
from itertools import permutations
from matplotlib import pyplot as plt
from numpy import linspace, pi, degrees, arange, asarray
import logging
from Option_utility import get_timestamp
from Buffs import Buffs
from Utility import timeit


"""
last stand entry not working?
cards on_attack / on_dmg inflicted is messy

fear - not attack the whole turn (not miss but attack itself is not possible, support cards work tho)
sleep - whole round, shields are ignored
pumpkin should draw in last stand too

"""


def setup_logger():
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.DEBUG)
    handler = logging.FileHandler(f'axie_game_{get_timestamp().replace(" ", "__").replace(":", "-")}.log', 'w', 'utf-8')
    handler.setFormatter(logging.Formatter('%(message)s'))
    root_logger.addHandler(handler)


#setup_logger()


def debug(s=""):
    #print(s)
    #logging.debug(s)
    pass


def flatten(t):
    out = []
    for item in t:
        if isinstance(item, (list, tuple)):
            out.extend(flatten(item))
        else:
            out.append(item)
    return out


type_to_stats = {  # hp, speed, skill, morale
    "beast": (31, 35, 31, 43),
    "aqua": (39, 39, 35, 27),
    "plant": (43, 31, 31, 35),
    "dusk": (43, 39, 27, 31),
    "mech": (31, 39, 43, 27),
    "reptile": (39, 35, 31, 35),
    "bug": (35, 31, 35, 39),
    "bird": (27, 43, 35, 35),
    "dawn": (35, 35, 39, 31)
}

part_to_stats = {  # hp, speed, skill, morale
    "beast": (0, 1, 0, 3),
    "aqua": (1, 3, 0, 0),
    "plant": (3, 0, 0, 1),
    "reptile": (3, 1, 0, 0),
    "bug": (1, 0, 0, 3),
    "bird": (0, 3, 0, 1),
}

poison_dmg_per_stack = 2
miss_prob = 0
hand_card_limit = 10


def get_dmg_bonus(attacker, defender):
    group1 = ("reptile", "plant", "dusk")
    group2 = ("aqua", "bird", "dawn")
    group3 = ("mech", "beast", "bug")

    if attacker in group1:
        if defender in group1:
            return 0
        if defender in group2:
            return 0.15
        if defender in group3:
            return -0.15

    if attacker in group2:
        if defender in group1:
            return -0.15
        if defender in group2:
            return 0
        if defender in group3:
            return 0.15

    if attacker in group3:
        if defender in group1:
            return 0.15
        if defender in group2:
            return -0.15
        if defender in group3:
            return 0

    return 0


class Card(ABC):

    def __init__(self, card_name, body_part, part_name, element, cost, r, attack, defense, effect):
        self.card_name = card_name.title()
        self.body_part = body_part.lower()
        self.part_name = part_name.title()
        self.element = element.lower()
        self.cost = cost
        self.attack = attack
        self.defense = defense
        self.effect = effect
        self.range = r if attack > 0 else "support"
        self.triggered = False

        self.owner: Axie = None

    def __str__(self):
        # return f'{type(self).__name__} ({self.cost}/{self.attack}/{self.defense})'
        return f'{self.card_name} ({self.cost}/{self.attack}/{self.defense})'

    def __repr__(self):
        return self.__str__()

    def __deepcopy__(self, memodict={}):  # do not deepcopy cards
        tmp = self.owner
        self.owner = None
        to_ret = shallowcopy(self)
        self.owner = tmp
        return to_ret

    def detail(self):
        return f'{self.card_name} - {self.part_name} ({self.body_part}, {self.range})\n\t' \
               f'{self.element} - Cost: {self.cost}, Attack: {self.attack}, Defense: {self.defense}\n\t' \
               f'Effect: {self.effect}'

    def on_determine_order(self):
        """

        :return: always_first?
        """
        return False

    def on_play(self, match, cards_played_this_turn):
        pass

    def on_combo(self, match, cards_played_this_turn, combo_cards):
        pass

    def on_chain(self, match, cards_played_this_turn, chain_cards):
        pass

    def on_start_turn(self, match, cards_played_this_turn):
        pass

    def on_target_select(self, match, cards_played_this_turn, attacker, defender):  # returns to_omit, focus (sets)
        """

        :param match:
        :param cards_played_this_turn:
        :param attacker:
        :param defender:
        :return: to_omit, to_focus (sets of axies)
        """
        return set(), list()

    def attack_times(self, match, cards_played_this_turn, attacker, defender):
        """

        :param match:
        :param cards_played_this_turn:
        :param attacker:
        :param defender:
        :return: number of attacks to perform upon play
        """
        return 1

    def on_attack(self, match, cards_played_this_turn, attacker, defender, card) -> float:
        """

        :param card:
        :param match:
        :param cards_played_this_turn:
        :param attacker:
        :param defender:
        :return: atk multi (like 0.1 for 10% additional dmg) / dmg reduction (like 0.95 for 5% reduction)
        """
        return 0

    def on_defense(self, match, cards_played_this_turn, attacker, defender, card) -> float:
        """

        :param card:
        :param match:
        :param cards_played_this_turn:
        :param attacker:
        :param defender:
        :return: atk multi (like 0.1 for 10% additional dmg) / dmg reduction (like 0.95 for 5% reduction)
        """
        return 0

    def pre_crit(self, match, cards_played_this_turn, attacker, defender) -> Tuple:  # (crit_multi, crit disable)
        """
        :param match:
        :param cards_played_this_turn:
        :param attacker:
        :param defender:
        :return: crit multi (1), crit disable (False), add crit prob (0)
        """
        return 1, False, 0

    def on_crit(self, match, cards_played_this_turn, attacker, defender):  # effects like draw on crit
        return

    def on_shield_break(self, match, cards_played_this_turn, attacker, defender) -> bool:
        """

        :param match:
        :param cards_played_this_turn:
        :param attacker:
        :param defender:
        :return: None
        """
        pass

    def on_damage_inflicted(self, match, cards_played_this_turn, attacker, defender, amount, atk_card):
        """

        :param atk_card:
        :param match:
        :param cards_played_this_turn:
        :param attacker:
        :param defender:
        :param amount:
        :return: None
        """
        pass

    def on_pre_last_stand(self, match, cards_played_this_turn, attacker, defender):  # block last stand, end last stand
        """

        :param match:
        :param cards_played_this_turn:
        :param attacker:
        :param defender:
        :return: block_last_stand, end_last_stand
        """
        return False, False

    def on_last_stand(self, match, cards_played_this_turn, attacker, defender):
        pass

    def on_end_turn(self, match, cards_played_this_turn):
        pass


class Axie:
    hp_multi = 8.39

    @staticmethod
    def get_genetically_complete_pop(n=1):
        import numpy as np
        from Axie.cards import get_all_card_classes
        all_cc = get_all_card_classes()

        teams = []

        for _ in range(n):
            all_backs = sample(all_cc["back"], len(all_cc["back"]))
            all_horns = sample(all_cc["horn"], len(all_cc["horn"])) * 2
            all_tail = sample(all_cc["tail"], len(all_cc["tail"])) * 2
            all_mouths = sample(all_cc["mouth"], len(all_cc["mouth"])) * 2

            l = len(all_backs)
            for j in range(int(np.ceil(len(all_backs) / 3))):
                i = j * 3
                teams.append([Axie(Axie.rand_elem(True),
                                   all_mouths[i % l](),
                                   all_horns[i % l](),
                                   all_backs[i % l](),
                                   all_tail[i % l](), neutral=False),
                              Axie(Axie.rand_elem(True),
                                   all_mouths[(i + 1) % l](),
                                   all_horns[(i + 1) % l](),
                                   all_backs[(i + 1) % l](),
                                   all_tail[(i + 1) % l](), neutral=False),
                              Axie(Axie.rand_elem(True),
                                   all_mouths[(i + 2) % l](),
                                   all_horns[(i + 2) % l](),
                                   all_backs[(i + 2) % l](),
                                   all_tail[(i + 2) % l](), neutral=False)])
        """
        for player in [Player(team) for team in teams]:
            print(player.get_deck_string())"""

        return [Player(team) for team in teams]

    @staticmethod
    def rand_elem(rare_types=False):
        if not rare_types:
            return sample(("beast", "bird", "bug", "plant", "aqua", "reptile"), 1)[0]
        return sample(("beast", "bird", "bug", "plant", "aqua", "reptile", "dusk", "dawn", "mech"), 1)[0]

    @staticmethod
    def get_random(neutral=False):
        from Axie.cards import get_all_card_classes
        all_cc = get_all_card_classes()

        return Axie(Axie.rand_elem(),
                    sample(all_cc["mouth"], 1)[0](),
                    sample(all_cc["horn"], 1)[0](),
                    sample(all_cc["back"], 1)[0](),
                    sample(all_cc["tail"], 1)[0](),
                    neutral=neutral)

    @staticmethod
    def flip_single_gene(axie):
        from Axie.cards import get_all_card_classes
        all_cc = get_all_card_classes()

        r = random()
        if r < 0.2:
            return Axie(Axie.rand_elem(True), axie.mouth, axie.horn, axie.back, axie.tail, neutral=True)
        if r < 0.4:
            return Axie(axie.element, sample(all_cc["mouth"], 1)[0](), axie.horn, axie.back, axie.tail, neutral=True)
        if r < 0.6:
            return Axie(axie.element, axie.mouth, sample(all_cc["horn"], 1)[0](), axie.back, axie.tail, neutral=True)
        if r < 0.8:
            return Axie(axie.element, axie.mouth, axie.horn, sample(all_cc["back"], 1)[0](), axie.tail, neutral=True)
        else:
            return Axie(axie.element, axie.mouth, axie.horn, axie.back, sample(all_cc["tail"], 1)[0](), neutral=True)

    @staticmethod
    def pair(mom, dad):
        return deepcopy(Axie(mom.element if random() > 0.5 else dad.element,
                             mom.mouth if random() > 0.5 else dad.mouth,
                             mom.horn if random() > 0.5 else dad.horn,
                             mom.back if random() > 0.5 else dad.back,
                             mom.tail if random() > 0.5 else dad.tail,
                             neutral=True))

    @staticmethod
    def from_simple_axie_model(s):
        from Axie.cards import part_name_to_card_name, instantiate
        from importlib import import_module
        module = import_module("Axie.cards")

        def to_class_name(x):
            return x.replace("'", "").replace("-", " ").title().replace(" ", "")

        def make_class(name):
            return getattr(module, name)()

        m = make_class(to_class_name(part_name_to_card_name(s.mouth.part_name, "mouth")))
        h = make_class(to_class_name(part_name_to_card_name(s.horn.part_name, "horn")))
        b = make_class(to_class_name(part_name_to_card_name(s.back.part_name, "back")))
        t = make_class(to_class_name(part_name_to_card_name(s.tail.part_name, "tail")))

        return Axie(s.element.lower() if s.element.lower() != "aquatic" else "aqua",
                    m, h, b, t,
                    s.eyes.part_type.lower() if s.eyes.part_type.lower() != "aquatic" else "aqua",
                    s.ears.part_type.lower() if s.ears.part_type.lower() != "aquatic" else "aqua")

    @staticmethod
    def from_id(axie_id):
        from axie_graphql import get_genes_by_id
        from cards import part_name_to_card_name
        from importlib import import_module
        module = import_module("Axie.cards")

        def to_class_name(x):
            return x.replace("'", "").replace("-", " ").title().replace(" ", "")

        def make_class(name):
            return getattr(module, name)()

        genes = get_genes_by_id(axie_id=axie_id)

        m = make_class(to_class_name(part_name_to_card_name(genes["mouth"]["d"], "mouth")))
        h = make_class(to_class_name(part_name_to_card_name(genes["horn"]["d"], "horn")))
        b = make_class(to_class_name(part_name_to_card_name(genes["back"]["d"], "back")))
        t = make_class(to_class_name(part_name_to_card_name(genes["tail"]["d"], "tail")))

        return Axie(genes["element"].lower() if genes["element"].lower() != "aquatic" else "aqua",
                    m, h, b, t,
                    genes["eyes"]["element"].lower() if genes["eyes"]["element"].lower() != "aquatic" else "aqua",
                    genes["ears"]["element"].lower() if genes["ears"]["element"].lower() != "aquatic" else "aqua")

    def __init__(self, element: str, mouth: Card, horn: Card, back: Card, tail: Card, eyes=None, ears=None,
                 neutral=False):

        self.player: Player = None

        self.element = element.lower() if element else self.rand_elem(rare_types=True)

        try:
            assert mouth.body_part == "mouth"
            assert back.body_part == "back"
            assert tail.body_part == "tail"
            assert horn.body_part == "horn"
        except AssertionError:
            # try fix it
            parts = (mouth, back, tail, horn)
            m = [p for p in parts if p.body_part == "mouth"]
            b = [p for p in parts if p.body_part == "back"]
            h = [p for p in parts if p.body_part == "horn"]
            t = [p for p in parts if p.body_part == "tail"]

            if m and b and h and t:
                try:
                    mouth = m[0]
                    back = b[0]
                    horn = h[0]
                    tail = t[0]
                except IndexError:
                    print("Body parts are mixed up, automatic fix failed! You miss at least 1 part")
                    exit(-1)

        self.mouth = mouth
        mouth.owner = self

        self.horn = horn
        horn.owner = self

        self.back = back
        back.owner = self

        self.tail = tail
        tail.owner = self

        self.eyes = eyes if eyes else self.rand_elem()
        self.ears = ears if ears else self.rand_elem()

        self.cards = {mouth, horn, back, tail}

        bhp, bsp, bsk, bm = type_to_stats[self.element]

        for card in self.cards:
            php, psp, psk, pm = part_to_stats[card.element]
            bhp += php
            bsp += psp
            bsk += psk
            bm += pm

        if not neutral:
            ehp, esp, esk, em = part_to_stats[self.eyes]
            php, psp, psk, pm = part_to_stats[self.ears]

            bhp += php + ehp
            bsp += psp + esp
            bsk += psk + esk
            bm += pm + em

        # if axie is idle and not in battle, element stats + card stats included
        self.base_hp = int(bhp * self.hp_multi)
        self.base_morale = bm
        self.base_speed = bsp
        self.base_skill = bsk

        # self.buffs = dict()  # buff name: stacks, turns remaining
        # self.buff_q = dict()
        self.buffs = Buffs()

        self.disabilities = dict()  # type to disable: number of turns, e.b. mouth: 1, auqa: 1
        self.disabilities_q = dict()

        # self.apply_stat_eff_changes = []

        self.hp = self.base_hp
        self.speed = self.base_speed
        self.skill = self.base_skill
        self.morale = self.base_morale

        self.last_stand = False
        self.last_stand_ticks = 1 if self.base_morale <= 38 else 2 if self.base_morale <= 49 else 3

        self.shield = 0
        self.shield_broke_this_turn = False

    def __repr__(self):
        hp_stat = f'(H:{self.hp}+{int(self.shield)}/{self.base_hp}, S:{self.speed})' if self.alive() else f'(dead)'
        stand_stat = f'L {self.last_stand_ticks}'
        x = self.buffs.to_str()
        return f'{" " * (7 - len(self.element)) + self.element.capitalize()} Axie #{str(id(self))[-4:]} ' \
               f'[{str(self.player)}] ' \
               f'{hp_stat if not self.last_stand else stand_stat} ({x})'

    def disable(self, part, turns, asap=False):
        if asap:
            self.disabilities.update({part: max(self.disabilities_q.get(part, 0), turns)})
        else:
            self.disabilities_q.update({part: max(self.disabilities_q.get(part, 0), turns)})

    def reset(self):
        self.buffs = Buffs()
        # self.buff_q = dict()
        self.disabilities = dict()  # type to disable: number of turns, e.b. mouth: 1, auqa: 1
        # self.apply_stat_eff_changes = []

        self.hp = self.base_hp
        self.speed = self.base_speed
        self.skill = self.base_skill
        self.morale = self.base_morale

        self.last_stand = False
        self.last_stand_ticks = 1 if self.base_morale <= 38 else 2 if self.base_morale <= 49 else 3

        self.shield = 0
        self.shield_broke_this_turn = False

        for card in self.cards:
            card.owner = self

    def long(self):
        hp_stat = f'(H:{self.hp}+{int(self.shield)}/{self.base_hp}, S:{self.speed})' if self.alive() else f'(dead)'
        return f'{" " * (7 - len(self.element)) + self.element.capitalize()} Axie #{str(id(self))[-4:]} ' \
               f'[{str(self.player)}] {hp_stat} - (' \
               f'{self.back}, ' \
               f'{self.mouth}, ' \
               f'{self.horn}, ' \
               f'{self.tail})'

    def on_death(self):
        try:
            self.hp = 0
            self.last_stand = False
            self.shield = 0
            self.buffs = Buffs()
            self.disabilities = dict()

            self.player.discard_pile += self.player.hand[self]
            del self.player.hand[self]

            self.player.deck_pile = [card for card in self.player.deck_pile if card.owner.alive()]
            self.player.discard_pile = [card for card in self.player.discard_pile if card.owner.alive()]

            debug(f'{self} died!')
            debug("Hand discarded sucessfully!")
        except KeyError as e:
            print(f'Key error in "on_death": {e}')
            print_exc()
            exit()

    def apply_damage(self, battle, cards_to_play, attacker, atk, double_shield_dmg, true_dmg, atk_card):

        if not self.alive():
            return

        morale_multi, block_last_stand = self.buffs.on_dmg_calc()
        end_last_stand = False
        fctp = flatten(list(cards_to_play.values()))
        sleep = False

        atk = int(atk)
        orig_atk = atk
        self.shield = int(self.shield)
        inflicted_dmg = 0

        debug(f'\tATK: {atk}')

        """if "sleep" in self.buffs.keys() and self.buffs["sleep"][0] > 0:
            sleep = True
            self.reduce_stat_eff("sleep", "all")"""

        # on pre last stand
        for _card in fctp:
            block_last_stand = block_last_stand or _card.on_pre_last_stand(battle, cards_to_play, attacker, self)

        def on_dmg_infl():
            debug(f'Damage inflicted: {orig_atk}')
            if atk_card:
                for c in fctp:
                    c.on_damage_inflicted(battle, cards_to_play, attacker, self, orig_atk, atk_card)

        if self.last_stand:

            if end_last_stand:
                self.last_stand = False
                self.last_stand_ticks = 0
                debug("Last stand ended by effect")
            else:

                self.last_stand_ticks -= 1
                inflicted_dmg += atk
                on_dmg_infl()

                if self.last_stand_ticks <= 0:
                    self.last_stand = False  # ded
                    self.on_death()
                    return

        # handle shield
        elif not true_dmg and not sleep and self.shield > 0:

            # shield break
            if (double_shield_dmg and atk * 2 >= self.shield) or atk >= self.shield:

                if double_shield_dmg:
                    atk -= int(self.shield / 2)
                    inflicted_dmg += int(self.shield / 2)
                else:
                    atk -= self.shield
                    inflicted_dmg += self.shield
                self.shield = 0

                debug(f'Shield broke')

                block_additional_dmg = False

                # on sb
                for _card in fctp:
                    block_additional_dmg = block_additional_dmg or _card.on_shield_break(battle, cards_to_play,
                                                                                         attacker, self)

                self.shield_broke_this_turn = True

            else:  # shield not breaking

                rem_shield = max(0, self.shield - atk)
                inflicted_dmg = atk
                if double_shield_dmg:
                    atk -= int(self.shield / 2)
                else:
                    atk -= self.shield
                self.shield = rem_shield
                on_dmg_infl()
                return

        # after shield break

        # last stand? TODO not working yet????
        if not block_last_stand and self.hp - atk < 0 and atk - self.hp < self.hp * self.morale * morale_multi / 100:

            # on last stand entry
            for _card in fctp:
                end_last_stand = end_last_stand or _card.on_last_stand(battle, cards_to_play, attacker, self)

            inflicted_dmg += self.hp
            on_dmg_infl()

            if not end_last_stand:
                self.enter_last_stand(morale_multi)
                return

        # lower hp
        else:

            inflicted_dmg += atk
            self.change_hp(-atk)
            on_dmg_infl()

    def change_hp(self, amount):
        if self.alive() and not self.last_stand:  # TODO this seemingly is called when self is already dead ...
            if "heal" in self.disabilities.keys():
                amount = min(amount, 0)
            self.hp = max(min(self.base_hp, self.hp + amount), 0)
            if self.hp <= 0:
                self.on_death()

    def enter_last_stand(self, multi=1):
        self.last_stand = True
        self.hp = 0
        # TODO check if this is true
        self.last_stand_ticks = int(((self.morale * multi) - 27) / 11)
        self.shield = 0
        debug(f'{self} entered last stand')

    def turn_tick(self):  # TODO each status effect should have a life time that ticks away

        # --- apply queue ---

        # apply buff q
        """for buff, (times, turns) in self.buff_q.items():
            ti, tu = self.buffs.get(buff, (0, 0))
            self.buffs[buff] = (ti + times, tu + turns)
        self.buff_q = dict()"""

        # apply disablity q
        for dis, turns in self.disabilities_q.items():
            self.disabilities.update({dis: max(self.disabilities.get(dis, 0), turns)})
        self.disabilities_q = dict()

        # --- tick ---

        # tick status effects (buffs, debuffs)
        """to_del = []
        for stat_eff, (times, turns) in self.buffs.items():

            if stat_eff == "attack down" or stat_eff == "attack up":
                continue

            if turns - 1 < 0 or times == 0:
                to_del.append(stat_eff)
            else:
                self.buffs[stat_eff] = (times, turns-1)

        for key in to_del:
            del self.buffs[key]"""
        self.buffs.on_turn()

        # tick disabilities (body parts, certain element cards)
        to_del = []
        for dis, turns in self.disabilities.items():
            if turns - 1 < 0:
                to_del.append(dis)
            else:
                self.disabilities[dis] = turns - 1

        for entry in to_del:
            del self.disabilities[entry]

        # reset shield
        self.shield = 0

    def action_tick(self):

        # tODO
        # tick last stand down
        if self.last_stand:
            self.last_stand_ticks -= 1

        # tick poison
        """if "poison" in self.buffs.keys():
            stacks, _ = self.buffs["poison"]
            self.change_hp(-stacks * poison_dmg_per_stack)"""
        self.change_hp(-self.buffs.on_tick())

        """for change in self.apply_stat_eff_changes:
            change()
        self.apply_stat_eff_changes = []"""

    def apply_stat_eff(self, eff, data):
        self.buffs.add(eff, data)

    """def apply_stat_eff(self, effect, times=1, turns=0):
        # which effects stack? if 2x fear for "next turn" does it stack to 2?

        
        TODO this is inaccurate because new stacks falsely extend the duration of old stacks
        :param effect: effect name, e.g. aroma
        :param times: stacks to apply, e.g. poison
        :param turns: 0 = until next turn, 1 = for next turn, 2 = for next 2 turns
        :return:
        

        stacking = {"attack up", "attack down", "morale up", "morale down", "speed up", "speed down", "poison"}
        only_for_next_round = {"jinx"}

        # effects that don't decay over time
        if effect in ("poison", "stun", "attack up", "attack down", "fragile", "lethal"):
            turns = 10000

        _times, _turns = self.buffs.get(effect, (0, 0))
        if effect in only_for_next_round:
            self.buff_q.update({effect: (1, turns)})
            return
        if effect not in stacking:
            self.buffs.update({effect: (1, max(_turns, turns))})
            return
        else:
            self.buffs.update({effect: (times + _times, max(_turns, turns))})
            return"""

    """def reduce_stat_eff(self, effect, stacks):

        def f():
            nonlocal stacks
            s, rounds = self.buffs.get(effect, (0, 0))
            if stacks == "all":
                stacks = s
            if rounds > 1000 and s-stacks <= 0:  # TODO NEW
                del self.buffs[effect]
                return
            self.buffs[effect] = (s-stacks, rounds)

        self.apply_stat_eff_changes.append(f)"""

    def alive(self):
        return self.hp > 0 or self.last_stand


class Player:
    start_energy = 3

    @staticmethod
    def get_random(agent=None):
        return Player(sorted([Axie.get_random() for _ in range(3)], key=lambda xe: xe.base_hp, reverse=True),
                      agent=agent)

    @staticmethod
    @timeit
    def get_top_ladder_decks(cached=True) -> list:
        from top_ladder_scraper import get_top_100_decks
        decks = get_top_100_decks(use_cached=cached)
        teams = [list() for _ in range(len(decks))]

        for i, team in enumerate(decks):
            for axie in team:
                teams[i].append(Axie.from_simple_axie_model(axie))

        return [Player(team) for team in teams]

    @staticmethod
    def get_top_ladder_deck_evaluations():
        eval_results = [deck.get_deck_evaluation() for deck in Player.get_top_ladder_decks()]
        scores_averages = {
            key: sum([eval_results[i][0][key] for i in range(len(eval_results))]) / len(eval_results)
            for key in eval_results[0][0].keys()
        }
        raw_data_averages = {
            "dmg_per_elem": {
                "plant/reptile": 0,
                "aqua/bird": 0,
                "beast/bug": 0,
                "per_axie": [[0, 0, 0], [0, 0, 0], [0, 0, 0]],
            },
            "shield_distribution": {
                "team": 0,
                "front": 0,
                "mid": 0,
                "back": 0,
            },
            "hp": {
                "team": 0,
                "front": 0,
                "mid": 0,
                "back": 0,
            }
        }
        for ii in range(len(eval_results)):
            for key, val in eval_results[ii][1].items():

                if key == "dmg_per_elem":

                    for k, v in val.items():

                        if k != "per_axie":
                            raw_data_averages[key][k] += (v / len(eval_results))
                        else:
                            for i in range(len(raw_data_averages[key][k])):
                                for j in range(3):
                                    raw_data_averages[key][k][j][i] += (v[j][i] / len(eval_results))

                elif key == "shield_distribution" or key == "hp":

                    for k, v in val.items():
                        raw_data_averages[key][k] += (v / len(eval_results))

        return scores_averages, raw_data_averages

    @staticmethod
    def from_team_ids(team_ids, agent=None, greedy=False):
        return Player([Axie.from_id(_id) for _id in team_ids], agent=agent, greedy=greedy)

    def __init__(self, team: List[Axie], agent=None, greedy=False):
        assert len(team) == 3
        for axie in team:
            if not type(axie) is Axie:
                print(team)
                raise AssertionError
        self.team = team  # team in order front to back line for simplicity
        self.deck = list()
        self.agent = agent
        self.greedy = greedy

        for axie in team:
            self.deck += list(axie.cards) + list(axie.cards)
            axie.player = self

        self.energy = self.start_energy
        self.deck_pile = shallowcopy(self.deck)
        self.discard_pile = list()
        self.hand = dict()

    def reset(self):
        for axie in self.team:
            axie.reset()
        self.energy = self.start_energy
        self.reset_deck()
        self.deck_pile = shallowcopy(self.deck)
        self.discard_pile = list()
        self.hand = dict()

    def reset_deck(self):
        self.deck = list()
        for axie in self.team:
            self.deck += list(axie.cards) + list(axie.cards)
            axie.player = self

    def get_deck_string(self):
        s = "\t"
        for axie in self.team:
            s += axie.long() + "\n\t"
        return s[:-1]

    def __repr__(self):
        return f'Player #{str(id(self))[-4:]}'

    def as_tuple(self, with_elem=False):
        if with_elem:
            return tuple(
                flatten([[axie.element] + sorted([card.card_name for card in axie.cards]) for axie in self.team]))
        return tuple(flatten([sorted([card.card_name for card in axie.cards]) for axie in self.team]))

    def diff(self, other, with_elem=False):
        t1 = self.as_tuple(with_elem=with_elem)
        t2 = other.as_tuple(with_elem=with_elem)
        d = sum([len(set(t1[i:i + (len(t1) // len(self.team))]).difference(set(t2[i:i + (len(t2) // len(self.team))])))
                 for i in range(len(self.team))])
        return d / len(t1)

        """
        return sum([a != b for a, b in zip(self.as_tuple(with_elem=with_elem), other.as_tuple(with_elem=with_elem))]) /\
            len(self.as_tuple(with_elem=with_elem))"""

    def __hash__(self):
        return id(self)

    def start_turn(self, cards_per_turn, energy_per_turn):

        for axie in self.team:
            if axie.alive():
                axie.turn_tick()

        self.draw_cards(cards_per_turn)
        self.gain_energy(energy_per_turn)

    def team_alive(self):
        for axie in self.team:
            if axie.alive():
                return True
        return False

    def random_discard(self):
        hand_cards = reduce(lambda x, y: x + y, list(self.hand.values()))

        if len(hand_cards) > 0:
            to_discard = choice(hand_cards)

            self.hand[to_discard.owner].remove(to_discard)
            self.discard_pile.append(to_discard)

    def draw_cards(self, n):

        if len(self.deck_pile) < n:
            self.deck_pile += shallowcopy(self.discard_pile)
            self.discard_pile = []

        shuffle(self.deck_pile)
        drawn_cards = []
        for i in range(n):
            try:
                drawn_cards.append(self.deck_pile.pop())
            except IndexError:
                debug("Index error while trying to draw")
                pass  # this happens if we have many cards in hand while the deck & dc pile are thin

        tmp = {owner: self.hand.get(owner, list()) + [c for c in drawn_cards if c.owner == owner] for owner in
               self.team}
        self.hand.update(tmp)

        hand_cards = reduce(lambda x, y: x + y, list(self.hand.values()))

        while len(hand_cards) > hand_card_limit:
            self.random_discard()
            hand_cards = reduce(lambda x, y: x + y, list(self.hand.values()))

        if len(hand_cards) + len(self.deck_pile) + len(self.discard_pile) != sum(
                map(lambda xe: xe.alive(), self.team)) * 8:
            print("Something went wrong!")
            print(f'Hand cards: {len(hand_cards)} Deck pile: {len(self.deck_pile)} Disc pile: {len(self.discard_pile)}')
            print("Card sum is wrong! Should be", sum(map(lambda xe: xe.alive(), self.team)) * 8, "but is",
                  len(hand_cards) + len(self.deck_pile) + len(self.discard_pile))
            for c in drawn_cards:
                print(c, c.owner)
            print()
            for c in self.team:
                print(c)
            print(self)
            exit(-1)
        debug(f'Hand:         {hand_cards} ({len(hand_cards)})')
        debug(f'Discard pile: ({len(self.discard_pile)})')
        debug(f'Deck pile:    ({len(self.deck_pile)})')
        for k, v in self.hand.items():
            debug(f'{k}: {v}')
        debug()

    def gain_energy(self, amount):
        self.energy = max(min(10, self.energy + amount), 0)

    def select_cards(self, game_state) -> Dict:
        for axie in self.team:
            if axie.alive():
                for dis, stacks, in axie.disabilities.items():
                    debug(f'Disabled: {dis} for {stacks} rounds')
                    
        return self.apply_selection(self.agent.select(self, game_state)) if self.agent else self.random_select(
            game_state)
    
    def get_many_possible_moves(self):
        """
        :param game_state: 
        :return: many legal moves the player can make, given by a dict {axie: cards_played}
        """

        # randomly choose an energy limit
        energy_limit = int(random() * self.energy)
        energy_used = 0

        def get_all_single_hand_selections(axie_hand, e_lim):
            """
            :param axie_hand:
            :param e_lim:
            :return: list of tuples (selected_cards_in_selected_order, cum_cost)
            """
            possible_selections = []
            for selection_len in range(5):
                for possible_sel in permutations(axie_hand, selection_len):
                    c = sum([c.cost for c in possible_sel])
                    if c <= e_lim:
                        possible_selections.append((possible_sel, c))
            return possible_selections

        def get_random_selection_from_single_axie_hand(axie_hand, lim):
            cards_to_select = sample(axie_hand, randint(0, 4))
            while sum([c.cost for c in cards_to_select]) > lim:
                cards_to_select = sample(axie_hand, randint(0, 4))
            return cards_to_select, sum([c.cost for c in cards_to_select])

        final_selection = {axie: [] for axie in self.team}
        shuffled_hand = shuffle([(owner,
                                  [card for card in cards if card.element not in card.owner.disabilities and
                                                             card.part not in card.owner.disabilities])
                                 for owner, cards in self.hand.items()])

        for axie, a_hand in shuffled_hand:

            if energy_used <= energy_limit:
                sel, cost = get_random_selection_from_single_axie_hand(a_hand, energy_limit-energy_used)
                final_selection[axie] = sel
                energy_used += cost

        return final_selection

    # TODO
    def get_all_possible_moves(self):
        """
        :return: many legal moves the player can make, given by a dict {axie: cards_played}
        """

        # randomly choose an energy limit
        energy_limit = int(random() * self.energy)
        energy_used = 0

        def get_all_single_hand_selections(axie_hand, e_lim):
            """
            :param axie_hand:
            :param e_lim:
            :return: list of tuples (selected_cards_in_selected_order, cum_cost)
            """
            # if owner is dead, return empty list
            if axie_hand and not axie_hand[0].owner.alive():
                return []

            possible_selections = []
            for selection_len in range(min(len(axie_hand), 5)):
                for possible_sel in permutations(axie_hand, selection_len):
                    c = sum([c.cost for c in possible_sel])
                    if c <= e_lim:
                        possible_selections.append((possible_sel, c))
            return possible_selections

        final_selection = {axie: [] for axie in self.team}
        shuffled_hand = shuffle([(owner,
                                  [card for card in cards if card.element not in card.owner.disabilities and
                                   card.part not in card.owner.disabilities])
                                 for owner, cards in self.hand.items()])

        single_hand_selections = [get_all_single_hand_selections(self.hand[axie], energy_limit) for axie in self.team]

        for hand_1_selection in single_hand_selections[0]:

            for hand_2_selection in single_hand_selections[1]:

                ...



        return final_selection

    def apply_selection(self, cards_picked) -> dict:

        # check if selection is ok
        used_energy = 0
        for axie, cards in cards_picked.items():
            assert len(cards) < 5
            for card in cards:
                assert card.owner.alive()
                assert card.element not in card.owner.disabilities.keys()
                assert card.body_part not in card.owner.disabilities.keys()
                assert card.range not in card.owner.disabilities.keys()
                used_energy += card.cost
        assert used_energy <= self.energy

        # apply it to game state

        cards_to_play = {axie: [] for axie in self.hand.keys()}  # to be returned)

        for c in flatten(cards_picked.values()):
            # put into return dict
            cards_to_play[c.owner].append(c)

            # remove card from hand
            self.hand[c.owner].remove(c)

        debug(f'\tCards to play: {[(str(id(axie))[-4:], card) for axie, card in cards_to_play.items()]}')
        debug(f'\tUsed energy: {used_energy}, Energy remaining: {self.energy - used_energy}')
        debug(f'\tTeam order: {[a for a in self.team]}')

        self.gain_energy(-used_energy)

        return cards_to_play

    def random_select(self, game_state) -> Dict:

        hand_cards = flatten(list(self.hand.values()))
        hand_card_number = len(hand_cards)

        c = 0
        used_energy = 0
        energy_this_turn = self.energy

        cards_to_play = {axie: [] for axie in self.hand.keys()}
        cards_per_axie = {axie: 0 for axie in self.hand.keys()}

        pick_threshold = 0.2 * game_state.round if not self.greedy else 0

        while c < hand_card_number and used_energy < energy_this_turn:

            if random() > pick_threshold \
                    and hand_cards[c].owner.alive() \
                    and hand_cards[c].cost <= energy_this_turn - used_energy \
                    and cards_per_axie[hand_cards[c].owner] < 4 \
                    and hand_cards[c].element not in hand_cards[c].owner.disabilities.keys() \
                    and hand_cards[c].body_part not in hand_cards[c].owner.disabilities.keys() \
                    and hand_cards[c].range not in hand_cards[c].owner.disabilities.keys():
                # put into return dict
                cards_to_play[hand_cards[c].owner].append(hand_cards[c])
                # remove card from hand
                self.hand[hand_cards[c].owner].remove(hand_cards[c])
                # remember energy
                used_energy += hand_cards[c].cost
                # count cards per axie
                cards_per_axie[hand_cards[c].owner] += 1

            c += 1

        # use up all energy used by cards
        self.gain_energy(-used_energy)

        debug(f'\tCards to play: {[(str(id(axie))[-4:], card) for axie, card in cards_to_play.items()]}')
        debug(f'\tUsed energy: {used_energy}, Energy remaining: {energy_this_turn - used_energy}')
        debug(f'\tTeam order: {[a for a in self.team]}')

        return cards_to_play

    def get_relative_team_hp(self):
        return sum([float(axie.hp) / float(axie.base_hp) for axie in self.team]) / len(self.team)

    # plotting & deck stats

    def get_abs_dmg_per_elem(self):
        group1 = ("reptile", "plant", "dusk")
        group2 = ("aqua", "bird", "dawn")
        group3 = ("mech", "beast", "bug")

        # <editor-fold desc="dmg types relative">
        # 1 - (result / 133); result is diff of each type from 33%
        g1, g2, g3 = 0, 0, 0
        ab = [[0 for _ in range(3)] for _ in range(3)]
        for i, axie in enumerate(sorted(self.team, key=lambda x: x.base_hp, reverse=True)):

            for card in axie.cards:
                if card.element in group1:
                    g1 += card.attack
                    ab[i][0] += card.attack
                if card.element in group2:
                    g2 += card.attack
                    ab[i][1] += card.attack
                if card.element in group3:
                    g3 += card.attack
                    ab[i][2] += card.attack

        return {
            "plant/reptile": g1,
            "aqua/bird": g2,
            "beast/bug": g3,
            "per_axie": ab
        }

    def get_dmg_type_per_axie(self):

        group1 = ("reptile", "plant", "dusk")
        group2 = ("aqua", "bird", "dawn")
        group3 = ("mech", "beast", "bug")

        axie_points = [0 for _ in range(3)]
        for ii, axie in enumerate(self.team):
            if axie.element in group1:
                for card in axie.cards:
                    if card in group1 or card in group2:
                        axie_points[ii] += 1
            elif axie.element in group2:
                for card in axie.cards:
                    if card in group2 or card in group3:
                        axie_points[ii] += 1
            elif axie.element in group3:
                for card in axie.cards:
                    if card in group3 or card in group1:
                        axie_points[ii] += 1

        return {
            "plant/reptile": axie_points[0],
            "aqua/bird": axie_points[1],
            "beast/bug": axie_points[2],
        }

    def get_deck_evaluation(self):

        """
        TODO return scores in comparison to average meta deck from top 100
        TODO

        :return:
        """

        results = dict()
        raw_data = dict()

        # add hints in case of shortcomings in certain categories
        hints = []  # (importance, hint)

        results["hp_score"] = sum([axie.hp/Axie.hp_multi for axie in self.team]) / 183
        raw_data["hp"] = {
            "team": sum([xe.hp for xe in self.team]),
            "front": self.team[0].base_hp,
            "mid": self.team[1].base_hp,
            "back": self.team[2].base_hp,
        }

        results["damage_score"] = sum([sum([card.attack for card in axie.cards]) for axie in self.team]) / (540 * 3)
        results["shield_score"] = sum([sum([card.defense for card in axie.cards]) for axie in self.team]) / (375 * 3)
        results["speed_score"] = sum([axie.speed for axie in self.team]) / 183

        # dmg per type

        # 1 - (result / 133); result is diff of each type from 33%
        dmg_per_elem = self.get_abs_dmg_per_elem()
        g1, g2, g3 = dmg_per_elem["plant/reptile"], dmg_per_elem["aqua/bird"], dmg_per_elem["beast/bug"]
        rel = [elem / (g1 + g2 + g3) for elem in (g1, g2, g3)]

        results["damage_diversity_score"] = 1 - (sum([abs(elem - 0.33) for elem in rel]) / 1.33)
        raw_data["dmg_per_elem"] = dmg_per_elem

        # dmg type per axie TODO overthink this

        # axie_points = list(self.get_dmg_type_per_axie().values())
        # only use assumtion of 3 dmg cards per axie to not rule out utility cards
        # raw_data["dmg_distribution_score"] = sum([min(1.0, axie_points[ii] / 3) for ii in range(3)]) / 3

        # shield distribution
        shield_per_axie = [sum([c.defense for c in axie.cards]) for axie in self.team]
        cum_shield = sum(shield_per_axie)

        raw_data["shield_distribution"] = {
            "team": cum_shield,
            "front": shield_per_axie[0],
            "mid": shield_per_axie[1],
            "back": shield_per_axie[2]
        }

        # <editor-fold desc="Relative to META">

        # energy gain
        ...

        # energy steal
        ...

        # backdoors (types)
        ...

        # draw power
        ...

        # disables
        ...

        # buffs / debuffs
        ...

        # 1 zero cost per axie
        ...

        # </editor-fold>

        return results, raw_data

    # ---

    def plot_deck_stats_overview(self):

        plt.style.use('Solarize_Light2')

        fig = plt.figure()

        ax1 = fig.add_subplot(221, projection="polar")
        ax2 = fig.add_subplot(222)
        # ax3 = fig.add_subplot(223)
        ax4 = fig.add_subplot(224)

        eval_results, raw_data = self.get_deck_evaluation()
        top_ladder_eval_results, top_ladder_raw_data = Player.get_top_ladder_deck_evaluations()

        # TODO add meta average
        Player.radar_plot(ax1, "Deck scores", Player.deck_eval_to_dict([eval_results, top_ladder_eval_results]))

        Player.bar_plot_dmg_types_distribution(ax2, "Your deck: Damage types/shield",
                                               Player.dmg_types_to_plot_format(raw_data["dmg_per_elem"],
                                                                               raw_data["shield_distribution"],
                                                                               raw_data["hp"],
                                                                               self.team))

        Player.bar_plot_dmg_types_distribution(ax4, "Top ladder: Damage types/shield",
                                               Player.dmg_types_to_plot_format(top_ladder_raw_data["dmg_per_elem"],
                                                                               top_ladder_raw_data["shield_distribution"],
                                                                               top_ladder_raw_data["hp"]))

        #plt.get_current_fig_manager().window.showMaximized()
        fig.tight_layout()
        plt.tight_layout()

        plt.show()

    @staticmethod
    def dmg_types_to_plot_format(dmg_types, shield_dist, hp_dist, team=None):
        deck_dmg_types = dmg_types
        dmg_type_per_axie = deck_dmg_types["per_axie"]

        return [
            [deck_dmg_types["plant/reptile"], dmg_type_per_axie[0][0], dmg_type_per_axie[1][0], dmg_type_per_axie[2][0]],
            [deck_dmg_types["aqua/bird"], dmg_type_per_axie[0][1], dmg_type_per_axie[1][1], dmg_type_per_axie[2][1]],
            [deck_dmg_types["beast/bug"], dmg_type_per_axie[0][2], dmg_type_per_axie[1][2], dmg_type_per_axie[2][2]],
            [xe.element for xe in team] if team else ["N/A" for _ in range(3)],
            [shield_dist["team"], shield_dist["front"], shield_dist["mid"], shield_dist["back"]],
            [hp_dist["team"], hp_dist["front"], hp_dist["mid"], hp_dist["back"]]
        ]

    @staticmethod
    def bar_plot_dmg_types_distribution(axes, name, data):
        """
        :param name:
        :param axes:
        :param data: [[team.plant, team.beast, team.aqua],
                      [front.plant, front.beast, front.aqua, front.axie_type],
                      ...]
        :return:
        """
        type_to_color = {
            "plant": '#006600',
            "reptile": '#9933ff',
            "beast": '#f0c800',
            "bug": '#ff3300',
            "bird": '#ff99cc',
            "aqua": '#ccffff',
            "N/A": '#888888'
        }

        X = arange(4)

        axes.set_title(name)
        #axes.get_xaxis().set_visible(False)

        from pprint import pprint as pp
        pp(data)

        # TODO this is buggy

        axes.bar(X - 0.3, data[0], color=type_to_color["plant"], width=0.2, label="Plant/Reptile")
        axes.bar(X - 0.1, data[1], color=type_to_color["aqua"], width=0.2, label="Aqua/Bird")
        axes.bar(X + 0.1, data[2], color=type_to_color["beast"], width=0.2, label="Beast/Bug")
        axes.bar(X + 0.30625, data[5], color="#00cc00", width=0.075, label="HP")  # aqua
        axes.bar(X + 0.38125, data[4], color="#0066cc", width=0.075, label="Shield")  # shield

        axes.bar(1, -20, color=type_to_color[data[3][0]], width=0.9)  # axie types
        axes.bar(2, -20, color=type_to_color[data[3][1]], width=0.9)  # axie types
        axes.bar(3, -20, color=type_to_color[data[3][2]], width=0.9)  # axie types

        axes.set_xticks(list(range(4)))
        axes.set_xticklabels(["Team", "Front", "Mid", "Back"])

        axes.legend(fancybox=True, shadow=True)


        """rects = axes.patches
        labels = ["Front", "Mid", "Back"]
        for rect, label in zip(rects[-3:], labels):
            height = rect.get_height()
            axes.text(rect.get_x() + rect.get_width() / 2, height - 20, label, ha="center", va="bottom")"""

        return axes

    @staticmethod
    def radar_plot(axes, name, input_data):
        """
        :param axes:
        :param name:
        :param input_data: {A: [attack_scores], B: [shield_scores], ...}
        :return:
        """

        num_data_samples = len(list(input_data.values())[0])
        data = [[] for _ in range(num_data_samples)]
        categories = []
        for key, val in input_data.items():
            for i in range(len(val)):
                data[i].append(val[i])
            categories.append(key.capitalize().replace("_", " "))

        categories = [*categories, categories[0]]

        prep_data = [[] for _ in range(num_data_samples)]
        for i in range(num_data_samples):
            prep_data[i] = [*data[i], data[i][0]]

        # ---------------------------------------------

        label_loc = linspace(start=0, stop=2*pi, num=len(categories))
        axes.set_title(name, y=1.05)

        lines, labels = axes.set_thetagrids(degrees(label_loc), labels=categories)
        axes.set_ylim(bottom=0, top=1)
        axes.set_rgrids(arange(0, 1, step=0.2))

        for i in range(num_data_samples):
            axes.plot(label_loc, prep_data[i],
                      label=f'{"Your deck" if i == 0 else "Top ladder average"}',
                      color="#888888" if i == 1 else "#00cc00")

        axes.legend(fancybox=True, shadow=True, bbox_to_anchor=(1.5, 1.0))

        return axes

    @staticmethod
    def deck_eval_to_dict(evals):

        if isinstance(evals, dict):
            return {key: [val] for key, val in evals.items()}

        elif isinstance(evals, list):
            keys = evals[0].keys()
            values = [[] for _ in range(len(keys))]
            for entry in evals:
                for i, key in enumerate(keys):
                    values[i].append(entry[key])
            return {key: val for key, val in zip(keys, values)}


class Match:
    cards_per_round = 3
    start_hand = 6
    energy_per_turn = 2

    def __init__(self, player1, player2):
        self.player1 = player1
        self.player2 = player2
        self.round = 0
        self.chain_cards = []  # cards chained with currently resolving card
        self.axies_in_attack_order = None

    def run_simulation(self):

        debug(f'Starting simulation [{self.player1} vs. {self.player2}]')

        self.player1.reset()
        self.player2.reset()
        self.round = 1

        debug("\nPlayer teams:")

        for axie in self.player1.team:
            debug(f'{axie.long()}')
        debug()
        for axie in self.player2.team:
            debug(f'{axie.long()}')

        debug()

        round_counter = 0

        while self.game_running():
            self.run_round(round_counter)
            debug(f'\nRound {round_counter + 1} finished\n')
            round_counter += 1

        res = self.get_result()

        self.player1.reset()
        self.player2.reset()
        self.round = 0

        return res

    def game_running(self):
        return self.player1.team_alive() and self.player2.team_alive()

    def run_round(self, round_counter):

        # tODO if round > sth apply auto dmg to axies
        if round_counter >= 10:
            for axie in self.player1.team + self.player2.team:
                if axie.alive():
                    axie.change_hp(-((round_counter - 10) * 30 + 20))

            if not self.game_running():
                return

        # start round, reset shield to 0, remove buffs/debuffs \ poison, draw cards
        debug("Player 1 hand:")
        self.player1.start_turn(self.cards_per_round if round_counter > 0 else self.start_hand,
                                self.energy_per_turn if round_counter > 0 else 0)
        debug("Player 2 hand:")
        self.player2.start_turn(self.cards_per_round if round_counter > 0 else self.start_hand,
                                self.energy_per_turn if round_counter > 0 else 0)

        debug("Player 1 cards:")
        cards_p1 = self.player1.select_cards(self)
        debug("\nPlayer 2 cards:")
        cards_p2 = self.player2.select_cards(self)

        for card in cards_p1.values():
            self.player1.discard_pile += card
        for card in cards_p2.values():
            self.player2.discard_pile += card

        cards_p1.update(cards_p2)

        self.fight(cards_p1)

        for axie in self.player1.team + self.player2.team:
            if axie.alive():
                axie.shield_broke_this_turn = False

    def get_result(self):

        """
        :return: [-1, 1]; 1 if player 1 wins with full team hp, -1 if player 2 wins with full team hp
        """

        if not (self.player1.team_alive() or self.player2.team_alive()):
            debug("Game ended in a draw")
            return 0  # draw

        if self.player1.team_alive():
            debug("Player 1 wins!")
            return self.player1.get_relative_team_hp()

        if self.player2.team_alive():
            debug("Player 2 wins!")
            return -self.player2.get_relative_team_hp()

    def fight(self, cards_to_play):

        # determine attack order based on current stats
        # (debuffs, buffs, e.g. aroma etc, effects e.g. always attack first)

        flat_cards_to_play = flatten(list(cards_to_play.values()))
        self.axies_in_attack_order = self.get_axies_in_attack_order(cards_to_play)

        if not self.axies_in_attack_order:
            return

        debug(f'\nAxies in attack order:')
        for x in self.axies_in_attack_order:
            t = "\t" if x.player is self.player1 else "\t\t"
            debug(f'{t}{x}: {cards_to_play[x]}')

        for card in flat_cards_to_play:
            card.on_start_turn(self, cards_to_play)

        # apply shield
        for axie in self.axies_in_attack_order:

            for card in cards_to_play[axie]:

                if axie.alive():

                    shield_to_give = card.defense
                    shield_mult = 1

                    if shield_to_give > 0:

                        debug()

                        # element bonus
                        if axie.element == card.element:
                            shield_mult += 0.1
                            debug(f'Shield mult raised by 0.1 due to element synergy')

                        # if the card is in a chain with another of same element
                        if card.element in [c.element for c in flatten([cards_to_play[xe] for xe in axie.player.team])]:
                            shield_mult += 0.05
                            debug(f'\nShield mult raised by 0.05 due to chain bonus')

                        axie.shield += int(shield_mult * shield_to_give)
                        debug(f'{axie} shield raised by {shield_mult:.2f} * {shield_to_give} = '
                              f'{int(shield_mult * shield_to_give)}')

        for axie in self.axies_in_attack_order:

            if not cards_to_play[axie] or not axie.alive() or not self.game_running():
                continue

            owner = None
            opp = None
            if axie.player is self.player1:
                owner = self.player1
                opp = self.player2
            if axie.player is self.player2:
                owner = self.player2
                opp = self.player1

            # ----------------------------------------------------

            def select_attack_target():

                to_omit = set()
                focus = []

                # card effect
                for _c in flat_cards_to_play:
                    _omit, _focus = _c.on_target_select(self, cards_to_play, axie, opp.team[0])
                    to_omit.union(_omit)
                    focus = _focus + focus
                    
                # status effects
                """for xe in opp.team:
                    if xe.alive():
                        for buff, count in xe.buffs.items():
                            if buff == "aroma":
                                focus.append(xe)
                            if buff == "stench":
                                to_omit.add(xe)"""
                for xe in opp.team:
                    if xe.alive():
                        foc, om = xe.buffs.on_target_select()
                        if foc:
                            focus = [xe] + focus
                        if om:
                            to_omit.add(xe)

                focus = [elem for elem in focus if elem.alive()]
                to_omit = {elem for elem in to_omit if elem.alive()}

                # default select = closest

                # if focus, attack focused
                for _xe in focus:
                    if _xe not in to_omit:
                        return _xe

                # attack closest (teams are list in order front to back)
                for _xe in opp.team:
                    if _xe not in to_omit and _xe.alive():
                        return _xe

                return None  # no valid target present

            # select attack target
            attack_target = select_attack_target()

            debug("_" * 350)
            debug(f'\n>>{axie}<<\nattacking\n>>{attack_target}<<\nwith\n{cards_to_play[axie]}\n')

            def perform_attack(attacking_card, attacker, defender):
                debug(f'Resolving: [{attacking_card}]')

                base_atk = attacking_card.attack
                atk_multi = 1
                skip_dmg_calc = False
                dmg_reduction = 0  # for this like gecko

                """if base_atk > 0 and "fear" in attacker.buffs.keys() and attacker.buffs["fear"][0] > 0:
                    skip_dmg_calc = True
                    attacker.reduce_stat_eff("fear", 1)
                    debug(f'Skipping damage calculation because attacker is feared')"""
                if not attacker.buffs.can_hit():
                    skip_dmg_calc = True
                    debug(f'Skipping damage calculation because attacker is feared/stunned')

                if not skip_dmg_calc and attack_target and attack_target.alive():

                    # TODO think of this
                    # miss = False
                    double_shield_dmg = False
                    true_dmg = False
                    crit_multi = 2
                    crit_disable = False

                    # element syn card / attacker
                    if attacking_card.element == attacker.element:
                        atk_multi += 0.1
                        debug(f'Atk multi set to {atk_multi:.2f} due to element synergy')

                    # element atk / def
                    atk_multi += get_dmg_bonus(attacking_card.element, defender.element)
                    debug(f'Atk multi set to {atk_multi:.2f} due to rpc system')

                    # buff / debuffs atk
                    """for buff, (stacks, rounds) in attacker.buffs.items():
                        for s in range(stacks):  # TODO maybe +1 bc as long as stacks not <0 it is not removed

                            if buff == "attack down":
                                atk_multi -= 0.2
                                attacker.reduce_stat_eff("attack down", 1)
                                debug(f'Atk multi set to {atk_multi:.2f} due to attack down')
                            if buff == "stun":
                                miss = True
                                attacker.reduce_stat_eff("stun", "all")
                                debug("Attack missed due to stun")
                            if buff == "attack up":
                                atk_multi += 0.2
                                attacker.reduce_stat_eff("attack up", 1)
                                debug(f'Atk multi set to {atk_multi:.2f} due to attack up')

                            if buff == "morale up":
                                morale_multi += 0.2
                                # attacker.reduce_stat_eff("morale up", 1)
                            if buff == "morale down":
                                morale_multi -= 0.2
                                # attacker.reduce_stat_eff("morale down", 1)"""
                    atk_multi_attacker, morale_multi_attacker, is_lethal, has_jinx = attacker.buffs.on_atk()
                    atk_multi += atk_multi_attacker
                    morale_multi = 1 + morale_multi_attacker
                    crit_disable = crit_disable or has_jinx

                    # TODO confirm calculation
                    if not is_lethal:
                        crit_prob = (axie.morale * morale_multi) / 457.0
                    else:
                        crit_prob = 1

                    # buff / debuff def
                    """for buff, (stacks, rounds) in defender.buffs.items():
                        for s in range(stacks):

                            if buff == "fragile":
                                double_shield_dmg = True
                                defender.reduce_stat_eff("fragile", "all")
                            if buff == "stun":
                                true_dmg = True
                                defender.reduce_stat_eff("stun", "all")"""
                    double_shield_dmg, sleeping, stunned_def = defender.buffs.on_defense()
                    true_dmg = sleeping or stunned_def

                    self.comb_cards = cards_to_play[attacker]

                    # on chain effects
                    self.chain_cards = []
                    for xe in owner.team:
                        if xe.alive() and xe is not axie:
                            other_cards = cards_to_play.get(xe, None)
                            if other_cards and attacking_card.element in [c.element for c in other_cards]:
                                self.chain_cards += cards_to_play[xe]
                    """
                    if len(chain_cards) > 0:
                        attacking_card.on_chain(self, cards_to_play, chain_cards)
                    """

                    # card effects atk
                    for _card in flat_cards_to_play:
                        if _card is card:
                            y = _card.on_attack(self, cards_to_play, attacker, defender, card)
                            atk_multi += y
                            if y != 0:
                                debug(f'Atk multi set to {atk_multi:.2f} due to {_card.part_name} effect')

                    # card effects def (only gecko right now?)
                    for _card in flat_cards_to_play:
                        dmg_reduction = _card.on_defense(self, cards_to_play, attacker, defender, card)

                    # combos
                    if len(self.comb_cards) > 1:
                        combo_bonus = int(base_atk * atk_multi * attacker.skill / 500)
                    else:
                        combo_bonus = 0
                    debug(f'Combo bonus: {combo_bonus}')

                    # calc damage now
                    atk = (base_atk * atk_multi + combo_bonus) * (1 - dmg_reduction)

                    # attacks on dead never hit
                    if attack_target.alive():

                        # pre crit effects, adjust crit dmg or disable crits
                        for _card in flat_cards_to_play:
                            _crit_multi, _crit_disable, _crit_prob = _card.pre_crit(self, cards_to_play, attacker,
                                                                                    defender)
                            crit_multi *= _crit_multi
                            crit_disable = crit_disable or _crit_disable
                            crit_prob += _crit_prob

                        # determine crit / normal hit
                        if not crit_disable and random() <= crit_prob:

                            # on_crit effects
                            for _card in flat_cards_to_play:
                                _card.on_crit(self, cards_to_play, attacker, defender)

                            atk *= crit_multi

                            debug(f'Critical hit for {int(atk)}')

                        defender.apply_damage(self, cards_to_play, attacker, atk, double_shield_dmg, true_dmg, card)

                    else:
                        debug("Attack missed!")

                        # tick actions
                        for xe in self.axies_in_attack_order:
                            if xe.alive():
                                xe.action_tick()

                        return False

                # apply card effect
                attacking_card.on_play(self, cards_to_play)

                # tick actions
                for xe in self.axies_in_attack_order:
                    if xe.alive():
                        xe.action_tick()

                return True

            # ------------------------------------------------------

            for card in cards_to_play[axie]:

                if self.game_running() and attack_target:

                    attack_times = 1
                    for c in flat_cards_to_play:
                        t = c.attack_times(self, cards_to_play, axie, attack_target)
                        if t:
                            attack_times += (t - 1)

                    missed = False
                    for _ in range(attack_times):

                        if missed:
                            continue

                        if self.game_running() and (attack_target.alive() or card.attack == 0):
                            missed = not perform_attack(card, axie, attack_target)

                        debug(f'\tAttack target after attack: {attack_target}')
                        debug(f'\t     Attacker after attack: {axie}\n')

                    for xe in self.player1.team+self.player2.team:
                        if xe.alive():
                            xe.buffs.on_action()

            # reset "once per round" effects
            for card in flat_cards_to_play:
                card.triggered = False

    def get_axies_in_attack_order(self, cards_to_play) -> List[Axie]:

        if not cards_to_play:
            return []

        axies = [axie for axie in cards_to_play.keys() if axie.alive() and cards_to_play.get(axie, None)]

        init_speeds = [(axie, axie.speed) for axie in axies]
        final_speeds = {axie: 0 for axie in axies}

        for axie, speed in init_speeds:
            # status effects
            """for buff, (count, turns) in axie.buffs.items():
                if buff == "speed up":
                    speed_mult += 0.2 * count
                if buff == "speed down":
                    speed_mult -= 0.2 * count"""
            sp, mor = axie.buffs.on_determine_order()

            # this breaks ties by hp & morale
            final_speeds[axie] = (speed * (1 + sp), axie.hp, axie.morale * (1 + mor))

        ranked_axies = sorted(axies, key=lambda axie: final_speeds[axie], reverse=True)

        always_first = []
        for axie in axies:
            for card in cards_to_play[axie]:
                if card.on_determine_order():
                    always_first.append(axie)
                    ranked_axies.remove(axie)


        return always_first + ranked_axies


class Agent(ABC):

    @abstractmethod
    def select(self, player: Player, game_state: Match) -> dict:
        """

        :param player:
        :param game_state:
        :return:
        """
        pass

    @abstractmethod
    def sinle_select(self, player: Player, game_state: Match):
        pass


class HumanAgent(Agent):

    def select(self, player, game_state) -> dict:
        print("\nEnergy:", player.energy)
        print("Select your cards (e.g. '1,2,5'):")
        n = 1
        number_to_card = {i + 1: None for i in range(len(flatten(player.hand.values())))}
        for axie, cards in player.hand.items():
            hand_card_string = ""
            for card in cards:
                hand_card_string += f'[{n}] {card} '
                number_to_card[n] = card
                n += 1
            disability_string = ','.join([f'<{dis} for {turns} turns>' for dis, turns in axie.disabilities.items()])
            print(f'{axie} (Disabilities: {disability_string}) - {hand_card_string}')

        inp = input("> ")
        if len(inp) > 0:
            selected = [int(x) for x in inp.split(",")]
            return {axie: [number_to_card[x] for x in selected if number_to_card[x].owner is axie] for axie in
                    player.team}
        else:
            return {axie: [] for axie in player.team}

    def sinle_select(self, game_state, player):
        pass
