from abc import abstractmethod, ABC
from typing import List, Dict
from random import shuffle, sample, random


def flatten(t):
    return [item for sublist in t for item in sublist]


type_to_stats = { # hp, speed, skill, morale
    "Beast": (31, 35, 31, 43),
    "Aqua": (39, 39, 35, 27),
    "Plant": (43, 31, 31, 35),
    "Dusk": (43, 39, 27, 31),
    "Mech": (31, 39, 43, 27),
    "Reptile": (39, 35, 31, 35),
    "Bug": (35, 31, 35, 39),
    "Bird": (27, 43, 35, 35),
    "Dawn": (35, 35, 39, 31)
}

part_to_stats = { # hp, speed, skill, morale
    "Beast": (0, 1, 0, 3),
    "Aqua": (1, 3, 0, 0),
    "Plant": (3, 0, 0, 1),
    "Reptile": (3, 1, 0, 0),
    "Bug": (1, 0, 0, 3),
    "Bird": (0, 3, 0, 1),
}

poison_dmg_per_stack = 2


def get_dmg_bonus(attacker, defender):

    group1 = ("Reptile", "Plant", "Dusk")
    group2 = ("Aqua", "Bird", "Dawn")
    group3 = ("Mech", "Beast", "Bug")

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

    def __init__(self, card_name, body_part, part_name, element, cost, attack, defense, effect):
        self.card_name = card_name
        self.body_part = body_part
        self.part_name = part_name
        self.element = element
        self.cost = cost
        self.attack = attack
        self.defense = defense
        self.effect = effect

        self.owner = None

    def on_play(self, match, cards_played_this_turn):
        pass

    def on_comb(self, match, cards_played_this_turn, combo_cards):
        pass

    def on_chain(self, match, cards_played_this_turn, chain_cards):
        pass

    def on_start_turn(self, match, cards_played_this_turn):
        pass

    def on_target_select(self, match, cards_played_this_turn, attacker, defender):  # returns to_omit, focus (sets)
        return set(), set()

    def attack_times(self, match, cards_played_this_turn, attacker, defender):
        return 1

    def on_attack(self, match, cards_played_this_turn, attacker, defender):
        pass

    def on_shield_break(self, match, cards_played_this_turn, attacker, defender):
        pass

    def on_damage_inflicted(self, match, cards_played_this_turn, attacker, defender, amount):
        pass

    def on_pre_last_stand(self, match, cards_played_this_turn, attacker, defender, amount):
        pass

    def on_end_turn(self, match, cards_played_this_turn):
        pass


class Axie:

    def __init__(self, element: str, card1: Card, card2: Card, card3: Card, card4: Card, eyes: str, pattern: str):

        self.player = None

        self.element = element
        self.mouth = card1
        card1.owner = self
        self.horn = card2
        card2.owner = self
        self.back = card3
        card3.owner = self
        self.tail = card4
        card4.owner = self
        self.eyes = eyes
        self.pattern = pattern

        self.cards = {card1, card2, card3, card4}

        bhp, bsp, bsk, bm = type_to_stats[self.element]

        for card in self.cards:
            php, psp, psk, pm = part_to_stats[card.element]
            bhp += php
            bsp += psp
            bsk += psk
            bm += pm

        ehp, esp, esk, em = part_to_stats[eyes]
        php, psp, psk, pm = part_to_stats[eyes]

        bhp += php + ehp
        bsp += psp + esp
        bsk += psk + esk
        bm += pm + em

        # if axie is idle and not in battle, element stats + card stats included
        self.base_hp = bhp
        self.base_morale = bm
        self.base_speed = bsp
        self.base_skill = bsk

        self.buffs = dict()  # buff name: stacks, turns remaining
        self.buff_q = dict()
        self.disabilities = dict()  # type to disable: number of turns, e.b. mouth: 1, auqa: 1

        self.hp = self.base_hp
        self.morale = self.base_morale
        self.skill = self.base_skill
        self.morale = self.base_morale

        self.last_stand = False
        self.last_stand_ticks = None  # TODO does this depend on HP?

        self.shield = 0

    def turn_tick(self):  # TODO each status effect should have a life time that ticks away

        # tick status effects (buffs, debuffs)
        for stat_eff, (times, turns) in self.buffs.items():
            if turns - 1 < 0:
                del self.buffs[stat_eff]
            else:
                self.buffs[stat_eff] = (times, turns-1)

        # apply q
        for buff, (times, turns) in self.buff_q.items():
            ti, tu = self.buffs.get(buff, (0, 0))
            self.buffs[buff] = (ti + times, tu + turns)
        self.buff_q = dict()

        # tick disabilities (body parts, certain element cards)
        for dis in self.disabilities.keys():
            if self.disabilities[dis] - 1 < 0:
                del self.disabilities[dis]
            else:
                self.disabilities[dis] = self.disabilities[dis] - 1

    def action_tick(self):

        # tODO
        # tick last stand down
        # tick poison
        ...

    def apply_stat_eff(self, effect, turns, next_turn=False):  # which effects stack? if 2x fear for "next turn" does it stack to 2?

        """
        :param effect:
        :param turns:
        :param next_turn: true if effect is applied next n turns
        :return:
        """

        if next_turn:
            self.buffs.update({effect: max(self.buffs.get(effect, 0), 1)})
        else:  # stack
            self.buffs.update({effect: self.buffs.get(effect, 0) + turns})

    def alive(self):
        return self.hp > 0 or self.last_stand

    def reset(self):
        self.hp = self.base_hp
        self.morale = self.base_morale
        self.skill = self.base_skill
        self.morale = self.base_morale
        self.buffs = dict()


class Player:

    start_energy = 3
    cards_per_turn = 3
    start_cards = 6

    def __init__(self, team: List[Axie]):
        assert len(team) == 3
        self.team = team  # team in order front to back line for simplicity
        self.deck = list()

        for axie in team:
            self.deck += list(axie.cards) + list(axie.cards)
            axie.player = self

        self.energy = self.start_energy
        self.deck_pile = self.deck
        self.discard_pile = list()
        self.hand = dict()

    def start_turn(self, n):

        for axie in self.team:
            if axie.alive():
                axie.turn_tick()

        self.draw_cards(n)

    def team_alive(self):
        for axie in self.team:
            if axie.alive():
                return True
        return False

    def draw_cards(self, n):
        shuffle(self.deck_pile)
        drawn_cards = []
        for i in range(n):
            drawn_cards.append(self.deck_pile.pop())
        self.hand.update({card.owner: card for card in drawn_cards})

        # TODO discard when too many

    def select_cards(self) -> Dict:
        return self.random_select()

    def random_select(self) -> Dict:

        hand_cards = flatten(list(self.hand.values()))
        hand_card_number = len(hand_cards)

        c = 0
        used_energy = 0
        energy_this_turn = self.energy

        cards_to_play = {axie: [] for axie in self.hand.keys()}
        cards_per_axie = {axie: 0 for axie in self.hand.keys()}

        while c < hand_card_number and used_energy < energy_this_turn:

            if random.random() > 1 / hand_card_number \
                    and hand_cards[c].cost <= energy_this_turn-used_energy \
                    and cards_per_axie[hand_cards[c].owner] < 4\
                    and cards_per_axie[hand_cards[c].element] not in hand_cards[c].owner.disabilities.keys():

                cards_to_play[hand_cards[c].owner].append(hand_cards[c])
                used_energy += hand_cards[c].cost
                cards_per_axie[hand_cards[c].owner] += 1
            c += 1

        return cards_to_play


class Event:

    def __init__(self, etype, attacker=None, defender=None):
        # attack(a,b), shield_break, end_turn, start_turn, target_select
        self.etype = etype
        self.attacker = attacker
        self.defender = defender


class Match:

    cards_per_round = 3

    def __init__(self, player1, player2):
        self.player1 = player1
        self.player2 = player2
        self.round = 1

    def run_round(self):

        # tODO if round > sth apply auto dmg to axies

        # start round, reset shield to 0, remove buffs/debuffs \ poison, draw cards
        self.player1.start_turn(self.cards_per_round)
        self.player2.start_turn(self.cards_per_round)

        cards_p1 = self.player1.select_cards()
        cards_p2 = self.player2.select_cards()

        cards_to_play = cards_p1.update(cards_p2)

        self.fight(cards_to_play)

    def fight(self, cards_to_play):

        # determine attack order based on current stats
        # (debuffs, buffs, e.g. aroma etc, effects e.g. always attack first)

        axies_in_attack_order = self.determine_order(cards_to_play)

        # apply shield
        for axie in axies_in_attack_order:
            # TODO add shield bonus of 5% if multiple same element cards on different axies
            for card in cards_to_play[axie]:
                if axie.alive():
                    axie.shield += card.defense

        for axie in axies_in_attack_order:

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
                focus = set()

                # status effects
                for xe in opp.team:
                    for buff, count in xe.buffs.items():
                        if buff == "aroma":
                            focus.add(xe)
                        if buff == "stench":
                            to_omit.add(xe)

                # card effect
                for _c in list(cards_to_play.values()):
                    _omit, _focus = _c.on_target_select(self, cards_to_play, axie, attack_target)
                    to_omit += _omit
                    focus += _focus

                # default select = closest

                # if focus, attack focused
                if focus and focus - to_omit:
                    return focus.pop()

                # attack closest (teams are list in order front to back)
                for _xe in opp.team:
                    if _xe not in to_omit and axie.alive():
                        return _xe

                return None  # no valid target present

            # select attack target
            attack_target = select_attack_target()

            def perform_attack(attacking_card, attacker, defender):
                base_atk = attacking_card.attack
                atk_multi = 1
                skip_dmg_calc = False

                if base_atk > 0 and "fear" in attacker.buffs.keys() and attacker.buffs["fear"] > 0:
                    skip_dmg_calc = True

                if not skip_dmg_calc:

                    # TODO think of this
                    miss = False
                    double_shield_dmg = False
                    true_dmg = False

                    # element syn card / attacker
                    if attacking_card.element == attacker.element:
                        atk_multi += 0.1

                    # element atk / def
                    atk_multi += get_dmg_bonus(attacker.element, defender.element)

                    # buff / debuffs atk
                    for buff, (stacks, rounds) in attacker.buffs.items():
                        for s in stacks:

                            if buff == "attack down":
                                atk_multi -= 0.2
                            if buff == "stun":
                                miss = True
                            if buff == "attack up":
                                atk_multi += 0.2

                            # for "next hit" effects
                            attacker.buffs[buff] = (stacks - 1, rounds)

                    # buff / debuff def
                    for buff, (stacks, rounds) in defender.buffs.items():
                        for s in stacks:
                            if buff == "fragile":
                                double_shield_dmg = True
                            if buff == "stun":
                                true_dmg = True

                            defender.buffs[buff] = (stacks-1, rounds)

                    # on combo effects
                    if len(cards_to_play[attacker]) > 1:
                        attacking_card.on_combo(self, cards_to_play, cards_to_play[attacker])

                    # on chain effects
                    chain_cards = []
                    for xe in owner.team:
                        if xe.alive():
                            other_card = cards_to_play.get(xe, None)
                            if other_card and other_card.element == attacking_card.element:
                                chain_cards += cards_to_play[xe]
                    if len(chain_cards) > 0:
                        attacking_card.on_chain(self, cards_to_play, chain_cards)

                    # card effects atk
                    for _card in cards_to_play:
                        dmg_multi = _card.on_attack(self, cards_to_play, attacker, defender)
                        atk_multi += dmg_multi

                    # card effects def

                    # combos
                    combo_bonus = int(base_atk * atk_multi * attacker.skill / 500)

                    # crits, dodge/miss

                    # last stand

                    # on_attack, on_shield_break, on_last_stand

                # apply card effect
                attacking_card.on_play(self, cards_to_play)

                # tick actions
                for xe in axies_in_attack_order:
                    if xe.alive():
                        xe.action_tick()

                return

            # ------------------------------------------------------

            for card in cards_to_play[axie]:

                attack_times = 1
                for c in list(cards_to_play.values()):
                    t = c.attack_times(self, cards_to_play, axie, attack_target)
                    if t:
                        attack_times += t

                for _ in range(attack_times):
                    perform_attack(card, axie, attack_target)

    def determine_order(self, cards_to_play) -> List[Axie]:

        # TODO
        #  ...

        return []