from abc import abstractmethod, ABC
from typing import List, Dict, Tuple
from random import shuffle, sample, random, choice
from copy import deepcopy, copy as shallowcopy
from functools import reduce
from traceback import print_exc


def debug(s=""):
    print(s)


def flatten(t):
    out = []
    for item in t:
        if isinstance(item, (list, tuple)):
            out.extend(flatten(item))
        else:
            out.append(item)
    return out


type_to_stats = { # hp, speed, skill, morale
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

part_to_stats = { # hp, speed, skill, morale
    "beast": (0, 1, 0, 3),
    "aqua": (1, 3, 0, 0),
    "plant": (3, 0, 0, 1),
    "reptile": (3, 1, 0, 0),
    "bug": (1, 0, 0, 3),
    "bird": (0, 3, 0, 1),
}

poison_dmg_per_stack = 2
miss_prob = 0.025
last_stand_tick_hp = 30  # this is wrong, it's always inf
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
        self.card_name = card_name
        self.body_part = body_part
        self.part_name = part_name
        self.element = element
        self.cost = cost
        self.attack = attack
        self.defense = defense
        self.effect = effect
        self.range = r

        self.owner: Axie = None

    def __str__(self):
        return f'{type(self).__name__} ({self.attack}/{self.defense})'

    def __repr__(self):
        return self.__str__()

    def detail(self):
        return f'{self.card_name} - {self.part_name} ({self.body_part}, {self.range})\n\t' \
            f'{self.element} - Cost: {self.cost}, Attack: {self.attack}, Defense: {self.defense}\n\t' \
            f'Effect: {self.effect}'

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
        return set(), set()

    def attack_times(self, match, cards_played_this_turn, attacker, defender):
        """

        :param match:
        :param cards_played_this_turn:
        :param attacker:
        :param defender:
        :return: number of attacks to perform upon play
        """
        return 1

    def on_attack(self, match, cards_played_this_turn, attacker, defender) -> float:
        """

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

    def on_shield_break(self, match, cards_played_this_turn, attacker, defender):
        """

        :param match:
        :param cards_played_this_turn:
        :param attacker:
        :param defender:
        :return: None
        """
        pass

    def on_damage_inflicted(self, match, cards_played_this_turn, attacker, defender, amount):
        """

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

    hp_multi = 8.72

    @staticmethod
    def rand_elem(rare_types=False):
        if not rare_types:
            return sample(("beast", "bird", "bug", "plant", "aqua", "reptile"), 1)[0]
        return sample(("beast", "bird", "bug", "plant", "aqua", "reptile", "dusk", "dawn", "mech"), 1)[0]

    @staticmethod
    def get_random():
        from Axie.cards import get_all_card_classes
        all_cc = get_all_card_classes()
        return Axie(Axie.rand_elem(), sample(all_cc["mouth"], 1)[0](), sample(all_cc["horn"], 1)[0](),
                    sample(all_cc["back"], 1)[0](), sample(all_cc["tail"], 1)[0]())

    def __init__(self, element: str, mouth: Card, horn: Card, back: Card, tail: Card, eyes=None, ears=None):

        self.player: Player = None

        self.element = element if element else self.rand_elem(rare_types=True)

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

        self.buffs = dict()  # buff name: stacks, turns remaining
        self.buff_q = dict()
        self.disabilities = dict()  # type to disable: number of turns, e.b. mouth: 1, auqa: 1
        self.apply_stat_eff_changes = []

        self.hp = self.base_hp
        self.speed = self.base_speed
        self.skill = self.base_skill
        self.morale = self.base_morale

        self.last_stand = False
        self.last_stand_ticks = 1 if self.base_morale <= 38 else 2 if self.base_morale <= 49 else 3

        self.shield = 0

    def __repr__(self):
        hp_stat = f'(H:{self.hp}+{int(self.shield)}/{self.base_hp}, S:{self.speed})' if self.alive() else f'(dead)'
        stand_stat = f'L {self.last_stand_ticks}'
        return f'{self.element.capitalize()} Axie #{str(id(self))[-4:]} [{str(self.player)}] {hp_stat if not self.last_stand else stand_stat}'

    def long(self):
        hp_stat = f'(H:{self.hp}+{int(self.shield)}/{self.base_hp}, S:{self.speed})' if self.alive() else f'(dead)'
        return f'{self.element} Axie #{str(id(self))[-4:]} [{str(self.player)}] {hp_stat} (' \
            f'{self.back}, ' \
            f'{self.mouth}, ' \
            f'{self.horn}, ' \
            f'{self.tail})'

    def on_death(self):
        try:
            self.player.discard_pile += self.player.hand[self]
            del self.player.hand[self]
            debug("Hand discarded sucessfully!")
        except KeyError as e:
            print(f'Key error in "on_death": {e}')
            print_exc()
            exit()

    def apply_damage(self, battle, cards_to_play, attacker, atk, double_shield_dmg, true_dmg):

        if not self.alive():
            return

        block_last_stand = False
        end_last_stand = False
        fctp = flatten(list(cards_to_play.values()))

        atk = int(atk)
        self.shield = int(self.shield)

        debug(f'\tATK: {atk}')

        # on pre last stand
        for _card in fctp:
            block_last_stand = block_last_stand or _card.on_pre_last_stand(battle, cards_to_play, attacker, self)

        if self.last_stand:

            if end_last_stand:
                self.last_stand = False
                self.last_stand_ticks = 0
            else:

                self.last_stand_ticks -= 1

                if self.last_stand_ticks <= 0:
                    self.last_stand = False  # ded
                    self.on_death()
                    return

        # handle shield
        elif not true_dmg and self.shield > 0:

            # shield break
            if (double_shield_dmg and atk * 2 >= self.shield) or atk >= self.shield:

                if double_shield_dmg:
                    atk -= int(self.shield / 2)
                else:
                    atk -= self.shield
                self.shield = 0

                # on sb
                for _card in fctp:
                    _card.on_shield_break(battle, cards_to_play, attacker, self)

            else:
                rem_shield = max(0, self.shield)
                if double_shield_dmg:
                    atk -= int(self.shield / 2)
                else:
                    atk -= self.shield
                self.shield = rem_shield

        # after shield break

        # last stand?
        if not block_last_stand and self.hp - atk < 0 and atk - self.hp < self.hp * self.morale / 100:

            # on last stand entry
            for _card in fctp:
                end_last_stand = end_last_stand or _card.on_last_stand(battle, cards_to_play, attacker, self)

            if not end_last_stand:
                self.enter_last_stand()
                return

        # lower hp
        else:

            self.hp -= atk

            if self.hp <= 0:
                self.on_death()
                return

    def heal(self, amount):
        if not self.last_stand:
            self.hp = max(self.base_hp, self.hp+amount)

    def enter_last_stand(self):
        self.last_stand = True
        self.hp = 0
        # TODO check if this is true
        self.last_stand_ticks = int((self.morale - 27) / 11)
        self.shield = 0

    def turn_tick(self):  # TODO each status effect should have a life time that ticks away

        to_del = []
        # tick status effects (buffs, debuffs)
        for stat_eff, (times, turns) in self.buffs.items():

            if stat_eff == "attack down" or stat_eff == "attack up":
                continue

            if turns - 1 < 0:
                to_del.append(stat_eff)
            else:
                self.buffs[stat_eff] = (times, turns-1)

        for key in to_del:
            del self.buffs[key]

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

        # reset shield
        self.shield = 0

    def action_tick(self):

        # tODO
        # tick last stand down
        if self.last_stand:
            self.last_stand_ticks -= 1

        # tick poison
        if "poison" in self.buffs.keys():
            stacks, _ = self.buffs["poison"]
            self.hp -= stacks * poison_dmg_per_stack
            if self.hp <= 0:
                self.on_death()

        for change in self.apply_stat_eff_changes:
            change()
        self.apply_stat_eff_changes = []

    def apply_stat_eff(self, effect, times=1, turns=0, next_turn=False):  # which effects stack? if 2x fear for "next turn" does it stack to 2?

        """
        :param effect:
        :param times:
        :param turns:
        :param next_turn: true if effect is applied next n turns
        :return:
        """

        if next_turn:
            self.buffs.update({effect: (times, max(self.buffs.get(effect, (0, 0)[1]), 1))})
        else:  # stack
            self.buffs.update({effect: (times, self.buffs.get(effect, (0, 0))[1] + turns)})

    def reduce_stat_eff(self, effect, stacks):

        def f():
            s, rounds = self.buffs.get(effect, (0, 0))
            self.buffs[effect] = (s-stacks, rounds)

        self.apply_stat_eff_changes.append(f)

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

    @staticmethod
    def get_random():
        return Player([Axie.get_random() for _ in range(3)])

    def __init__(self, team: List[Axie]):
        assert len(team) == 3
        self.team = team  # team in order front to back line for simplicity
        self.deck = list()

        for axie in team:
            self.deck += list(axie.cards) + list(axie.cards)
            axie.player = self

        self.energy = self.start_energy
        self.deck_pile = shallowcopy(self.deck)
        self.discard_pile = list()
        self.hand = dict()

    def __repr__(self):
        return f'Player #{str(id(self))[-4:]}'

    def start_turn(self, cards_per_turn, energy_per_turn):

        for axie in self.team:
            if axie.alive():
                axie.turn_tick()

        self.draw_cards(cards_per_turn)
        self.energy += energy_per_turn

    def team_alive(self):
        for axie in self.team:
            if axie.alive():
                return True
        return False

    def random_discard(self):
        hand_cards = reduce(lambda x, y: x+y, list(self.hand.values()))

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
            drawn_cards.append(self.deck_pile.pop())

        self.hand.update({owner: self.hand.get(owner, list()) + [c for c in drawn_cards if c.owner == owner]
                          for owner in self.team})

        hand_cards = reduce(lambda x, y: x+y, list(self.hand.values()))

        while len(hand_cards) > hand_card_limit:
            self.random_discard()
            hand_cards = reduce(lambda x, y: x + y, list(self.hand.values()))

        if len(hand_cards) + len(self.deck_pile) + len(self.discard_pile) != 24:
            debug("Somoething went wrong!")
            debug(f'Hand cards: {len(hand_cards)} Deck pile: {len(self.deck_pile)} Disc pile: {len(self.discard_pile)}')
            exit()
        debug(f'Hand:         {hand_cards} ({len(hand_cards)})')
        debug(f'Discard pile: ({len(self.discard_pile)})')
        debug(f'Deck pile:    ({len(self.deck_pile)})')
        for k, v in self.hand.items():
            debug(f'{k}: {v}')
        debug()

    def gain_energy(self, amount):
        self.energy = max(min(10, self.energy+amount), 0)

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

        pick_threshold = 1.0 / hand_card_number
        pick_threshold = 0.5

        while c < hand_card_number and used_energy < energy_this_turn:

            if random() > pick_threshold \
                    and hand_cards[c].owner.alive() \
                    and hand_cards[c].cost <= energy_this_turn-used_energy \
                    and cards_per_axie[hand_cards[c].owner] < 4\
                    and hand_cards[c].element not in hand_cards[c].owner.disabilities.keys() \
                    and hand_cards[c].body_part not in hand_cards[c].owner.disabilities.keys():

                cards_to_play[hand_cards[c].owner].append(hand_cards[c])
                self.hand[hand_cards[c].owner].remove(hand_cards[c])
                used_energy += hand_cards[c].cost
                cards_per_axie[hand_cards[c].owner] += 1

            c += 1

        self.energy -= used_energy

        debug(f'\tCards to play: {[(str(id(axie))[-4:], card) for axie, card in cards_to_play.items()]}')
        debug(f'\tUsed energy: {used_energy}, Energy remaining: {energy_this_turn-used_energy}')
        debug(f'\tTeam order: {[a for a in self.team]}')
        return cards_to_play


class Match:

    cards_per_round = 3
    start_hand = 6
    energy_per_turn = 2

    def __init__(self, player1, player2):
        self.player1 = player1
        self.player2 = player2
        self.round = 1

        self.game_status = None

    def run_simulation(self):

        debug("Starting simulation")

        debug("\nPlayer teams:")

        for axie in self.player1.team:
            debug(f'{axie.long()}')
        debug()
        for axie in self.player2.team:
            debug(f'{axie.long()}')

        debug()

        round_counter = 0

        while self.run_round(round_counter) not in (0, 1, 2):
            debug(f'\nRound {round_counter+1} finished\n')
            round_counter += 1

    def run_round(self, round_counter) -> int:  # 1 = player 1 wins, 2 = player 2 wins, 0 = draw

        # tODO if round > sth apply auto dmg to axies

        # start round, reset shield to 0, remove buffs/debuffs \ poison, draw cards
        debug("Player 1 hand:")
        self.player1.start_turn(self.cards_per_round if round_counter > 0 else self.start_hand,
                                self.energy_per_turn if round_counter > 0 else 0)
        debug("Player 2 hand:")
        self.player2.start_turn(self.cards_per_round if round_counter > 0 else self.start_hand,
                                self.energy_per_turn if round_counter > 0 else 0)

        debug("Player 1 cards:")
        cards_p1 = self.player1.select_cards()
        debug("\nPlayer 2 cards:")
        cards_p2 = self.player2.select_cards()

        for card in cards_p1.values():
            self.player1.discard_pile += card
        for card in cards_p2.values():
            self.player2.discard_pile += card

        cards_p1.update(cards_p2)

        self.fight(cards_p1)

        return self.check_win()

    def check_win(self):

        if not self.player1.team_alive():
            if not self.player2.team_alive():
                debug(f'Game ended in a draw')
                return 0
            else:
                debug("Player 2 wins")
                return 2
        if not self.player2.team_alive():
            if not self.player1.team_alive():
                debug("Game ended in a draw")
                return 0
            else:
                debug("Player 1 wins")
                return 1

    def fight(self, cards_to_play):

        # determine attack order based on current stats
        # (debuffs, buffs, e.g. aroma etc, effects e.g. always attack first)

        flat_cards_to_play = flatten(list(cards_to_play.values()))
        axies_in_attack_order = self.determine_order(cards_to_play)

        if not axies_in_attack_order:
            return

        debug(f'\nAxies in attack order:')
        for x in axies_in_attack_order:
            debug(f'\t{x}: {cards_to_play[x]}')

        # apply shield
        for axie in axies_in_attack_order:

            for card in cards_to_play[axie]:

                if axie.alive():

                    shield_to_give = card.defense
                    shield_mult = 1

                    if shield_to_give > 0:
                        # element bonus
                        if axie.element == card.element:
                            shield_mult += 0.1

                        # if the card is in a chain with another of same element
                        if card.element in [c.element for c in flatten([cards_to_play[xe] for xe in axie.player.team])]:
                            shield_mult += 0.05

                    """
                    # chain
                    if len([c for c in cards_to_play.values()
                            if c.owner.player == axie.player and c.owner is not axie]) > 0:
                        shield_mult += 0.5
                    """

                    axie.shield += int(shield_mult * shield_to_give)

        for axie in axies_in_attack_order:

            if not cards_to_play[axie] or not axie.alive() or self.game_status in range(3):
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

                # status effects
                for xe in opp.team:
                    if xe.alive():
                        for buff, count in xe.buffs.items():
                            if buff == "aroma":
                                focus.append(xe)
                            if buff == "stench":
                                to_omit.add(xe)

                # card effect
                for _c in flat_cards_to_play:
                    _omit, _focus = _c.on_target_select(self, cards_to_play, axie, opp.team[0])
                    to_omit.union(_omit)
                    focus.extend(_focus)

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

            debug(f'\nAxie >>{axie}<< attacking >>{attack_target}<< with {cards_to_play[axie]}\n')

            def perform_attack(attacking_card, attacker, defender):
                base_atk = attacking_card.attack
                atk_multi = 1
                skip_dmg_calc = False
                dmg_reduction = 0  # for this like gecko

                # todo comparison won't work "> 0"
                if base_atk > 0 and "fear" in attacker.buffs.keys() and attacker.buffs["fear"] > 0:
                    skip_dmg_calc = True
                    debug(f'Skipping damage calculation because attacker is feared')

                if not skip_dmg_calc and attack_target and attack_target.alive():

                    # TODO think of this
                    miss = False
                    double_shield_dmg = False
                    true_dmg = False
                    morale_multi = 1

                    # element syn card / attacker
                    if attacking_card.element == attacker.element:
                        atk_multi += 0.1
                        debug(f'Atk multi set to {atk_multi} due to element synergy')

                    # element atk / def
                    atk_multi += get_dmg_bonus(attacker.element, defender.element)
                    debug(f'Atk multi set to {atk_multi} due to rpc system')

                    # buff / debuffs atk
                    for buff, (stacks, rounds) in attacker.buffs.items():
                        for s in range(stacks):

                            if buff == "attack down":
                                atk_multi -= 0.2
                                attacker.reduce_stat_eff("attack down", 1)
                                debug(f'Atk multi set to {atk_multi} due to attack down')
                            if buff == "stun":
                                miss = True
                            if buff == "attack up":
                                atk_multi += 0.2
                                attacker.reduce_stat_eff("attack up", 1)
                                debug(f'Atk multi set to {atk_multi} due to attack up')
                            if buff == "morale up":
                                morale_multi += 0.2
                                attacker.reduce_stat_eff("morale up", 1)
                            if buff == "morale down":
                                morale_multi -= 0.2
                                attacker.reduce_stat_eff("morale down", 1)

                            # TODO modification in iteration?
                            # for "next hit" effects
                            attacker.buffs[buff] = (s - 1, rounds)

                    # buff / debuff def
                    for buff, (stacks, rounds) in defender.buffs.items():
                        for s in range(stacks):
                            if buff == "fragile":
                                double_shield_dmg = True
                            if buff == "stun":
                                true_dmg = True

                            defender.buffs[buff] = (s-1, rounds)

                    # on combo effects
                    if len(cards_to_play[attacker]) > 1:
                        attacking_card.on_combo(self, cards_to_play, cards_to_play[attacker])

                    # on chain effects
                    chain_cards = []
                    for xe in owner.team:
                        if xe.alive():
                            other_cards = cards_to_play.get(xe, None)
                            if other_cards and attacking_card.element in [c.element for c in other_cards]:
                                chain_cards += cards_to_play[xe]
                    if len(chain_cards) > 0:
                        attacking_card.on_chain(self, cards_to_play, chain_cards)

                    # card effects atk
                    for _card in flat_cards_to_play:
                        atk_multi += _card.on_attack(self, cards_to_play, attacker, defender)
                        #debug(f'Atk multi set to {atk_multi} due to on_attack effect')

                    # card effects def (only gecko right now?)
                    for _card in flat_cards_to_play:
                        dmg_reduction = _card.on_attack(self, cards_to_play, attacker, defender)

                    # combos
                    if len(cards_to_play[attacker]) > 1:
                        combo_bonus = int(base_atk * atk_multi * attacker.skill / 500)
                    else:
                        combo_bonus = 0
                    debug(f'Combo bonus: {combo_bonus}')

                    # calc damage now
                    atk = (base_atk * atk_multi + combo_bonus) * (1 - dmg_reduction)

                    # crits, dodge/miss
                    # TODO confirm probabilities
                    # crit prob = moral / 457
                    # miss prob = 2.5%

                    # attacks on dead never hit
                    if not attack_target.alive():
                        miss = True

                    # determine miss / hit
                    if not miss and random() > miss_prob:

                        # TODO confirm calculation
                        crit_prob = (axie.morale * morale_multi) / 457.0

                        crit_multi = 2
                        crit_disable = False

                        # pre crit effects, adjust crit dmg or disable crits
                        for _card in flat_cards_to_play:
                            _crit_multi, _crit_disable, _crit_prob = _card.pre_crit(self, cards_to_play, attacker, defender)
                            crit_multi *= _crit_multi
                            crit_disable = crit_disable or _crit_disable
                            crit_prob += _crit_prob

                        # determine crit / normal hit
                        if not crit_disable and random() <= crit_prob:

                            # on_crit effects
                            for _card in flat_cards_to_play:
                                _card.on_crit(self, cards_to_play, attacker, defender)

                            atk *= crit_multi

                            debug(f'Critical hit for {atk}')

                        # TODO if ded, discard hand for that axie
                        defender.apply_damage(self, cards_to_play, attacker, atk, double_shield_dmg, true_dmg)
                    else:
                        debug("Attack missed!")

                        # tick actions
                        for xe in axies_in_attack_order:
                            if xe.alive():
                                xe.action_tick()

                        self.game_status = self.check_win()

                        return False

                # apply card effect
                attacking_card.on_play(self, cards_to_play)

                # tick actions
                for xe in axies_in_attack_order:
                    if xe.alive():
                        xe.action_tick()

                self.game_status = self.check_win()

                return True

            # ------------------------------------------------------

            for card in cards_to_play[axie]:

                if self.game_status not in range(3) and attack_target:

                    attack_times = 1
                    for c in flat_cards_to_play:
                        t = c.attack_times(self, cards_to_play, axie, attack_target)
                        if t:
                            attack_times += (t-1)

                    missed = False
                    for _ in range(attack_times):

                        if missed:
                            continue

                        if self.game_status not in range(3) and (attack_target.alive() or card.attack == 0):
                            missed = not perform_attack(card, axie, attack_target)

                        debug(f'\tAttack target after attack: {attack_target}')
                        debug(f'\t     Attacker after attack: {axie}\n')

    def get_axies_in_attack_order(self, cards_to_play) -> List[Axie]:

        if not cards_to_play:
            return []

        return self.determine_order(cards_to_play.keys())

    def determine_order(self, axies):

        init_speeds = [(axie, axie.speed) for axie in axies]
        final_speeds = {axie: 0 for axie in axies}

        for axie, speed in init_speeds:

            speed_mult = 1

            # status effects
            for buff, (count, turns) in axie.buffs.items():
                if buff == "speed up":
                    speed_mult += 0.2 * count
                if buff == "speed down":
                    speed_mult -= 0.2 * count

            # this breaks ties by hp & morale
            final_speeds[axie] = (speed * speed_mult, axie.hp, axie.morale)

        return sorted(axies, key=lambda axie: final_speeds[axie], reverse=True)