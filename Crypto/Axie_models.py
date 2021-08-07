from abc import abstractmethod, ABC
from typing import List, Dict, Tuple
from random import shuffle, sample, random, choice
from copy import deepcopy, copy as shallowcopy
from functools import reduce


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

        self.owner = None

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
        return set(), set()

    def attack_times(self, match, cards_played_this_turn, attacker, defender):
        return 1

    def on_attack(self, match, cards_played_this_turn, attacker, defender) -> float:
        return 0

    def pre_crit(self, match, cards_played_this_turn, attacker, defender) -> Tuple:  # (crit_multi, crit disable)
        return 0, False

    def on_crit(self, match, cards_played_this_turn, attacker, defender):  # effects like draw on crit
        return

    def on_shield_break(self, match, cards_played_this_turn, attacker, defender):
        pass

    def on_damage_inflicted(self, match, cards_played_this_turn, attacker, defender, amount):
        pass

    def on_pre_last_stand(self, match, cards_played_this_turn, attacker, defender):  # block last stand, end last stand
        return False, False

    def on_last_stand(self, match, cards_played_this_turn, attacker, defender):
        pass

    def on_end_turn(self, match, cards_played_this_turn):
        pass


class Axie:

    hp_multi = 8.72

    def __init__(self, element: str, mouth: Card, horn: Card, back: Card, tail: Card, eyes: str, ears: str):

        self.player = None

        self.element = element
        self.mouth = mouth
        mouth.owner = self
        self.horn = horn
        horn.owner = self
        self.back = back
        back.owner = self
        self.tail = tail
        tail.owner = self

        self.eyes = eyes
        self.ears = ears

        self.cards = {mouth, horn, back, tail}

        bhp, bsp, bsk, bm = type_to_stats[self.element]

        for card in self.cards:
            php, psp, psk, pm = part_to_stats[card.element]
            bhp += php
            bsp += psp
            bsk += psk
            bm += pm

        ehp, esp, esk, em = part_to_stats[eyes]
        php, psp, psk, pm = part_to_stats[ears]

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

        self.hp = self.base_hp
        self.speed = self.base_speed
        self.skill = self.base_skill
        self.morale = self.base_morale

        self.last_stand = False
        self.last_stand_ticks = 1 if self.base_morale <= 38 else 2 if self.base_morale <= 49 else 3

        self.shield = 0

    def short(self):
        return f'Axie #{str(id(self))[-4:]}'

    def __repr__(self):
        return f'{self.element} Axie #{str(id(self))[-4:]} (H:{self.hp}+{int(self.shield)}/{self.base_hp}, S:{self.speed}) (' \
            f'{self.back}, ' \
            f'{self.mouth}, ' \
            f'{self.horn}, ' \
            f'{self.tail})'

    def on_death(self):
        try:
            self.player.discard_pile += self.player.hand[self]
            del self.player.hand[self]
        except KeyError as e:
            print(f'Key error: {e}')
            exit()

    def apply_damage(self, battle, cards_to_play, attacker, atk, double_shield_dmg, true_dmg):

        if not self.alive():
            return

        block_last_stand = False
        end_last_stand = False
        fctp = flatten(list(cards_to_play.values()))

        atk = int(atk)
        self.shield = int(self.shield)

        # on pre last stand
        for _card in fctp:
            block, end = _card.on_pre_last_stand(battle, cards_to_play, attacker, self)
            block_last_stand = block_last_stand or block
            end_last_stand = end_last_stand or end

        if self.last_stand:

            if block_last_stand:
                self.last_stand = False
                self.last_stand_ticks = 0
            else:

                self.last_stand_ticks -= 1

                if self.last_stand_ticks <= 0:
                    self.last_stand = False  # ded
                    self.on_death()
                    return

        # handle shield
        elif not true_dmg:

            if self.shield > 0:

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

        # after shield break

        # last stand?
        if not block_last_stand and atk < self.hp * self.morale / 100:

            # on last stand entry
            for _card in fctp:
                _card.on_last_stand(battle, cards_to_play, attacker, self)

            self.last_stand = True

        # lower hp
        else:

            self.hp -= atk

            if self.hp <= 0:
                self.on_death()
                return

    def enter_last_stand(self):
        self.last_stand = True
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
        ...

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
        debug(f'\tTeam order: {[a.short() for a in self.team]}')
        return cards_to_play


class Match:

    cards_per_round = 3
    start_hand = 6
    energy_per_turn = 2

    def __init__(self, player1, player2):
        self.player1 = player1
        self.player2 = player2
        self.round = 1

    def run_simulation(self):

        debug("Starting simulation")

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

                    axie.shield = shield_mult * shield_to_give

        for axie in axies_in_attack_order:

            if not cards_to_play[axie] or not axie.alive():
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
                focus = set()

                # status effects
                for xe in opp.team:
                    if xe.alive():
                        for buff, count in xe.buffs.items():
                            if buff == "aroma":
                                focus.add(xe)
                            if buff == "stench":
                                to_omit.add(xe)

                # card effect
                for _c in flat_cards_to_play:
                    _omit, _focus = _c.on_target_select(self, cards_to_play, axie, None)
                    to_omit.union(_omit)
                    focus.union(_focus)

                focus = {elem for elem in focus if elem.alive()}
                to_omit = {elem for elem in to_omit if elem.alive()}

                # default select = closest

                # if focus, attack focused
                if focus and focus - to_omit:
                    return focus.pop()

                # attack closest (teams are list in order front to back)
                for _xe in opp.team:
                    if _xe not in to_omit and _xe.alive():
                        return _xe

                return None  # no valid target present

            # select attack target
            attack_target = select_attack_target()

            debug(f'\nAxie >{axie}<\n\tattacking >{attack_target}<\n\twith {cards_to_play[axie]}\n')

            def perform_attack(attacking_card, attacker, defender):
                base_atk = attacking_card.attack
                atk_multi = 1
                skip_dmg_calc = False
                dmg_reduction = 0  # for this like gecko

                if base_atk > 0 and "fear" in attacker.buffs.keys() and attacker.buffs["fear"] > 0:
                    skip_dmg_calc = True

                if not skip_dmg_calc and attack_target is not None:

                    # TODO think of this
                    miss = False
                    double_shield_dmg = False
                    true_dmg = False
                    morale_multi = 1

                    # element syn card / attacker
                    if attacking_card.element == attacker.element:
                        atk_multi += 0.1

                    # element atk / def
                    atk_multi += get_dmg_bonus(attacker.element, defender.element)

                    # buff / debuffs atk
                    for buff, (stacks, rounds) in attacker.buffs.items():
                        for s in range(stacks):

                            if buff == "attack down":
                                atk_multi -= 0.2
                            if buff == "stun":
                                miss = True
                            if buff == "attack up":
                                atk_multi += 0.2
                            if buff == "morale up":
                                morale_multi += 0.2
                            if buff == "morale down":
                                morale_multi -= 0.2

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

                    # card effects def (only gecko right now?)
                    for _card in flat_cards_to_play:
                        dmg_reduction = _card.on_attack(self, cards_to_play, attacker, defender)

                    # combos
                    combo_bonus = int(base_atk * atk_multi * attacker.skill / 500)

                    # calc damage now
                    atk = (base_atk * atk_multi + combo_bonus) * (1 - dmg_reduction)

                    # crits, dodge/miss
                    # TODO confirm probabilities
                    # crit prob = moral / 457
                    # miss prob = 2.5%

                    # determine miss / hit
                    if not miss and random() > miss_prob:

                        # TODO confirm calculation
                        crit_prob = (axie.morale * morale_multi) / 457.0

                        crit_multi = 2
                        crit_disable = False

                        # pre crit effects, adjust crit dmg or disable crits
                        for _card in flat_cards_to_play:
                            _crit_multi, _crit_disable = _card.pre_crit(self, cards_to_play, attacker, defender)
                            crit_multi *= _crit_multi
                            crit_disable = crit_disable or _crit_disable

                        # determine crit / normal hit
                        if not crit_disable and random() <= crit_prob:

                            # on_crit effects
                            for _card in flat_cards_to_play:
                                _card.on_crit(self, cards_to_play, attacker, defender)

                            atk *= crit_multi

                        # TODO if ded, discard hand for that axie
                        defender.apply_damage(self, cards_to_play, attacker, atk, double_shield_dmg, true_dmg)

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
                for c in flat_cards_to_play:
                    t = c.attack_times(self, cards_to_play, axie, attack_target)
                    if t:
                        attack_times += (t-1)

                for _ in range(attack_times):
                    if card.part_name == "Dark swoop":
                        continue
                    perform_attack(card, axie, attack_target)
                    debug(f'\tAttack target after attack: {attack_target}')
                    debug(f'\t     Attacker after attack: {axie}\n')

    def determine_order(self, cards_to_play) -> List[Axie]:  # TODO wrong, account for health etc to resolve conflicts

        if not cards_to_play:
            return []

        init_speeds = [(axie, axie.speed) for axie in cards_to_play.keys()]
        final_speeds = {axie: 0 for axie in cards_to_play.keys()}

        for axie, speed in init_speeds:

            speed_mult = 1

            # status effects
            for buff, count in axie.buffs.items():
                if buff == "speed up":
                    speed_mult += 0.2 * count
                if buff == "speed down":
                    speed_mult -= 0.2 * count

            final_speeds[axie] = speed * speed_mult

        return sorted(list(cards_to_play.keys()), key=lambda axie: final_speeds[axie], reverse=True)
