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

    @abstractmethod
    def play(self, game_state):  # game_state is a Battle

        # on each action, check if any triggers are triggered and apply their effects
        ...

        # on each action, tick poison
        ...


class Axie:

    def __init__(self, element: str, card1: Card, card2: Card, card3: Card, card4: Card, eyes: str, pattern: str):
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

        self.status_effects = list()
        self.hp = self.base_hp
        self.morale = self.base_morale
        self.skill = self.base_skill
        self.morale = self.base_morale

        self.last_stand = False
        self.last_stand_ticks = None  # TODO does this depend on HP?

        self.shield = 0

    def alive(self):
        return self.hp > 0 or self.last_stand

    def reset(self):
        self.hp = self.base_hp
        self.morale = self.base_morale
        self.skill = self.base_skill
        self.morale = self.base_morale
        self.status_effects = list()


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

        self.energy = self.start_energy
        self.deck_pile = self.deck
        self.discard_pile = list()
        self.hand = dict()

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
                    and cards_per_axie[hand_cards[c].owner] < 4:

                cards_to_play[hand_cards[c].owner].append(hand_cards[c])
                used_energy += hand_cards[c].cost
                cards_per_axie[hand_cards[c].owner] += 1
            c += 1

        return cards_to_play


class Match:

    def __init__(self, player1, player2):
        self.player1 = player1
        self.player2 = player2
        self.round = 1
        self.triggers = list()

    def run_round(self):

        # tODO if round > sth apply auto dmg to axies

        cards_p1 = self.player1.select_cards()
        cards_p2 = self.player2.select_cards()

        cards_to_play = cards_p1.update(cards_p2)

        self.fight(cards_to_play)

    def fight(self, cards_to_play):

        # determine attack order based on current stats (debuffs, buffs, e.g. aroma etc, effects e.g. always attack first)

        axies_in_attack_order = self.determine_order(cards_to_play)

        # TODO apply shield

        for axie in axies_in_attack_order:

            attack_target = None

            # select attack target (for all cards)

            # default select = closest
            # check status effects
            # check attacker card
            # check defender cards

            for card in cards_to_play[axie]:

                if axie.alive():

                    # select attack target (for all following cards)

                    # default select
                    # check attacker card
                    # check defender cards

                    # game state changes include hp changes, status changes
                    game_state_changes, trigger = card.play(self)
                    self.apply_game_state_changes(game_state_changes)

    def apply_game_state_changes(self, gsc):
        # TODO
        ...

    def determine_order(self, cards_to_play) -> List[Axie]:

        # TODO
        #  ...

        return []