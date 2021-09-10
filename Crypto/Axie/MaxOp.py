from copy import copy, deepcopy
from random import sample
from Utility import timeit


class Node:

    def __init__(self, parent):
        self.children = []
        self.parent = parent

    def add_child(self, child):
        self.children.append(child)

    def add_children(self, children):
        for child in children:
            self.add_child(child)


class ActionNode(Node):

    def __init__(self, action, parent):
        super().__init__(parent)
        self.action = action

    def get_value(self):
        if not self.children:
            raise RuntimeError("Action has no determinizations yet")
        return sum([child.get_value() for child in self.children]) / len(self.children)

    def __repr__(self):
        return f'\nActionNode(children={len(self.children)}, parent={self.parent})'


class StateNode(Node):

    def __init__(self, state, parent, naiive_value):
        super().__init__(parent)
        self.state = state
        self.naiive_value = naiive_value

    def get_value(self):
        if self.children:
            return max([v.get_value() for v in self.children])
        else:
            return self.naiive_value

    def __repr__(self):
        return f'\nStateNode(children={len(self.children)}, value={self.naiive_value}, parent={self.parent})'


class MaxOp:

    def __init__(self):
        self.player = None

    @staticmethod
    def get_team_status(team):
        """
        incorporate team health and status effect and map to value between 0 and 1

        debuffs are punished; TODO more so on healthy axies

        :param team:
        :return: number between 0 and 1
        """
        inverse_hp_scaling_debuffs = \
            ["attack down", "morale down", "speed down", "fear", "chill", "stun", "poison", "fragile", "lethal",
             "sleep"]

        buff_adjustment = 0.05
        score = 0

        for axie in [axie for axie in team if axie.alive()]:
            buffs = axie.buffs.count_buffs() - axie.buffs.count_debuffs()
            hp_score = axie.hp / axie.base_hp
            score += min(1.2, max(0, hp_score + buffs * buff_adjustment))

        score *= sum([1 for axie in team if axie.alive()])
        score /= 10.8

        return max(0.001, score)

    def weighted_game_state_eval(self, s, team_weight=0.6):
        t, o = self.evaluate_game_state(s)

        return team_weight * t + (1-team_weight) * o

    def evaluate_game_state(self, state):
        """
        incorporate team status, number of options and opp team status & number of options

        (we determinize the cards & options of opponent in a fictional state)

        :param state:
        :return:
        """

        if state is None:
            pass

        if self.player == 1:
            # > 0: good; < 0: bad
            try:
                teams_score = (MaxOp.get_team_status(state.player1.team) / MaxOp.get_team_status(state.player2.team)) - 1
            except:
                pass

            opp_moves = state.player2.get_all_possible_moves()
            if opp_moves:
                options_score = (len(state.player1.get_all_possible_moves()) / len(opp_moves)) - 1
            else:
                options_score = 10

        else:
            # > 0: good; < 0: bad
            teams_score = (MaxOp.get_team_status(state.player2.team) / MaxOp.get_team_status(state.player1.team)) - 1

            opp_moves = state.player1.get_all_possible_moves()
            if opp_moves:
                options_score = (len(state.player2.get_all_possible_moves()) / len(opp_moves)) - 1
            else:
                options_score = 10

        return teams_score, options_score

    def flatten(self, t):
        out = []
        for item in t:
            if isinstance(item, (list, tuple)):
                out.extend(self.flatten(item))
            else:
                out.append(item)
        return out

    def copy_and_randomize(self, state):
        """
        create a copy of the given state and randomize any information not visible to observer

        :param state:
        :return:
        """

        clone = deepcopy(state)

        if self.player == 1:
            opp = clone.player2
        else:
            opp = clone.player1

        # shuffle opponent hand into deck and let him draw new
        old_opp_hand = self.flatten(opp.hand.values())
        opp.deck_pile += old_opp_hand
        opp.hand = {axie: [] for axie in opp.team if axie.alive()}
        opp.draw_cards(len(old_opp_hand))

        return clone

    @timeit
    def get_action(self, game_state, lookahead=3, action_search_width=50, determinization_width=20,
                   determinization_decay=0.3):
        """
        maximise oppertunities & team status for yourself and minimize those for opp

        :param determinization_decay: decay the determinization_width by this amount per level of the tree
        :param determinization_width: number of possible opponent hand+move combinations to consider, influences
            evaluation quality
        :param action_search_width: max number of own actions to consider
        :param lookahead:
        :param game_state:
        :return:
        """

        base_state_copy = self.copy_and_randomize(game_state)
        root = StateNode(base_state_copy, None, self.weighted_game_state_eval(base_state_copy))

        self.generate_subtree(root, lookahead, action_search_width, determinization_width, determinization_decay)

        return self.map_action_to_original_gamestate(
            game_state, max(root.children, key=lambda child: child.get_value()).action
        )

    def map_action_to_original_gamestate(self, orig_state, action):

        # selected move needs to be transformed back to reference actual game axies / cards instead of clones
        def clone_to_real_axie(c):
            if self.player == 1:
                return [axie for axie in orig_state.player1.team if axie.axie_id == c.axie_id][0]
            else:
                return [axie for axie in orig_state.player2.team if axie.axie_id == c.axie_id][0]

        def clone_to_real_card(c):
            try:
                if self.player == 1:
                    orig_hand_cards = self.flatten(list(orig_state.player1.hand.values()))
                else:
                    orig_hand_cards = self.flatten(list(orig_state.player2.hand.values()))

                return [card for card in orig_hand_cards if card.card_id == c.card_id][0]
            except IndexError:
                raise RuntimeError("Clone mapping of card failed")
            # use cards in hand to actually get the copy in hand and not elsewhere

        return {
            clone_to_real_axie(axie): [clone_to_real_card(card) for card in action[axie]]
            for axie in action.keys()
        }

    def generate_subtree(self, state_node, lookahead, action_search_width, determinization_width,
                         determinization_decay):

        if lookahead == 0:
            raise RuntimeError("Lookahead of 0 is illegal")

        if lookahead == 2:
            print("Generating subtree")

        # expand 1
        self.expand_state_node(state_node, action_search_width, determinization_width)

        if lookahead - 1 > 0:
            # for all leafs: f(lookahead-1)
            for action_node in state_node.children:
                for child_state in action_node.children:
                    self.generate_subtree(child_state, lookahead-1, action_search_width,
                                          determinization_width * (1 - determinization_decay), determinization_decay)

    def expand_state_node(self, state_node, action_search_width, determinization_width):
        """
        selects action_search_width actions at random and for each determinises determinization_width opponent actions,
        expanding the state node by the action nodes which are expanded by the resulting states from the actions+opp
        actions

        :param state_node:
        :param action_search_width:
        :param determinization_width:
        :return:
        """

        # get moves from determinized base state copy so we don't peek into opponents cards
        possible_actions = state_node.state.player1.get_all_possible_moves() if self.player == 1 \
            else state_node.state.player2.get_all_possible_moves()

        for action in sample(possible_actions, min(action_search_width, len(possible_actions))):

            resulting_states = \
                [self.apply_move(action, self.copy_and_randomize(state_node.state))
                 for _ in range(max(int(determinization_width), 1))]

            action_node = ActionNode(action, state_node)
            action_node.add_children([StateNode(s, action_node, self.weighted_game_state_eval(s))
                                      for s in resulting_states])
            state_node.add_child(action_node)

    def apply_move(self, move, state):

        if self.player == 1:
            cards_p1 = move
            cards_p2 = state.player2.new_random_select()
        else:
            cards_p2 = move
            cards_p1 = state.player1.new_random_select()

        copy(cards_p1).update(cards_p2)

        state.fight(cards_p1)

        for axie in state.player1.team + state.player2.team:
            if axie.alive():
                axie.shield_broke_this_turn = False

        state.round += 1

        # ------------------------------------------------------------

        if state.game_running():

            # start next turn
            if state.round >= 10:
                for axie in state.player1.team + state.player2.team:
                    if axie.alive():
                        axie.change_hp(-((state.round - 10) * 30 + 20))

                if not state.game_running():
                    return state

            # reset shield to 0, remove buffs/debuffs/poison, draw cards
            state.player1.start_turn(state.cards_per_round if state.round > 1 else state.start_hand,
                                     state.energy_per_turn if state.round > 1 else 0)
            state.player2.start_turn(state.cards_per_round if state.round > 1 else state.start_hand,
                                     state.energy_per_turn if state.round > 1 else 0)

        return state

    def permutate_move(self, move):
        ...

    def __repr__(self):
        return f'MaxOp(player={self.player})'
