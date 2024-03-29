from math import *
from random import choice, shuffle
import sys
from copy import deepcopy, copy
from typing import Dict, List
from Utility import timeit
from pprint import pprint as pp


# TODO adjust for player 1 / player 2, currently mostly only coded as player 1


def flatten(t):
    out = []
    for item in t:
        if isinstance(item, (list, tuple)):
            out.extend(flatten(item))
        else:
            out.append(item)
    return out


# abstract game state class
class GameState:
    """ A state of the game, i.e. the game board. These are the only functions which are
        absolutely necessary to implement ISMCTS in any imperfect information game,
        although they could be enhanced and made quicker, for example by using a
        GetRandomMove() function to generate a random move during rollout.
        By convention the players are numbered 1, 2, ..., self.numberOfPlayers.
    """

    def __init__(self):
        self.numberOfPlayers = 2
        self.playerToMove = 1

    def GetNextPlayer(self, p):
        """ Return the player to the left of the specified player
        """
        return (p % self.numberOfPlayers) + 1

    def Clone(self):
        """ Create a deep clone of this game state.
        """
        st = GameState()
        st.playerToMove = self.playerToMove
        return st

    def CloneAndRandomize(self, observer):
        """ Create a deep clone of this game state, randomizing any information not visible to the specified observer player.
        """
        return self.Clone()

    def DoMove(self, move):
        """ Update a state by carrying out the given move.
            Must update playerToMove.
        """
        self.playerToMove = self.GetNextPlayer(self.playerToMove)

    def GetMoves(self):
        """ Get all possible moves from this state.
        """
        raise NotImplementedError

    def GetResult(self, player):
        """ Get the game result from the viewpoint of player.
        """
        raise NotImplementedError

    def __repr__(self):
        """ Don't need this - but good style.
        """
        pass

'''
class Card:
    """ A playing card, with rank and suit.
        rank must be an integer between 2 and 14 inclusive (Jack=11, Queen=12, King=13, Ace=14)
        suit must be a string of length 1, one of 'C' (Clubs), 'D' (Diamonds), 'H' (Hearts) or 'S' (Spades)
    """

    def __init__(self, rank, suit):
        if rank not in range(2, 14 + 1):
            raise Exception("Invalid rank")
        if suit not in ['C', 'D', 'H', 'S']:
            raise Exception("Invalid suit")
        self.rank = rank
        self.suit = suit

    def __repr__(self):
        return "??23456789TJQKA"[self.rank] + self.suit

    def __eq__(self, other):
        return self.rank == other.rank and self.suit == other.suit

    def __ne__(self, other):
        return self.rank != other.rank or self.suit != other.suit


class KnockoutWhistState(GameState):
    """ A state of the game Knockout Whist.
        See http://www.pagat.com/whist/kowhist.html for a full description of the rules.
        For simplicity of implementation, this version of the game does not include the "dog's life" rule
        and the trump suit for each round is picked randomly rather than being chosen by one of the players.
    """

    def __init__(self, n):
        """ Initialise the game state. n is the number of players (from 2 to 7).
        """
        super().__init__()
        self.numberOfPlayers = n
        self.playerToMove = 1
        self.tricksInRound = 7
        self.playerHands = {p: [] for p in range(1, self.numberOfPlayers + 1)}
        self.discards = []  # Stores the cards that have been played already in this round
        self.currentTrick = []
        self.trumpSuit = None
        self.tricksTaken = {}  # Number of tricks taken by each player this round
        self.knockedOut = {p: False for p in range(1, self.numberOfPlayers + 1)}
        self.Deal()

    def Clone(self):
        """ Create a deep clone of this game state.
        """
        st = KnockoutWhistState(self.numberOfPlayers)
        st.playerToMove = self.playerToMove
        st.tricksInRound = self.tricksInRound
        st.playerHands = deepcopy(self.playerHands)
        st.discards = deepcopy(self.discards)
        st.currentTrick = deepcopy(self.currentTrick)
        st.trumpSuit = self.trumpSuit
        st.tricksTaken = deepcopy(self.tricksTaken)
        st.knockedOut = deepcopy(self.knockedOut)
        return st

    def CloneAndRandomize(self, observer):
        """ Create a deep clone of this game state, randomizing any information not visible to the specified observer player.
        """
        st = self.Clone()

        # The observer can see his own hand and the cards in the current trick, and can remember the cards played in previous tricks
        seenCards = st.playerHands[observer] + st.discards + [card for (player, card) in st.currentTrick]
        # The observer can't see the rest of the deck
        unseenCards = [card for card in st.GetCardDeck() if card not in seenCards]

        # Deal the unseen cards to the other players
        shuffle(unseenCards)
        for p in range(1, st.numberOfPlayers + 1):
            if p != observer:
                # Deal cards to player p
                # Store the size of player p's hand
                numCards = len(self.playerHands[p])
                # Give player p the first numCards unseen cards
                st.playerHands[p] = unseenCards[: numCards]
                # Remove those cards from unseenCards
                unseenCards = unseenCards[numCards:]

        return st

    def GetCardDeck(self):
        """ Construct a standard deck of 52 cards.
        """
        return [Card(rank, suit) for rank in range(2, 14 + 1) for suit in ['C', 'D', 'H', 'S']]

    def Deal(self):
        """ Reset the game state for the beginning of a new round, and deal the cards.
        """
        self.discards = []
        self.currentTrick = []
        self.tricksTaken = {p: 0 for p in range(1, self.numberOfPlayers + 1)}

        # Construct a deck, shuffle it, and deal it to the players
        deck = self.GetCardDeck()
        shuffle(deck)
        for p in range(1, self.numberOfPlayers + 1):
            self.playerHands[p] = deck[: self.tricksInRound]
            deck = deck[self.tricksInRound:]

        # Choose the trump suit for this round
        self.trumpSuit = choice(['C', 'D', 'H', 'S'])

    def GetNextPlayer(self, p):
        """ Return the player to the left of the specified player, skipping players who have been knocked out
        """
        n = (p % self.numberOfPlayers) + 1
        # Skip any knocked-out players
        while n != p and self.knockedOut[n]:
            n = (n % self.numberOfPlayers) + 1
        return n

    def DoMove(self, move):
        """ Update a state by carrying out the given move.
            Must update playerToMove.
        """
        # Store the played card in the current trick
        self.currentTrick.append((self.playerToMove, move))

        # Remove the card from the player's hand
        self.playerHands[self.playerToMove].remove(move)

        # Find the next player
        self.playerToMove = self.GetNextPlayer(self.playerToMove)

        # If the next player has already played in this trick, then the trick is over
        if any(True for (player, card) in self.currentTrick if player == self.playerToMove):
            # Sort the plays in the trick: those that followed suit (in ascending rank order), then any trump plays (in ascending rank order)
            (leader, leadCard) = self.currentTrick[0]
            suitedPlays = [(player, card.rank) for (player, card) in self.currentTrick if card.suit == leadCard.suit]
            trumpPlays = [(player, card.rank) for (player, card) in self.currentTrick if card.suit == self.trumpSuit]
            sortedPlays = sorted(suitedPlays, key=lambda x: [1]) + sorted(trumpPlays, key=lambda x: x[1])
            # The winning play is the last element in sortedPlays
            trickWinner = sortedPlays[-1][0]

            # Update the game state
            self.tricksTaken[trickWinner] += 1
            self.discards += [card for (player, card) in self.currentTrick]
            self.currentTrick = []
            self.playerToMove = trickWinner

            # If the next player's hand is empty, this round is over
            if not self.playerHands[self.playerToMove]:
                self.tricksInRound -= 1
                self.knockedOut = {p: (self.knockedOut[p] or self.tricksTaken[p] == 0) for p in
                                   range(1, self.numberOfPlayers + 1)}
                # If all but one players are now knocked out, the game is over
                if len([x for x in self.knockedOut.values() if not x]) <= 1:
                    self.tricksInRound = 0

                self.Deal()

    def GetMoves(self):
        """ Get all possible moves from this state.
        """
        hand = self.playerHands[self.playerToMove]
        if not self.currentTrick:
            # May lead a trick with any card
            return hand
        else:
            (leader, leadCard) = self.currentTrick[0]
            # Must follow suit if it is possible to do so
            cardsInSuit = [card for card in hand if card.suit == leadCard.suit]
            if cardsInSuit:
                return cardsInSuit
            else:
                # Can't follow suit, so can play any card
                return hand

    def GetResult(self, player):
        """ Get the game result from the viewpoint of player.
        """
        return 0 if (self.knockedOut[player]) else 1

    def __repr__(self):
        """ Return a human-readable representation of the state
        """
        result = "Round %i" % self.tricksInRound
        result += " | P%i: " % self.playerToMove
        result += ",".join(str(card) for card in self.playerHands[self.playerToMove])
        result += " | Tricks: %i" % self.tricksTaken[self.playerToMove]
        result += " | Trump: %s" % self.trumpSuit
        result += " | Trick: ["
        result += ",".join(("%i:%s" % (player, card)) for (player, card) in self.currentTrick)
        result += "]"
        return result
'''


class AxieGameState:
    """ A state of the game, i.e. the game board. These are the only functions which are
        absolutely necessary to implement ISMCTS in any imperfect information game,
        although they could be enhanced and made quicker, for example by using a
        GetRandomMove() function to generate a random move during rollout.
        By convention the players are numbered 1, 2, ..., self.numberOfPlayers.
    """

    def __init__(self, s):
        self.numberOfPlayers = 2
        self.playerToMove = 1  # 1 = me; 2 = opp
        self.state = s

    def GetNextPlayer(self, p):
        """ Return the player to the left of the specified player
        """
        return (p % self.numberOfPlayers) + 1

    def Clone(self):
        """ Create a deep clone of this game state.
        """
        st = AxieGameState(deepcopy(self.state))
        st.playerToMove = self.playerToMove
        return st

    def CloneAndRandomize(self, observer: int):
        """
        Create a deep clone of this game state,
        randomizing any information not visible to the specified observer player.
        """
        clone = self.Clone()  # beware: clone contains reference to MCTS agent as agents are part of game state

        if observer == 1:
            opp = clone.state.player2
        else:
            opp = clone.state.player1

        # shuffle opponent hand into deck and let him draw new
        old_opp_hand = flatten(opp.hand.values())
        opp.deck_pile += old_opp_hand
        opp.hand = {axie: [] for axie in opp.team if axie.alive()}
        opp.draw_cards(len(old_opp_hand))

        return clone

    def DoMove(self, move: Dict):
        """
        Update a state by carrying out the given move.
        Must update playerToMove.

        we must reorder things here, start with applying moves, then check if game is over; if not start new turn,
        distribute energy, draw, etc.

        """

        cards_p1 = move
        cards_p2 = self.state.player2.new_random_select()  # automatically uses random if no agent supplied

        copy(cards_p1).update(cards_p2)

        self.state.fight(cards_p1)

        for axie in self.state.player1.team + self.state.player2.team:
            if axie.alive():
                axie.shield_broke_this_turn = False

        self.state.round += 1

        # -------------------------------------------------------------

        self.playerToMove = self.GetNextPlayer(self.playerToMove)

        # ------------------------------------------------------------

        if self.state.game_running():

            # start next turn
            if self.state.round >= 10:
                for axie in self.state.player1.team + self.state.player2.team:
                    if axie.alive():
                        axie.change_hp(-((self.state.round - 10) * 30 + 20))

                if not self.state.game_running():
                    return

            # reset shield to 0, remove buffs/debuffs/poison, draw cards
            self.state.player1.start_turn(self.state.cards_per_round if self.state.round > 1 else self.state.start_hand,
                                          self.state.energy_per_turn if self.state.round > 1 else 0)
            self.state.player2.start_turn(self.state.cards_per_round if self.state.round > 1 else self.state.start_hand,
                                          self.state.energy_per_turn if self.state.round > 1 else 0)

    def GetMoves(self):
        """ Get all possible moves from this state.
        """

        if self.playerToMove == 1:  # AIs turn
            if not self.state.game_running():
                return []  # [{axie: [] for axie in self.state.player1.team}]
            return self.state.player1.get_all_possible_moves()
        else:
            if not self.state.game_running():
                return []  # [{axie: [] for axie in self.state.player2.team}]
            return self.state.player2.get_all_possible_moves()

    def GetResult(self, player: int):
        """ Get the game result from the viewpoint of player.
        """
        if not self.state.game_running():
            r = self.state.get_result()
            if player == 1:
                return r
            else:
                return -r

        """for axie in self.state.player1.team+self.state.player2.team:
            print(axie.long())

        print(len(self.state.player1.get_all_possible_moves()+self.state.player2.get_all_possible_moves()))"""

        raise RuntimeError("Trying to get result of non-terminal state")


class Node:
    """ A node in the game tree. Note wins is always from the viewpoint of playerJustMoved.
    """

    def __init__(self, move=None, parent=None, playerJustMoved=None):
        self.move = move  # the move that got us to this node - "None" for the root node
        self.parentNode = parent  # "None" for the root node
        self.childNodes = []
        self.wins = 0
        self.visits = 0
        self.avails = 1
        self.playerJustMoved = playerJustMoved  # the only part of the state that the Node needs later

    def GetUntriedMoves(self, legalMoves):
        """ Return the elements of legalMoves for which this node does not have children.
        """

        # Find all moves for which this node *does* have children
        triedMoves = [child.move for child in self.childNodes]

        # Return all moves that are legal but have not been tried yet
        return [move for move in legalMoves if move not in triedMoves]

    def UCBSelectChild(self, legalMoves, exploration=0.7):
        """ Use the UCB1 formula to select a child node, filtered by the given list of legal moves.
            exploration is a constant balancing between exploitation and exploration, with default value 0.7 (approximately sqrt(2) / 2)
        """

        # Filter the list of children by the list of legal moves
        legalChildren = [child for child in self.childNodes if child.move in legalMoves]

        # Get the child with the highest UCB score
        s = max(legalChildren,
                key=lambda c: float(c.wins) / float(c.visits) + exploration * sqrt(log(c.avails) / float(c.visits)))

        # Update availability counts -- it is easier to do this now than during backpropagation
        for child in legalChildren:
            child.avails += 1

        # Return the child selected above
        return s

    def AddChild(self, m, p):
        """ Add a new child node for the move m.
            Return the added child node
        """
        n = Node(move=m, parent=self, playerJustMoved=p)
        self.childNodes.append(n)
        return n

    def Update(self, terminalState):
        """ Update this node -
        increment the visit count by one, and increase the win count by the result of terminalState for self.playerJustMoved.
        """
        self.visits += 1
        if self.playerJustMoved is not None:
            self.wins += terminalState.GetResult(self.playerJustMoved)

    def __repr__(self):
        return "[M:%s W/V/A: %4i/%4i/%4i]" % (self.move, self.wins, self.visits, self.avails)

    def TreeToString(self, indent):
        """ Represent the tree as a string, for debugging purposes.
        """
        s = self.IndentString(indent) + str(self)
        for c in self.childNodes:
            s += c.TreeToString(indent + 1)
        return s

    def IndentString(self, indent):
        s = "\n"
        for i in range(1, indent + 1):
            s += "| "
        return s

    def ChildrenToString(self):
        s = ""
        for c in self.childNodes:
            s += str(c) + "\n"
        return s


def ISMCTS(rootstate: AxieGameState, itermax=1000, verbose=False):
    """ Conduct an ISMCTS search for itermax iterations starting from rootstate.
        Return the best move from the rootstate.
    """

    # TODO beware to differ between [] as turn = no legal moves = game over AND
    #  [{axie: [] for axie in...}] = pass = game running

    rootnode = Node()

    for i in range(itermax):
        node = rootnode

        # Determinize
        axie_game_state = rootstate.CloneAndRandomize(rootstate.playerToMove)

        # Select
        possible_moves = axie_game_state.GetMoves()
        while possible_moves != [] and node.GetUntriedMoves(possible_moves) == []:  # node is fully expanded and non-terminal
            #print("ISMCTS: selecting")
            node = node.UCBSelectChild(possible_moves)
            axie_game_state.DoMove(node.move)
            possible_moves = axie_game_state.GetMoves()

        # Expand
        untriedMoves = node.GetUntriedMoves(possible_moves)
        if untriedMoves:  # if we can expand (i.e. state/node is non-terminal)
            m = choice(untriedMoves)
            #print("ISMCTS: expanding with move:")
            #pp(m)
            player = axie_game_state.playerToMove
            axie_game_state.DoMove(m)
            node = node.AddChild(m, player)  # add child and descend tree

        # Simulate
        possible_moves = axie_game_state.GetMoves()
        while possible_moves:  # while state is non-terminal
            #print("ISMCTS: simulating")
            axie_game_state.DoMove(choice(possible_moves))
            possible_moves = axie_game_state.GetMoves()

        # Backpropagate
        while node is not None:  # backpropagate from the expanded node and work back to the root node
            #print("ISMTS: backpropagating")
            node.Update(axie_game_state)
            node = node.parentNode

    """# Output some information about the tree - can be omitted
    if verbose:
        print(rootnode.TreeToString(0))
    else:
        print(rootnode.ChildrenToString())"""

    selected_move = max(rootnode.childNodes, key=lambda c: c.visits).move  # return the move that was most visited

    # selected move needs to be transformed back to reference actual game axies / cards instead of clones
    def clone_to_real_axie(c):
        return [axie for axie in rootstate.state.player1.team if axie.axie_id == c.axie_id][0]

    def clone_to_real_card(c):
        try:
            orig_hand_cards = flatten(list(rootstate.state.player1.hand.values()))
            return [card for card in orig_hand_cards if card.card_id == c.card_id][0]
        except IndexError:
            raise RuntimeError("Clone mapping of card failed")
        # use cards in hand to actually get the copy in hand and not elsewhere

    uncloned_move = {
        clone_to_real_axie(axie): [clone_to_real_card(card) for card in selected_move[axie]]
        for axie in selected_move.keys()
    }

    return uncloned_move


def PlayGame():
    """ Play a sample game between two ISMCTS players.
    """
    state = AxieGameState()

    while state.GetMoves():
        #print(str(state))
        # Use different numbers of iterations (simulations, tree nodes) for different players
        if state.playerToMove == 1:
            m = ISMCTS(rootstate=state, itermax=1000, verbose=False)
        else:
            m = ISMCTS(rootstate=state, itermax=100, verbose=False)
        print("Best Move: " + str(m) + "\n")
        state.DoMove(m)

    someoneWon = False
    for p in range(1, state.numberOfPlayers + 1):
        if state.GetResult(p) > 0:
            print("Player " + str(p) + " wins!")
            someoneWon = True
    if not someoneWon:
        print("Nobody wins!")


if __name__ == "__main__":
    PlayGame()
