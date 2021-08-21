from Axie.Axie_models import *
from Axie.cards import *
from Utility import timeit

"""
p1 = Player([Axie("bird", DarkSwoop(), Eggbomb(), Blackmail(), RiskyFeather()),
             Axie("bird", DarkSwoop(), Eggbomb(), Blackmail(), RiskyFeather()),
             Axie("bird", DarkSwoop(), Eggbomb(), Blackmail(), RiskyFeather())])

p2 = Player([Axie("bird", DarkSwoop(), Eggbomb(), Blackmail(), RiskyFeather()),
             Axie("bird", DarkSwoop(), Eggbomb(), Blackmail(), RiskyFeather()),
             Axie("bird", DarkSwoop(), Eggbomb(), Blackmail(), RiskyFeather())])
"""


def single_fight():

    p1, p2 = Player.get_random(), Player.get_random()

    print("Player 1 team:")
    for axie in p1.team:
        print(axie.long())

    print("Player 2 team:")
    for axie in p2.team:
        print(axie.long())

    print(Match(p1, p2).run_simulation())


def human_fight_test(deck=None):

    hooman = HumanAgent()
    p2 = Player.get_random()
    if deck:
        p1 = Player(deck, hooman)
    else:
        p1 = Player.get_random(hooman)

    print("Player 1 team:")
    for axie in p1.team:
        print(axie.long())

    print(p1.team[0].tail is p1.team[1].tail)
    print(p1.team[0].tail == p1.team[1].tail)

    print("Player 2 team:")
    for axie in p2.team:
        print(axie.long())

    print(Match(p1, p2).run_simulation())


def deck_fight():

    p1, p2 = Player.get_random(), Player.get_random()

    print("Player 1 team:")
    for axie in p1.team:
        print(axie.long())

    print("Player 2 team:")
    for axie in p2.team:
        print(axie.long())

    results = []
    n = 1000

    for i in range(n):
        if i % 100 == 0:
            print("Starting game", i)
        results.append(Match(p1, p2).run_simulation())

    print()
    print("Avg:", sum(results)/len(results))
    print("Winrate:", 100*len([x for x in results if x > 0])/n)

    """
    from matplotlib import pyplot as plt
    
    plt.hist(results, bins=50)
    plt.show()
    #"""


@timeit
def sample_tournament(n_decks=100, matches_per_matchup=100):

    players: List[Player] = [Player.get_random() for _ in range(n_decks)]
    deck_results: Dict[Player, int] = {p: 0 for p in players}

    from itertools import combinations

    for j, (one, two) in enumerate(combinations(players, 2)):
        print("Matchup:", j+1)
        m = Match(one, two)
        r = sum([m.run_simulation() for _ in range(matches_per_matchup)])/matches_per_matchup
        deck_results[one] += r
        deck_results[two] += -r

    for p in players:
        p.reset()

    decks_by_score = sorted(players, key=lambda p: deck_results[p]/(n_decks*(n_decks-1)*0.5), reverse=True)
    for i, p in enumerate(decks_by_score):
        print("Place", i+1, "with", deck_results[p]/(n_decks*(n_decks-1)*0.5), "points")
        print(p.get_deck_string())


def player_hash_test():

    p1 = Player.get_random()
    p2 = deepcopy(p1)

    print(p1)
    print(p2)

    print(p1.as_tuple())
    print(p2.as_tuple())
    print(hash(p1))
    print(hash(p2))

    p2.team[2] = Axie.get_random()

    print(p1 == p2)
    print(p1 is p2)
    print(hash(p1))
    print(hash(p2))


def plot_test():
    from matplotlib import pyplot as plt
    from numpy import convolve, ones
    from random import random

    def get_sma(ser, n):
        return convolve(ser, ones(n) / n, mode='valid')

    data = [random() for _ in range(100)]
    length = len(data)
    sma = get_sma(data, 10)

    plt.plot(range(length), data)
    plt.plot(range(length - len(sma), length), sma)
    plt.show()


human_fight_test([Axie("dusk", AquaVitality(), AngryLam(), AirForceOne(), TailSlap()),
                  Axie("dusk", AquaVitality(), AngryLam(), AirForceOne(), TailSlap()),
                  Axie("dusk", AquaVitality(), AngryLam(), AirForceOne(), TailSlap())])

