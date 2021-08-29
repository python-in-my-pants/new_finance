from Axie.Axie_models import *
from Axie.cards import *
from Utility import timeit
from matplotlib import pyplot as plt
from numpy import convolve, ones
from random import random


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


def match_significance():

    results = []

    matchups = 100
    for j in range(matchups):
        print("Matchup:", j)
        p1, p2 = Player.get_random(), Player.get_random()
        m = Match(p1, p2)
        matches_per_matchup = 100
        results.append(sum([m.run_simulation() for _ in range(matches_per_matchup)]) / matches_per_matchup)

    s = sum([1 for r in results if abs(r) >= 0.1])
    print(f'Result is significantly different from 0 (10%) in '
          f'{100 * s / len(results):.2f} % ({s}) of cases')

    plt.plot(sorted(results), range(len(results)))
    plt.tight_layout()
    plt.show()


def plot_vs_random():

    p1, p2 = Player.get_random(), Player.get_random()

    print("Player 1 team:")
    for axie in p1.team:
        print(axie.long())

    print("Player 2 team:")
    for axie in p2.team:
        print(axie.long())

    m = Match(p1, p2)

    results = [m.run_simulation() for _ in range(100)]
    rand = [random()*2-1 for _ in range(len(results))]

    def get_sma(ser, n):
        return convolve(ser, ones(n) / n, mode='valid')

    sma_len = 10
    sma_res = get_sma(results, sma_len)
    sma_rand = get_sma(rand, sma_len)

    #plt.plot(range(len(results)), results, label="results")
    #plt.plot(range(len(results)), rand, label="random")
    plt.plot(range(len(results) - len(sma_res), len(results)), sma_res, label="results")
    plt.plot(range(len(results) - len(sma_rand), len(results)), sma_rand, label="random")

    plt.plot(range(len(results)), [sum(results) / len(results) for _ in range(len(results))], label="res avg")
    plt.plot(range(len(results)), [sum(rand) / len(results) for _ in range(len(results))], label="rand avg")
    plt.legend()
    plt.tight_layout()
    plt.show()


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


def premade_decks_fight():

    deck1 = [Axie("plant", VegetalBite(), Disguise(), AquaStock(), SpicySurprise()),
                        Axie("beast", NutCrack(), SinisterStrike(), RevengeArrow(), NutThrow()),
                        Axie("bird", DarkSwoop(), Eggbomb(), Blackmail(), AllOutShot())]
    deck2 = [Axie("dusk",  FishHook(), Disarm(), NileStrike(), CattailSlap()),
             Axie("dusk",  BloodTaste(), Disarm(), NileStrike(), TailSlap()),
             Axie("dusk",  FishHook(), SinisterStrike(), BalloonPop(), TailSlap())]
    p1, p2 = Player(deck1), Player(deck2)

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
    print("Avg:", sum(results) / len(results))
    print("Winrate:", 100 * len([x for x in results if x > 0]) / n)


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

    def get_sma(ser, n):
        return convolve(ser, ones(n) / n, mode='valid')

    data = [random() for _ in range(100)]
    length = len(data)
    sma = get_sma(data, 10)

    plt.plot(range(length), data)
    plt.plot(range(length - len(sma), length), sma)
    plt.show()


def top_ladder_decks_test():
    for p in Player.get_top_ladder_decks():
        print(p.as_tuple())


"""human_fight_test([Axie("dusk", AquaVitality(), AngryLam(), AirForceOne(), TailSlap()),
                  Axie("dusk", AquaVitality(), AngryLam(), AirForceOne(), TailSlap()),
                  Axie("dusk", AquaVitality(), AngryLam(), AirForceOne(), TailSlap())])"""

top_ladder_decks_test()
