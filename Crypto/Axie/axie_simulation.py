from Axie.Axie_models import *
from Axie.cards import *

"""
p1 = Player([Axie("bird", DarkSwoop(), Eggbomb(), Blackmail(), RiskyFeather()),
             Axie("bird", DarkSwoop(), Eggbomb(), Blackmail(), RiskyFeather()),
             Axie("bird", DarkSwoop(), Eggbomb(), Blackmail(), RiskyFeather())])

p2 = Player([Axie("bird", DarkSwoop(), Eggbomb(), Blackmail(), RiskyFeather()),
             Axie("bird", DarkSwoop(), Eggbomb(), Blackmail(), RiskyFeather()),
             Axie("bird", DarkSwoop(), Eggbomb(), Blackmail(), RiskyFeather())])
"""


def deck_fight():

    p1, p2 = Player.get_random(), Player.get_random()

    print("Player 1 team:")
    for axie in p1.team:
        print(axie.long())

    print("Player 2 team:")
    for axie in p2.team:
        print(axie.long())

    results = []
    n = 100

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


def sample_tournament(n_decks=100, matches_per_matchup=100):

    players = [Player.get_random() for _ in range(n_decks)]
    deck_results = {p: [] for p in players}

    from itertools import combinations

    for j, (one, two) in enumerate(combinations(players, 2)):
        print("Matchup:", j)
        m = Match(one, two)
        r = sum([m.run_simulation() for _ in range(matches_per_matchup)])/matches_per_matchup
        deck_results[one].append(r)
        deck_results[two].append(r)

    for p in players:
        p.reset()

    decks_by_score = sorted(players, key=lambda p: sum(deck_results[p])/len(deck_results[p]), reverse=True)
    for i, p in enumerate(decks_by_score):
        print("Place", i+1, "with", sum(deck_results[p])/len(deck_results[p]), "points")
        print(p.get_deck_string())


sample_tournament(50)

