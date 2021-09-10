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


def ai_show_battle():

    p1 = Player.get_random(agent=MaxOpAgent(lookahead=1, action_search_width=50, determinization_width=20,
                                            determinization_decay=0))
    p2 = Player.get_random(agent=MaxOpAgent(lookahead=1, action_search_width=50,  determinization_width=20,
                                            determinization_decay=0))
    p2.team = deepcopy(p1.team)

    results1 = []
    results2 = []
    results3 = []
    results4 = []

    def inner(match):
        nonlocal results1, results2, results3, results4
        t, o = p1.agent.maxOpInstance.evaluate_game_state(match)
        results1.append(t)
        results2.append(o)
        t, o = p2.agent.maxOpInstance.evaluate_game_state(match)
        results3.append(t)
        results4.append(o)

    m = Match(p1, p2, inner)

    print(m.run_simulation())

    plt.plot(range(len(results1)), results1, color="#ff9999", label="Team 1 status")
    plt.plot(range(len(results2)), results2, color="#800000", label="Team 1 options")
    plt.plot(range(len(results2)), [(a+b)/2 for a, b in zip(results1, results2)], "--",
             color="#ff0000", label="State value 1")

    plt.plot(range(len(results3)), results3, color="#b3b3ff", label="Team 2 status")
    plt.plot(range(len(results4)), results4, color="#000080", label="Team 2 options")
    plt.plot(range(len(results3)), [(a + b) / 2 for a, b in zip(results3, results4)], "--",
             color="#0000ff", label="State value 2")

    plt.legend()
    plt.grid(True)
    plt.show()


def ai_vs_random():

    p1 = Player.get_random(agent=MaxOpAgent(lookahead=1, action_search_width=50, determinization_width=20,
                                            determinization_decay=0))
    p2 = Player.get_random()

    """print("Player 1 team:")
    for axie in p1.team:
        print(axie.long())

    print("Player 2 team:")
    for axie in p2.team:
        print(axie.long())"""

    m = Match(p1, p2)
    results = [m.run_simulation() for _ in range(100)]

    p1.agent = None
    m2 = Match(p1, p2)
    results2 = [m2.run_simulation() for _ in range(100)]

    print(sum(results), sum(results2))

    plt.plot(range(len(results)), results, label="AI")
    plt.plot(range(len(results2)), results2, label="Random")
    plt.legend()
    plt.grid(True)
    plt.show()


def team_status_test():

    p1, p2 = Player.get_random(), Player.get_random()

    results1 = []
    results2 = []

    def inner(match):
        nonlocal results1, results2
        results1.append(MaxOp.get_team_status(match.player1.team))
        results2.append(MaxOp.get_team_status(match.player2.team))

    m = Match(p1, p2, inner)
    print(m.run_simulation())

    plt.plot(range(len(results1)), results1, label="results 1")
    plt.plot(range(len(results2)), results2, label="results 2")
    plt.legend()
    plt.grid(True)
    plt.show()


def plot_action_width():

    p1, p2 = Player.get_random(), Player.get_random()

    actions1, actions2 = [], []
    alive1, alive2 = [], []

    def inner(match):
        actions1.append(len(match.player1.get_all_possible_moves()))
        actions2.append(len(match.player2.get_all_possible_moves()))
        alive1.append(100*sum([1 for axie in match.player1.team if axie.alive()]))
        alive2.append(100*sum([1 for axie in match.player2.team if axie.alive()]))

    m = Match(p1, p2, inner)
    print(m.run_simulation())

    plt.plot(range(len(actions1)), actions1, label="actions 1")
    plt.plot(range(len(actions2)), actions2, label="actions 2")
    plt.plot(range(len(alive1)), alive1, label="alive 1")
    plt.plot(range(len(alive2)), alive2, label="avlive 2")

    plt.legend()
    plt.grid(True)
    plt.show()


def plot_deck_stats_test():

    p = Player.from_team_ids([4358428, 2949043, 4557896])
    # p = Player.from_team_ids([1386868, 2949043, 4759059])

    for xe in p.team:
        print(xe.long())

    p.plot_deck_stats_overview()


def eval_against_top_ladder():
    print(Player.from_team_ids([4358428, 2949043, 4557896]).\
          evaluate_against_top_ladder(games_per_matchup=10, verbose=True))


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
    print(f'Average score of one random deck over another is significantly different from 0 (10%) in '
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
    ai = MaxOpAgent(lookahead=1, action_search_width=50, determinization_width=20,
                    determinization_decay=0)

    if deck:
        p1 = Player(deck, hooman)
        p2 = Player(deepcopy(deck), agent=ai)
    else:
        p1 = Player.get_random(hooman)
        p2 = Player.get_random(agent=ai)

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


if __name__ == "__main__":
    human_fight_test([Axie("dusk", AquaVitality(), AngryLam(), AirForceOne(), TailSlap()),
                      Axie("dusk", AquaVitality(), AngryLam(), AirForceOne(), TailSlap()),
                      Axie("dusk", AquaVitality(), AngryLam(), AirForceOne(), TailSlap())])
