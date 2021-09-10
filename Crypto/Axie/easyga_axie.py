from itertools import combinations
from functools import lru_cache
import my_pyeasyga
from Axie_models import *
from cards import *
from Utility import timeit
from sys import stdout
from matplotlib import pyplot as plt
from numpy import arange

print("Getting top ladder decks...")
top_ladder_decks = Player.get_top_ladder_decks()
print("Done!")


def calc_pop_fitness(pop: List[Player], matches_per_matchup=50, matchup_lim_fraction=0.8, method="v2",
                     matchups_per_deck=0.25):
    """

    :param pop:
    :param matches_per_matchup: regulates the accuracy of a matchup result
    :param matchup_lim_fraction: for v1: fraction of possible matchups that are simulated; regulates the accuracy of
                                         fitness in population
    :param method:
    :param matchups_per_deck: for v2: regulates the accuracy of  fitness in population
    :return:
    """

    print("Calculating fitness ...")
    deck_results: Dict[Player, float] = {p: 0.0 for p in pop}
    t = int(len(pop)*(len(pop)-1)*0.5)
    deck_matchup_counter = {deck: 0 for deck in pop}

    shuffle(pop)

    def v1():  # extensive evaluation, most accurate and most expensive
        i = 0
        for one, two in sample([(o, k) for o, k in combinations(pop, 2)], int(t*matchup_lim_fraction)):
            # print(f'Starting matchup {i} of {t*matchup_lim_fraction} [{one} vs. {two}]')
            m = Match(one, two)
            r = sum([m.run_simulation() for _ in range(matches_per_matchup)]) / matches_per_matchup
            deck_results[one] += r
            deck_results[two] += -r
            i += 1

    def v2():  # limit the portion of matchups for faster calc
        nonlocal deck_matchup_counter
        for j, deck in enumerate(pop):
            one = deck
            # print(f'Starting {deck} ({j+1}) matches')
            for k in range(int(min((matchups_per_deck * len(pop))//2, len(pop)-1))):
                two = pop[(j+k+1) % len(pop)]
                m = Match(one, two)
                r = sum([m.run_simulation() for _ in range(matches_per_matchup)]) / matches_per_matchup
                deck_results[one] += r
                deck_results[two] += -r
                deck_matchup_counter[one] += 1
                deck_matchup_counter[two] += 1

    def v3():  # like v2, but purley based on wins
        nonlocal deck_matchup_counter
        for j, deck in enumerate(pop):
            one = deck
            # print(f'Starting {deck} ({j+1}) matches')
            for k in range(int(min((matchups_per_deck * len(pop)) // 2, len(pop) - 1))):
                two = pop[(j + k + 1) % len(pop)]
                m = Match(one, two)
                r = sum([m.run_simulation() for _ in range(matches_per_matchup)]) / matches_per_matchup
                deck_results[one] += 1 if r > 0 else 0
                deck_results[two] += 1 if r < 0 else 0
                deck_matchup_counter[one] += 1
                deck_matchup_counter[two] += 1

    if method == "v1":
        v1()
        return {player: deck_results[player] / t for player in pop}
    if method == "v2":
        v2()
        return {player: deck_results[player] / deck_matchup_counter[player] for player in pop}
    if method == "v3":
        v3()
        return {player: deck_results[player] / deck_matchup_counter[player] for player in pop}

    raise NotImplementedError


def ladder_fitness_factory(top_ladder_limit=100, matches_per_matchup=1):

    #@timeit
    #@lru_cache()
    def fitness_against_top_ladder(v, _):
        #print("Calculating fitness for", v)
        stdout.write(".")
        fit = 0
        for ladder_deck in sample(top_ladder_decks, min(len(top_ladder_decks), top_ladder_limit)):
            m = Match(v, ladder_deck)
            fit += sum([m.run_simulation() for _ in range(matches_per_matchup)]) / matches_per_matchup
        return fit

    #@timeit
    #@lru_cache()
    def fitness_against_top_ladder_comprehension(v, _):
        #print("Calculating fitness for", v)
        stdout.write(".")
        return sum(
            [sum([Match(v, ladder_deck).run_simulation() for _ in range(matches_per_matchup)]) / matches_per_matchup
             for ladder_deck in sample(top_ladder_decks, min(len(top_ladder_decks), top_ladder_limit))]
        ) / top_ladder_limit

    return fitness_against_top_ladder_comprehension


# define and set function to create a candidate solution representation
def create_individual(data):
    return Player.get_random()


# define and set the GA's crossover operation
def crossover1(parent_1, parent_2):
    mom, dad = deepcopy(parent_1), deepcopy(parent_2)
    shuffle(mom.team)
    shuffle(dad.team)
    return Player([Axie.pair(mom.team[0], dad.team[0]),
                   Axie.pair(mom.team[1], dad.team[1]),
                   Axie.pair(mom.team[2], dad.team[2])]), \
           Player([Axie.pair(mom.team[0], dad.team[0]),
                   Axie.pair(mom.team[1], dad.team[1]),
                   Axie.pair(mom.team[2], dad.team[2])])


def crossover2(parent1, partent2):
    mom, dad = parent1, partent2
    index = sample(range(3), 1)[0]
    mom_team, dad_team = mom.team, dad.team
    mom_team[index], dad_team[index] = dad_team[index], mom_team[index]
    return Player(mom_team), Player(dad_team)


def crossover_mix(parent1, parent2):
    return crossover1(parent1, parent2) if random() > 0.5 else crossover2(parent1, parent2)


# define and set the GA's mutation operation
def mutate(indiv):
    individual = deepcopy(indiv)
    to_mutate = sample(range(3), 1)[0]
    mutated = Axie.flip_single_gene(individual.team[to_mutate])

    if to_mutate == 0:
        individual.genes = Player([mutated] + individual.team[1:])
    elif to_mutate == 1:
        individual.genes = Player([individual.team[0], mutated, individual.team[2]])
    elif to_mutate == 2:
        individual.genes = Player(individual.team[:-1] + [mutated])


# define a fitness function
def fitness(individual, data):
    return population_fitness_table[individual]


def on_generation_factory(matchups_per_deck=0.3, matches_per_matchup=50, matchup_lim_fraction=0.8, method="v2"):

    def inner(model):
        global population_fitness_table
        population_fitness_table = calc_pop_fitness([chromosome.genes for chromosome in model.current_generation],
                                                    matchups_per_deck=matchups_per_deck,
                                                    matches_per_matchup=matches_per_matchup,
                                                    matchup_lim_fraction=matchup_lim_fraction,
                                                    method=method)

    return inner


def on_generation(model):
    global population_fitness_table
    population_fitness_table = calc_pop_fitness([chromosome.genes for chromosome in model.current_generation],
                                                matchups_per_deck=0.3, matches_per_matchup=50)


def get_premade_decks():
    decks = list()

    double_aqua = [Axie("plant", VegetalBite(), PricklyTrap(), OctoberTreat(), AquaDeflect()),
                   Axie("aqua", AngryLam(), HerosBane(), Shipwreck(), TailSlap()),
                   Axie("aqua", FishHook(), StarShuriken(), HeroicReward(), UpstreamSwim())]
    decks.append(double_aqua)

    beast_bird_plant = [Axie("plant", VegetalBite(), Disguise(), AquaStock(), SpicySurprise()),
                        Axie("beast", NutCrack(), SinisterStrike(), RevengeArrow(), NutThrow()),
                        Axie("bird", DarkSwoop(), Eggbomb(), Blackmail(), AllOutShot())]
    decks.append(beast_bird_plant)

    return [Player(x) for x in decks]


# mutation function
# fitness function
# crossover function    mix
# crossover prob        1.0     0.7
# mutation prob         0.1     0.25
# generations
# init populaion
# premade decks
# elem in tuple         no
# bot agent greed       1       1
ga = my_pyeasyga.GeneticAlgorithm(list(),
                                  initial_population=Axie.get_genetically_complete_pop(5),
                                  # population_size=50,
                                  generations=100,
                                  crossover_probability=1,
                                  mutation_probability=0.1,
                                  elitism=False,
                                  maximise_fitness=True)

population_fitness_table = None

ga.create_individual = create_individual
ga.crossover_function = crossover1
ga.mutate_function = mutate
ga.fitness_function = ladder_fitness_factory(top_ladder_limit=100, matches_per_matchup=1)
"""ga.on_generation = on_generation_factory(matchups_per_deck=0.7, matches_per_matchup=30,
                                         matchup_lim_fraction=0.8, method="v2")"""

try:
    ga.run()
finally:
    print("_"*150)
    for f, i in ga.last_generation()[:10]:

        print("Fitness:", f)
        print(i, hash(i))
        print(i.get_deck_string())

    print("Hall Of Fame")
    print("_"*150)
    for indiv in ga.hall_of_fame:

        f, i = indiv.fitness, indiv.genes

        print(f)
        print(i, hash(i))
        print(i.get_deck_string())

    ga.plot()
