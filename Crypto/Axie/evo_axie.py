import pygad
from Axie_models import *
import atexit
from Option_utility import get_timestamp
import numpy as np
from itertools import combinations
from random import sample


def calc_pop_fitness(pop: List[Player], matches_per_matchup=100):

    deck_results: Dict[Player, int] = {p: 0 for p in pop}

    for one, two in combinations(pop, 2):
        m = Match(one, two)
        r = sum([m.run_simulation() for _ in range(matches_per_matchup)]) / matches_per_matchup
        deck_results[one] += r
        deck_results[two] += -r

    return {player: deck_results[player] / (len(pop)*(len(pop)-1)*0.5) for player in pop}


def run_model(model: pygad.GA):

    run_complete = False

    def save_state():
        if not run_complete:
            if input("Save? [Y/N] ") in ("y", "Y"):
                model.save(f'pygad_model_{get_timestamp().replace(" ", "_")}.save')

    atexit.register(save_state)

    model.run()

    run_complete = True

    solution, solution_fitness, solution_idx = model.best_solution()
    print("Parameters of the best solution : {solution}".format(solution=solution))
    print("Fitness value of the best solution = {solution_fitness}".format(solution_fitness=solution_fitness))
    print("Index of the best solution : {solution_idx}".format(solution_idx=solution_idx))

    model.plot_fitness()


def get_model(init_pop_size=100, generations=1000, stop_if_fitness_reached=0.2, stop_if_stale_for=10):

    ancestors = [p.team for p in [Player.get_random() for _ in range(init_pop_size)]]
    population_fitness_table = None

    def fitness_function(soltion, solution_idx):
        return population_fitness_table[soltion]

    def on_start(ga_instance: pygad.GA):
        """
        populate fitness table for this population
        :return: None
        """
        nonlocal population_fitness_table
        population_fitness_table = calc_pop_fitness(ga_instance.population)

    def crossover_func1(parents: List[Player], offspring_size, ga_instance):
        """
        Switches 1 axie with the other team

        :param parents: The selected parents.
        :param offspring_size: The size of the offspring as a tuple of 2 numbers: (the offspring size, number of genes)
        :param ga_instance: The instance from the pygad.GA class. This instance helps to retrieve any property like
                            population, gene_type, gene_space, etc
        :return: a NumPy array of shape equal to the value passed to the second parameter
        """
        mom, dad = parents
        children = []
        while len(children) < offspring_size:
            index = sample(range(3), 1)[0]
            mom_team, dad_team = mom.team, dad.team
            mom_team[index], dad_team[index] = dad_team[index], mom_team[index]
            children.extend([Player(mom_team), Player(dad_team)])

        return np.array(children[:offspring_size])

    def crossover_func2(parents, offspring_size, ga_instance):
        """
        Pairs each Axie with the corresponding axie of the other team

        :param parents: The selected parents.
        :param offspring_size: The size of the offspring as a tuple of 2 numbers: (the offspring size, number of genes)
        :param ga_instance: The instance from the pygad.GA class. This instance helps to retrieve any property like
                            population, gene_type, gene_space, etc
        :return: a NumPy array of shape equal to the value passed to the second parameter
        """
        mom, dad = parents
        children = []
        while len(children) < offspring_size:
            children.append(Player([Axie.pair(mom.team[0], dad.team[0]),
                                    Axie.pair(mom.team[1], dad.team[1]),
                                    Axie.pair(mom.team[2], dad.team[2])]))

        return np.array(children[:offspring_size])

    def mutation_func(offspring: Player, ga_instance):
        """
        :param offspring: The offspring to be mutated.
        :param ga_instance: The instance from the pygad.GA class. This instance helps to retrieve any property like
                            population, gene_type, gene_space, etc.
        :return: offspring
        """
        to_mutate = sample(range(3), 1)[0]
        mutated = Axie.flip_single_gene(offspring.team[to_mutate])

        if to_mutate == 0:
            return Player([mutated] + offspring.team[1:])
        elif to_mutate == 1:
            return Player([offspring.team[0], mutated, offspring.team[2]])
        elif to_mutate == 2:
            return Player(offspring.team[:-1] + [to_mutate])

        print("Mutation function did not return anything")
        exit(-1)

    # not needed I think
    def parent_selection_func(fitness, num_parents, ga_instance):
        """
        :param fitness: The fitness values of the current population.
        :param num_parents: The number of parents needed.
        :param ga_instance: The instance from the pygad.GA class. This instance helps to retrieve any property like
                            population, gene_type, gene_space, etc.
        :return:
            1. The selected parents as a NumPy array. Its shape is equal to (the number of selected parents, num_genes).
               Note that the number of selected parents is equal to the value assigned to the second input parameter
            2. The indices of the selected parents inside the population. It is a 1D list with length equal to the
               number of selected parents
        """
        ...
        return

    on_stop = [f'reach_{stop_if_fitness_reached}', f'saturate_{stop_if_stale_for}']

    # TODO use adaptive mutation
    ga = pygad.GA(initial_population=np.array(ancestors),
                  num_generations=generations,
                  num_parents_mating=2,
                  fitness_func=fitness_function,
                  sol_per_pop=init_pop_size,
                  on_start=on_start,
                  parent_selection_type=pygad.GA.tournament_selection,
                  crossover_type=crossover_func1,
                  mutation_type=mutation_func,
                  on_stop=on_stop)

    return ga


run_model(get_model(init_pop_size=10, generations=100))
