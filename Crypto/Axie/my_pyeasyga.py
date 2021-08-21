# -*- coding: utf-8 -*-
"""
    pyeasyga module

"""

import random
import copy
from operator import attrgetter
from Axie_models import Axie, flatten
from six.moves import range
from Utility import timeit


class GeneticAlgorithm(object):
    """Genetic Algorithm class.

    This is the main class that controls the functionality of the Genetic
    Algorithm.

    A simple example of usage:

    >>> # Select only two items from the list and maximise profit
    >>> from pyeasyga.pyeasyga import GeneticAlgorithm
    >>> input_data = [('pear', 50), ('apple', 35), ('banana', 40)]
    >>> easyga = GeneticAlgorithm(input_data)
    >>> def fitness (member, data):
    >>>     return sum([profit for (selected, (fruit, profit)) in
    >>>                 zip(member, data) if selected and
    >>>                 member.count(1) == 2])
    >>> easyga.fitness_function = fitness
    >>> easyga.run()
    >>> print easyga.best_individual()

    """

    def __init__(self,
                 seed_data,
                 initial_population=None,
                 population_size=50,
                 generations=100,
                 crossover_probability=0.8,
                 mutation_probability=0.2,
                 elitism=True,
                 maximise_fitness=True,
                 on_generation=lambda _: None,
                 sim_meth="similarity",
                 max_sim=0.9):
        """Instantiate the Genetic Algorithm.

        :param seed_data: input data to the Genetic Algorithm
        :type seed_data: list of objects
        :param int population_size: size of population
        :param int generations: number of generations to evolve
        :param float crossover_probability: probability of crossover operation
        :param float mutation_probability: probability of mutation operation

        """

        self.seed_data = seed_data
        self.initial_population = initial_population
        self.population_size = population_size if not initial_population else len(initial_population)
        self.generations = generations
        self.crossover_probability = crossover_probability
        self.mutation_probability = mutation_probability
        self.elitism = elitism
        self.maximise_fitness = maximise_fitness

        self.current_generation = []

        def create_individual(_):
            raise NotImplementedError

        def crossover(parent_1, parent_2):
            """Crossover (mate) two parents to produce two children.

            :param parent_1: candidate solution representation (list)
            :param parent_2: candidate solution representation (list)
            :returns: tuple containing two children

            """
            index = random.randrange(1, len(parent_1))
            child_1 = parent_1[:index] + parent_2[index:]
            child_2 = parent_2[:index] + parent_1[index:]
            return child_1, child_2

        def mutate(individual):
            """Reverse the bit of a random index in an individual."""
            mutate_index = random.randrange(len(individual))
            individual[mutate_index] = (0, 1)[individual[mutate_index] == 0]

        def random_selection(population):
            """Select and return a random member of the population."""
            return random.choice(population)

        def tournament_selection(population):
            """Select a random number of individuals from the population and
            return the fittest member of them all.
            """
            if self.tournament_size == 0:
                self.tournament_size = 2
            members = random.sample(population, self.tournament_size)
            members.sort(key=attrgetter('fitness'), reverse=self.maximise_fitness)
            return members[0]

        self.fitness_function = None
        self.tournament_selection = tournament_selection
        self.tournament_size = self.population_size // 10
        self.random_selection = random_selection
        self.create_individual = create_individual
        self.crossover_function = crossover
        self.mutate_function = mutate
        self.selection_function = self.tournament_selection
        self.on_generation = on_generation

        self.similarity_method = sim_meth
        self.max_sim = max_sim

        self.top_individual_fitness_hist = list()
        self.worst_individual_fitness_hist = list()
        self.avg_pop_fitness_hist = list()
        self.hall_of_fame = list()
        self.diversity_hist = list()
        self.prev_best = None

    def create_initial_population(self):
        """Create members of the first population randomly.
        """
        if not self.initial_population:
            initial_population = []
            for _ in range(self.population_size):
                genes = self.create_individual(self.seed_data)
                individual = Chromosome(genes)
                initial_population.append(individual)
            self.current_generation = initial_population
        else:
            self.current_generation = [Chromosome(genes) for genes in self.initial_population]
        self.prev_best = self.current_generation[0].genes

    def calculate_population_fitness(self):
        """Calculate the fitness of every member of the given population using
        the supplied fitness_function.
        """
        for individual in self.current_generation:
            individual.fitness = self.fitness_function(individual.genes, self.seed_data)

    def rank_population(self):
        """Sort the population by fitness according to the order defined by
        maximise_fitness.
        """
        self.current_generation.sort(
            key=attrgetter('fitness'), reverse=self.maximise_fitness)

    def create_new_population(self):
        """Create a new population using the genetic operators (selection,
        crossover, and mutation) supplied.
        """
        new_population = []
        elite = copy.deepcopy(self.current_generation[0])
        selection = self.selection_function

        while len(new_population) < self.population_size:
            parent_1 = copy.deepcopy(selection(self.current_generation))
            parent_2 = copy.deepcopy(selection(self.current_generation))

            child_1, child_2 = parent_1, parent_2
            child_1.fitness, child_2.fitness = 0, 0

            can_crossover = random.random() < self.crossover_probability
            can_mutate = random.random() < self.mutation_probability

            if can_crossover:
                child_1.genes, child_2.genes = self.crossover_function(parent_1.genes, parent_2.genes)

            if can_mutate:
                self.mutate_function(child_1.genes)
                self.mutate_function(child_2.genes)

            new_population.append(child_1)
            if len(new_population) < self.population_size:
                new_population.append(child_2)

        new_population = self.remove_doubles(new_population)

        if self.elitism:
            new_population[0] = elite

        self.current_generation = new_population

    def remove_doubles(self, new_population, refill=True, with_elem=False, max_sim=0.9):
        """

        :param new_population:
        :param refill: fill up with new individuals after old ones are kicked
        :param method: tuple or similarity; used to determine equality of 2 individuals
        :param with_elem: use element in gene tuple
        :param max_sim: how much percent can 2 individuals genes share before they are considered equal
        :return:
        """
        # replace doubles with new individuals
        # print("Population size:", len(new_population))
        # print("(pre) Unique idividuals:", len(set([indiv.genes.as_tuple() for indiv in new_population])))

        method = self.similarity_method
        max_sim = self.max_sim

        ids = set()
        to_remove = set()
        for indiv in new_population:
            if method == "tuple":
                if indiv.genes.as_tuple(with_elem) in ids:
                    to_remove.add(indiv)
                else:
                    ids.add(indiv.genes.as_tuple(with_elem))
            if method == "similarity":
                rem = False
                for other in ids:
                    if indiv.genes.diff(other.genes) <= 1-max_sim:
                        #print("Diff:", indiv.genes.diff(other.genes) / len(indiv.genes.as_tuple(with_elem)))
                        #print(indiv.genes.as_tuple())
                        #print(other.genes.as_tuple())
                        to_remove.add(indiv)
                        rem = True
                        break
                if not rem:
                    ids.add(indiv)

        for entry in to_remove:
            new_population.remove(entry)
            if refill:
                new_population.append(Chromosome(self.create_individual(list())))
                # print("Individual added")

        # print("Population size:", len(new_population))
        # print("(after) Unique idividuals:", len(set([indiv.genes.as_tuple() for indiv in new_population])))

        return new_population

    def create_first_generation(self):
        """Create the first population, calculate the population's fitness and
        rank the population by fitness according to the order specified.
        """
        self.create_initial_population()
        self.on_generation(self)
        self.calculate_population_fitness()
        self.rank_population()

        f, b = self.best_individual()
        self.top_individual_fitness_hist.append(f)
        self.worst_individual_fitness_hist.append(self.current_generation[-1].fitness)
        self.avg_pop_fitness_hist.append(self.get_avg_fitness())
        self.diversity_hist.append(self.get_pop_diversity())
        print("Initial population generated!")

    def create_next_generation(self):
        """Create subsequent populations, calculate the population fitness and
        rank the population by fitness in the order specified.
        """
        self.create_new_population()
        self.on_generation(self)
        self.calculate_population_fitness()
        self.rank_population()

        f, b = self.best_individual()
        self.diversity_hist.append(self.get_pop_diversity())

        if b.diff(self.prev_best):
            print("Generation:", len(self.avg_pop_fitness_hist), "::: Diversity:", f'{self.diversity_hist[-1]:.2f}',
                  "::: Fitness:", f, "::: Hash:", hash(b))
            t = b.as_tuple()
            print("Tuple:")
            for elem in range(len(t)//3):
                print("   ", t[elem:elem+(len(t)//3)])
            print(" Diff:", int(b.diff(self.prev_best) * 100), "%")
            print(b.get_deck_string())

        self.prev_best = b
        self.top_individual_fitness_hist.append(f)
        self.worst_individual_fitness_hist.append(self.current_generation[-1].fitness)
        self.avg_pop_fitness_hist.append(self.get_avg_fitness())
        self.update_hof()

    @timeit
    def run(self):
        """Run the Genetic Algorithm."""
        self.create_first_generation()

        for _ in range(1, self.generations):
            self.create_next_generation()

        print("Done!")

    def best_individual(self):
        """Return the individual with the best fitness in the current
        generation.
        """
        best = self.current_generation[0]
        return best.fitness, best.genes

    def update_hof(self):
        hof = self.remove_doubles(self.current_generation+self.hall_of_fame, refill=False)
        self.hall_of_fame = sorted(hof, key=lambda x: x.fitness, reverse=True)[:self.population_size//10]

    def last_generation(self):
        return [(member.fitness, member.genes) for member in sorted(self.current_generation, key=lambda x: x.fitness, reverse=True)]

    def get_avg_fitness(self):
        return sum([member.fitness for member in self.current_generation]) / len(self.current_generation)

    def get_pop_diversity(self):
        s = set()
        for indiv in self.current_generation:
            s.update(indiv.get_gene_set())
        return len(s) / len(self.current_generation)

    def plot(self):
        from matplotlib import pyplot as plt
        from numpy import convolve, ones

        def get_sma(ser, n):
            return convolve(ser, ones(n) / n, mode='valid')

        div_hist = [x/max(self.diversity_hist) for x in self.diversity_hist]

        length = len(self.top_individual_fitness_hist)
        sma = get_sma(self.top_individual_fitness_hist, 10)

        plt.plot(range(length), self.top_individual_fitness_hist, label="Top Fitness")
        plt.plot(range(length), self.worst_individual_fitness_hist, label="Worst fitness")
        plt.plot(range(length-len(sma), length), sma, label="Fitness SMA10")
        plt.plot(range(len(div_hist)), div_hist, label="Diversity")

        plt.plot(range(len(self.avg_pop_fitness_hist)), self.avg_pop_fitness_hist, label="Average population fitness")

        plt.grid(True)
        plt.legend()
        plt.show()


class Chromosome(object):
    """ Chromosome class that encapsulates an individual's fitness and solution
    representation.
    """
    def __init__(self, genes):
        """Initialise the Chromosome."""
        self.genes = genes
        self.fitness = 0

    def get_gene_set(self):
        return set(flatten([[axie.element]+[card.card_name for card in axie.cards] for axie in self.genes.team]))

    def __repr__(self):
        """Return initialised Chromosome representation in human readable form.
        """
        return repr((self.fitness, self.genes))
