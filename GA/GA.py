import random
import copy
import numpy as np
import time
from matplotlib import pyplot as plt
import torch
from matplotlib.patches import Circle

class Chromosome:
    """
    Class Chromosome represents one chromosome which consists of genetic code and value of
    fitness function.
    Genetic code represents potential solution to problem - the list of locations that are selected
    as medians.
    """

    def __init__(self, content, fitness):
        self.content = content
        self.fitness = fitness

    def __str__(self): return "%s f=%d" % (self.content, self.fitness)

    def __repr__(self): return "%s f=%d" % (self.content, self.fitness)


class GeneticAlgorithm:

    def __init__(self, n, m, p, cost_matrix, apply_hypermutation=True):

        self.time = None
        # self.num_facilities = num_facilities
        self.user_num = n
        self.fac_num = m
        self.p = p
        self.cost_matrix = cost_matrix

        self.apply_hypermutation = apply_hypermutation

        self.iterations = 100  # Maximal number of iterations
        self.current_iteration = 0
        self.generation_size = 25  # Number of individuals in one generation
        self.reproduction_size = 10  # Number of individuals for reproduction

        self.mutation_prob = 0.3  # Mutation probability
        self.hypermutation_prob = 0.03  # Hypermutation probability
        self.hypermutation_population_percent = 10

        self.top_chromosome = None  # Chromosome that represents solution of optimization process

    def mutation(self, chromosome):
        """
        Applies mutation over chromosome with probability self.mutation_prob
        In this process, a randomly selected median is replaced with a randomly selected demand point.
        """

        mp = random.random()
        if mp < self.mutation_prob:
            # index of randomly selected median:
            i = random.randint(0, len(chromosome) - 1)
            # demand points without current medians:
            demand_points = [element for element in range(0, self.fac_num) if element not in chromosome]
            # replace selected median with randomly selected demand point:
            chromosome[i] = random.choice(demand_points)

        return chromosome

    def crossover(self, parent1, parent2):

        identical_elements = [element for element in parent1 if element in parent2]

        # If the two parents are equal to each other, one of the parents is reproduced unaltered for the next generation
        # and the other parent is deleted, to avoid that duplicate individuals be inserted into the population.
        if len(identical_elements) == len(parent1):
            return parent1, None

        exchange_vector_for_parent1 = [element for element in parent1 if element not in identical_elements]
        exchange_vector_for_parent2 = [element for element in parent2 if element not in identical_elements]

        c = random.randint(0, len(exchange_vector_for_parent1) - 1)

        for i in range(c):
            exchange_vector_for_parent1[i], exchange_vector_for_parent2[i] = exchange_vector_for_parent2[i], \
                                                                             exchange_vector_for_parent1[i]

        child1 = identical_elements + exchange_vector_for_parent1
        child2 = identical_elements + exchange_vector_for_parent2

        return child1, child2

    def cost_to_nearest_median(self, user, medians):
        """ For given facility, returns cost to its nearest median """
        min_cost = self.cost_matrix[user, medians[0]]
        for median in medians:
            if min_cost > self.cost_matrix[user, median]:
                min_cost = self.cost_matrix[user, median]
        return min_cost

    def fitness(self, chromosome):
        """ Calculates fitness of given chromosome """
        cost_sum = 0
        N = self.user_num
        for i in range(N):
            cost_sum += self.cost_to_nearest_median(i, chromosome)
        return cost_sum

    def initial_random_population(self):
        """
        Creates initial population by generating self.generation_size random individuals.
        Each individual is created by randomly choosing p facilities to be medians.
        """

        init_population = []
        for k in range(self.generation_size):
            rand_medians = []
            facilities = list(range(self.fac_num))
            for i in range(self.p):
                rand_median = random.choice(facilities)
                rand_medians.append(rand_median)
                facilities.remove(rand_median)
            init_population.append(rand_medians)

        init_population = [Chromosome(content, self.fitness(content)) for content in init_population]
        self.top_chromosome = min(init_population, key=lambda chromo: chromo.fitness)
        print("Current top solution: %s" % self.top_chromosome)
        return init_population

    def selection(self, chromosomes):
        """Ranking-based selection method"""

        # Chromosomes are sorted ascending by their fitness value
        chromosomes.sort(key=lambda x: x.fitness)
        L = self.reproduction_size
        selected_chromosomes = []

        for i in range(self.reproduction_size):
            j = L - np.floor((-1 + np.sqrt(1 + 4 * random.uniform(0, 1) * (L ** 2 + L))) / 2)
            selected_chromosomes.append(chromosomes[int(j)])
        return selected_chromosomes

    def create_generation(self, for_reproduction):
        """
        Creates new generation from individuals that are chosen for reproduction,
        by applying crossover and mutation operators.
        Size of the new generation is same as the size of previous.
        """
        new_generation = []

        while len(new_generation) < self.generation_size:
            parents = random.sample(for_reproduction, 2)
            child1, child2 = self.crossover(parents[0].content, parents[1].content)

            self.mutation(child1)
            new_generation.append(Chromosome(child1, self.fitness(child1)))

            if child2 is not None and len(new_generation) < self.generation_size:
                self.mutation(child2)
                new_generation.append(Chromosome(child2, self.fitness(child2)))

        return new_generation

    def nearest_median(self, facility, medians):
        """ Returns the nearest median for given facility """
        min_cost = self.cost_matrix[facility, medians[0]]
        nearest_med = medians[0]
        for median in medians:
            if min_cost > self.cost_matrix[facility, median]:
                nearest_med = median
        return nearest_med

    # def initial_population_with_center_point(self):
    #     """
    #     Creates initial population.
    #     Based on paper: Oksuz, Satoglu, Kayakutlu: 'A Genetic Algorithm for the P-Median Facility Location Problem'
    #     """
    #
    #     init_population = []
    #     for k in range(self.generation_size):
    #         N = self.user_num
    #         M = self.fac_num
    #         # Randomly select p-medians
    #         medians = []
    #         facilities = list(range(M))
    #         for i in range(self.p):
    #             rand_median = random.choice(facilities)
    #             medians.append(rand_median)
    #             facilities.remove(rand_median)
    #
    #         # Assign all demand points to nearest median
    #         median_nearestpoints_map = dict((el, []) for el in medians)
    #         for i in range(N):
    #             median_nearestpoints_map[self.nearest_median(i, medians)].append(i)
    #
    #         n = len(medians)
    #         # For each median
    #         for i in range(n):
    #             median = medians[i]
    #             # Determine the center point which has minimum distance to all demand points
    #             # that assigned this median
    #             min_dist = float(np.inf)
    #             center_point = median
    #
    #             cluster = [median] + median_nearestpoints_map[median]
    #             for point in cluster:
    #                 dist = 0
    #                 for other_point in cluster:
    #                     dist += self.cost_matrix[point, other_point]
    #                 if dist < min_dist:
    #                     min_dist = dist
    #                     center_point = point
    #
    #             # Replace the median with center point
    #             medians[i] = center_point
    #
    #         init_population.append(medians)
    #
    #     init_population = [Chromosome(content, self.fitness(content)) for content in init_population]
    #     self.top_chromosome = min(init_population, key=lambda chromo: chromo.fitness)
    #     print("Current top solution: %s" % self.top_chromosome)
    #     return init_population

    def optimize(self):

        start_time = time.time()

        chromosomes = self.initial_random_population()

        while self.current_iteration < self.iterations:
            # print("Iteration: %d" % self.current_iteration)

            # From current population choose individuals for reproduction
            for_reproduction = self.selection(chromosomes)

            # Create new generation from individuals that are chosen for reproduction
            chromosomes = self.create_generation(for_reproduction)

            if self.apply_hypermutation:
                hp = random.random()
                if hp < self.hypermutation_prob:
                    print("Hypermutation...")

                    chromosomes_content = [chromo.content for chromo in chromosomes]

                    # choose individuals on which hypermutation will be applied
                    k = int(self.generation_size * self.hypermutation_population_percent / 100)
                    individuals_subset = random.sample(chromosomes_content, k)

                    for individual in individuals_subset:
                        chromosomes_content.remove(individual)

                    new_individuals_subset = self.hypermutation(individuals_subset)

                    for individual in new_individuals_subset:
                        chromosomes_content.append(individual)

                    chromosomes = [Chromosome(chromo_content, self.fitness(chromo_content)) for chromo_content in
                                   chromosomes_content]

            self.current_iteration += 1

            chromosome_with_min_fitness = min(chromosomes, key=lambda chromo: chromo.fitness)
            if chromosome_with_min_fitness.fitness < self.top_chromosome.fitness:
                self.top_chromosome = chromosome_with_min_fitness

        end_time = time.time()
        self.time = end_time - start_time
        hours, rem = divmod(end_time - start_time, 3600)
        minutes, seconds = divmod(rem, 60)

        print()
        print("Final top solution: %s" % self.top_chromosome)
        print('Time: {:0>2}:{:0>2}:{:05.4f}'.format(int(hours), int(minutes), seconds))

    def hypermutation(self, individuals_subset):

        N = self.fac_num
        n = len(individuals_subset)

        # FOR EACH individual X from selected individuals DO
        for idx in range(n):
            X = individuals_subset[idx]

            # Let H be the set of facility indexes that are not currently present
            # in the genotype of individual X
            H = [element for element in range(N) if element not in X]

            # FOR EACH facility index “i” included in set H DO
            for i in H:

                best = X

                # FOR EACH facility index “j” that is currently present in the genotype of
                # the individual X DO
                for j in X:

                    # Let Y be a new individual with the set of facilities given by: (X – {j}) ∪ {i}
                    Y = copy.deepcopy(X)
                    Y.remove(j)
                    Y = Y + [i]

                    if self.fitness(Y) < self.fitness(best):
                        best = Y

                if self.fitness(best) < self.fitness(X):
                    # Insert the new X into the population, replacing the old X
                    individuals_subset[idx] = best

        return individuals_subset


if __name__ == '__main__':
    # torch.manual_seed(1234)
    # n_users = 100
    # n_facilities = 50
    # n_centers = 15
    # users = [(random.random(), random.random()) for i in range(n_users)]
    # facilities = [(random.random(), random.random()) for i in range(n_facilities)]
    #
    # users, facilities = np.array(users), np.array(facilities)
    # distance = np.sum((users[:, np.newaxis, :] - facilities[np.newaxis, :, :]) ** 2, axis=-1) ** 0.5
    #
    # start_time = time.time()
    # genetic = GeneticAlgorithm(n_users, n_facilities, n_centers, distance)
    # genetic.optimize()
    # obj = genetic.top_chromosome.fitness
    # centers = genetic.top_chromosome.content
    # time = genetic.time
    # print("The Set of centers are: %s" % centers)
    # print("The objective is: %s" % obj)
    import pickle
    with open('./Test/PM/PM15.pkl', 'rb') as f:
        loaded_data = pickle.load(f)
    ls = loaded_data[0]["users"]
    sitedf = loaded_data[0]["facilities"]
    p = 15
    dist = (ls[:, None, :] - sitedf[None, :, :]).norm(p=2, dim=-1)
    genetic = GeneticAlgorithm(len(ls), len(sitedf), p, dist)
    genetic.optimize()
    obj = genetic.top_chromosome.fitness
    centers = genetic.top_chromosome.content
    time = genetic.time

    print("The Set of centers are: %s" % centers)
    print("The objective is: %s" % obj)