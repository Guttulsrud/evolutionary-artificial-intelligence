from typing import List
import random
from classes.Individual import Individual
from classes.cartpole import run_cart
from utils.general_utils import get_max_rule
from utils.evolution import get_criterion_function, get_config_dict
from scipy.special import softmax
import numpy as np


class Population:
    def __init__(self, config):
        self.individuals = []
        self.config = config
        self.create()

    def create(self):

        for x in range(self.config['population_limit']):
            individual = Individual(config=self.config)

            self.individuals.append(individual)

    def get_individuals(self) -> List[Individual]:
        return self.individuals

    def get_best_individual(self):
        sorted_individuals = sorted(self.individuals, key=lambda i: i.get_fitness_score(), reverse=True)
        return sorted_individuals[0]

    def run_generation(self):
        for individual in self.individuals:
            run_cart(individual, self.config)

        # print(max([x.get_fitness_score() for x in self.individuals]),
        #       min([x.get_fitness_score() for x in self.individuals]))
        survivors = self.select_survivors(self.individuals)

        n = self.config['population_limit'] - len(survivors)

        logged_list = softmax([np.log(individual.get_fitness_score()) for individual in self.individuals])

        new_population = survivors
        for _ in range(n):
            parent = np.random.choice(self.individuals, size=1, replace=False, p=logged_list)[0]
            child = parent.reproduce()
            new_population.append(child)

        self.individuals = sorted(new_population, key=lambda i: i.get_fitness_score(), reverse=True)

    def select_survivors(self, individuals):

        individuals = sorted(individuals, key=lambda i: i.get_fitness_score(), reverse=True)
        criterion_function = get_criterion_function(self.config['selection_criterion'])
        survivors = list(criterion_function(individuals, self.config['survival_rate']))

        return survivors
