from typing import List
import random
from classes.Individual import Individual
from classes.cartpole import run_cart
from utils.general_utils import get_max_rule
from utils.evolution import get_criterion_function
from scipy.special import softmax
import numpy as np


class Population:
    def __init__(self, population_limit):
        self.population_limit = population_limit
        self.individuals = []
        self.create()

    def create(self):
        for x in range(self.population_limit):
            kernel_size = 3
            rule_number = random.randrange(0, get_max_rule(kernel_size))

            individual = Individual({
                'time_steps': 5,
                'width': 5,
                'kernel_size': kernel_size,
                'angle_index': 0,
                'action_index': 2,
                'rule_number': rule_number
            })

            self.individuals.append(individual)

    def get_individuals(self) -> List[Individual]:
        return self.individuals

    def run_generation(self, render: bool = False):
        for individual in self.individuals:
            run_cart(individual, render)

        survivors = self.select_survivors()

        n = self.population_limit - len(survivors)
        soft_maxed_weights = softmax([individual.get_fitness_score() for individual in survivors])

        for _ in range(n):
            parent = np.random.choice(survivors, size=1, replace=False, p=soft_maxed_weights)[0]
            child = parent.reproduce()
            self.individuals.append(child)

    def select_survivors(self, survival_rate=0.2, criterion='fitness_proportional'):

        self.individuals = sorted(self.individuals, key=lambda i: i.get_fitness_score(), reverse=True)
        criterion_function = get_criterion_function(criterion)
        survivors = list(criterion_function(self.individuals, survival_rate))

        return survivors
