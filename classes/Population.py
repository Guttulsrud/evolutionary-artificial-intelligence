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
            width = random.randrange(40, 60)
            time_steps = 20
            rule_number = random.randrange(0, get_max_rule(kernel_size))
            individual = Individual({
                'time_steps': time_steps,
                'width': width,
                'kernel_size': kernel_size,
                'pole':
                    {
                        'angle': [
                            {'value': -0.4, 'index': 0},
                            {'value': -0.26, 'index': 1},
                            {'value': -0.13, 'index': 2},
                            {'value': 0.00, 'index': 3},
                            {'value': 0.13, 'index': 4},
                            {'value': 0.26, 'index': 5},
                            {'value': 0.4, 'index': 6},

                        ],
                        'velocity': [
                            {'value': -2, 'index': 7},
                            {'value': -1.5, 'index': 8},
                            {'value': -1, 'index': 9},
                            {'value': -0.5, 'index': 10},
                            {'value': 0, 'index': 11},
                            {'value': 0.5, 'index': 12},
                            {'value': 1, 'index': 13},
                            {'value': 1.5, 'index': 14},

                        ]
                    },

                'cart': {
                    'position': [
                        {'value': -4, 'index': 15},
                        {'value': -2.6, 'index': 16},
                        {'value': -1.3, 'index': 17},
                        {'value': 0, 'index': 18},
                        {'value': 1.3, 'index': 19},
                        {'value': 2.6, 'index': 20},
                        {'value': 4, 'index': 21},

                    ],
                    'velocity': [
                        {'value': -1, 'index': 22},
                        {'value': -0.6, 'index': 23},
                        {'value': -0.3, 'index': 24},
                        {'value': 0, 'index': 25},
                        {'value': 0.3, 'index': 26},
                        {'value': 0.6, 'index': 27},
                    ]
                },
                'action_index': random.randrange(0, width),
                'rule_number': rule_number
            })

            self.individuals.append(individual)

    def get_individuals(self) -> List[Individual]:
        return self.individuals

    def run_generation(self, render: bool = False):
        for individual in self.individuals:
            run_cart(individual, render)

        # print(max([x.get_fitness_score() for x in self.individuals]),
        #       min([x.get_fitness_score() for x in self.individuals]))
        survivors = self.select_survivors(self.individuals)

        n = self.population_limit - len(survivors)
        logged_list = softmax([np.log(individual.get_fitness_score()) for individual in self.individuals])

        new_population = survivors
        for _ in range(n):
            parent = np.random.choice(self.individuals, size=1, replace=False, p=logged_list)[0]
            child = parent.reproduce()
            new_population.append(child)

        self.individuals = sorted(new_population, key=lambda i: i.get_fitness_score(), reverse=True)

    def select_survivors(self, individuals, survival_rate=0.2, criterion='fitness_proportional'):

        individuals = sorted(individuals, key=lambda i: i.get_fitness_score(), reverse=True)
        criterion_function = get_criterion_function(criterion)
        survivors = list(criterion_function(individuals, survival_rate))

        return survivors
