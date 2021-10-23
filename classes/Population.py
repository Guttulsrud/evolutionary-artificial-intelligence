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
        template = get_config_dict()

        for x in range(self.config['population_limit']):
            kernel_size = random.choice(template['kernel_size'])
            width = random.randrange(template['width']['min'], template['width']['max'])
            time_steps = random.randrange(template['time_steps']['min'], template['time_steps']['max'])
            rule_number = random.randrange(0, get_max_rule(kernel_size))
            individual = Individual({
                'time_steps': time_steps,
                'width': width,
                'kernel_size': kernel_size,
                'action_index': random.randrange(0, width),
                'rule_number': rule_number,
                'pole':
                    {
                        'angle': [
                            {'value': -0, 'gt': True, 'index': random.randrange(0, width)},
                            {'value': -0, 'gt': False, 'index': random.randrange(0, width)},
                            {'value': -0.1, 'gt': True, 'index': random.randrange(0, width)},
                            {'value': -0.1, 'gt': False, 'index': random.randrange(0, width)},
                            {'value': -0.12, 'gt': True, 'index': random.randrange(0, width)},
                            {'value': -0.12, 'gt': False, 'index': random.randrange(0, width)},
                            {'value': -0.15, 'gt': True, 'index': random.randrange(0, width)},
                            {'value': -0.15, 'gt': False, 'index': random.randrange(0, width)},
                            {'value': 0, 'gt': True, 'index': random.randrange(0, width)},
                            {'value': 0, 'gt': False, 'index': random.randrange(0, width)},
                            {'value': 0.1, 'gt': True, 'index': random.randrange(0, width)},
                            {'value': 0.1, 'gt': False, 'index': random.randrange(0, width)},
                            {'value': 0.12, 'gt': True, 'index': random.randrange(0, width)},
                            {'value': 0.12, 'gt': False, 'index': random.randrange(0, width)},
                            {'value': 0.15, 'gt': True, 'index': random.randrange(0, width)},
                            {'value': 0.15, 'gt': False, 'index': random.randrange(0, width)},
                        ],
                        'velocity': [
                            {'value': -1, 'gt': True, 'index': random.randrange(0, width)},
                            {'value': -1, 'gt': False, 'index': random.randrange(0, width)},
                            {'value': -2, 'gt': True, 'index': random.randrange(0, width)},
                            {'value': -2, 'gt': False, 'index': random.randrange(0, width)},
                            {'value': 0, 'gt': True, 'index': random.randrange(0, width)},
                            {'value': 0, 'gt': False, 'index': random.randrange(0, width)},
                            {'value': 0.3, 'gt': True, 'index': random.randrange(0, width)},
                            {'value': 0.3, 'gt': False, 'index': random.randrange(0, width)},
                            {'value': 0.5, 'gt': True, 'index': random.randrange(0, width)},
                            {'value': 0.5, 'gt': False, 'index': random.randrange(0, width)},
                            {'value': 1, 'gt': True, 'index': random.randrange(0, width)},
                            {'value': 1, 'gt': False, 'index': random.randrange(0, width)},
                            {'value': 2, 'gt': True, 'index': random.randrange(0, width)},
                            {'value': 2, 'gt': False, 'index': random.randrange(0, width)},
                        ]
                    },

                'cart': {
                    'position': [
                        {'value': -0.01, 'gt': True, 'index': random.randrange(0, width)},
                        {'value': -0.01, 'gt': False, 'index': random.randrange(0, width)},
                        {'value': -0.02, 'gt': True, 'index': random.randrange(0, width)},
                        {'value': -0.03, 'gt': False, 'index': random.randrange(0, width)},
                        {'value': -0.03, 'gt': True, 'index': random.randrange(0, width)},
                        {'value': -0.03, 'gt': False, 'index': random.randrange(0, width)},
                        {'value': 0, 'gt': True, 'index': random.randrange(0, width)},
                        {'value': 0, 'gt': False, 'index': random.randrange(0, width)},
                        {'value': 0.01, 'gt': True, 'index': random.randrange(0, width)},
                        {'value': 0.01, 'gt': False, 'index': random.randrange(0, width)},
                        {'value': 0.02, 'gt': True, 'index': random.randrange(0, width)},
                        {'value': 0.03, 'gt': False, 'index': random.randrange(0, width)},
                        {'value': 0.03, 'gt': True, 'index': random.randrange(0, width)},
                        {'value': 0.03, 'gt': False, 'index': random.randrange(0, width)},
                    ],
                    'velocity': [
                        {'value': -0.2, 'gt': True, 'index': random.randrange(0, width)},
                        {'value': -0.2, 'gt': False, 'index': random.randrange(0, width)},
                        {'value': -0.3, 'gt': True, 'index': random.randrange(0, width)},
                        {'value': -0.3, 'gt': False, 'index': random.randrange(0, width)},
                        {'value': -0.5, 'gt': True, 'index': random.randrange(0, width)},
                        {'value': -0.5, 'gt': False, 'index': random.randrange(0, width)},
                        {'value': 0, 'gt': True, 'index': random.randrange(0, width)},
                        {'value': 0, 'gt': False, 'index': random.randrange(0, width)},
                        {'value': 0.2, 'gt': True, 'index': random.randrange(0, width)},
                        {'value': 0.2, 'gt': False, 'index': random.randrange(0, width)},
                        {'value': 0.3, 'gt': True, 'index': random.randrange(0, width)},
                        {'value': 0.3, 'gt': False, 'index': random.randrange(0, width)},
                        {'value': 0.5, 'gt': True, 'index': random.randrange(0, width)},
                        {'value': 0.5, 'gt': False, 'index': random.randrange(0, width)},
                    ]
                },

            }, config=self.config)

            self.individuals.append(individual)

    def get_individuals(self) -> List[Individual]:
        return self.individuals

    def run_generation(self):
        for individual in self.individuals:
            run_cart(individual, self.config)

        survivors = self.select_survivors()

        # survivors = self.individuals

        n = self.config['population_limit'] - len(survivors)
        soft_maxed_weights = softmax([individual.get_fitness_score() for individual in survivors])

        for _ in range(n):
            parent = np.random.choice(survivors, size=1, replace=False, p=soft_maxed_weights)[0]
            child = parent.reproduce()
            self.individuals.append(child)

    def select_survivors(self):

        self.individuals = sorted(self.individuals, key=lambda i: i.get_fitness_score(), reverse=True)
        criterion_function = get_criterion_function(self.config['selection_criterion'])
        survivors = list(criterion_function(self.individuals, self.config['survival_rate']))

        return survivors
