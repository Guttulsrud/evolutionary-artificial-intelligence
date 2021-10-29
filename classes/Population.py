from typing import List
import random
from classes.Individual import Individual
from classes.cartpole import run_cart
from utils.general_utils import get_max_rule
from utils.evolution import get_criterion_function
from scipy.special import softmax
import numpy as np


class Population:
    def __init__(self, config: dict):
        self.individuals = []
        self.config = config
        self.create()

    def create(self):
        population_limit = self.config['evolution']['population_limit']

        for x in range(population_limit):
            genotype = self.create_genotype()
            individual = Individual(genotype=genotype, config=self.config)

            self.individuals.append(individual)

    def create_genotype(self) -> dict:
        config = self.config['ca']

        width = random.randrange(config['width']['min'], config['width']['max'])
        kernel_size = random.choice(config['kernel_size'])
        genotype = {
            'time_steps': random.randrange(config['time_steps']['min'], config['time_steps']['max']),
            'width': width,
            'kernel_size': kernel_size,
            'action_index': random.randrange(0, width),
            'rule_number': random.randrange(0, get_max_rule(kernel_size)),
            'pole_angle': [{
                'value': random.randrange(config['pole_angle']['min'],
                                          config['pole_angle']['max']) / 100,
                'index': random.randrange(0, width)}],
            'pole_velocity': [{
                'value': random.randrange(config['pole_velocity']['min'],
                                          config['pole_velocity']['max']) / 100,
                'index': random.randrange(0, width)}],
            'cart_position': [{
                'value': random.randrange(config['cart_position']['min'],
                                          config['cart_position']['max']) / 100,
                'index': random.randrange(0, width)}],
            'cart_velocity': [{
                'value': random.randrange(config['cart_velocity']['min'],
                                          config['cart_velocity']['max']) / 100,
                'index': random.randrange(0, width)}]

        }

        return genotype

    def get_individuals(self) -> List[Individual]:
        return self.individuals

    def get_best_individual(self) -> Individual:
        sorted_individuals = sorted(self.individuals, key=lambda i: i.get_fitness_score(), reverse=True)
        return sorted_individuals[0]

    def run_generation(self):
        self.config['render_cart'] = True

        for individual in self.individuals:
            run_cart(individual, self.config)
            self.config['render_cart'] = False

        survivors = self.select_survivors(self.individuals)

        n = self.config['evolution']['population_limit'] - len(survivors)

        logged_list = softmax([np.log(individual.get_fitness_score()) for individual in self.individuals])

        new_population = survivors
        for _ in range(n):
            parents = np.random.choice(self.individuals, size=2, replace=False, p=logged_list)
            child = parents[0].reproduce(parents[1])
            new_population.append(child)

        self.individuals = sorted(new_population, key=lambda i: i.get_fitness_score(), reverse=True)

    def select_survivors(self, individuals: List[Individual]) -> List[Individual]:

        individuals = sorted(individuals, key=lambda i: i.get_fitness_score(), reverse=True)
        criterion_function = get_criterion_function(self.config['evolution']['selection_criterion'])
        survivors = list(criterion_function(individuals, self.config['evolution']['survival_rate']))

        return survivors
