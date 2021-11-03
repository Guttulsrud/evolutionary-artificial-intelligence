from typing import List
from classes.Individual import Individual
from classes.cartpole import run_cart
from utils.evolution import get_criterion_function
import numpy as np


class Population:
    def __init__(self, config: dict):
        self.individuals = []
        self.config = config
        self.create()

    def create(self):
        population_limit = self.config['evolution']['population_limit']

        for _ in range(population_limit):
            self.individuals.append(Individual(config=self.config))

    def get_individuals(self) -> List[Individual]:
        return self.individuals

    def get_best_individual(self) -> Individual:
        sorted_individuals = sorted(self.individuals, key=lambda i: i.get_fitness_score(), reverse=True)
        return sorted_individuals[0]

    def run_generation(self):
        for individual in self.individuals:
            run_cart(individual, self.config)

    def evolve_population(self):
        new_population = self.select_survivors(self.individuals)


        reproduction_criterion = get_criterion_function(self.config['evolution']['reproduction_selection_criterion'])

        parents_a = reproduction_criterion(self.individuals, 1 - self.config['evolution']['survival_rate'], True)
        parents_b = reproduction_criterion(self.individuals, 1 - self.config['evolution']['survival_rate'], True)
        np.random.shuffle(parents_a)

        for parent_a, parent_b in zip(parents_a, parents_b):
            child = parent_a.reproduce(parent_b)
            new_population.append(child)
        self.individuals = sorted(new_population, key=lambda i: i.get_fitness_score(), reverse=True)

    def select_survivors(self, individuals: List[Individual]) -> List[Individual]:

        individuals = sorted(individuals, key=lambda i: i.get_fitness_score(), reverse=True)
        criterion_function = get_criterion_function(self.config['evolution']['survival_selection_criterion'])
        survivors = list(criterion_function(individuals, self.config['evolution']['survival_rate']))

        return survivors
