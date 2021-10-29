from typing import List

from classes.CellularAutomaton import CellularAutomaton
import numpy as np
import random


class Individual:
    def __init__(self, genotype, config):
        self.genotype = genotype
        self.phenotype = CellularAutomaton(genotype=self.genotype, config=config)
        self.score_history = []
        self.time_step_survived = []
        self.config = config

    def run(self, observation) -> int:
        return self.phenotype.run(observation)

    def add_fitness_score(self, score: int, time_step: int) -> None:
        self.score_history.append(score)
        self.time_step_survived.append(time_step)

    def get_fitness_score(self) -> float:
        return np.mean(self.score_history) if self.score_history else 1

    def get_time_steps_survived(self) -> np.ndarray:
        return np.mean(self.time_step_survived)

    def get_phenotype(self) -> CellularAutomaton:
        return self.phenotype

    def get_genotype(self) -> dict:
        return self.genotype

    def reproduce(self, other_parent):
        child_genotype = self.mutate_genotype(other_parent)

        child = Individual(child_genotype, self.config)
        return child

    def mutate_genotype(self, other_parent) -> dict:
        mutated_genotype = self.genotype.copy()
        mutated_genotype['rule_number'] = self.mutate_rule_number(other_parent)

        self.mutate_gene('pole_angle')
        self.mutate_gene('pole_velocity')
        self.mutate_gene('cart_position')
        self.mutate_gene('cart_velocity')

        return mutated_genotype

    def mutate_gene(self, gene_type: str) -> None:
        self.genotype[gene_type] = self.mutate_index(self.genotype, gene_type)
        self.genotype[gene_type] = self.mutate_threshold(self.genotype, gene_type)

    def mutate_index(self, genotype: dict, gene_type: str):
        thresholds = genotype[gene_type]

        for idx, t in enumerate(thresholds):

            will_mutate = random.uniform(0, 1) < self.config['evolution']['mutation_rate']
            if will_mutate:
                thresholds[idx]['index'] = random.randrange(0, genotype['width'])

        return thresholds

    def mutate_threshold(self, genotype: dict, gene_type: str):
        thresholds = genotype[gene_type]

        for idx, t in enumerate(thresholds):
            will_mutate = random.uniform(0, 1) < self.config['evolution']['mutation_rate']

            if will_mutate:
                positive_mutation = random.choice([0, 1])
                thresholds[idx]['value'] += 0.01 if positive_mutation else -0.01

        return thresholds

    def mutate_rule_number(self, other_parent):

        rule_number_parent_one = list(np.binary_repr(self.genotype['rule_number']))
        rule_number_parent_two = list(np.binary_repr(other_parent.genotype['rule_number']))
        crossover_index = random.randrange(0, len(rule_number_parent_one))
        rule_number = rule_number_parent_one[:crossover_index] + rule_number_parent_two[crossover_index:]

        value_map = {
            '1': '0',
            '0': '1'
        }
        for idx, i in enumerate(rule_number):
            will_mutate = random.uniform(0, 1) < self.config['evolution']['mutation_rate']
            if will_mutate:
                rule_number[idx] = value_map[i]
        rule_number = ''.join(rule_number)
        rule_number = int(rule_number, 2)
        return rule_number
