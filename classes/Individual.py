from classes.CellularAutomaton import CellularAutomaton
import numpy as np
import random


class Individual:
    def __init__(self, genotype, config):
        self.genotype = genotype
        self.phenotype = CellularAutomaton(genotype=genotype, config=config)
        self.score_history = []
        self.config = config

    def run(self, observation) -> int:
        return self.phenotype.run(observation)

    def add_fitness_score(self, score):
        self.score_history.append(score)

    def get_fitness_score(self):
        return np.mean(self.score_history)

    def get_genotype(self):
        return self.genotype

    def reproduce(self):
        child_genotype = self.mutate_genotype()
        child = Individual(child_genotype, self.config)
        return child

    def mutate_genotype(self):
        mutated_genotype = self.genotype.copy()
        mutated_genotype['rule_number'] = self.mutate_rule_number()

        return mutated_genotype

    def mutate_rule_number(self):
        rule_number = list(np.binary_repr(self.genotype['rule_number']))
        value_map = {
            '1': '0',
            '0': '1'
        }
        for idx, i in enumerate(rule_number):
            will_mutate = random.uniform(0, 1) < self.config['mutation_rate']
            if will_mutate:
                rule_number[idx] = value_map[i]
        rule_number = ''.join(rule_number)
        rule_number = int(rule_number, 2)
        return rule_number
