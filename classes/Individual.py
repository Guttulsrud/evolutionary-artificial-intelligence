from classes.CellularAutomaton import CellularAutomaton
import numpy as np
import random


class Individual:
    def __init__(self, genotype):
        self.genotype = genotype
        self.phenotype = CellularAutomaton(genotype=genotype)
        self.score_history = []

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
        child = Individual(child_genotype)

        return child

    def mutate_genotype(self):
        mutated_genotype = self.genotype.copy()
        rule_number = list(np.binary_repr(self.genotype['rule_number']))
        temp = {
            '1': '0',
            '0': '1'
        }
        mutation_rate = 0.2
        for idx, i in enumerate(rule_number):
            will_mutate = random.uniform(0, 1) < mutation_rate

            if will_mutate:
                rule_number[idx] = temp[i]

        rule_number = ''.join(rule_number)
        mutated_genotype['rule_number'] = int(rule_number, 2)

        return mutated_genotype
