from classes.CellularAutomaton import CellularAutomaton
import numpy as np
import random

from utils.evolution import get_config_dict
from utils.general_utils import get_max_rule


class Individual:
    def __init__(self, config):
        self.genotype = self.create_genotype()
        self.phenotype = CellularAutomaton(genotype=self.genotype, config=config)
        self.score_history = []
        self.time_step_survived = []
        self.config = config

    def create_genotype(self):
        template = get_config_dict()
        kernel_size = random.choice(template['kernel_size'])
        width = random.randrange(template['width']['min'], template['width']['max'])
        time_steps = random.randrange(template['time_steps']['min'], template['time_steps']['max'])
        rule_number = random.randrange(0, get_max_rule(kernel_size))

        genotype = {
            'time_steps': time_steps,
            'width': width,
            'kernel_size': kernel_size,
            'action_index': random.randrange(0, width),
            'rule_number': rule_number,
            'pole_angle': [
                {'value': -0.4, 'index': 0},
                {'value': -0.26, 'index': 1},
                {'value': -0.13, 'index': 2},
                {'value': 0.00, 'index': 3},
                {'value': 0.13, 'index': 4},
                {'value': 0.26, 'index': 5},
                {'value': 0.4, 'index': 6},
            ],
            'pole_velocity': [
                {'value': -2, 'index': 7},
                {'value': -1.5, 'index': 8},
                {'value': -1, 'index': 9},
                {'value': -0.5, 'index': 10},
                {'value': 0, 'index': 11},
                {'value': 0.5, 'index': 12},
                {'value': 1, 'index': 13},
                {'value': 1.5, 'index': 14},
            ],
            'cart_position': [
                {'value': -4, 'index': 15},
                {'value': -2.6, 'index': 16},
                {'value': -1.3, 'index': 17},
                {'value': 0, 'index': 18},
                {'value': 1.3, 'index': 19},
                {'value': 2.6, 'index': 20},
                {'value': 4, 'index': 21},
            ],
            'cart_velocity': [
                {'value': -1, 'index': 22},
                {'value': -0.6, 'index': 23},
                {'value': -0.3, 'index': 24},
                {'value': 0, 'index': 25},
                {'value': 0.3, 'index': 26},
                {'value': 0.6, 'index': 27},
            ],

        }

        return genotype

    def run(self, observation) -> int:
        return self.phenotype.run(observation)

    def add_fitness_score(self, score, time_step):
        self.score_history.append(score)
        self.time_step_survived.append(time_step)

    def get_fitness_score(self):
        return np.mean(self.score_history) if self.score_history else 1

    def get_time_steps_survived(self):
        return np.mean(self.time_step_survived)

    def get_phenotype(self):
        return self.phenotype

    def get_genotype(self):
        return self.genotype

    def reproduce(self, other_parent):

        child_genotype = self.mutate_genotype(other_parent)

        child = Individual(child_genotype, self.config)
        return child

    def mutate_genotype(self, other_parent):
        mutated_genotype = self.genotype.copy()
        mutated_genotype['rule_number'] = self.mutate_rule_number(other_parent)

        self.mutate_gene('pole_angle')
        self.mutate_gene('pole_velocity')
        self.mutate_gene('cart_position')
        self.mutate_gene('cart_velocity')

        return mutated_genotype

    def mutate_width(self):
        will_mutate = random.uniform(0, 1) < self.config['mutation_rate']
        width = self.genotype['width']
        if will_mutate:
            positive_mutation = random.choice([0, 1])
            width += 1 if positive_mutation else -1

        return width

    def mutate_gene(self, gene_type):
        self.genotype[gene_type] = self.mutate_index(self.genotype, gene_type)
        self.genotype[gene_type] = self.mutate_threshold(self.genotype, gene_type)

    def mutate_index(self, genotype, type):
        thresholds = genotype[type]

        for idx, t in enumerate(thresholds):

            will_mutate = random.uniform(0, 1) < self.config['mutation_rate']
            if will_mutate:
                thresholds[idx]['index'] = random.randrange(0, genotype['width'])

        return thresholds

    def mutate_threshold(self, genotype, type):
        thresholds = genotype[type]

        for idx, t in enumerate(thresholds):
            will_mutate = random.uniform(0, 1) < self.config['mutation_rate']

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
            will_mutate = random.uniform(0, 1) < self.config['mutation_rate']
            if will_mutate:
                rule_number[idx] = value_map[i]
        rule_number = ''.join(rule_number)
        rule_number = int(rule_number, 2)
        return rule_number
