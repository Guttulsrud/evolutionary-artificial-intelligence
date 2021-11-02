import numpy as np
from utils.general_utils import get_max_rule
import random


class CellularAutomaton:
    def __init__(self, config: dict, genotype: dict = None):
        self.history = []
        self.genotype = genotype
        self.config = config
        self.mutated_genotype = None
        if not genotype:
            self.genotype = self.create_genotype()
        self.rule_map = self.make_rule_map()

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

    def get_history(self):
        return self.history

    def run(self, observation: dict) -> int:
        step_range = self.genotype['time_steps']
        ca = self.create(observation=observation)
        self.history = [ca]

        for _ in range(step_range):
            ca = self.step(vector=ca)
            self.history.append(ca)

        if self.config['ca']['action_type'] == 'mean':
            mean = np.mean([int(x) for x in ca])

            action = 1 if mean > 0.5 else 0
        else:
            action = int(ca[self.genotype['action_index']])

        return action

    def create(self, observation: dict) -> dict:
        pole_angle = observation['pole_angle']
        pole_angular_velocity = observation['pole_angular_velocity']
        cart_position = observation['cart_position']
        cart_velocity = observation['cart_velocity']

        vector = np.random.choice(['0', '1'], size=(self.genotype['width'],))

        for threshold in self.genotype['pole_angle']:
            vector[threshold['index']] = determine_threshold(threshold, pole_angle)

        for threshold in self.genotype['pole_velocity']:
            vector[threshold['index']] = determine_threshold(threshold, pole_angular_velocity)

        for threshold in self.genotype['cart_position']:
            vector[threshold['index']] = determine_threshold(threshold, cart_position)

        for threshold in self.genotype['cart_velocity']:
            vector[threshold['index']] = determine_threshold(threshold, cart_velocity)

        return vector

    def step(self, vector: list) -> list:
        kernel_size = self.genotype['kernel_size']
        low = int(-(kernel_size / 2))
        high = int((kernel_size / 2))

        shift_amounts = range(low, high + 1)
        shifted_vectors = [np.roll(vector, shift_amount) for shift_amount in shift_amounts]
        shifted_vectors = np.flipud(shifted_vectors)

        new_vector = []
        for key in zip(*shifted_vectors):
            key_as_str = ''.join(key)
            output = self.rule_map[key_as_str]
            new_vector.append(output)

        return new_vector

    def make_rule_map(self):
        rule_number = self.genotype['rule_number']
        kernel_size = self.genotype['kernel_size']
        max_rule = get_max_rule(kernel_size)

        if kernel_size % 2 == 0:
            print(f'n_neighbours has to be even. Was {kernel_size}. Exiting....')
            exit()

        n_configurations = 2 ** kernel_size

        if rule_number > max_rule:
            print(f'Rule number "{rule_number}" is out of bounds. \n '
                  f'With {kernel_size} neighbours, max value is {max_rule}. Exiting....')
            exit()

        binary_keys = [np.binary_repr(x, kernel_size) for x in range(n_configurations)]
        binary_keys = np.flipud(binary_keys)
        rule = np.binary_repr(rule_number, width=n_configurations)
        rule_map = {binary_keys[i]: rule[i] for i in range(n_configurations)}

        return rule_map

    def mutate(self, other_parent) -> dict:
        self.mutated_genotype = self.genotype.copy()

        self.mutated_genotype['rule_number'] = self.mutate_rule_number(other_parent)

        self.mutate_gene('pole_angle')
        self.mutate_gene('pole_velocity')
        self.mutate_gene('cart_position')
        self.mutate_gene('cart_velocity')

        return self.mutated_genotype

    def mutate_gene(self, gene_type: str) -> None:
        self.mutated_genotype[gene_type] = self.mutate_index(self.genotype, gene_type)
        self.mutated_genotype[gene_type] = self.mutate_threshold(self.genotype, gene_type)

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


def determine_threshold(threshold_object, value):
    threshold_value = threshold_object['value']
    return '1' if value > threshold_value else '0'
