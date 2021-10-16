import numpy as np
from utils.general_utils import get_max_rule
from utils.plot import render


class CellularAutomaton:
    def __init__(self, genotype: dict):
        self.genotype = genotype
        self.rule_map = self.make_rule_map()

    def run(self, observation: dict, render_ca: bool = False) -> int:
        step_range = self.genotype['time_steps']
        ca = self.create(observation=observation)
        history = [ca]

        for _ in range(step_range):
            ca = self.step(vector=ca)
            history.append(ca)
        if render_ca:
            render(history)

        action = int(ca[self.genotype['action_index']])
        return action

    def create(self, observation: dict) -> dict:
        vector = np.random.choice(['0', '1'], size=(self.genotype['width'],))
        angle_index_value = '1' if observation['pole_angle'] > 0 else '0'
        vector[self.genotype['angle_index']] = angle_index_value

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