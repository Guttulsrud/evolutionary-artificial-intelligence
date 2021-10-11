import numpy as np

from src.cellular_automata.config import get_max_rule
from src.utils.plot import render


class CellularAutomataController:
    def __init__(self, config: dict):
        self.config = config
        self.rule_map: dict = self.make_rule_map()

    def run(self, observation: dict, render=None) -> int:
        step_range = self.config['time_steps']
        ca = self.createCellularAutomaton()
        history = [ca]

        for _ in range(step_range):
            ca = self.step(vector=ca)
            history.append(ca)
        if render:
            render(history)

        action = ca[self.config['action_index']]
        return int(action)

    def createCellularAutomaton(self):
        vector = np.random.choice(['0'], size=(self.config['width'],))
        return vector

    def step(self, vector: list) -> list:
        kernel_size = self.config['kernel_size']
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
        rule_number = self.config['rule_number']
        kernel_size = self.config['kernel_size']
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

        return {binary_keys[i]: rule[i] for i in range(n_configurations)}
