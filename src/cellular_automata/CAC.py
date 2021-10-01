import numpy as np

from src.utils.plot import render


class CellularAutomataController:
    def __init__(self, config: dict):
        self.config = config
        self.rule_map: dict = self.make_rule_map()

    def run(self, observation, render=None):
        step_range = self.config['time_steps']
        ca = self.createCellularAutomaton()
        history = [ca]

        for _ in range(step_range):
            ca = self.step(vector=ca)
            history.append(ca)
        if render:
            render(history)

    def createCellularAutomaton(self):
        vector = np.random.choice(['0'], size=(self.config['width'],))
        return vector

    def step(self, vector: list) -> list:
        n_neighbours = self.config['n_neighbours']
        low = int(-(n_neighbours / 2))
        high = int((n_neighbours / 2))

        shift_amounts = range(low, high + 1)
        shifted_vectors = [np.roll(vector, shift_amount) for shift_amount in shift_amounts]
        shifted_vectors = np.flipud(shifted_vectors)

        new_vector = []
        for key in zip(*shifted_vectors):
            key_as_str = ''.join(key)
            output = self.rule_map[key_as_str]
            new_vector.append(output)

        return new_vector

    def getMaxRule(self):
        return 2 ** (2 ** (self.config['n_neighbours'] + 1))

    def make_rule_map(self):
        rule_number = self.config['rule_number']
        n_neighbours = self.config['n_neighbours']
        max_rule = self.getMaxRule()

        if n_neighbours % 2:
            print(f'n_neighbours has to be even. Was {n_neighbours}. Exiting....')
            exit()

        n_configurations = 2 ** (n_neighbours + 1)

        if rule_number > max_rule:
            print(f'Rule number "{rule_number}" is out of bounds. \n '
                  f'With {n_neighbours} neighbours, max value is {max_rule}. Exiting....')
            exit()

        binary_keys = [np.binary_repr(x, n_neighbours + 1) for x in range(n_configurations)]
        binary_keys = np.flipud(binary_keys)
        rule = np.binary_repr(rule_number, width=n_configurations)

        return {binary_keys[i]: rule[i] for i in range(n_configurations)}
