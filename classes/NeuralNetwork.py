import numpy as np
from itertools import islice
import random


class NeuralNetwork:
    def __init__(self, genotype: dict, config: dict):
        self.genotype = genotype
        self.config = config
        self.nodes_per_layer = []
        self.layer_shapes = []
        self.layer_weight_count = []
        self.network = []

        self.build_network()

    def build_network(self):
        self.nodes_per_layer = [4, *genotype['layers'], 2]

        for count, layer_node_count in enumerate(self.nodes_per_layer[1:]):
            self.layer_weight_count.append(self.nodes_per_layer[count] * layer_node_count)
            self.layer_shapes.append((self.nodes_per_layer[count], layer_node_count))

        if not self.genotype.get('weights'):
            total_weights = sum(self.layer_weight_count)
            self.genotype['weights'] = np.random.random(total_weights)


        weights_iter = iter(genotype['weights'])
        network = [list(islice(weights_iter, elem)) for elem in self.layer_weight_count]
        self.network = [np.reshape(weights, shape) for weights, shape in zip(network, self.layer_shapes)]

    def run(self, observation: dict) -> int:
        layer_input = list(observation.values())

        for layer in self.network:
            output = self.genotype['activation_function'](np.dot(layer_input, layer))
            layer_input = output
        action = output.argmax(axis=0)
        return action

    def mutate(self):
        step_size = self.config['step_size']
        weights = self.genotype['weights']
        print(weights)
        for index, weight in enumerate(self.genotype['weights']):
            will_mutate = random.uniform(0, 1) < self.config['mutation_rate']  # This line is a duplicate
            if will_mutate:
                min = weight - step_size
                max = weight + step_size
                new_value = random.uniform(min, max)
                weights[index] = new_value

        return weights


if __name__ == '__main__':
    config = {
        'mutation_rate': 0.1,
        'step_size': 0.1
    }

    genotype = {
        'layers': [4, 4],
        'activation_function': lambda n: np.maximum(n, 0),
    }

    nn = NeuralNetwork(genotype, config)

    observation = {1: 1, 2: 1, 3: 1, 4: 1}
    action = nn.run(observation)
    print(nn.mutate())
