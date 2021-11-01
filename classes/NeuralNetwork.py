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
        self.genotype['weights'] = []
        self.activation_function = lambda n: np.maximum(n, 0)

        self.build_network()

    def build_network(self):
        self.nodes_per_layer = [4, *self.genotype['hidden_layers'], 2]

        for count, current_layer_node_count in enumerate(self.nodes_per_layer[1:]):
            previous_layer_node_count = self.nodes_per_layer[count]
            self.layer_weight_count.append(previous_layer_node_count * current_layer_node_count)
            self.layer_shapes.append((previous_layer_node_count, current_layer_node_count))

        if not len(self.genotype['weights']):
            total_weights = sum(self.layer_weight_count)
            self.genotype['weights'] = np.random.random(total_weights)

        weights_iter = iter(self.genotype['weights'])
        network = [list(islice(weights_iter, elem)) for elem in self.layer_weight_count]
        self.network = [np.reshape(weights, shape) for weights, shape in zip(network, self.layer_shapes)]

    def run(self, observation: dict) -> int:
        layer_input = list(observation.values())
        for layer in self.network:
            output = self.activation_function(np.dot(layer_input, layer))
            layer_input = output
        action = output.argmax(axis=0)
        return action

    def mutate(self):
        step_size = self.config['nn']['step_size']
        weights_copy = self.genotype['weights']
        for index, weight in enumerate(self.genotype['weights']):
            will_mutate = random.uniform(0, 1) < self.config['evolution']['mutation_rate']  # This line is a duplicate
            if will_mutate:
                min = weight - step_size
                max = weight + step_size
                new_value = random.uniform(min, max)
                weights_copy[index] = new_value

        return weights_copy
