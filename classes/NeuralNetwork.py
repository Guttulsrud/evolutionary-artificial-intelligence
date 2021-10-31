import numpy as np


class NeuralNetwork:
    def __init__(self, genotype: dict, config: dict):
        # self.shapes = [np.random.rand(4, genotype['layers'][0])]

        self.shapes = [(4, genotype['layers'][0])]
        weights_count = self.shapes[0][0]*self.shapes[0][1]
        for count, layer_node_count in enumerate(genotype['layers'][1:]):
            weights_count += genotype['layers'][count] * layer_node_count
            self.shapes.append((genotype['layers'][count], layer_node_count))

        weights = np.random.random(weights_count)

        network = np.reshape(weights, self.shapes)
        print(network)
        print(self.shapes)
        print(weights)
        self.genotype = genotype
        self.config = config

    def run(self, observation: dict) -> int:
        input = list(observation.values())

        layer_1_output = self.genotype['activation_function'](np.dot(input, layer_1_weights))
        layer_2_output = self.genotype['activation_function'](np.dot(layer_1_output, layer_2_weights))
        output = self.genotype['activation_function'](np.dot(layer_2_output, output_weights))

        action = output.argmax(axis=0)
        return action

    def mutate(self):
        pass


if __name__ == '__main__':
    config = {
        'nn': {}
    }

    genotype = {
        'layers': [4, 4, 2],
        'activation_function': lambda n: np.maximum(n, 0),
    }

    nn = NeuralNetwork(genotype, config)

    observation = {1: 1, 2: 1, 3: 1, 4: 1}
    action = nn.run(observation)
