import numpy as np


class NeuralNetwork:
    def __init__(self, genotype: dict, config: dict):
        self.genotype = genotype
        self.config = config

    def run(self, observation: dict) -> int:
        input = list(observation.values())
        layer_1_weights = self.genotype['layer_1_weights']
        layer_2_weights = self.genotype['layer_2_weights']
        output_weights = self.genotype['output_weights']
        activation_function = self.genotype['activation_function']

        layer_1_output = activation_function(np.dot(input, layer_1_weights))
        layer_2_output = activation_function(np.dot(layer_1_output, layer_2_weights))
        output = activation_function(np.dot(layer_2_output, output_weights))

        print(f'{input=}')
        print(f'{layer_1_weights=}')
        print(f'{layer_1_output=}')
        print(f'{layer_2_weights=}')
        print(f'{layer_2_output=}')
        print(f'{output_weights=}')
        print(f'{output=}')

        action = output.argmax(axis=0)
        return action


if __name__ == '__main__':
    config = {
        'nn': {}
    }

    genotype = {
        'layer_1_weights': np.random.rand(4, 4),
        'layer_2_weights': np.random.rand(4, 4),
        'output_weights': np.random.rand(4, 2),
        'activation_function': lambda n: np.maximum(n, 0),
    }

    nn = NeuralNetwork(genotype, config)

    observation = {1: 1, 2: 1, 3: 1, 4: 1}
    action = nn.run(observation)
