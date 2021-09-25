import numpy as np
import matplotlib.pyplot as plt
import random
import functools


def render(vector):
    vector_as_integers = np.array(vector, dtype=int)
    plt.imshow(vector_as_integers, vmin=0, vmax=1, cmap='BuGn')
    plt.axis('off')
    plt.show()


def createCellularAutomaton(width):
    vector = np.random.choice(['0', '1'], size=(width,))
    return vector


def step(vector, rule_map, boundary_condition):
    n_neighbours = len(list(rule_map.keys())[0]) - 1
    low = int(-(n_neighbours / 2))
    high = int((n_neighbours / 2))

    shifted_vectors = [np.roll(vector, i) for i in range(high, low - 1, -1)]
    new_vector = [rule_map["".join(input_vector)] for input_vector in zip(*shifted_vectors)]

    return new_vector


def getMaxRule(n_neighbours):
    return 2 ** (2 ** (n_neighbours + 1))


@functools.lru_cache()
def make_rule_map(rule_number, n_neighbours=2):
    if n_neighbours % 2:
        print(f'n_neighbours has to be even. Was {n_neighbours}. Exiting....')
        exit()

    n_configurations = 2 ** (n_neighbours + 1)
    max_rule = getMaxRule(n_neighbours)

    if rule_number > max_rule:
        print(f'Rule number "{rule_number}" is out of bounds. \n '
              f'With {n_neighbours} neighbours, max value is {max_rule}. Exiting....')
        exit()

    binary_keys = [np.binary_repr(x, n_neighbours + 1) for x in range(n_configurations)]
    binary_keys = np.flipud(binary_keys)
    rule = np.binary_repr(rule_number, width=n_configurations)
    return {binary_keys[i]: rule[i] for i in range(n_configurations)}


config = {
    'n_neighbours': 4,
    'width': 150,
    'rule': 136123,
    'boundary_condition': 'periodic'
}

# Generate random in sample size
config['rule'] = random.randint(0, getMaxRule(config['n_neighbours']))

boundary_condition = 'periodic'
rule_map = make_rule_map(config['rule'], config['n_neighbours'])
ca = createCellularAutomaton(width=config['width'])
history = [ca]
for _ in range(150):
    ca = step(ca, rule_map, config['boundary_condition'])
    history.append(ca)
render(history)

# 1. Familiarize yourself with the models (CA, networks). DONE
# 2. Implement in Python a cellular automaton which receives argument(s) to define its
# rule.
# 3. Familiarize yourself with the cart-pole balancing environment. You can install and
# prepare this environment by following the instructions on
# this link: https://gym.openai.com/docs/.
#
# 4. Come up with a method to encode input (environment observations) into the CA and
# to decode the CA state into output (action).
# 5. Come up with a fitness function that tracks the performance of the controller.
# 6. Evolve the rule of the CA to improve its control of the cart.
# 7. Expand it to a network model (a simple neural network model with binary neurons).
# Then, evolve its parameters to improve the controller.
