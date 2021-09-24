import numpy as np
import matplotlib.pyplot as plt


def render(vector):
    print(vector)
    return
    vector = np.expand_dims(vector, 0)
    plt.imshow(vector, vmin=0, vmax=1, cmap='BuGn')
    plt.axis('off')
    plt.show()


def createCellularAutomaton(width):
    vector = np.random.choice([0, 1], size=(width,))

    return vector


def step(vector, rule_lookup, boundary_condition):


    print(vector)
    print(rule_lookup)
    return vector


def make_rule_lookup(rule_number):
    if rule_number > 255:
        return False

    binary_keys = [np.binary_repr(x, 3) for x in range(8)]
    binary_keys = np.flipud(binary_keys)

    rule = np.binary_repr(rule_number, width=8)

    return {binary_keys[i]: rule[i] for i in range(8)}


boundary_condition = 'periodic'
rule_map = make_rule_lookup(91)
ca = createCellularAutomaton(width=12)

step(ca, rule_map, boundary_condition)



# np.random.choice([0, 1], size=(width,))


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
