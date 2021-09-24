import numpy as np
import matplotlib.pyplot as plt


def render(vector):
    print(vector)
    return
    vector = np.expand_dims(vector, 0)
    plt.imshow(vector, vmin=0, vmax=1, cmap='BuGn')
    plt.axis('off')
    plt.show()


def createCellularAutomaton(width, boundary_condition, update_rule):
    vector = np.random.choice([0, 1], size=(width,))

    vector = step(vector=vector, update_rule=update_rule)
    render(vector)
    pass


def step(vector, update_rule):
    return vector


def make_rule_lookup(rule_number):
    binary_keys = [np.binary_repr(x, 3) for x in range(8)]
    binary_keys = np.flipud(binary_keys)

    rule = np.binary_repr(rule_number, width=8)

    return {binary_keys[i]: rule[i] for i in range(8)}


boundary_condition = 'periodic'
# createCellularAutomaton(width=12, boundary_condition=boundary_condition, update_rule=rule)

print(make_rule_lookup(110))