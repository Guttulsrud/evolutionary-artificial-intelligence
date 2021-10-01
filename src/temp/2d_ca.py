import numpy as np
import matplotlib.pyplot as plt

from src.update_rules import get_update_rules

rules = get_update_rules()


def render2D(grid):
    plt.imshow(grid, vmin=0, vmax=1)
    plt.show()


def create2DCellularAutomaton(side_length, update_rule, time_steps):
    grid = np.random.rand(side_length, side_length)

    for x in range(time_steps):
        render2D(grid)
        grid = step2D(grid, update_rule)


def step2D(grid, update_rule):
    update_value = rules[update_rule]
    new_grid = np.zeros(shape=grid.shape)

    for row_index, row in enumerate(grid):
        for column_index, value in enumerate(row):
            new_value = update_value(grid, row_index, column_index)

            new_grid[row_index][column_index] = new_value

    return new_grid

# create2DCellularAutomaton(side_length=12, update_rule='move_down', time_steps=10)
