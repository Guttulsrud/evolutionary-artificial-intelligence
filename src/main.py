import numpy as np
import matplotlib.pyplot as plt

from src.update_rules import get_update_rules

rules = get_update_rules()


def render(grid):
    plt.imshow(grid, vmin=0, vmax=1)
    plt.show()


def createCellularAutomaton(side_length, update_rule):
    grid = np.random.rand(side_length, side_length)

    for x in range(5):
        render(grid)
        grid = step(grid, update_rule)


def step(grid, update_rule):
    update_value = rules[update_rule]
    new_grid = np.zeros(shape=grid.shape)

    for row_index, row in enumerate(grid):
        for column_index, value in enumerate(row):
            new_value = update_value(grid, row_index, column_index)

            new_grid[row_index][column_index] = new_value

    return new_grid


createCellularAutomaton(side_length=20, update_rule='move_down')
