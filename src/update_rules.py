def move_up(grid, row, column):
    return grid[(row + 1) % 10][column]


def move_down(grid, row, column):
    return grid[(row - 1) % 10][column]


def get_update_rules():
    return {
        'move_up': move_up,
        'move_down': move_down
    }
