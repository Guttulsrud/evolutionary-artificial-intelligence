import random
import numpy as np
from src.cartpole import run_cart
from src.cellular_automata.config import get_max_rule
from src.evolutionary.config import get_config_dict


def init_population(population_size):
    population = []
    config: dict = get_config_dict()

    for x in range(population_size):
        props = config['independent_properties']
        time_steps = random.randrange(props['time_steps']['min'], props['time_steps']['max'])
        width = random.randrange(props['width']['min'], props['width']['max'])
        kernel_size = random.choice(props['kernel_size']['values'])
        angle_index = random.randrange(0, width)
        action_index = random.randrange(0, width)
        rule_number = random.randrange(0, get_max_rule(kernel_size))

        population.append({
            'config': {
                'time_steps': time_steps,
                'width': width,
                'kernel_size': kernel_size,
                'angle_index': angle_index,
                'action_index': action_index,
                'rule_number': rule_number
            },
        })

    return population


def main():
    population = init_population(10)
    n_generations = 100

    # 1. build n offspring from the m parents
    # 2. obtain an n + m population by merging parens and offspring
    # 3. select m individuals to survive

    for _ in range(n_generations):
        for individual in population:
            individual['score_history'] = run_cart(individual['config'])
            individual['score'] = np.mean(individual['score_history'])

        population = sorted(population, key=lambda d: d['score'], reverse=True)

        break


if __name__ == '__main__':
    main()
