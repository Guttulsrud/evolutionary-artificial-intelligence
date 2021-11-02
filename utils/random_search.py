import random
from classes.Config import Config
from utils.general_utils import get_config


# Run random search for configs
def run_random_search(iterations=300):
    print(f'Running random search for {iterations} iterations. See config.json for available parameters.')
    config = get_config()

    mutation_rates = [0.01, 0.001, 0.0001, 0.00001]
    selection_criterions = ['fitness_proportional', 'elitist', 'tournament']
    fitness_functions = ['position_and_angle_based',
                         'position_based',
                         'angle_based',
                         'angle_and_time_based',
                         'time_based',
                         'total_time_steps']
    survival_rates = [0.1, 0.2]
    episodes_per_individual = [2, 3, 5]

    for _ in range(iterations):
        config['stats'] = {}
        config['mutation_rate'] = random.choice(mutation_rates)
        config['selection_criterion'] = random.choice(selection_criterions)
        config['fitness_function'] = random.choice(fitness_functions)
        config['reproduction_criterion'] = random.choice(selection_criterions)
        config['survival_rate'] = random.choice(survival_rates)
        config['episodes_per_individual'] = random.choice(episodes_per_individual)
        print(f'-------Running config {_ + 1}/{iterations}-------')
        Config(config).run()
