import random
from classes.Config import Config
from utils.general_utils import get_config
from tqdm import tqdm


# Run random search for configs
def run_random_search(iterations=100):
    print(f'Running random search for {iterations} iterations. See config.json for available parameters.')
    config = get_config()

    mutation_rates = [0.01, 0.003, 0.001]
    selection_criteria = ['fitness_proportional',
                          'elitist',
                          'rank_proportional',
                          'uniform',
                          'tournament'
                          ]
    fitness_functions = ['position_and_angle_based',
                         'position_based',
                         'angle_based',
                         'angle_and_time_based',
                         'time_based',
                         'total_time_steps']
    survival_rates = [0, 0.1, 0.2]

    for count in range(iterations):
        config['stats'] = {}
        config['evolution']['mutation_rate'] = random.choice(mutation_rates)
        config['evolution']['survival_selection_criterion'] = random.choice(selection_criteria)
        config['evolution']['fitness_function'] = random.choice(fitness_functions)
        config['evolution']['reproduction_selection_criterion'] = random.choice(selection_criteria)
        config['evolution']['survival_rate'] = random.choice(survival_rates)
        print(f'\n-------Running config {count + 1}/{iterations}-------\n')
        Config(config).run()
