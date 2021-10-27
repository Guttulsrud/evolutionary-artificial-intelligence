from classes.Population import Population
from utils.general_utils import save_to_file, append_stats
from utils.plot import render
import numpy as np
import json
import random


def main(config):
    results = {
        'best': {
            'score': 0,
            'generation': 0,
            'genotype': {}
        },
    }
    time_step_history = []
    population = Population(config)

    log_file = save_to_file({'config': config, 'generations': []})
    stats = None
    for generation in range(config['n_generations']):

        population.run_generation()

        # todo: Make me into a function
        best_individual = population.get_best_individual()

        if config['render_ca']:
            render(best_individual.get_phenotype().get_history())

        if best_individual.get_time_steps_survived() > results['best']['score']:
            results['best']['score'] = best_individual.get_time_steps_survived()
            results['best']['generation'] = generation + 1
            results['best']['genotype'] = best_individual.get_genotype()

            stats = {'Generation': generation + 1,
                     'Time steps survived': best_individual.get_time_steps_survived(),
                     'Best individual': best_individual.get_genotype(),
                     'rule': best_individual.get_genotype()['rule_number'],
                     }
            append_stats(file_name=log_file, data=stats)
        time_step_history.append(best_individual.get_time_steps_survived())
        # print({'G': generation + 1,
        #        'steps': best_individual.get_time_steps_survived(),
        #        'width': best_individual.get_genotype()['width'],
        #        'bin': np.binary_repr(best_individual.get_genotype()['rule_number']),
        #        'rule': best_individual.get_genotype()['rule_number'],
        #        'p_ang': best_individual.get_genotype()['pole_angle'],
        #        'p_vel': best_individual.get_genotype()['pole_velocity'],
        #        'c_pos': best_individual.get_genotype()['cart_position'],
        #        'c_vel': best_individual.get_genotype()['cart_velocity'],
        #        })

    config['stats'] = {
        'Best generation': stats['Generation'],
        'Time steps': stats['Time steps survived'],
        'Rule': stats['rule'],
        'time_steps_history': time_step_history
    }
    with open('new_res.json', 'r+') as file:
        # First we load existing data into a dict.
        file_data = json.load(file)
        # Join new_data with file_data inside emp_details
        file_data['configs'].append(config)
        # Sets file's current position at offset.
        file.seek(0)
        # convert back to json.
        json.dump(file_data, file, indent=4)


if __name__ == '__main__':

    # selection_criterion
    config = {
        'population_limit': 100,
        'n_generations': 100,
        'render_cart': False,
        'render_ca': False,
        'fitness_function': 'position_and_angle_based',
        'selection_criterion': 'fitness_proportional',
        'mutation_rate': 0.003,
        'survival_rate': 0.2,
        'episodes_per_individual': 2,
        'action_type': 'index',
        'ca': {
            'time_steps': {'min': 5, 'max': 6},
            'width': {'min': 5, 'max': 6},
            'kernel_size': [3],
            'pole_angle': {
                'min': -3,
                'max': 3
            },
            'pole_velocity': {
                'min': -3,
                'max': 3
            },
            'cart_position': {
                'min': -3,
                'max': 3
            },
            'cart_velocity': {
                'min': -3,
                'max': 3
            },

            'dependent_properties': {
                'inputs': [{
                    'name': 'angle_index'
                }, ],
                'output': [{
                    'name': 'action_index'
                }, ],
                'rule_number': {
                    'min': 0,
                },
            }
        }
    }

    mutation_rates = [0.01, 0.001, 0.0001, 0.00001]
    selection_criterions = ['fitness_proportional', 'elitist', 'tournament']
    fitness_functions = ['position_and_angle_based',
                         'position_based',
                         'angle_based',
                         'angle_and_time_based',
                         'time_based',
                         'total_time_steps']
    survival_rates = [0.1, 0.2]
    action_types = ['mean', 'index']
    population_limits = [300, 1000]
    episodes_per_individual = [2, 5]

    for _ in range(50):
        config['stats'] = {}
        config['mutation_rate'] = random.choice(mutation_rates)
        config['selection_criterion'] = random.choice(selection_criterions)
        config['fitness_function'] = random.choice(fitness_functions)
        config['survival_rate'] = random.choice(survival_rates)
        config['action_type'] = random.choice(action_types)
        config['population_limit'] = random.choice(population_limits)
        config['episodes_per_individual'] = random.choice(episodes_per_individual)
        print(f'Testing config #{_ + 1}:')
        print(json.dumps(config, indent=4))
        main(config)
