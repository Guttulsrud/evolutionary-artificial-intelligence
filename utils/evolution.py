from scipy.special import softmax
import numpy as np


def get_config_dict() -> dict:
    return {
        'independent_properties': {
            'time_steps': {
                'max': 10,
                'min': 10,
            },
            'kernel_size': {
                'values': [3],
            },
            'width': {
                'max': 50,
                'min': 10,
            },
        },
        'dependent_properties': {
            'inputs': [{
                'name': 'angle_index'
            }, ],
            'output': [{
                'name': 'action_index'
            }, ],
            'rule_number': {
                'max': None,
                'min': 0,
            },
        }
        # 'boundary_condition': 'periodic'
    }


def get_criterion_function(criterion_name: str):
    criterion_functions = {
        'fitness_proportional': fitness_proportional_selection
    }

    return criterion_functions[criterion_name]


def fitness_proportional_selection(population: list, survival_rate: float = 0.2):
    logged_list = softmax([np.log(individual.get_fitness_score()) for individual in population])
    size = int(survival_rate * len(population))
    survivors = np.random.choice(population, size=size, replace=False, p=logged_list)
    return survivors
