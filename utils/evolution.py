from scipy.special import softmax
import numpy as np


def get_config_dict() -> dict:
    return {
        'time_steps': {
            'max': 50,
            'min': 10,
        },
        'kernel_size': [3, 5],
        'width': {
            'max': 50,
            'min': 10,
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
        # 'boundary_condition': 'periodic'
    }


def get_criterion_function(criterion_name: str):
    criterion_functions = {
        'fitness_proportional': fitness_proportional_selection
    }

    return criterion_functions[criterion_name]


def fitness_proportional_selection(population: list, survival_rate: float = 0.2):
    soft_maxed_weights = softmax([individual.get_fitness_score() for individual in population])
    size = int(survival_rate * len(population))
    survivors = np.random.choice(population, size=size, replace=False, p=soft_maxed_weights)
    return survivors
