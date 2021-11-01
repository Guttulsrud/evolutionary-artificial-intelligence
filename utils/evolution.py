from scipy.special import softmax
import numpy as np


def get_criterion_function(criterion_name: str):
    criterion_functions = {
        'fitness_proportional': fitness_proportional_selection,
        'tournament': tournament,
        'elitist': elitist
    }

    return criterion_functions[criterion_name]


def fitness_proportional_selection(population: list, survival_rate: float, replace=False):
    logged_list = softmax(np.log([individual.get_fitness_score() + 1 for individual in population]))
    size = int(survival_rate * len(population))
    survivors = np.random.choice(population, size=size, replace=replace, p=logged_list)

    return survivors


def tournament(population: list, survival_rate: float, replace=False):
    size = int(len(population) / 2)
    survivors = np.random.choice(population, size=size, replace=replace)
    best_survivors = sorted(survivors, key=lambda i: i.get_fitness_score(), reverse=True)
    survivors = best_survivors[:int(survival_rate * len(best_survivors))]
    return survivors


def elitist(population: list, survival_rate: float):
    size = int(survival_rate * len(population))
    best_survivors = sorted(population, key=lambda i: i.get_fitness_score(), reverse=True)

    survivors = best_survivors[:size]
    return survivors
