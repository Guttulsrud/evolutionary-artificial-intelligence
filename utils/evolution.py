from scipy.special import softmax
import numpy as np


def get_criterion_function(criterion_name: str):
    criterion_functions = {
        'fitness_proportional': fitness_proportional_selection,
        'rank_proportional': rank_proportional,
        'tournament': tournament,
        'elitist': elitist,
        'uniform': uniform
    }

    return criterion_functions[criterion_name]


def fitness_proportional_selection(population: list, survival_rate: float, replace=False):
    logged_list = softmax(np.log([individual.get_fitness_score() + 1 for individual in population]))
    size = int(survival_rate * len(population))
    selected_population = np.random.choice(population, size=size, replace=replace, p=logged_list)

    return selected_population


def tournament(population: list, survival_rate: float, replace=False):
    size = int(survival_rate * len(population))

    selected_population = []
    for _ in range(size):
        copy = population
        contenders = np.random.choice(copy, size=20, replace=False)
        winner = sorted(contenders, key=lambda i: i.get_fitness_score(), reverse=True)[0]
        selected_population.append(winner)
        population.remove(winner)

    return selected_population


def elitist(population: list, survival_rate: float, replace=False):
    size = int(survival_rate * len(population))
    best_survivors = sorted(population, key=lambda i: i.get_fitness_score(), reverse=True)

    selected_population = best_survivors[:size]
    return selected_population


def rank_proportional(population: list, survival_rate: float, replace=False):
    population = sorted(population, key=lambda i: i.get_fitness_score(), reverse=True)
    soft_maxed_indices = softmax([idx + 1 for idx, _ in enumerate(population)][::-1])

    size = int(survival_rate * len(population))
    selected_population = np.random.choice(population, size=size, replace=replace, p=soft_maxed_indices)

    return selected_population


def uniform(population: list, survival_rate: float, replace=False):
    size = int(survival_rate * len(population))

    selected_population = np.random.choice(population, size=size, replace=replace)
    return selected_population
