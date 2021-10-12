import json
import random
import numpy as np
from src.cartpole import run_cart
from src.cellular_automata.config import get_max_rule
from src.evolutionary.config import get_config_dict
from scipy.special import softmax


def init_population(population_size):
    population = []
    # config: dict = get_config_dict()

    for x in range(population_size):
        # props = config['independent_properties']
        # time_steps = random.randrange(props['time_steps']['min'], props['time_steps']['max'])
        # width = random.randrange(props['width']['min'], props['width']['max'])
        # kernel_size = random.choice(props['kernel_size']['values'])
        kernel_size = 3
        # angle_index = random.randrange(0, 10)
        # action_index = random.randrange(0, width)
        rule_number = random.randrange(0, get_max_rule(kernel_size))
        population.append({
            'genotype': {
                'time_steps': 5,
                'width': 5,
                'kernel_size': kernel_size,
                'angle_index': 0,
                'action_index': 2,
                'rule_number': rule_number
            },
        })

    return population


def main():
    population_limit = 100
    n_generations = 100
    population = init_population(population_limit)

    for generation in range(n_generations):

        for individual in population:
            individual['score_history'] = run_cart(individual['genotype'])
            individual['score'] = np.mean(individual['score_history'])

        population = sorted(population, key=lambda d: d['score'], reverse=True)
        survivors = select_survivors(population)
        # print(survivors[0]['score'], survivors[-1]['score'])
        population = list(survivors)

        n = population_limit - len(survivors)
        soft_maxed_weights = softmax([x['score'] for x in survivors])

        for _ in range(n):
            parent = np.random.choice(survivors, size=1, replace=False, p=soft_maxed_weights)[0]
            child_genotype = mutate_genotype(parent['genotype'])
            population.append({'genotype': child_genotype})


        best_rule_number = population[0]["genotype"]["rule_number"]
        print(f'Gen: {generation+1}. Rule: {best_rule_number}. Score: {population[0]["score"]}.')
        # print(population[0[]])


def mutate_genotype(genotype: dict) -> dict:
    mutated_genotype = genotype.copy()
    rule_number = list(np.binary_repr(genotype['rule_number']))
    temp = {
        '1': '0',
        '0': '1'
    }
    mutation_rate = 0.25
    for idx, i in enumerate(rule_number):
        will_mutate = random.uniform(0, 1) < mutation_rate

        if will_mutate:
            rule_number[idx] = temp[i]

    rule_number = ''.join(rule_number)
    mutated_genotype['rule_number'] = int(rule_number, 2)

    return mutated_genotype


def get_criterion_function(criterion_name: str):
    criterion_functions = {
        'fitness_proportional': fitness_proportional_selection
    }

    return criterion_functions[criterion_name]


def fitness_proportional_selection(population: list, survival_rate: float = 0.2):
    soft_maxed_weights = softmax([x['score'] for x in population])
    size = int(survival_rate * len(population))
    survivors = np.random.choice(population, size=size, replace=False, p=soft_maxed_weights)
    return survivors


def select_survivors(population, survival_rate=0.2, criterion='fitness_proportional'):
    criterion_function = get_criterion_function(criterion)

    population = criterion_function(population)
    return population


if __name__ == '__main__':
    main()
