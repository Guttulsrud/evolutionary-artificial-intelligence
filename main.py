from classes.Population import Population
from utils.general_utils import save_to_file
from utils.plot import render

results = {
    'best': {
        'score': 0,
        'generation': 0,
        'genotype': {}
    },
}

config = {
    'population_limit': 300,
    'n_generations': 150,
    'render_cart': False,
    'render_ca': False,
    'fitness_function': 'total_time_steps',
    'selection_criterion': 'test',
    'mutation_rate': 0.3,
    'survival_rate': 0.3,
    'episodes_per_individual': 1,
    'cart_max_steps': 800
}


# fitness_function options:
# total_time_steps, position_based, angle_based, time_based, angle_and_time_based


def main():
    population = Population(config)
    for generation in range(config['n_generations']):
        population.run_generation()

        # todo: Make me into a function
        best_individual = population.get_best_individual()

        print(f'Gen: {generation + 1}. '
              f'Rule: {best_individual.get_genotype()["rule_number"]}. '
              f'Score: {best_individual.get_fitness_score()}.')

        if config['render_ca']:
            render(best_individual.get_phenotype().get_history())

        if best_individual.get_fitness_score() > results['best']['score']:
            results['best']['score'] = best_individual.get_fitness_score()
            results['best']['generation'] = generation + 1
            results['best']['genotype'] = best_individual.get_genotype()

    save_to_file(results)


if __name__ == '__main__':
    main()
