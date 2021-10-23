from classes.Population import Population
from utils.general_utils import save_to_file

results = {
    'best': {
        'score': 0,
        'generation': 0,
        'genotype': {}
    },
}

config = {
    'population_limit': 10,
    'n_generations': 150,
    'render_cart': False,
    'render_ca': False,
    'fitness_function': 'total_time_steps',
    'selection_criterion': 'fitness_proportional',
    'mutation_rate': 0.05,
    'survival_rate': 0.2,
    'episodes_per_individual': 1,
    'cart_max_steps': 100
}
# fitness_function options:
# total_time_steps, position_based, angle_based, time_based, angle_and_time_based


def main():
    population = Population(config)
    for generation in range(config['n_generations']):
        population.run_generation()

        # todo: Make me into a function
        individuals = population.get_individuals()
        best_rule_number = individuals[0].get_genotype()["rule_number"]
        best_gen_score = individuals[0].get_fitness_score()
        print(f'Gen: {generation + 1}. Rule: {best_rule_number}. Score: {best_gen_score}.')
        # print([str(x) for x in individuals])

        if best_gen_score > results['best']['score']:
            results['best']['score'] = best_gen_score
            results['best']['generation'] = generation + 1
            results['best']['genotype'] = individuals[0].get_genotype()

    save_to_file(results)


if __name__ == '__main__':
    main()
