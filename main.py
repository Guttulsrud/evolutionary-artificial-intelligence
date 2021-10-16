from classes.Population import Population
from utils.general_utils import save_to_file

population_limit = 10
n_generations = 150
results = {
    'best': {
        'score': 0,
        'generation': 0,
        'genotype': {}
    },
}


def main():
    population = Population(population_limit)
    for generation in range(n_generations):
        population.run_generation()

        # todo: Make me into a function
        individuals = population.get_individuals()
        best_rule_number = individuals[0].get_genotype()["rule_number"]
        best_gen_score = individuals[0].get_fitness_score()
        print(f'Gen: {generation + 1}. Rule: {best_rule_number}. Score: {best_gen_score}.')

        if best_gen_score > results['best']['score']:
            results['best']['score'] = best_gen_score
            results['best']['generation'] = generation + 1
            results['best']['genotype'] = individuals[0].get_genotype()

    save_to_file(results)


if __name__ == '__main__':
    main()
