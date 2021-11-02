import json

from classes.Population import Population
from utils.general_utils import save_to_file, append_stats
from utils.plot import render
from tqdm import tqdm


class Config:
    def __init__(self, options: dict):
        self.time_step_history = []
        self.results = {'best': {'score': 0, 'generation': 0, 'genotype': {}}, }
        self.options = options
        self.log_file = save_to_file({'config': options, 'generations': []})
        self.stats = None

    def run(self):
        options = self.options

        population = Population(options)

        for generation in tqdm(range(options['evolution']['n_generations'])):

            population.run_generation()
            best_individual = population.get_best_individual()
            individuals = population.get_individuals()
            population.evolve_population()

            if options['general']['render_ca']:
                render(best_individual.get_phenotype().get_history())

            if best_individual.get_time_steps_survived() > self.results['best']['score']:
                self.save_best_score(best_individual, generation)

            self.time_step_history.append(best_individual.get_time_steps_survived())
            append_stats(file_name=self.log_file,
                         data={'individuals': [individual.get_fitness_score() for individual in individuals]})

        self.save_results()

    def save_best_score(self, best_individual, generation):
        self.results['best']['score'] = best_individual.get_time_steps_survived()
        self.results['best']['generation'] = generation + 1
        self.results['best']['genotype'] = best_individual.get_genotype()

        self.stats = {'Generation': generation + 1,
                      'Time steps survived': best_individual.get_time_steps_survived(),
                      'Best individual': best_individual.get_genotype(),
                      }

        append_stats(file_name=self.log_file, data=self.stats)

    def save_results(self):
        self.options['stats'] = {
            'Best generation': self.stats['Generation'],
            'Time steps': self.stats['Time steps survived'],
            'Time step history': self.time_step_history
        }
        with open('final_results.json', 'r+') as file:
            file_data = json.load(file)
            file_data['configs'].append(self.options)
            file.seek(0)
            json.dump(file_data, file, indent=4)
