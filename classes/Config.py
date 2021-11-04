import json

from classes.Population import Population
from utils.general_utils import create_result_file, append_stats
from utils.plot import render
from tqdm import tqdm


class Config:
    def __init__(self, config: dict):
        self.time_step_history = []
        self.results = {'best': {'score': 0, 'generation': 0, 'genotype': {}}, }
        self.config = config
        self.log_file = create_result_file({'config': config, 'generations': [], 'individual_scores': []})
        self.stats = None

    def run(self):
        config = self.config
        population = Population(config)
        for generation in tqdm(range(config['evolution']['n_generations'])):

            population.run_generation()
            best_individual = population.get_best_individual()
            individuals = population.get_individuals()
            population.evolve_population()

            if config['general']['render_ca']:
                render(best_individual.get_phenotype().get_history())

            if best_individual.get_time_steps_survived() > self.results['best']['score']:
                self.save_best_score(best_individual, generation)
            self.time_step_history.append(best_individual.get_time_steps_survived())

            append_stats(file_name=self.log_file,
                         data={'individuals': [individual.get_fitness_score() for individual in individuals]},
                         type='individual_scores')
            append_stats(file_name=self.log_file, data=self.stats, type='generations')

        self.save_results()

    def save_best_score(self, best_individual, generation):
        self.results['best']['score'] = best_individual.get_time_steps_survived()

        self.stats = {'generation': generation + 1,
                      'time_steps_survived': best_individual.get_time_steps_survived(),
                      'best_individual': best_individual.get_genotype(), }

    def save_results(self):
        self.config['stats'] = {
            'best_generation': self.stats['generation'],
            'time_steps': self.stats['time_steps_survived'],
            'best_individual': self.stats['best_individual'],
            'time_step_history': self.time_step_history
        }
        with open('results/final-results.json', 'r+') as file:
            file_data = json.load(file)
            file_data['configs'].append(self.config)
            file.seek(0)
            json.dump(file_data, file, indent=4)
