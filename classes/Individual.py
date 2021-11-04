from classes.NeuralNetwork import NeuralNetwork
from classes.CellularAutomaton import CellularAutomaton
import numpy as np
import random


class Individual:
    def __init__(self, config, parent_genotype=None):
        self.config = config
        self.genotype = parent_genotype
        self.phenotype = self.create_phenotype()
        self.score_history = []
        self.time_step_survived = []
        self.create()

    def create(self):
        if not self.genotype:
            self.genotype = self.phenotype.genotype

    def create_phenotype(self):
        config = self.config

        if config['general']['phenotype'] == 'nn':
            phenotype = NeuralNetwork(config=config, genotype=self.genotype)
        else:
            phenotype = CellularAutomaton(config=config, genotype=self.genotype)

        return phenotype

    def run(self, observation) -> int:
        self.score_history = []
        self.time_step_survived = []
        return self.phenotype.run(observation)

    def add_fitness_score(self, score: int, time_step: int) -> None:
        self.score_history.append(score)
        self.time_step_survived.append(time_step)

    def get_fitness_score(self) -> float:
        return np.mean(self.score_history) if self.score_history else 0

    def get_time_steps_survived(self) -> np.ndarray:
        return np.mean(self.time_step_survived)

    def get_phenotype(self) -> CellularAutomaton or NeuralNetwork:
        return self.phenotype

    def get_genotype(self) -> dict:
        if self.config['general']['phenotype'] == 'nn':
            self.genotype['weights'] = list(self.genotype['weights'])
        return self.genotype

    def reproduce(self, other_parent):
        genotype = self.phenotype.mutate(other_parent)

        child = Individual(parent_genotype=genotype, config=self.config)
        return child

