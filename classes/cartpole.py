import gym
import numpy as np

from classes.Individual import Individual

env = gym.make('CartPole-v0')
max_steps = 15_000
# env._max_episode_steps = max_steps


def run_cart(individual: Individual, config: dict) -> [int]:
    for i_episode in range(config['evolution']['episodes_per_individual']):
        observation = format_observation(env.reset())

        fitnesses = np.array([])
        for t in range(100):
            if config['general']['render_cart']:
                env.render()

            action = individual.run(observation=observation)

            observation, reward, done, info = env.step(action)
            observation = format_observation(observation)
            fitnesses = np.append(fitnesses,
                                  calculate_time_step_fitness(observation, t,
                                                              fitness_function=config['evolution']['fitness_function']))
            if done or abs(observation['cart_position']) > 4:
                individual.add_fitness_score(np.sum(fitnesses), t)
                break


def calculate_time_step_fitness(observation, total_time_steps, fitness_function='total_time_steps'):
    fitness_function_map = {
        'total_time_steps': 1,
        'position_based': np.log(1 / abs(observation['cart_position']) + 1),
        'angle_based': np.log(1 / abs(observation['pole_angle']) + 1),
        'time_based': np.log(total_time_steps + 1),
        'angle_and_time_based': total_time_steps + np.log(1 / abs(observation['pole_angle']) + 1),
        'position_and_angle_based': np.log(1 / abs(observation['cart_position']) + 1) +
                                    np.log(1 / abs(observation['pole_angle']) + 1)
    }
    return fitness_function_map[fitness_function]


def format_observation(observation):
    return {
        'cart_position': observation[0],
        'cart_velocity': observation[1],
        'pole_angle': observation[2],
        'pole_angular_velocity': observation[3]
    }
