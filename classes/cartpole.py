import gym

from classes.Individual import Individual

env = gym.make('CartPole-v0')


def run_cart(individual: Individual, render: bool = False) -> [int]:
    episodes = 3
    time_steps = 100

    for i_episode in range(episodes):
        observation = env.reset()
        for t in range(time_steps):
            if render:
                env.render()

            observation = format_observation(observation)
            action = individual.run(observation=observation)

            observation, reward, done, info = env.step(action)
            fitness = calculate_fitness(observation, t)
            if done:
                individual.add_fitness_score(fitness)
                break


def calculate_fitness(observation, time):
    return time + 1


def format_observation(observation):
    return {
        "cart_position": observation[0],
        "cart_velocity": observation[1],
        "pole_angle": observation[2],
        "pole_angular_velocity": observation[3]
    }
