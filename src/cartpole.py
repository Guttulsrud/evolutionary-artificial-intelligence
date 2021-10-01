import gym

from src.cellular_automata.CAC import CellularAutomataController

env = gym.make('CartPole-v0')

config = {
    "time_steps": 10,
    "high": 1,
    "high_threshold": 0.1,
    "low": 1,
    "low_threshold": -0.1,
    "output": 1,
    "n_neighbours": 2,
    "width": 12,
    "rule_number": 191,
    "boundary_condition": "periodic"
}

CAC = CellularAutomataController(config=config)


class Gym:
    def run(self):
        for i_episode in range(30):
            observation = env.reset()
            for t in range(100):
                env.render()

                action = CAC.run(observation=format_observation(observation))

                observation, reward, done, info = env.step(action)
                if done:
                    print("Episode finished after {} timesteps".format(t + 1))
                    break
            break
        env.close()


def format_observation(observation):
    return {
        "cart_position": observation[0],
        "cart_velocity": observation[1],
        "pole_angle": observation[2],
        "pole_angular_velocity": observation[3]
    }


Gym().run()
