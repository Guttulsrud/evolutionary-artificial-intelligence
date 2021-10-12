import gym

from src.cellular_automata.CAC import CellularAutomataController

env = gym.make('CartPole-v0')


def run_cart(config: dict, render: bool = False) -> [int]:
    CAC = CellularAutomataController(config=config)

    episodes = 5
    time_steps = 100

    list_of_scores = []

    for i_episode in range(episodes):
        observation = env.reset()
        for t in range(time_steps):
            if render:
                env.render()

            observation = format_observation(observation)
            action = CAC.run(observation=observation, render_ca=render)

            observation, reward, done, info = env.step(action)
            score = t + 1
            if done:
                list_of_scores.append(score)
                break

    env.close()
    return list_of_scores


def format_observation(observation):
    return {
        "cart_position": observation[0],
        "cart_velocity": observation[1],
        "pole_angle": observation[2],
        "pole_angular_velocity": observation[3]
    }


# run_cart({'time_steps': 5, 'width': 5, 'kernel_size': 3, 'angle_index': 0, 'action_index': 2,
#           'rule_number': 121}, render=True)
