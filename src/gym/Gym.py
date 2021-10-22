import gym

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
