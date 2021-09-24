import gym

env = gym.make('CartPole-v0')


def getAction(observation):
    return env.action_space.sample()


for i_episode in range(30):
    observation = env.reset()
    for t in range(100):
        env.render()
        action_space = env.action_space

        action = getAction(observation)

        observation, reward, done, info = env.step(action)

        # print({
        #     'observation': observation,
        #     'reward': reward,
        #     'done': done,
        #     'info': info
        # })
        if done:
            print("Episode finished after {} timesteps".format(t + 1))
            # break
env.close()
