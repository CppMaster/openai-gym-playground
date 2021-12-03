import gym

env = gym.make('MountainCarContinuous-v0')

for i_episode in range(20):
    observation = env.reset()
    t = 0
    for t in range(1000):
        env.render()
        print(observation)
        action = env.action_space.sample()
        observation, reward, done, info = env.step(action)
        if done:
            print("Episode finished after {} timesteps".format(t+1))
            break
env.close()
