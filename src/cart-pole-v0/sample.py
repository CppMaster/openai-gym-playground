import gym
import numpy as np

env = gym.make('CartPole-v0')

total_rewards = []

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
    total_rewards.append(t)
env.close()

print(np.mean(total_rewards))
