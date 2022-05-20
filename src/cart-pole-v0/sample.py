import gym
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


# load environment
env = gym.make('CartPole-v0')

# make it deterministic
seed = 42
env.seed(seed)
env.action_space.seed(seed)

# track total reward for each episode
total_rewards = []

# run 10 episodes
for i_episode in range(10):

    # start the episode
    observation = env.reset()
    episode_reward = 0.0

    # run maximum of 200 steps
    for timestep in range(200):

        # render current step
        env.render()
        print(observation)

        # choose a random action
        action = env.action_space.sample()

        # perform the action
        observation, reward, done, info = env.step(action)

        # accumulate the reward
        episode_reward += reward

        # end the episode
        if done:
            print(f"Episode finished after {timestep + 1} timesteps. Reward: {episode_reward}")
            break

    total_rewards.append(episode_reward)

env.close()

# show statistics
mean_total_reward = np.mean(total_rewards)
print(f"Mean total reward: {mean_total_reward}")

# plot rewards
g = sns.lineplot(range(len(total_rewards)), total_rewards)
# draw mean line
plt.axhline(mean_total_reward, c="r")
plt.show()
