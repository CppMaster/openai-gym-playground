import gym
import numpy as np
import tensorflow as tf
import seaborn as sns
import matplotlib.pyplot as plt
import time

# load environment
from keras.models import load_model

env = gym.make('CartPole-v0')


class Agent:
    def __init__(self, path: str):
        self.model = load_model(path)

    def policy(self, _state) -> int:
        state_tensor = tf.convert_to_tensor(_state)
        state_tensor = tf.expand_dims(state_tensor, 0)
        action_values = self.model(state_tensor, training=False)
        return tf.argmax(action_values[0]).numpy()


# path of the model (None for random actions)
# model_path = "runs_for_workshop/cartpole-deep-q.h5"
model_path = None
if model_path:
    agent = Agent(model_path)
    policy = agent.policy
else:
    policy = lambda _: env.action_space.sample()


# track total reward for each episode
total_rewards = []

# run 10 episodes
for i_episode in range(10):

    # start the episode
    state = env.reset()
    episode_reward = 0.0

    # run maximum of 200 steps
    for timestep in range(200):

        # render current step
        env.render()
        print(state)
        time.sleep(0.1)

        # choose an action
        action = policy(state)

        # perform the action
        state, reward, done, info = env.step(action)

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
