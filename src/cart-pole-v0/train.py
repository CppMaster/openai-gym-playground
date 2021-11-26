import gym
import numpy as np
from keras.layers import Dense
from keras.models import Sequential

env = gym.make('CartPole-v0')

x_obs_arr = []
x_action_arr = []
y_reward_arr = []

n_simulations = 100
max_step = 1000

for i_episode in range(n_simulations):
    observation = env.reset()
    t = 0
    for t in range(max_step):
        action = env.action_space.sample()
        x_obs_arr.append(observation)
        x_action_arr.append([action])
        observation, reward, done, info = env.step(action)
        if done:
            break
    total_reward = t + 1
    y_reward_arr.extend([[total_reward]] * total_reward)
env.close()

x_obs = np.array(x_obs_arr)
x_action = np.array(x_action_arr)
y_reward = np.array(y_reward_arr)

x = np.concatenate([x_obs, x_action], axis=1)

model = Sequential([
    Dense(units=8, input_dim=x.shape[1], activation="relu"),
    Dense(units=4, activation="relu"),
    Dense(units=1, activation="relu")
])
model.compile("adam", "mse")
model.fit(x, y_reward, batch_size=100, epochs=1000)

