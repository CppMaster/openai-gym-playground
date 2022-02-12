import gym
import numpy as np
from keras.callbacks import ModelCheckpoint
from keras.layers import Dense
from keras.models import Sequential
from keras.optimizer_v2.adam import Adam

x_obs_arr = []
x_action_arr = []
y_reward_arr = []

n_simulations = 100000
max_step = 200

env = gym.make('CartPole-v0')

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
model.compile(Adam(lr=0.0001), "mse")
model.fit(x, y_reward, batch_size=100, epochs=100, callbacks=ModelCheckpoint("temp/cartpole.h5", monitor="loss"))

# model = load_model("temp/cartpole.h5")

env = gym.make('CartPole-v0')

scores = []
for i_episode in range(10):
    observation = env.reset()
    t = 0
    for t in range(max_step):

        x_action = np.concatenate([np.array([observation, observation]), np.array([[0.0], [1.0]])], axis=1)
        pred = model.predict(x_action, batch_size=2)
        action = np.argmax(pred, axis=0)[0]

        # env.render()
        observation, reward, done, info = env.step(action)
        if done:
            break
    total_reward = t + 1
    print(f"Score: {total_reward}")
    scores.append(total_reward)

print(f"Avr score: {np.mean(scores)}")
env.close()

