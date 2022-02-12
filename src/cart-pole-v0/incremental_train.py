import gym
import numpy as np
from keras.callbacks import ModelCheckpoint
from keras.layers import Dense
from keras.models import Sequential


x_obs_arr = []
x_action_arr = []
y_reward_arr = []
sim_len_arr = []

n_warmup_sim = 100
n_incremental_sim = 100
max_step = 200
keep_n_simulations = 100
trainings_steps = 20
score_n_simulations = 10
chance_for_random = 0.5
epochs = 1000
batch_size = 50

env = gym.make('CartPole-v0')

# Warm up
for i_episode in range(n_warmup_sim):
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
    sim_len_arr.append(total_reward)


model = Sequential([
    Dense(units=8, input_dim=5, activation="relu"),
    Dense(units=4, activation="relu"),
    Dense(units=1, activation="relu")
])
model.compile("adam", "mse")


def create_training_data():
    x_obs = np.array(x_obs_arr)
    x_action = np.array(x_action_arr)
    y_reward = np.array(y_reward_arr)

    return np.concatenate([x_obs, x_action], axis=1), y_reward


x, y = create_training_data()
callbacks = [ModelCheckpoint("temp/cartpole.h5", monitor="loss")]
model.fit(x, y, batch_size=50, epochs=epochs, verbose=2, callbacks=callbacks)


def score_model():

    scores = []
    for i_episode in range(score_n_simulations):
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

    score = np.mean(scores)
    print(f"Avr score: {score}")
    return score

score_model()

# Incremental training
for training_step in range(trainings_steps):
    print(f"Training step: {training_step+1}/{trainings_steps}")
    for i_sim in range(n_incremental_sim):
        print(f"Sim: {i_sim+1}/{n_incremental_sim}")
        observation = env.reset()
        t = 0
        for t in range(max_step):

            if np.random.random() > chance_for_random:
                x_action = np.concatenate([np.array([observation, observation]), np.array([[0.0], [1.0]])], axis=1)
                pred = model.predict(x_action, batch_size=2)
                action = np.argmax(pred, axis=0)[0]
            else:
                action = env.action_space.sample()

            x_obs_arr.append(observation)
            x_action_arr.append([action])

            # env.render()
            observation, reward, done, info = env.step(action)
            if done:
                break
        total_reward = t + 1
        print(f"Score: {total_reward}")
        y_reward_arr.extend([[total_reward]] * total_reward)

    x, y = create_training_data()
    model.fit(x, y, batch_size=50, epochs=epochs, verbose=2, callbacks=callbacks)
    score_model()

env.close()
