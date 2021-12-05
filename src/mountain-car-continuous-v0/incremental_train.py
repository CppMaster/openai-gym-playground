from typing import List, Union

import gym
import numpy as np
from keras.layers import Dense
from keras.models import Sequential
from keras.optimizer_v2.adam import Adam
from keras.callbacks import ReduceLROnPlateau, EarlyStopping, Callback
from tqdm import tqdm


x_obs_arr = []
x_action_arr = []
y_reward_arr = []

n_warmup_sim = 1000000
min_pos_episodes = 20
at_most_n_neg_episodes_per_pos = 1
n_incremental_sim = 10
max_step = 200
extreme_action_chance = 1.0
keep_n_simulations = 100
trainings_steps = 20
score_n_simulations = 5
chance_for_random = 0.5
epochs = 10000
batch_size = 20
render_every_n_step: Union[None, int] = None
score_every_n_epochs = 200
min_delta = 0.1
monitor = "loss"


env = gym.make('MountainCarContinuous-v0')


def try_reject_episode(obs: List[np.ndarray]) -> bool:

    return False

    max_speeds: List[float] = list()
    curr_max_speed = 0.0
    is_speed_pos_old = None

    for o in obs:
        speed = o[1]
        is_speed_pos_cur = speed > 0.0
        if is_speed_pos_old is None:
            is_speed_pos_old = is_speed_pos_cur
        if is_speed_pos_old == is_speed_pos_cur:
            if is_speed_pos_cur:
                if speed > curr_max_speed:
                    curr_max_speed = speed
            else:
                if speed < curr_max_speed:
                    curr_max_speed = speed
        else:
            if curr_max_speed:
                max_speeds.append(curr_max_speed)
                if curr_max_speed > 0.0:
                    if curr_max_speed < np.max(max_speeds):
                        return True
                else:
                    if curr_max_speed > np.min(max_speeds):
                        return True

            curr_max_speed = 0.0
        is_speed_pos_old = is_speed_pos_cur

    return False


# Warm up
pos_episodes = 0
neg_episodes = 0
for i_episode in tqdm(range(n_warmup_sim)):
    observation = env.reset()
    total_reward = 0
    t = 0
    observations = []
    actions = []
    for t in range(max_step):

        if np.random.random() < extreme_action_chance:
            action = np.array([1.0 if np.random.random() > 0.5 else -1.0])
        else:
            action = env.action_space.sample()
        observations.append(observation)
        actions.append(action)
        observation, reward, done, info = env.step(action)
        total_reward += reward
        if done:
            break

    if total_reward > 0.0:
        x_obs_arr.extend(observations)
        x_action_arr.extend(actions)
        y_reward_arr.extend([[total_reward]] * len(observations))
        pos_episodes += 1
        print(f"Successful episode {pos_episodes}/{min_pos_episodes} with reward: {total_reward}")
        if pos_episodes >= min_pos_episodes:
            break
    elif neg_episodes < at_most_n_neg_episodes_per_pos * (pos_episodes + 1):
        x_obs_arr.extend(observations)
        x_action_arr.extend(actions)
        y_reward_arr.extend([[total_reward]] * len(observations))
        neg_episodes += 1

print(f"Successful episodes: {pos_episodes}")

value_model = Sequential([
    Dense(units=8, input_dim=env.observation_space.shape[0] + env.action_space.shape[0], activation="relu"),
    Dense(units=4, activation="relu"),
    Dense(units=1, activation="relu")
])
value_model.compile(Adam(learning_rate=0.0001), "mse")


def create_training_data():
    x_obs = np.array(x_obs_arr)
    x_action = np.array(x_action_arr)
    y_reward = np.array(y_reward_arr)

    return np.concatenate([x_obs, x_action], axis=1), y_reward


x, y = create_training_data()


def score_model():

    scores = []
    for i_episode in range(score_n_simulations):
        observation = env.reset()
        total_reward = 0
        observations = []
        for t in range(max_step):

            observations.append(observation)
            if try_reject_episode(observations):
                total_reward = -20.0
                break

            x_action = np.concatenate([np.array([observation, observation]), np.array([[-1.0], [1.0]])], axis=1)
            pred = value_model.predict(x_action, batch_size=2)
            action_index = np.argmax(pred, axis=0)[0]
            action = [-1.0 if action_index == 0 else 1.0]

            if render_every_n_step and (t + 1) % render_every_n_step == 0:
                env.render()
            observation, reward, done, info = env.step(action)
            total_reward += reward
            if done:
                break
        print(f"Score: {total_reward}")
        scores.append(total_reward)

    score = np.mean(scores)
    print(f"Avr score: {score}")
    return score


# score_model()


class ScoreModel(Callback):
    def __init__(self, period=10):
        super().__init__()
        self.period = period

    def on_epoch_end(self, epoch, logs=None):
        if (epoch + 1) % self.period == 0:
            score_model()

callbacks = [
    ReduceLROnPlateau(patience=10, min_delta=min_delta, monitor=monitor, verbose=1),
    EarlyStopping(patience=30, min_delta=min_delta, monitor=monitor, verbose=1),
    ScoreModel(period=score_every_n_epochs)
]
value_model.fit(x, y, batch_size=batch_size, epochs=epochs, verbose=2, callbacks=callbacks)

# TODO: Add action_model that learns based on value_model
