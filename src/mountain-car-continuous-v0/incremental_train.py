import os.path
import pickle
from typing import List, Optional

import gym
import numpy as np
import yaml
from keras.layers import Dense
from keras.models import Sequential, load_model, Model
from keras.optimizer_v2.adam import Adam
from keras.callbacks import ReduceLROnPlateau, EarlyStopping, Callback
from tqdm import tqdm

from src.utils.gpu import set_memory_growth


with open("config.yml") as config_file:
    config = yaml.load(config_file, Loader=yaml.FullLoader)


set_memory_growth()

env_config = config["env"]
env = gym.make(env_config["name"])
max_step: int = env_config["max_step"]
obs_size = env.observation_space.shape[0]

reject_episode = False


def try_reject_episode(obs: List[np.ndarray]) -> bool:

    if not reject_episode:
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

value_config = config["value"]
n_warmup_sim: int = value_config["n_sim"]
min_pos_episodes: int = value_config["min_pos_episodes"]
at_most_n_neg_episodes_per_pos: float = value_config["at_most_n_neg_episodes_per_pos"]
extreme_action_chance: float = value_config["extreme_action_chance"]


def get_warmup_dataset():

    print("Creating warmup value dataset")

    x_obs_arr = []
    x_action_arr = []
    y_reward_arr = []

    pos_episodes = 0
    neg_episodes = 0
    for i_episode in tqdm(range(n_warmup_sim)):

        if pos_episodes >= min_pos_episodes:
            break

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

    return x_obs_arr, x_action_arr, y_reward_arr


def extend_and_save_arr(arr: List, path: str):
    if os.path.exists(path):
        with open(path, "rb") as f:
            arr.extend(pickle.load(f))
    pickle.dump(arr, open(path, "wb"))


x_obs_arr, x_action_arr, y_reward_arr = get_warmup_dataset()

os.makedirs("temp", exist_ok=True)
extend_and_save_arr(x_obs_arr, "temp/x_obs_arr.p")
extend_and_save_arr(x_action_arr, "temp/x_action_arr.p")
extend_and_save_arr(y_reward_arr, "temp/y_reward_arr.p")

value_scale: float = value_config["label_scale"]


def create_training_data():
    x_obs = np.array(x_obs_arr)
    x_action = np.array(x_action_arr)
    y_reward = np.array(y_reward_arr) * value_scale

    return np.concatenate([x_obs, x_action], axis=1), y_reward


x, y = create_training_data()


def get_value_model():
    value_model = Sequential([
        Dense(units=8, input_dim=env.observation_space.shape[0] + env.action_space.shape[0], activation="relu"),
        Dense(units=4, activation="relu"),
        Dense(units=1, activation="relu")
    ])
    value_model.compile(Adam(learning_rate=value_config["lr"]), "mse")
    return value_model


value_model = get_value_model()


eval_config = config["eval"]

score_n_simulations = eval_config["n_sim"]
render_every_n_step: Optional[int] = eval_config["render_every_n_step"]


def score_value_model():

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


# score_value_model()


class ScoreValueModel(Callback):
    def __init__(self, period=10):
        super().__init__()
        self.period = period

    def on_epoch_end(self, epoch, logs=None):
        if (epoch + 1) % self.period == 0:
            score_value_model()


min_delta_value = value_config["min_delta"]
monitor = value_config["monitor"]

value_callbacks = [
    ReduceLROnPlateau(patience=10, min_delta=min_delta_value, monitor=monitor, verbose=1),
    EarlyStopping(patience=30, min_delta=min_delta_value, monitor=monitor, verbose=1),
    ScoreValueModel(period=value_config["score_period"])
]
value_model.fit(
    x, y, batch_size=value_config["batch_size"], epochs=value_config["epochs"], verbose=2, callbacks=value_callbacks
)
value_model.save("temp/car_value-model.h5")
# value_model: Model = load_model("temp/car_value-model.h5")
# score_value_model()

policy_config = config["policy"]


def get_policy_model():
    policy_model = Sequential([
        Dense(units=8, input_dim=env.observation_space.shape[0], activation="relu"),
        Dense(units=4, activation="relu"),
        Dense(units=env.action_space.shape[0], activation="tanh")
    ])
    policy_model.compile(Adam(learning_rate=policy_config["lr"]), "mse")
    return policy_model


policy_model: Model = get_policy_model()


n_policy_sim = policy_config["n_sim"]
n_policy_samples = policy_config["n_policy_samples"]
sample_n_actions = policy_config["n_action_samples"]


def get_policy_dataset():

    print("Creating policy dataset")
    xq_obs_arr = []
    yq_action_arr = []

    for i_episode in tqdm(range(n_policy_sim)):
        observation = env.reset()
        for t in range(max_step):

            query_actions = [env.action_space.sample() for _ in range(sample_n_actions)] \
                            + [np.array([-1.0]), [np.array([1.0])]]
            query = np.array([np.concatenate([observation, query_actions[i]]) for i in range(sample_n_actions)])
            pred = value_model.predict(query, batch_size=sample_n_actions)
            best_action = query_actions[np.argmax(pred, axis=0)[0]]

            xq_obs_arr.append(observation)
            yq_action_arr.append(best_action)

            observation, reward, done, info = env.step(best_action)

    if n_policy_samples:
        observations = [env.observation_space.sample() for _ in range(n_policy_samples)]
        random_query = []
        for s_i in range(n_policy_samples):
            query_actions = [env.action_space.sample() for _ in range(sample_n_actions)] \
                            + [np.array([-1.0]), [np.array([1.0])]]
            observation = observations[s_i]
            query = [np.concatenate([observation, query_actions[i]]) for i in range(sample_n_actions)]
            random_query.extend(query)
        random_query = np.array(random_query)

        pred = value_model.predict(random_query, batch_size=n_policy_samples*sample_n_actions)
        situations = np.split(pred, n_policy_samples)
        for s_i, situation in enumerate(situations):
            best_action_index = np.argmax(situation, axis=0)
            query = random_query[best_action_index + s_i * sample_n_actions][0]
            best_action = query[obs_size:]
            observation = query[:obs_size]
            xq_obs_arr.append(observation)
            yq_action_arr.append(best_action)

    return xq_obs_arr, yq_action_arr


xq_obs_arr, yq_action_arr = get_policy_dataset()
extend_and_save_arr(xq_obs_arr, "temp/xq_obs_arr.p")
extend_and_save_arr(yq_action_arr, "temp/yq_action_arr.p")


def score_policy_model():

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

            action = policy_model.predict(np.array([observation]))[0]

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


class ScorePolicyModel(Callback):
    def __init__(self, period=10):
        super().__init__()
        self.period = period

    def on_epoch_end(self, epoch, logs=None):
        if (epoch + 1) % self.period == 0:
            score_policy_model()


min_delta_policy: float = policy_config["min_delta"]
policy_callbacks = [
    ReduceLROnPlateau(patience=10, min_delta=min_delta_policy, monitor=monitor, verbose=1),
    EarlyStopping(patience=30, min_delta=min_delta_policy, monitor=monitor, verbose=1),
    ScorePolicyModel(period=policy_config["score_period"])
]

# score_policy_model()

policy_model.fit(
    np.array(xq_obs_arr), np.array(yq_action_arr), batch_size=policy_config["batch_size"],
    epochs=policy_config["epochs"], verbose=2, callbacks=policy_callbacks
)
policy_model.save("temp/car_policy-model.h5")

score_policy_model()
