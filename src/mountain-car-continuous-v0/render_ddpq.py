import numpy as np
import gym
import tensorflow as tf
from keras.applications.densenet import layers
from keras.models import load_model
from src.utils.gpu import set_memory_growth


set_memory_growth()

env = gym.make('MountainCarContinuous-v0')

num_states = env.observation_space.shape[0]
upper_bound = env.action_space.high[0]
lower_bound = env.action_space.low[0]


def get_actor():
    # Initialize weights between -3e-3 and 3-e3
    last_init = tf.random_uniform_initializer(minval=-0.003, maxval=0.003)

    inputs = layers.Input(shape=(num_states,))
    out = layers.Dense(256, activation="relu")(inputs)
    out = layers.Dense(256, activation="relu")(out)
    outputs = layers.Dense(1, activation="tanh", kernel_initializer=last_init)(out)

    # Our upper bound is 2.0 for Pendulum.
    outputs = outputs * upper_bound
    model = tf.keras.Model(inputs, outputs)
    return model

actor_model = get_actor()
actor_model.load_weights("pendulum_actor_3.h5")

def policy(state):
    sampled_actions = tf.squeeze(actor_model(state))

    # We make sure action is within bounds
    legal_action = np.clip(sampled_actions, lower_bound, upper_bound)

    return [np.squeeze(legal_action)]


for i_episode in range(20):
    observation = env.reset()
    t = 0
    total_rewards = 0.0
    for t in range(1000):
        env.render()
        state_tensor = tf.convert_to_tensor(observation)
        state_tensor = tf.expand_dims(state_tensor, 0)
        action_probs = policy(state_tensor)
        # Take best action
        print(t, " : ", observation, " -> ", action_probs)
        observation, reward, done, info = env.step(action_probs)
        total_rewards += reward
        if done:
            print(f"Episode finished after {t+1} timesteps")
            break
    print(f"Total reward: {total_rewards}")
env.close()
