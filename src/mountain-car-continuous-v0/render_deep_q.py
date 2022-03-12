import numpy as np
import gym
import tensorflow as tf
from keras.models import load_model
from src.utils.gpu import set_memory_growth


set_memory_growth()

num_actions = 5
actions = np.linspace(-1.0, 1.0, num=num_actions)
model = load_model("temp/keras_deep_q_sample/model-tmp.h5")

env = gym.make('MountainCarContinuous-v0')

for i_episode in range(20):
    observation = env.reset()
    t = 0
    for t in range(1000):
        env.render()
        state_tensor = tf.convert_to_tensor(observation)
        state_tensor = tf.expand_dims(state_tensor, 0)
        action_probs = model(state_tensor, training=False)
        # Take best action
        action_index = tf.argmax(action_probs[0]).numpy()
        print(t, " : ", observation, " -> ", actions[action_index])
        observation, reward, done, info = env.step([actions[action_index]])
        if done:
            print("Episode finished after {} timesteps".format(t+1))
            break
env.close()
