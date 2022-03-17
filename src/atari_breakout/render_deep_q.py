import time

import numpy as np
import gym
import tensorflow as tf
from baselines.common.atari_wrappers import make_atari, wrap_deepmind
from keras.models import load_model
from src.utils.gpu import set_memory_growth


set_memory_growth()

max_steps_per_episode = 10000
model = load_model("temp/sample/model_0.h5")

env = make_atari("BreakoutNoFrameskip-v4")
env = wrap_deepmind(env, frame_stack=True, scale=True)

for i_episode in range(20):
    observation = env.reset()
    t = 0
    total_rewards = 0.0
    for t in range(max_steps_per_episode):
        env.render()
        time.sleep(0.01)
        state_tensor = tf.convert_to_tensor(observation)
        state_tensor = tf.expand_dims(state_tensor, 0)
        action_probs = model(state_tensor, training=False)
        # Take best action
        action_index = tf.argmax(action_probs[0]).numpy()
        # print(t, " : ", observation, " -> ", action_index)
        observation, reward, done, info = env.step(action_index)
        total_rewards += reward
        if done:
            print(f"Episode finished after {t+1} timesteps")
            break
    print(f"Total reward: {total_rewards}")
env.close()
