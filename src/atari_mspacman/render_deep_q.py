import gym
import numpy as np
import tensorflow as tf
from baselines.common.atari_wrappers import MaxAndSkipEnv, EpisodicLifeEnv, WarpFrame, ScaledFloatFrame, FrameStack
from keras.models import load_model
import time
import shutil

from src.utils.gpu import set_memory_growth


set_memory_growth()

skip_frames = 4
stack_frames = 4
reward_scale = 1/10
max_steps_per_episode = 10000
frame_width = 210
frame_height = 160
grayscale = False

env_name = "MsPacmanNoFrameskip-v4"
env = gym.make(env_name)
if skip_frames > 1:
    env = MaxAndSkipEnv(env, skip=skip_frames)
env = EpisodicLifeEnv(env)
env = WarpFrame(env, width=frame_width, height=frame_height, grayscale=grayscale)
env = ScaledFloatFrame(env)
if stack_frames > 1:
    env = FrameStack(env, stack_frames)

shutil.copy("temp/model_deep-q-1_rgb_res-210-160.h5", "temp/render_last.h5")
model = load_model("temp/render_last.h5")

while True:
    state = np.array(env.reset())
    episode_reward = 0

    for timestep in range(1, max_steps_per_episode):
        env.render()

        state_tensor = tf.convert_to_tensor(state)
        state_tensor = tf.expand_dims(state_tensor, 0)
        action_probs = model(state_tensor, training=False)
        # Take best action
        action = tf.argmax(action_probs[0]).numpy()

        state, reward, done, _ = env.step(action)
        reward *= reward_scale
        print(f"Action: {action}, Reward: {reward}")

        episode_reward += reward

        time.sleep(0.01)

        if done:
            break

    print(f"Episode reward: {episode_reward}")
