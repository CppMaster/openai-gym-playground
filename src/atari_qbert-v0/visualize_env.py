import gym
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib

from baselines.common.atari_wrappers import MaxAndSkipEnv, EpisodicLifeEnv, WarpFrame, ScaledFloatFrame, FrameStack


seed = 42
skip_frames = 8
stack_frames = 4
reward_scale = 1/25
env_name = "QbertNoFrameskip-v4"
output_dir = f"temp/images/{env_name}_skip-frames-{skip_frames}_stack-frames-{stack_frames}"
os.makedirs(output_dir, exist_ok=True)
matplotlib.use('Agg')

env = gym.make(env_name)
if skip_frames > 1:
    env = MaxAndSkipEnv(env, skip=skip_frames)
env = EpisodicLifeEnv(env)
env = WarpFrame(env)
env = ScaledFloatFrame(env)
if stack_frames > 1:
    env = FrameStack(env, stack_frames)

env.seed(seed)
num_actions = env.action_space.n

frame_count = 0
while True:
    state = env.reset()

    while True:
        fig, axs = plt.subplots(2, 4, clear=True)
        for x in range(stack_frames):
            axs[0, x].imshow(state[:, :, x])
            if x < stack_frames - 1:
                axs[1, x].imshow(np.abs(state[:, :, x + 1] - state[:, :, x]))
        axs[1, stack_frames - 1].imshow(np.max(state, axis=2))
        plt.savefig(f"{output_dir}/frame_{frame_count}.png")
        plt.close()

        action = np.random.choice(num_actions)
        state, reward, done, _ = env.step(action)

        frame_count += 1

        if done:
            break
