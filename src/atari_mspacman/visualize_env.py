import gym
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib

from baselines.common.atari_wrappers import MaxAndSkipEnv, EpisodicLifeEnv, WarpFrame, ScaledFloatFrame, FrameStack


seed = 42
skip_frames = 8
stack_frames = 4
reward_scale = 1/10
frame_width = 210
frame_height = 160
grayscale = False
env_name = "MsPacmanNoFrameskip-v4"
output_dir = f"temp/images/{env_name}_skip-frames-{skip_frames}_stack-frames-{stack_frames}_resolution-{frame_width}-{frame_height}_grayscale-{grayscale}"
os.makedirs(output_dir, exist_ok=True)
matplotlib.use('Agg')

env = gym.make(env_name)
if skip_frames > 1:
    env = MaxAndSkipEnv(env, skip=skip_frames)
env = EpisodicLifeEnv(env)
env = WarpFrame(env, width=frame_width, height=frame_height, grayscale=grayscale)
env = ScaledFloatFrame(env)
if stack_frames > 1:
    env = FrameStack(env, stack_frames)

env.seed(seed)
num_actions = env.action_space.n

frame_count = 0
while True:
    state = env.reset()

    while True:
        fig, axs = plt.subplots(2, 4, clear=True, figsize=(10, 5), dpi=400)
        if grayscale:
            for x in range(stack_frames):
                axs[0, x].imshow(state[:, :, x])
                if x < stack_frames - 1:
                    axs[1, x].imshow(np.abs(state[:, :, x + 1] - state[:, :, x]))
            axs[1, stack_frames - 1].imshow(np.max(state, axis=2))
        else:
            for x in range(stack_frames):
                axs[0, x].imshow(state[:, :, x*3:(x+1)*3])
                if x < stack_frames - 1:
                    axs[1, x].imshow(np.abs(state[:, :, (x+1)*3:(x+2)*3] - state[:, :, x*3:(x+1)*3]))
            max_frame = np.zeros(shape=(frame_height, frame_width, 3))
            for d in range(3):
                max_frame[:, :, d] = np.max(np.concatenate([state[:,:, x:x+1] for x in range(d, 3 * stack_frames, 3)], axis=2), axis=2)
            axs[1, stack_frames - 1].imshow(max_frame)
        plt.savefig(f"{output_dir}/frame_{frame_count}.png")
        plt.close()

        action = np.random.choice(num_actions)
        state, reward, done, _ = env.step(action)

        print(reward, done)

        frame_count += 1

        if done:
            break
