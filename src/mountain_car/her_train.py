import gym
from stable_baselines3 import DDPG, HerReplayBuffer, DQN, SAC, TD3, A2C
from stable_baselines3.her.her import HER

from src.mountain_car.her_env import HerEnv

suffix = "DQN_no-hyper_min-pos_no-her_no-random-goal"

env = HerEnv(random_goal=False)
env_eval = HerEnv(random_goal=False)

# replay_buffer_class = HerReplayBuffer
# replay_buffer_kwargs = dict(n_sampled_goal=4, goal_selection_strategy="future", max_episode_length=200,
#                             online_sampling=True)
replay_buffer_class = None
replay_buffer_kwargs = None
model = DQN('MultiInputPolicy', env, verbose=1, replay_buffer_class=replay_buffer_class,
            replay_buffer_kwargs=replay_buffer_kwargs, tensorboard_log=f"temp/her/{suffix}",
            )
            # learning_rate=4e-3, batch_size=128, buffer_size=10000, learning_starts=1000,
            # gamma=0.98, target_update_interval=600, train_freq=16, gradient_steps=8, exploration_fraction=0.2,
            # exploration_initial_eps=0.07, policy_kwargs=dict(net_arch=[256, 256]))
# model = A2C("MultiInputPolicy", env, verbose=1, replay_buffer_class=replay_buffer_class,
#             replay_buffer_kwargs=replay_buffer_kwargs, tensorboard_log=f"temp/her/{suffix}")

model.learn(total_timesteps=10000000, eval_log_path=f"temp/her/{suffix}", eval_freq=10000,
            eval_env=env_eval)
