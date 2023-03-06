from sb3_contrib import MaskablePPO
from stable_baselines3.common.env_util import make_vec_env
import torch as th

from src.yatzee.valid_observation_env import YahtzeeValidObservationEnv

env = make_vec_env(lambda: YahtzeeValidObservationEnv(), n_envs=16)

model = MaskablePPO("MultiInputPolicy", env, verbose=True, tensorboard_log="logs/leakyrelu_arch-256-256-256",
                    ent_coef=0.01, n_steps=2 ** 13, batch_size=2 ** 9, policy_kwargs=dict(
        activation_fn=th.nn.LeakyReLU, net_arch=dict(pi=[256, 256, 256], vf=[256, 256, 256])
    ))
model.learn(total_timesteps=1000000000)
