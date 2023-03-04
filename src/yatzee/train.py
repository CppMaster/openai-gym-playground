from sb3_contrib import MaskablePPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.monitor import Monitor

from src.yatzee.valid_observation_env import YahtzeeValidObservationEnv

# env = YahtzeeValidObservationEnv()
env = make_vec_env(lambda: YahtzeeValidObservationEnv(), n_envs=16)

old_model = MaskablePPO.load("models/vec_env_8")
model = MaskablePPO("MultiInputPolicy", env, verbose=True, tensorboard_log="logs/ven_env_16_ent_001",
                    ent_coef=0.01, n_steps=2**12)
model.policy = old_model.policy
model.learn(total_timesteps=1000000000)
