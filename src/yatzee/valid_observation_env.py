import numpy as np
from gym import spaces
from gym_yahtzee.envs import YahtzeeSingleEnv
from gym_yahtzee.envs.yahtzee_env import GameType
from pyhtzee import Rule


class YahtzeeValidObservationEnv(YahtzeeSingleEnv):
    def __init__(self,
                 rule: Rule = Rule.YAHTZEE_FREE_CHOICE_JOKER,
                 game_type: GameType = GameType.RETRY_ON_WRONG_ACTION,
                 seed=None):
        super().__init__(rule, game_type, seed)
        self.parent_observation_space = self.observation_space
        self.observation_space = spaces.Dict({
            "round": spaces.MultiDiscrete([13, 4]),
            "dies": spaces.Box(low=1, high=6, shape=(5,), dtype=np.uint8),
            "scores": spaces.Box(low=-1., high=1., shape=(15,), dtype=np.float)
        })

    def get_observation_space(self) -> dict:
        parent_observation = super().get_observation_space()
        return {
            "round": np.array([parent_observation[0], parent_observation[1]]),
            "dies": np.array([parent_observation[i] for i in range(2, 7)]),
            "scores": np.array([
                float(parent_observation[i] / self.parent_observation_space.spaces[i].high)
                if parent_observation[i] >= 0 else -1.0
                for i in range(7, 22)
            ])
        }

    def reset(self) -> dict:
        super().reset()
        return self.get_observation_space()

    def action_masks(self) -> np.ndarray:
        mask = np.zeros(shape=(self.action_space.n,), dtype=np.bool)
        for action_index in self.pyhtzee.get_possible_actions():
            mask[action_index] = True
        return mask
