import gymnasium as gym
import numpy as np
from gymnasium import spaces

from rubiks_cube.gym_env.cube import RubiksCube


class CubeEnv(gym.Env):
    def __init__(self, render_mode=None, max_episode_steps: int = 1000):
        self.observation_space = spaces.MultiDiscrete(
            np.array(
                [
                    ((6, 6), (6, 6)),
                    ((6, 6), (6, 6)),
                    ((6, 6), (6, 6)),
                    ((6, 6), (6, 6)),
                    ((6, 6), (6, 6)),
                    ((6, 6), (6, 6)),
                ]
            )
        )

        self.action_space = spaces.Discrete(18)

        self.render_mode = render_mode

        self.window = None
        self.clock = None
        self._cube = None
        self.count = None

        self._max_episode_steps = max_episode_steps
        self._elapsed_steps = None

    def reset(self, seed=None, options=None):
        # We need the following line to seed self.np_random
        super().reset(seed=seed)

        ini_cube = RubiksCube()
        ini_cube.shuffle(1, 14)

        self._cube = ini_cube

        self._elapsed_steps = 0

        observation = np.array(self._cube.return_cube())
        info = {"Off by": self._cube.compare()}

        return observation, info

    def step(self, action):
        # L
        if action == 0:
            self._cube.vertical_twist(0, 0)
        # F
        elif action == 1:
            self._cube.side_twist(1, 1)
        # R
        elif action == 2:
            self._cube.vertical_twist(1, 1)
        # B
        elif action == 3:
            self._cube.side_twist(0, 0)
        # U
        elif action == 4:
            self._cube.horizontal_twist(0, 0)
        # D
        elif action == 5:
            self._cube.horizontal_twist(1, 1)
        # L'
        elif action == 6:
            self._cube.vertical_twist(0, 1)
        # F'
        elif action == 7:
            self._cube.side_twist(1, 0)
        # R'
        elif action == 8:
            self._cube.vertical_twist(1, 0)
        # B'
        elif action == 9:
            self._cube.side_twist(0, 1)
        # U'
        elif action == 10:
            self._cube.horizontal_twist(0, 1)
        # D'
        elif action == 11:
            self._cube.horizontal_twist(1, 0)
        # U2
        elif action == 12:
            self._cube.horizontal_twist(0, 0)
            self._cube.horizontal_twist(0, 0)
        # D2
        elif action == 13:
            self._cube.horizontal_twist(1, 1)
            self._cube.horizontal_twist(1, 1)
        # L2
        elif action == 14:
            self._cube.vertical_twist(0, 0)
            self._cube.vertical_twist(0, 0)
        # F2
        elif action == 15:
            self._cube.side_twist(1, 1)
            self._cube.side_twist(1, 1)
        # R2
        elif action == 16:
            self._cube.vertical_twist(1, 1)
            self._cube.vertical_twist(1, 1)
        # B2
        elif action == 17:
            self._cube.side_twist(0, 0)
            self._cube.side_twist(0, 0)

        terminated = self._cube.solved()
        reward = 1 if terminated else 0  # Binary sparse rewards
        observation = np.array(self._cube.return_cube())
        info = {"Off by": self._cube.compare()}

        self._elapsed_steps += 1

        # Returning False, since I'm using Tune to limit max_episode_steps rather than gym
        return observation, reward, terminated, False, info
