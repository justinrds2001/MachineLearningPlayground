# Example agent for multiplayer snake

import logging
import gymnasium as gym
from gymsnake.envs.controller import Action  # performs the registration of snake-v1 with gymnasium

log = logging.getLogger("snake_challenge")
log.setLevel(logging.INFO)
if not log.hasHandlers(): log.addHandler(logging.StreamHandler())

env = gym.make('snake-v1', n_snakes=2, disable_env_checker=True)
obs, info = env.reset()
env.render()

log.info("start game")
done = False
for i in range(1, 12):
    if not done:
        obs, reward, terminated, truncated, info = env.step([Action(i % 4), Action(i % 4)])
        done = terminated or truncated
        env.render()
        log.info(f"reward: {reward}")
env.close()
