# Example competition using agents trained with RL library Stable-Baselines3

import time
import logging
import gymnasium as gym
from gymsnake.envs.controller import Action  # performs the registration of snake-v1 with gymnasium
from stable_baselines3 import DQN, PPO


def swap(obs0):
    # clone to avoid changing the original observation
    obs1 = obs0.copy()
    temp_value = -1
    obs1[obs1 == head_representation0] = temp_value
    obs1[obs1 == head_representation1] = head_representation0
    obs1[obs1 == temp_value] = head_representation1
    return obs1


log = logging.getLogger("snake_challenge")
log.setLevel(logging.INFO)
if not log.hasHandlers(): log.addHandler(logging.StreamHandler())

# game parameters with which both snakes have been trained
head_representation0 = 3
head_representation1 = 4
env = gym.make("snake-v1", n_snakes=2,
               head_representations=((head_representation0, head_representation1), ('H', 'I'), (0, 100)), 
               disable_env_checker=True)

model0 = PPO.load("learned_models/multisnake_agent0")
model1 = PPO.load("learned_models/multisnake_agent1")  # does not need to be the same algorithm or same hyperparameters

obs, info = env.reset()
env.render()
time.sleep(5)  # sufficient time to start screen capture
log.info("start game")
done = False
timestamp = 0
while not done:
    log.info(f"(snake 0 length: {env.controller.snakes[0].len()} and snake 1 length: {env.controller.snakes[1].len()}"
             f" (both including head)")
    action0, _states1 = model0.predict(obs)
    '''
    action0, _states1 = model0.predict(obs, deterministic=True)
    '''
    action1, _states2 = model1.predict(swap(obs))
    '''
    action1, _states2 = model1.predict(swap(obs), deterministic=True)
    '''
    log.info(f"predicted action0: {action0} and predicted action1: {action1} for timestamp {timestamp}")
    obs, rewards, terminated, truncated, info = env.step([Action(action0), Action(action1)])
    done = terminated or truncated
    log.info(f"reward0: {rewards[0]} and reward1: {rewards[1]}")
    env.render()
    timestamp += 1
env.close()
