# Multiplayer agent for gymsnake using RL library Stable-Baselines3
# using use_bots=True for training and playing

from os.path import exists
import time
import logging
import gymnasium as gym
from gymsnake.envs.controller import Action  # performs the registration of snake-v1 with gymnasium
from stable_baselines3 import DQN, PPO
from stable_baselines3.common.monitor import Monitor

log = logging.getLogger("snake_challenge")
log.setLevel(logging.INFO)
if not log.hasHandlers(): log.addHandler(logging.StreamHandler())

# use_bots=True: one learning AI-snake, whereas the other snake(s) are bots
env = gym.make("snake-v1", n_snakes=2, use_bots=True,
               head_representations=((3, 4), ('H', 'I'), (0, 100)), # digit-grid, letter-grid, pixel-grid
               disable_env_checker=True)

'''
# example: settings for 3 snakes:
env = gym.make("snake-v1", n_snakes=3, use_bots=True, 
               head_representations=((110, 111, 112), ('H', 'I', 'J'), (0, 100, 200)), # digit-grid, letter-grid, pixel-grid
               disable_env_checker=True)
'''

obs, info = env.reset()
env.render()

training = True
if training:
    # using n_learning_runs, total_timesteps does not have to be super high, meaning that bots can quickly benefit from
    # a better model, whereas the AI-snake doesn't need to start learning from scratch every run again
    n_learning_runs = 1000
    for _ in range(n_learning_runs):
        if exists("learned_models/multisnake_agent.zip"):
            model = PPO.load("learned_models/multisnake_agent", env=env)
            log.info("load pre-trained model and continue learning")
        else:
            model = PPO("MlpPolicy", env, device="cpu", verbose=1, tensorboard_log="tensorboard_logs/multisnake_agent/")
            log.info("no pre-trained model found, start learning from scratch")
        model.learn(total_timesteps=20000, reset_num_timesteps=False)
        model.save("learned_models/multisnake_agent")
else:  # no learning, only predicting
    model = PPO.load("learned_models/multisnake_agent")

log.info("finished training, now use the model to predict the optimal moves and render the env")
obs, info = env.reset()
env.render()
time.sleep(5)  # sufficient time to start screen capture

log.info("start game")
done = False
timestamp = 0
while not done:
    log.info(f"(snake length including head is {env.controller.snakes[0].len()})")
    action, _states = model.predict(obs, deterministic=True)
    log.info(f"predicted action: {action} for timestamp {timestamp}")
    obs, reward, terminated, truncated, info = env.step(Action(action))
    done = terminated or truncated
    log.info(f"reward: {reward}")
    env.render()
    timestamp += 1
env.close()
