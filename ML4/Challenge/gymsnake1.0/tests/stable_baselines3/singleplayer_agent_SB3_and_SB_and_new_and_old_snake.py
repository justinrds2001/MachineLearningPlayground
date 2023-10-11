# Single-player agent using RL library Stable-Baselines3

import logging
import gymnasium as gym
# performs the registration of snake-v1 with gym
from gymsnake.envs.controller import Action
from stable_baselines3 import DQN, PPO
# from stable_baselines import DQN, PPO2

log = logging.getLogger("snake_challenge")
log.setLevel(logging.INFO)
if not log.hasHandlers():
    log.addHandler(logging.StreamHandler())

env = gym.make("snake-v1", disable_env_checker=True)
# from gym_snake.envs.snake_env import SnakeEnv
# env = SnakeEnv(snake_size=2, n_snakes=1, n_foods=1)

obs = env.reset()
env.render()

'''
# some algorithm of SB3 require a vectorized environment. Here's an example how to use DummyVecEnv
from stable_baselines3.common.vec_env import DummyVecEnv
env = DummyVecEnv([lambda: gym.make("snake-v1", n_snakes = 2, disable_env_checker=True)])  

# if you want to use gymsnake source code as part of your project, here's an example how to do this, using DummyVecEnv
from stable_baselines3.common.vec_env import DummyVecEnv
env = DummyVecEnv([lambda: SnakeEnv(n_snakes=2)])
'''

training = True
if training:
    # model = PPO2("MlpPolicy", env, verbose=1, tensorboard_log="tensorboard_logs/snake_dqn_agent/")
    model = PPO("MlpPolicy", env, device="cpu", verbose=1,
                tensorboard_log="tensorboard_logs/snake_dqn_agent/")
    # example how to tweak hyperparameters:
    # model = DQN("MlpPolicy", env, verbose=1, learning_rate=0.0005, gamma=0.99, policy_kwargs=dict(layers=[64, 64]),
    #             device="cpu", tensorboard_log="tensorboard_logs/snake_dqn_agent/")

    model.learn(total_timesteps=2000000, reset_num_timesteps=False)
    model.save("learned_models/snake_dqn_agent")
else:
    model = PPO2.load("learned_models/snake_dqn_agent")
    # TODO: model = PPO.load("learned_models/snake_dqn_agent")

log.info("finished training, now use the model to predict the optimal moves and render the env")
obs = env.reset()
env.render()
log.info("start game")
done = False
timestamp = 0
while not done:
    action, _states = model.predict(obs)
    log.info(f"predicted action: {action} for timestamp {timestamp}")
    obs, reward, done, info = env.step(action)
    # TODO:obs, reward, done, info = env.step(Action(action))
    log.info(f"reward: {reward}")
    env.render()
    timestamp += 1
env.close()
