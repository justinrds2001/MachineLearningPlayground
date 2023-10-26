# Single-player agent for gymsnake using RL library Stable-Baselines3

import time
import logging
import gymnasium as gym
from gymsnake.envs.controller import Action, ObsType  # performs the registration of snake-v1 with gymnasium
from stable_baselines3 import DQN, PPO

log = logging.getLogger("snake_challenge")
log.setLevel(logging.INFO)
if not log.hasHandlers(): log.addHandler(logging.StreamHandler())

env = gym.make("snake-v1", disable_env_checker=True)

'''
# this snake has the Markov property, but will learn slower due to larger state space
env = gym.make("snake-v1", unicolor_body=False, disable_env_checker=True)

# some algorithms of SB3 require a vectorized environment. Here's an example how to use 
# DummyVecEnv
from stable_baselines3.common.vec_env import DummyVecEnv
env = DummyVecEnv([lambda: gym.make("snake-v1", n_snakes = 2, disable_env_checker=True)])  

# if you want to use gymsnake source code as part of your project instead of 
# installed as a package, here's an example how to do this, using DummyVecEnv.
# Advantage of using it this way is that modifications in the source code of gymsnake are
# immediately effective in your project, without needing to restart the Jupyter notebook
# kernel. Note that in this way the environment is not wrapped by 
# gymnasium.wrappers.time_limit.TimeLimit, so no step limit of 400 # steps is enforced. 
# You have to add this yourself in the code of the environment!
from stable_baselines3.common.vec_env import DummyVecEnv
env = DummyVecEnv([lambda: SnakeEnv(n_snakes=2)])
'''

obs, info = env.reset()
env.render()

training = True
if training:
    model = PPO("MlpPolicy", env, device="cpu", verbose=1, tensorboard_log="tensorboard_logs/snake_agent/")
    '''
    # example how to tweak hyperparameters:
    model = DQN("MlpPolicy", env, verbose=1, learning_rate=0.0005, gamma=0.99, policy_kwargs=dict(layers=[64, 64]),
                device="cpu", tensorboard_log="tensorboard_logs/singlesnake_agent/")
    '''

    # benchmark: PPO with standard hyperparameter values and 2000000 gives good results (~ snake length 18)
    model.learn(total_timesteps=200000, reset_num_timesteps=False)
    model.save("learned_models/singlesnake_agent")
else:
    model = PPO.load("learned_models/singlesnake_agent")

log.info("finished training, now use the model to predict the optimal moves and render the env")
obs, info = env.reset()
env.render()
time.sleep(5)  # sufficient time to start screen capture

log.info("start game")
done = False
timestamp = 0
while not done:
    action, _states = model.predict(obs)  # some exploring: sometimes stupid actions, but endless loops are interrupted
    '''
    action, _states = model.predict(obs, deterministic=True)  # no exploring: so no stupid actions, but possibly endless loops
    '''
    log.info(f"(snake length including head is {env.controller.snakes[0].len()})")
    log.info(f"predicted action: {action} for timestamp {timestamp}")
    obs, reward, terminated, truncated, info = env.step(Action(action))
    done = terminated or truncated
    log.info(f"reward: {reward}")
    env.render()
    timestamp += 1
env.close()
