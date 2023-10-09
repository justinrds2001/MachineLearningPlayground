# Expert agent following the E-pattern
# image of E-pattern https://datagenetics.com/blog/april42013/g.png

import logging
import gymnasium as gym
# performs the registration of snake-v1 with gymnasium
from gymsnake.envs.controller import Action

log = logging.getLogger("snake_challenge")
log.setLevel(logging.INFO)
if not log.hasHandlers():
    log.addHandler(logging.StreamHandler())

env = gym.make('snake-v1', disable_env_checker=True)
obs, info = env.reset()
env.render()


def move(action):
    global env
    global total_reward
    global done
    if not done:
        obs_, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        total_reward += reward
        env.render()
    return done


# we know that snake always start head pointing downwards
# move to the start position, which is the bottom-left corner
def move_to_start_position():
    global env
    snake = env.controller.snakes[0]
    while snake.head[1] > 0:
        move(Action.DOWN)
    while snake.head[0] > 0:
        move(Action.LEFT)


def follow_e_pattern():
    global env
    snake = env.controller.snakes[0]
    while not done:
        # move up
        while snake.head[1] < env.grid_size[1] - 1:
            if move(Action.UP):
                return
        # do the squirming
        for i in range(env.grid_size[1]//2):
            while snake.head[0] < env.grid_size[0] - 1:
                if move(Action.RIGHT):
                    return
            if move(Action.DOWN):
                return
            while snake.head[0] > 1:
                if move(Action.LEFT):
                    return
            if i < env.grid_size[1]//2 - 1:
                if move(Action.DOWN):
                    return
        if move(Action.LEFT):
            return


assert env.grid_size[1] % 2 == 0, "grid_size in y-direction must be even"
log.info("start game")
done = False
total_reward = 0
move_to_start_position()
follow_e_pattern()
print(f"total reward: {total_reward}")
