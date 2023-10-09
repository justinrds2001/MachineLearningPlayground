from gymnasium.envs.registration import register

register(
    id='snake-v1',
    entry_point='gymsnake.envs:SnakeEnv',
    # apparently the gym wrapper is not used by gym-snake, so max_episode_steps is ignored
    # instead max_episode_steps has been hard-coded in the snake_env
    max_episode_steps=400,
)
print("gym register called for snake-v1!")
