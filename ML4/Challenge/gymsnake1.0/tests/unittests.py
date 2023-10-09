import unittest
import gymnasium as gym
from gymsnake.envs import SnakeEnv
from gymsnake.envs import ObsType, Action


class TestSnakeEnv(unittest.TestCase):
    @staticmethod
    def prepare():
        snake_env = gym.make('snake-v1', grid_size=(10, 10), n_snakes=2, obs_type=ObsType.PIXEL_GRID, disable_env_checker=True)
        snake_env.reset()
        snake_env.render()
        return snake_env

    def test_snake_env_init_singleplayer(self):
        """
        snakes are initialized in random order, but head representation remains the same
        so sometimes the orange snake is on the left side, sometimes on the right side
        snakes always initialized at same location, head pointing down
        food always initialized at same location
        """
        snake_env = gym.make('snake-v1', disable_env_checker=True)
        snake_env.reset()
        snake_env.render()
        obs, rewards, terminated, truncated, info = snake_env.step(Action.DOWN)  # pass Action as type
        assert isinstance(rewards, int)  # single value int, so not a list
        obs, rewards, terminated, truncated, info = snake_env.step(2)  # pass int as type
        assert isinstance(rewards, int)  # single value int, so not a list

    def test_snake_env_init_multiplayer(self):
        """
        snakes are initialized in random order, but head representation remains the same
        so sometimes the orange snake is on the left side, sometimes on the right side
        snakes always initialized at same location, head pointing down
        food always initialized at same location
        """
        snake_env = gym.make('snake-v1', body_start_length=3, n_snakes=2, n_foods=3, obs_type=ObsType.PIXEL_GRID, disable_env_checker=True)
        snake_env.reset()
        snake_env.render()
        assert snake_env.head_representations[0][snake_env.controller.snakes[0].id] == 110
        assert snake_env.head_representations[0][snake_env.controller.snakes[1].id] == 111

    def test_eat_food(self):
        """
        result: * the right snake eats food and new food appears on random empty square
                * due to random positioning of the snakes on the grid at initialization, sometimes the right snake
                  is the other snake and that one gets the reward
        """
        snake_env = self.prepare()

        obs, rewards, terminated, truncated, info = snake_env.step([Action.LEFT, Action.LEFT])
        done = terminated or truncated
        assert rewards == [0, 0] and not done
        snake_env.render()
        obs, rewards, terminated, truncated, info = snake_env.step([Action.UP, Action.UP])
        snake_env.render()
        obs, rewards, terminated, truncated, info = snake_env.step([Action.UP, Action.UP])
        snake_env.render()
        obs, rewards, terminated, truncated, info = snake_env.step([Action.UP, Action.UP])
        done = terminated or truncated
        snake_env.render()
        assert rewards == [0, 1] or rewards == [1, 0] and not done

    def test_collision_with_self(self):
        """
        both snakes should die and a new piece of food should emerge on a random square
        """
        snake_env = self.prepare()

        obs, rewards, terminated, truncated, info = snake_env.step([Action.UP, Action.UP])
        done = terminated or truncated
        snake_env.render()
        assert rewards == [-1, -1] and done

    def test_collision_with_other_snake(self):
        """
        setup: two snakes move to the same empty square in the same action
        result: * one snake should die, one should survive
                * as the order in which snake moves are evaluated is random, sometimes the 1st snake survives,
                  sometimes the 2nd
                * due to random positioning of the snakes on the grid at initialization, sometimes the snakes move
                  away from each other
        """
        snake_env = self.prepare()

        obs, rewards, terminated, truncated, info = snake_env.step([Action.RIGHT, Action.LEFT])
        snake_env.render()
        obs, rewards, terminated, truncated, info = snake_env.step([Action.RIGHT, Action.LEFT])
        done = terminated or truncated
        snake_env.render()
        print(rewards)
        print(done)
        assert (rewards == [-1, 0] or rewards == [0, -1] and not done) or (rewards == [0, 0] and not done)

    def test_snake_off_grid(self):
        """
        setup: both snakes walk off the grid
        result: * both die
                * due to random positioning of the snakes on the grid at initialization, sometimes the snakes move
                  towards each other
        """
        snake_env = self.prepare()

        obs, rewards, terminated, truncated, info = snake_env.step([Action.LEFT, Action.RIGHT])
        snake_env.render()
        obs, rewards, terminated, truncated, info = snake_env.step([Action.LEFT, Action.RIGHT])
        snake_env.render()
        obs, rewards, terminated, truncated, info = snake_env.step([Action.LEFT, Action.RIGHT])
        snake_env.render()
        obs, rewards, terminated, truncated, info = snake_env.step([Action.LEFT, Action.RIGHT])
        snake_env.render()
        done = terminated or truncated
        print(rewards)
        print(done)
        assert (rewards == [-1, -1] and done) or (rewards == [0, 0] and not done)

    def test_snakes_keep_identity(self):
        """
        if the middle snake dies the others should keep having their identity
        needs to be visually verified (no assert)
        """
        snake_env = gym.make('snake-v1', grid_size=(10, 10), body_start_length=3, n_snakes=3, n_foods=3,
                             rendering_obs=ObsType.LETTER_GRID,
                             head_representations=((10, 11, 12), ('H', 'I', 'J'), (0, 100, 200)), disable_env_checker=True)
        snake_env.reset()
        snake_env.render()

        obs, rewards, terminated, truncated, info = snake_env.step([Action.LEFT, Action.DOWN, Action.RIGHT])
        snake_env.render()
        obs, rewards, terminated, truncated, info = snake_env.step([Action.DOWN, Action.DOWN, Action.DOWN])
        snake_env.render()
        obs, rewards, terminated, truncated, info = snake_env.step([Action.DOWN, Action.DOWN, Action.DOWN])
        snake_env.render()
        obs, rewards, terminated, truncated, info = snake_env.step([Action.DOWN, Action.DOWN, Action.DOWN])
        snake_env.render()

    def test_coord_obs(self):
        """
        shows the COORD observation representation
        needs to be visually verified (no assert)
        """
        snake_env = gym.make('snake-v1', grid_size=(10, 10), body_start_length=3, n_snakes=2, n_foods=3,
                             rendering_obs=ObsType.COORDS, obs_type = ObsType.COORDS, disable_env_checker=True)
        obs_, info = snake_env.reset()
        print("machine learning representation:")
        snake_env.render()
        print(obs_)

        snake_env.env.obs_type = ObsType.DIGIT_GRID
        obs_, info = snake_env.reset()
        print("human-readable representation:")
        snake_env.render()

        # example how to read a coord obs:
        obs = ([(2, 7), (5, 7), (7, 7)], [(0, (3, 2), [(3, 3), (3, 4), (3, 5)]), (1, (6, 2), [(6, 3), (6, 4), (6, 5)])])
        print('foods', obs[0])
        for snake in obs[1]:
            print('snake id', snake[0])
            print('snake head', snake[1])
            print('snake body', snake[2])

    def test_time_limit(self):
        """
        the wrapper gym.wrappers.time_limit.TimeLimit should enforce the step limit of 1500
        """
        snake_env = self.prepare()

        for _ in range(1500):
            obs, rewards, terminated, truncated, info = snake_env.step([Action.LEFT, Action.LEFT])
            obs, rewards, terminated, truncated, info = snake_env.step([Action.UP, Action.UP])
            obs, rewards, terminated, truncated, info = snake_env.step([Action.RIGHT, Action.RIGHT])
            obs, rewards, terminated, truncated, info = snake_env.step([Action.DOWN, Action.DOWN])
        assert truncated and not terminated


if __name__ == "__main__":
    unittest.main()
