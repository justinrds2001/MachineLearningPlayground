import logging
import matplotlib.pyplot as plt
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from gymnasium.utils import seeding
from gymsnake.envs.controller import Controller, ObsType, Action
from typing import Optional

log = logging.getLogger("snake_env")
log.setLevel(logging.INFO)
if not log.hasHandlers():
    log.addHandler(logging.StreamHandler())


class SnakeEnv(gym.Env):
    """
    SnakeEnv is concerned with communication with the outside world; it is like a presentation layer
    """
    metadata = {'render_modes': ['human'], "render_fps": 30}

    def __init__(self, grid_size=(6, 6), body_start_length=1, n_snakes=1, n_foods=1, random_food_init=False,
                 obs_type=ObsType.DIGIT_GRID, rendering_obs=ObsType.PIXEL_GRID,
                 head_representations=((110, 111), ('H', 'I'), (0, 100)), unicolor_body=True, use_bots=False):

        super(SnakeEnv, self).__init__()
        self.grid_size = grid_size
        self.body_start_length = body_start_length
        self.n_snakes = n_snakes
        self.n_foods = n_foods
        self.random_food_init = random_food_init
        self.square_size = 10  # pixels
        self.obs_type = obs_type
        self.rendering_obs = rendering_obs
        self.head_representations = head_representations
        self.unicolor_body = unicolor_body
        self.digit_body_representation_offset = 2
        self.multicolor_body_delta = 10  # 8-bit color space
        self.use_bots = use_bots
        self.env_state = None
        self.controller = None
        self.bot_model = None
        self.viewer = None
        self.fig = None

        assert self.n_snakes <= 19, "too many snakes, not sufficient letters in alphabet to represent snake heads"
        assert self.n_snakes < self.grid_size[0] - \
            2, "too many snakes for grid size (x-direction)"
        assert self.body_start_length < self.grid_size[1] - \
            2, "body start length too large for grid size (y-direction)"
        assert self.n_snakes <= len(self.head_representations[0]) \
            or self.n_snakes <= len(self.head_representations[1]) \
            or self.n_snakes <= len(self.head_representations[2]), \
            "not sufficient head representations provided for number of snakes"
        assert min(self.head_representations[0]) > self.digit_body_representation_offset, \
            "digit head representations overlap with body representation"
        assert self.unicolor_body or min(self.head_representations[0]) > self.grid_size[0] * self.grid_size[1] \
            + self.digit_body_representation_offset, \
            "digit head representations too low numbers, so risk of overlap with multicolor body representations"

        # Stable-Baselines3 Zoo requires definition of action and observation space to be done
        # in __init__ and not in reset
        self.action_space = spaces.Discrete(4)
        match self.obs_type:
            case ObsType.COORDS:
                self.observation_space = spaces.Box(low=0, high=max(self.grid_size[0], self.grid_size[1]),
                                                    shape=(
                                                        2 * self.grid_size[0] * self.grid_size[1] + self.n_snakes,),
                                                    dtype=np.uint8)
            case ObsType.DIGIT_GRID:
                self.observation_space = spaces.Box(low=0, high=max(self.head_representations[0]),
                                                    shape=(self.grid_size[0], self.grid_size[1]), dtype=np.uint8)
            case ObsType.LETTER_GRID:
                self.observation_space = spaces.Box(low=0, high=max(self.head_representations[1]),
                                                    shape=(self.grid_size[0], self.grid_size[1]), dtype=np.uint8)
            case ObsType.PIXEL_GRID:
                self.observation_space = spaces.Box(low=0, high=255, shape=(self.grid_size[0] * self.square_size,
                                                    self.grid_size[1] * self.square_size, 3), dtype=np.uint8)
            case _:
                assert False, "unsupported observation type value"

        self.seed()

    def step(self, actions) -> tuple[np.ndarray, list[int], bool, bool, dict[str, int]]:
        # OpenAI Gym interface requires actions to be int's, so OPTIONALLY map int's to instances of Action type, as
        # the Controller uses Action type to represent actions
        #
        # SB3 shows strange behavior: predict always returns an action with type np.int64, but as last action
        # it returns an action with type nd.array with ndim=0
        if isinstance(actions, list) or (isinstance(actions, np.ndarray) and actions.ndim != 0):  # collection type
            if not isinstance(actions[0], Action):
                actions = list(map(lambda action: Action(action), actions))
        # non-collection type like int, np.int32, np.int64 or ndarray with ndim=0
        elif not isinstance(actions, Action):
            actions = Action(actions)

        # convert to list, in case agent passed single value instead of list (only for single-player snake)
        use_value_instead_of_list = False
        if isinstance(actions, Action):
            actions = [actions]
            use_value_instead_of_list = True

        if self.use_bots:
            assert self.n_snakes > 1 and len(
                actions) == 1, "multiplayer bot-mode not correctly used"
            # first snake is the agent that is trained and used for prediction; the other snakes (2 to n) are bots
            actions += self.get_bots_action()
            log.debug(
                f"actions (1st one is the AI; other ones are the bots): {actions}")

        foods, alive_snakes, rewards, terminated, info = self.controller.step(
            actions)
        # in bot-mode stop the game if the AI-bot died, even if other snakes are still alive
        if self.use_bots and not terminated:
            terminated = alive_snakes[0].id != 0

        self.env_state = (foods, alive_snakes)

        if self.use_bots:
            # list with only one entry; in bot-mode rewards of bots are discarded
            rewards = [rewards[0]]

        # convert list back to single value, in case agent passed single value instead of list
        # (only for single-player snake)
        if use_value_instead_of_list:
            rewards = rewards[0]

        return self.observation(self.obs_type, self.head_representations), rewards, terminated, False, info

    def initialize_bots(self):
        """
        if there's no trained model yet, self.bot_model is set to None
        TODO: remove the dependency of gymsnake on SB3 by introducing an interface between this method and gymsnake
        """
        if not self.use_bots:
            return
        # not very efficient: with every reset of the environment the model is loaded from disk, whereas the model on
        # disk is only updated at the end of a learning run; however, training on a normal laptop showed low disk
        # access intensity
        from os.path import exists
        if exists("learned_models/multisnake_agent.zip"):
            log.debug(
                f"trained model found (learned_models/multisnake_agent.zip), so let's use it")
            from stable_baselines3 import PPO
            self.bot_model = PPO.load("learned_models/multisnake_agent")
        else:
            log.debug("no trained model yet, move every bot Action.DOWN")

    def get_bots_action(self) -> list[Action]:
        """
        uses the *same* trained model for all bots!!
        TODO remove the dependency of gymsnake on SB3 by introducing an interface between this method and gymsnake
        """
        if self.n_snakes == 1:
            return []  # no bots involved when single-player mode

        if self.bot_model is None:  # no trained model yet, move every bot Action.DOWN,
            return [Action.DOWN] * (self.n_snakes - 1)

        bot_actions = []
        for i in range(1, self.n_snakes):  # don't swap snake 0
            # important that a bot does a prediction using the head representation for which the model was trained
            # so swap the head representations of snake 0 and snake i
            swapped_head_repr = [list(self.head_representations[0]), list(self.head_representations[1]),
                                 list(self.head_representations[2])]
            swapped_head_repr[0][0], swapped_head_repr[0][i] = swapped_head_repr[0][i], swapped_head_repr[0][0]
            swapped_head_repr[1][0], swapped_head_repr[1][i] = swapped_head_repr[1][i], swapped_head_repr[1][0]
            swapped_head_repr[2][0], swapped_head_repr[2][i] = swapped_head_repr[2][i], swapped_head_repr[2][0]
            swapped_head_repr = tuple([tuple(swapped_head_repr[0]), tuple(swapped_head_repr[1]),
                                       tuple(swapped_head_repr[2])])

            bot_action, _states = self.bot_model.predict(self.observation(self.obs_type, swapped_head_repr),
                                                         deterministic=True)
            bot_actions.append(Action(bot_action))
        return bot_actions

    def reset(self,  *,
              seed: Optional[int] = None,
              options: Optional[dict] = None,) -> tuple[np.ndarray, dict[str, int]]:
        """
        initializes or re-initializes the environment
        also creates an observation and returns it
        """
        super().reset(seed=seed)

        self.controller = Controller(self.grid_size, self.body_start_length, self.n_snakes, self.n_foods,
                                     self.random_food_init, self.square_size)
        self.env_state = (self.controller.foods,
                          self.controller.alive_snakes())

        self.initialize_bots()

        return self.observation(self.obs_type, self.head_representations), {}

    def render(self, mode='human', close=False, frame_speed=.1) -> None:
        render_obs = self.observation(
            self.rendering_obs, self.head_representations)
        match self.rendering_obs:
            case ObsType.COORDS:
                print(render_obs)
            case ObsType.DIGIT_GRID:
                print(np.flip(np.transpose(render_obs), axis=0))
                print('')
            case ObsType.LETTER_GRID:
                print(np.flip(np.transpose(render_obs), axis=0))
                print('')
            case ObsType.PIXEL_GRID:
                if self.viewer is None:
                    self.fig = plt.figure()
                    self.viewer = self.fig.add_subplot(111)
                self.viewer.clear()
                # set origin to left bottom to allow easy interpretation
                self.viewer.imshow(render_obs, origin='lower')
                self.fig.show()
                plt.pause(frame_speed)
            case _:
                assert False, "unsupported observation type"

    def observation(self, obs_type, head_representations):
        """
        returns the complete state as observation
        """
        match obs_type:
            case ObsType.COORDS:
                return self.create_coords_obs()
            case ObsType.DIGIT_GRID:
                return self.create_grid(head_representations, as_digits=True)
            case ObsType.LETTER_GRID:
                return self.create_grid(head_representations, as_digits=False)
            case ObsType.PIXEL_GRID:
                return self.create_pixel_grid(head_representations)
            case _:
                assert False, "unsupported observation type value"

    def create_coords_obs(self):
        """
        creates a list representation of the state

        if the obs_type is ObsType.COORDS, it will return a representation good for machine learning,
        otherwise it will return a human-readable representation
        """
        foods, alive_snakes = self.env_state

        if not self.obs_type == ObsType.COORDS:
            coord_obs = []
            for snake in alive_snakes:
                coord_obs.append((snake.id, snake.head, list(snake.body)))
            return foods, coord_obs
        else:  # non-human-readable format good for machine learning
            coord_obs = []
            for food in foods:
                # extend automatically converts tuple to list
                coord_obs.extend(food)
            for snake in alive_snakes:
                coord_obs.append(snake.id)
                coord_obs.extend(snake.head)
                for body in snake.body:
                    coord_obs.extend(body)
            # padding with -1 up to observation space length
            padding_length = 2 * \
                self.grid_size[0] * self.grid_size[1] + \
                self.n_snakes - len(coord_obs)
            coord_obs.extend([-1] * padding_length)
            return coord_obs

    def create_grid(self, head_representations, as_digits) -> np.ndarray:
        """
        creates a digit or letter grid representation of the state
        """
        if as_digits:
            grid = np.zeros(self.grid_size, dtype=np.uint8)
            body_representation = self.digit_body_representation_offset
            food_representation = 1
            space_representation = 0
            head_representation = head_representations[0]
        else:  # as_letters
            # creates array with strings of max length 1 for good performance
            # initialized with empty strings, so elements with value ''
            grid = np.zeros(self.grid_size, dtype='U1')

            # these are strings of length one, not characters; python does not support characters
            assert self.unicolor_body, "multicolor body is nut supported for ObsType.LETTER_GRID , as there not" \
                                       "enough letters in the alphabet to represent every body part"
            body_representation = 'B'
            food_representation = 'F'
            space_representation = ' '
            head_representation = head_representations[1]

        grid[:, :] = space_representation

        foods, alive_snakes = self.env_state
        for food in foods:
            grid[food] = food_representation

        for snake in alive_snakes:
            grid[snake.head] = head_representation[snake.id]
            for i, part in enumerate(snake.body):
                grid[part] = body_representation if self.unicolor_body else body_representation + i

        return grid

    def create_pixel_grid(self, head_representations) -> np.ndarray:
        """
        creates pixel grid representation of the state
        """
        body_color = np.array([1, 0, 0], dtype=np.uint8)
        food_color = np.array([0, 0, 255], dtype=np.uint8)
        space_color = np.array([0, 255, 0], dtype=np.uint8)
        channels = 3

        width = self.grid_size[0] * self.square_size
        height = self.grid_size[1] * self.square_size
        pixel_grid = np.zeros((width, height, channels), dtype=np.uint8)
        pixel_grid[:, :, :] = space_color  # init all squares with space color

        foods, alive_snakes = self.env_state
        for food in foods:
            self.draw_square(food, food_color, pixel_grid)

        for snake in alive_snakes:
            head_color = np.array(
                [255, head_representations[2][snake.id], 0], dtype=np.uint8)
            self.draw_square(snake.head, head_color, pixel_grid)
            for i, part in enumerate(snake.body):
                assert self.unicolor_body or i * self.multicolor_body_delta <= 255, \
                    "not sufficient colors available per snake for this number of body parts"
                # this statement only allows for max 50 body parts per snake
                self.draw_square(part,
                                 body_color if self.unicolor_body else body_color + i * self.multicolor_body_delta,
                                 pixel_grid)

        return pixel_grid

    def draw_square(self, coord, color, pixel_grid):
        """
        receives a reference to the pixel_grid and modifies the pixel_grid
        """
        x = int(coord[0] * self.square_size)
        end_x = x + self.square_size
        y = int(coord[1] * self.square_size)
        end_y = y + self.square_size
        pixel_grid[y:end_y, x:end_x, :] = np.asarray(color, dtype=np.uint8)

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return None
