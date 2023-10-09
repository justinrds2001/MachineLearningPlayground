import numpy as np
import random
from collections import deque
from enum import Enum
from itertools import product


class Action(Enum):
    UP = 0
    RIGHT = 1
    DOWN = 2
    LEFT = 3


class ObsType(Enum):
    """
    four representation of the state are supported
    """
    COORDS = 0
    DIGIT_GRID = 1,
    LETTER_GRID = 2,
    PIXEL_GRID = 3


class Snake:
    """
    Stores the information of one snake. Note that the direction of an action is relative to the grid and not relative
    to the snake's head direction.
    """
    def __init__(self, head_coord_start, body_start_length, id_):
        """
        Initializes a snake with the head pointing down
        """
        self.is_alive = True
        self.head = head_coord_start
        self.id = id_
        # tail of the body is on the right side of the deque; the part close to the head on the left side
        # default order of iteration is from left to right, so starting close to the head and finishing at the tail
        # self.body[0] is the part close to the head; self.body[-1] is the tail of the body
        self.body = deque()
        for i in range(body_start_length):
            # initialized snakes are pointing head down (appending is done on the right side of the deque)
            self.body.append((self.head[0], self.head[1] + i + 1))

    def step_without_removing_tail_end(self, action) -> tuple[int, int]:
        """
        moves the head to the action direction, occupies the old head position by a body part, but does not remove the
        tail end. The snake has become one part longer. This is a temporary situation. If it turns out that the snake
        did not find food, the tail end is removed anyway at a later moment
        """
        assert 0 <= action.value < 4, "unsupported action value"

        self.body.appendleft(self.head)  # add the head to the left side of the deque
        match action:
            case Action.UP:
                self.head = self.head[0], self.head[1] + 1
            case Action.RIGHT:
                self.head = self.head[0] + 1, self.head[1]
            case Action.DOWN:
                self.head = self.head[0], self.head[1] - 1
            case Action.LEFT:
                self.head = self.head[0] - 1, self.head[1]
            case _:
                assert False, "unsupported action value"
        return self.head

    def remove_tail_end(self) -> None:
        """
        removes the tail end of the body of the snake
        """
        self.body.pop()  # remove the right element of the deque

    def len(self) -> int:
        """
        returns the length of the snake including the head
        """
        return len(self.body) + 1  # the '+ 1' is for the head


class Controller:
    """
    Controller contains the game logic
    """
    def __init__(self, grid_size, body_start_length, n_snakes, n_foods, random_food_init, square_size):

        # init snake world
        self.grid_size = grid_size
        self.square_size = square_size

        # init snakes
        self.snakes = []
        random_perm = np.random.permutation(n_snakes)
        for i, rnd in enumerate(random_perm):
            # place snakes in random order on grid, for the AI to learn each way of initialization
            # x-coord: divide space in n_snakes+1 parts and place each snake on the right side of a part
            # y-coord: place the middle of the snake one below the middle of the grid
            head_start_coord = ((rnd+1) * grid_size[0] // (n_snakes + 1), (grid_size[1] - body_start_length) // 2 - 1)
            self.snakes.append(Snake(head_start_coord, body_start_length, i))

        # init food
        self.foods = []
        if not random_food_init:
            for i in range(n_foods):
                if random_food_init:
                    self.place_new_food()
                else:
                    # x-coord: divide space in n_foods+1 parts and place each snake on the right side of a part
                    # y-coord: place food just above the tail of the snakes
                    food_coord = ((i+1) * grid_size[0] // (n_foods + 1), self.snakes[0].body[-1][1] + 2)
                    self.foods.append(food_coord)

    def n_empty_squares(self) -> int:
        """
        returns the number of empty squares on the grid
        """
        return self.grid_size[0] * self.grid_size[1] - len(self.foods) \
               - sum([snake.len() for snake in self.alive_snakes()])

    def n_alive_snakes(self) -> int:
        """
        returns the number of snakes that are alive
        """
        return sum(map(lambda snake: snake.is_alive, self.snakes))

    def alive_snakes(self):
        """
        returns list of alive snakes
        """
        return list(filter(lambda snake: snake.is_alive, self.snakes))

    def is_food_square(self, coord) -> bool:
        """
        returns if the square is occupied by a piece of food
        """
        return coord in self.foods

    def is_snake_square(self, coord) -> bool:
        """
        returns if the square is occupied by a snake
        """
        for snake in self.alive_snakes():
            if coord in snake.body or coord == snake.head:
                return True
        return False

    def is_snake_square_except_own_head(self, own_snake) -> bool:
        """
        checks whether square is occupied by head or body of other snakes,
        or occupied by body of own snake (so ignores head of own snake while checking)
        """
        coord = own_snake.head
        for snake in self.alive_snakes():
            if snake is not own_snake:
                if coord in snake.body or coord == snake.head:
                    return True
            else:
                if coord in snake.body:
                    return True

        return False

    def is_off_grid(self, coord) -> bool:
        """
        returns if coord is on the grid
        """
        return coord[0] < 0 or coord[0] >= self.grid_size[0] or coord[1] < 0 or coord[1] >= self.grid_size[1]

    def place_new_food(self) -> None:
        """
        places a piece of food on a random empty square, if an empty square is left
        """
        assert not self.n_empty_squares() == 0, "don't call place_new_food() if terminated is True"

        # two implementations with different efficiency to randomly select an empty square
        if self.n_empty_squares() / self.grid_size[0] * self.grid_size[1] > 0.2:  # more than 20% empty squares
            coord_found = False
            while not coord_found:
                coord = (np.random.randint(0, self.grid_size[0]), np.random.randint(0, self.grid_size[1]))
                # check if square is empty
                if not self.is_food_square(coord) and not self.is_snake_square(coord):
                    self.foods.append(coord)
                    coord_found = True
        else:  # less than 20% empty squares
            all_empty_coords = set(product(range(self.grid_size[0]), range(self.grid_size[1])))
            all_empty_coords -= set(self.foods)
            for i, snake in enumerate(self.alive_snakes()):
                all_empty_coords -= {snake.head}
                all_empty_coords -= set(snake.body)
            # random.choice does not work on sets so convert to tuple
            self.foods.append(random.choice(tuple(all_empty_coords)))

    def step_snake(self, action, snake_idx) -> int:
        """
        moves snake, if it is alive, and then checks for food and snake collisions
        """

        # cannot use self.alive_snakes() here as the list of actions contains entries for all snakes, also dead ones
        snake = self.snakes[snake_idx]

        # check if snake is alive
        if not snake.is_alive:
            return 0

        # move snake without removing tail end
        snake.step_without_removing_tail_end(action)

        # check for death of snake
        if self.is_off_grid(snake.head) or self.is_snake_square_except_own_head(snake):
            snake.is_alive = False
            reward = -1
        # check for food
        elif self.is_food_square(snake.head):
            self.foods.remove(snake.head)  # all food coords are unique
            self.place_new_food()
            reward = 1
        # ordinary step
        else:
            snake.remove_tail_end()
            reward = 0

        return reward

    def step(self, actions) -> tuple[list[(int, int)], list[Snake], list[int], bool, dict[str, int]]:
        """
        performs an action for each snake and collects their rewards. Also checks if the game has finished

        actions: list with actions corresponding to each snake, including dead snakes. the actions of the dead snakes
        are ignored
        returns: the observation, a list of rewards including dead snakes, whether terminated or not, info. rewards for
        dead snakes are 0
        """
        assert not self.n_alive_snakes() == 0 and not self.n_empty_squares() == 0, "don't call step() if terminated is True"

        # it's not fair if always the same snake can perform its action first, so randomly permute the
        # order in which snakes can perform their action
        rewards = [0] * len(self.snakes)  # to allow assignment by index instead of append
        random_perm = np.random.permutation(len(self.snakes))
        for i in random_perm:
            rewards[i] = self.step_snake(actions[i], i)

        # TODO: begin experiment: stimulate cooperation: AI-snake also gets reward if bot gets reward
        # only works for n_snakes = 2
        # if rewards[1] == 1:
        #     rewards[0] += 1
        # end experiment

        # TODO: begin experiment: game stops if one of both snakes die, to stimulate multiplayer learning
        # only works for n_snakes = 2
        terminated = self.n_alive_snakes() == 0 or self.n_empty_squares() == 0  # orig code
        # terminated = self.n_alive_snakes() <= 1 or self.n_empty_squares() == 0
        # end experiment

        # return the complete state as observation
        return self.foods, self.alive_snakes(), rewards, terminated, {"snakes_remaining": self.n_alive_snakes()}


