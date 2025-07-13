import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pygame
import random

class SnakeEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self, run_speed = 10):
        super(SnakeEnv, self).__init__()
        self.run_speed = run_speed
        self.action_space = spaces.Discrete(4)

        # Observation: [danger_straight, danger_right, danger_left, 
        #               food_direction_x, food_direction_y, 
        #               direction_left, direction_right, direction_up, direction_down]
        self.observation_space = spaces.Box(low=-1, high=1, shape=(9,), dtype=np.float32)

        # Game board settings
        self.grid_size = 12  # Grid : grid_size x grid_size dimensions
        self.snake_block = 20  # Size of each block in pixels
        self.width = self.grid_size * self.snake_block
        self.height = self.grid_size * self.snake_block

        # Initialize Pygame
        pygame.init()
        self.display = pygame.display.set_mode((self.width, self.height))
        pygame.display.set_caption('Snake')
        self.clock = pygame.time.Clock()

        # Initialize game state
        self.reset()

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)  # optional: sets the seed
        self.snake = [[self.grid_size // 2, self.grid_size // 2]] 
        self.snake_direction = random.choice(['UP', 'DOWN', 'LEFT', 'RIGHT'])
        self.snake_length = 1
        self._place_food()
        self.score = 0
        self.done = False

        return self._get_observation(), {}

    def step(self, action):
        # Execute one time step within the environment
        if self.done:
            return self.reset(), 0.0, True, False, {}

        # Update the direction
        self._update_direction(action)
        # Move the snake
        self._move_snake()
        # Check for collisions
        self.done = self._check_collision()
        # Calculate reward
        reward = self._calculate_reward()
        # Get observation
        observation = self._get_observation()

        terminated = self.done
        truncated = False

        return observation, reward, terminated, truncated, {}

    def render(self, mode='human'):
        # Render the environment to the screen
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.close()

        self.display.fill((0, 0, 0))

        # Draw the food
        pygame.draw.rect(
            self.display, (255, 0, 0),
            [self.food[0]*self.snake_block, self.food[1]*self.snake_block,
             self.snake_block, self.snake_block]
        )

        # Draw the snake
        for segment in self.snake:
            pygame.draw.rect(
                self.display, (0, 255, 0),
                [segment[0]*self.snake_block, segment[1]*self.snake_block,
                 self.snake_block, self.snake_block]
            )

        pygame.display.update()
        self.clock.tick(self.run_speed)  # Control the game speed

    def close(self):
        pygame.quit()

    def _place_food(self):
        # Place the food at a random position not occupied by the snake
        while True:
            x = random.randint(0, self.grid_size - 1)
            y = random.randint(0, self.grid_size - 1)
            if [x, y] not in self.snake:
                self.food = [x, y]
                break

    def _update_direction(self, action):
        # Update the snake's direction based on action
        direction_mapping = {0: 'LEFT', 1: 'RIGHT', 2: 'UP', 3: 'DOWN'}
        opposite_directions = {'LEFT': 'RIGHT', 'RIGHT': 'LEFT', 'UP': 'DOWN', 'DOWN': 'UP'}
        new_direction = direction_mapping[action]
        # Prevent the snake from reversing
        if new_direction != opposite_directions.get(self.snake_direction):
            self.snake_direction = new_direction

    def _move_snake(self):
        # Move the snake in the current direction
        head_x, head_y = self.snake[0]
        if self.snake_direction == 'LEFT':
            head_x -= 1
        elif self.snake_direction == 'RIGHT':
            head_x += 1
        elif self.snake_direction == 'UP':
            head_y -= 1
        elif self.snake_direction == 'DOWN':
            head_y += 1
        new_head = [head_x, head_y]
        self.snake.insert(0, new_head)
        if len(self.snake) > self.snake_length:
            self.snake.pop()

    def _check_collision(self):
        # Check for collisions with walls or self
        head = self.snake[0]
        if (head[0] < 0 or head[0] >= self.grid_size or
                head[1] < 0 or head[1] >= self.grid_size):
            return True
        if head in self.snake[1:]:
            return True
        return False

    def _calculate_reward(self):
        # Calculate reward
        if self.done:
            return -1  # Penalty for dying
        if self.snake[0] == self.food:
            self.snake_length += 1
            self.score += 1
            self._place_food()
            return 1  # Reward for eating food
        else:
            return 0  # Otherwise, no reward

    def _get_observation(self):
        # Get observation
        head = self.snake[0]
        # Danger straight, right, left
        danger_straight = self._danger_in_direction(self.snake_direction)
        danger_right = self._danger_in_direction(self._turn_direction('RIGHT'))
        danger_left = self._danger_in_direction(self._turn_direction('LEFT'))
        # Food direction
        food_direction = [0, 0]
        if self.food[0] > head[0]:
            food_direction[0] = 1  # Food is to the right
        elif self.food[0] < head[0]:
            food_direction[0] = -1  # Food is to the left
        if self.food[1] > head[1]:
            food_direction[1] = 1  # Food is down
        elif self.food[1] < head[1]:
            food_direction[1] = -1  # Food is up
        # Current direction (one-hot encoded)
        direction_one_hot = [0, 0, 0, 0]
        directions = ['LEFT', 'RIGHT', 'UP', 'DOWN']
        direction_index = directions.index(self.snake_direction)
        direction_one_hot[direction_index] = 1
        observation = [
            int(danger_straight), int(danger_right), int(danger_left),
            food_direction[0], food_direction[1]
        ] + direction_one_hot
        return np.array(observation, dtype=np.float32)

    def _danger_in_direction(self, direction):
        # Check if there is danger in the given direction
        head = self.snake[0]
        x, y = head
        if direction == 'LEFT':
            x -= 1
        elif direction == 'RIGHT':
            x += 1
        elif direction == 'UP':
            y -= 1
        elif direction == 'DOWN':
            y += 1
        if x < 0 or x >= self.grid_size or y < 0 or y >= self.grid_size:
            return True
        if [x, y] in self.snake:
            return True
        return False

    def _turn_direction(self, turn):
        # Get new direction after turning
        directions = ['UP', 'RIGHT', 'DOWN', 'LEFT']
        idx = directions.index(self.snake_direction)
        if turn == 'RIGHT':
            idx = (idx + 1) % 4
        elif turn == 'LEFT':
            idx = (idx - 1) % 4
        return directions[idx]
