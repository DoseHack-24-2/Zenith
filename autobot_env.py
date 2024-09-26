import gym
from gym import spaces
import numpy as np

# Define constants for directions
UP, RIGHT, DOWN, LEFT = 0, 1, 2, 3

class AutobotEnv(gym.Env):
    """Custom Environment for guiding multiple autobots in a rectangular grid with obstacles"""
    metadata = {'render.modes': ['human']}

    def __init__(self, grid):
        super(AutobotEnv, self).__init__()

        self.grid = grid
        self.grid_height, self.grid_width = len(grid), len(grid[0])
        self.num_autobots = sum(1 for row in grid for cell in row if cell.isupper() and cell != 'X')

        self.action_space = spaces.Discrete(5)

        self.observation_space = spaces.Dict({
            'positions': spaces.Box(low=0, high=max(self.grid_height-1, self.grid_width-1), shape=(self.num_autobots, 2), dtype=np.int32),
            'directions': spaces.MultiDiscrete([4] * self.num_autobots),
            'obstacles': spaces.MultiBinary(self.grid_height * self.grid_width),
            'destinations': spaces.Box(low=0, high=max(self.grid_height-1, self.grid_width-1), shape=(self.num_autobots, 2), dtype=np.int32)
        })

        self.starting_positions = []
        self.destinations = []
        self.obstacles = []
        self.correct_destinations = []  # To store the correct destinations

        for i, row in enumerate(grid):
            for j, cell in enumerate(row):
                if cell.isupper() and cell != 'X':
                    self.starting_positions.append((i, j))
                elif cell.islower():
                    self.destinations.append((i, j))
                    # Create a mapping for autobots to destinations
                    autobot_index = ord(cell.upper()) - ord('A')
                    while len(self.correct_destinations) <= autobot_index:
                        self.correct_destinations.append(None)
                    self.correct_destinations[autobot_index] = (i, j)
                elif cell == 'X':
                    self.obstacles.append((i, j))

        self.starting_positions = np.array(self.starting_positions, dtype=np.int32)
        self.destinations = np.array(self.destinations, dtype=np.int32)
        self.autobot_positions = self.starting_positions.copy()
        self.autobot_directions = np.zeros(self.num_autobots, dtype=np.int32)

    def reset(self):
        """Reset the environment to the starting state for each episode."""
        self.autobot_positions = self.starting_positions.copy()
        self.autobot_directions = np.zeros(self.num_autobots, dtype=np.int32)

        obstacle_grid = np.zeros((self.grid_height, self.grid_width), dtype=np.int32)
        for obs in self.obstacles:
            obstacle_grid[obs[0], obs[1]] = 1

        return {
            'positions': self.autobot_positions.copy(),
            'directions': self.autobot_directions.copy(),
            'obstacles': obstacle_grid,
            'destinations': np.array(self.correct_destinations, dtype=np.int32)  # Return correct destinations
        }

    def step(self, actions):
        """Perform actions for all autobots."""
        rewards = np.zeros(self.num_autobots)
        done = False

        # Store the original positions
        original_positions = self.autobot_positions.copy()

        # First, calculate the new positions for all autobots
        new_positions = []
        for i in range(self.num_autobots):
            if np.array_equal(self.autobot_positions[i], self.correct_destinations[i]):
                new_positions.append(self.autobot_positions[i].copy())
                continue

            new_pos = self.autobot_positions[i].copy()
            if actions[i] == 0:  # Forward
                new_pos = self._get_forward_position(i)
            elif actions[i] == 1:  # Reverse
                new_pos = self._get_reverse_position(i)
            elif actions[i] == 2:  # Turn left
                self.autobot_directions[i] = (self.autobot_directions[i] - 1) % 4
            elif actions[i] == 3:  # Turn right
                self.autobot_directions[i] = (self.autobot_directions[i] + 1) % 4
            new_positions.append(new_pos)

        # Now, check for collisions and resolve them
        move_order = list(range(self.num_autobots))
        np.random.shuffle(move_order)  # Randomize the order of movement to avoid bias

        for i in move_order:
            if np.array_equal(self.autobot_positions[i], self.correct_destinations[i]):
                continue

            new_pos = new_positions[i]
            collision = False

            # Check for collisions with other autobots
            for j in range(self.num_autobots):
                if i != j and np.array_equal(new_pos, new_positions[j]):
                    collision = True
                    break

            # Check for collisions with obstacles
            if tuple(new_pos) in self.obstacles:
                collision = True

            if not collision:
                self.autobot_positions[i] = new_pos
            else:
                rewards[i] -= 5  # Penalty for collision

        # Calculate rewards
        for i in range(self.num_autobots):
            if not np.array_equal(self.autobot_positions[i], self.correct_destinations[i]):
                if tuple(self.autobot_positions[i]) in self.obstacles:
                    rewards[i] -= 10  # Negative reward for hitting an obstacle
                elif np.array_equal(self.autobot_positions[i], original_positions[i]):
                    rewards[i] -= 2  # Small negative reward for not moving
                else:
                    rewards[i] -= 1  # Small negative reward for each step

        # Check if all autobots have reached their destinations
        done = np.all([np.array_equal(self.autobot_positions[i], self.correct_destinations[i]) for i in range(self.num_autobots)])

        obstacle_grid = np.zeros((self.grid_height, self.grid_width), dtype=np.int32)
        for obs in self.obstacles:
            obstacle_grid[obs[0], obs[1]] = 1

        return {
            'positions': self.autobot_positions.copy(),
            'directions': self.autobot_directions.copy(),
            'obstacles': obstacle_grid,
            'destinations': self.correct_destinations.copy()  # Use the correct destinations
        }, rewards, done, {}

    def _get_forward_position(self, i):
        """Calculate the forward position for an autobot."""
        new_position = self.autobot_positions[i].copy()
        if self.autobot_directions[i] == UP and new_position[0] > 0:
            new_position[0] -= 1
        elif self.autobot_directions[i] == DOWN and new_position[0] < self.grid_height - 1:
            new_position[0] += 1
        elif self.autobot_directions[i] == LEFT and new_position[1] > 0:
            new_position[1] -= 1
        elif self.autobot_directions[i] == RIGHT and new_position[1] < self.grid_width - 1:
            new_position[1] += 1
        return new_position

    def _get_reverse_position(self, i):
        """Calculate the reverse position for an autobot."""
        new_position = self.autobot_positions[i].copy()
        if self.autobot_directions[i] == UP and new_position[0] < self.grid_height - 1:
            new_position[0] += 1
        elif self.autobot_directions[i] == DOWN and new_position[0] > 0:
            new_position[0] -= 1
        elif self.autobot_directions[i] == LEFT and new_position[1] < self.grid_width - 1:
            new_position[1] += 1
        elif self.autobot_directions[i] == RIGHT and new_position[1] > 0:
            new_position[1] -= 1
        return new_position

    def render(self, mode='human'):
        """Render the grid for visualization."""
        grid = [['.' for _ in range(self.grid_width)] for _ in range(self.grid_height)]
        
        for obs in self.obstacles:
            grid[obs[0]][obs[1]] = 'X'
        
        for i, dest in enumerate(self.correct_destinations):
            grid[dest[0]][dest[1]] = f'D{i+1}'
        
        for i, pos in enumerate(self.autobot_positions):
            grid[pos[0]][pos[1]] = f'A{i+1}'
        
        for row in grid:
            print(' '.join(row))
        print()