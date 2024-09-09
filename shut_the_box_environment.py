import numpy as np  # Import NumPy for dtype
import torch
import gymnasium as gym
from gymnasium import spaces

class ShutTheBoxEnv(gym.Env):
    def __init__(self, render_mode=None):
        super(ShutTheBoxEnv, self).__init__()
        
        self.render_mode = render_mode
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Tiles are from 1 to 9, and can either be open (1) or shut (0)
        self.tiles = torch.ones(9, device=self.device)
        self.dice_sum = torch.tensor(0, device=self.device)  # Initialize dice_sum
        
        # Observation space: 9 for tiles + 1 for dice_sum, use np.float32
        self.observation_space = spaces.Box(low=0, high=1, shape=(10,), dtype=np.float32)  # Correct dtype
        
        # Action space: 512 possible actions (combinations of tiles to shut)
        self.action_space = spaces.Discrete(512)

        # To track invalid actions during training
        self.invalid_actions_count = 0


    def reset(self, seed=None, options=None):
        # Set random seed for reproducibility
        if seed is not None:
            torch.manual_seed(seed)
        
        # Reset all tiles to open and roll dice
        self.tiles = torch.ones(9, device=self.device)
        self.dice_sum = self.roll_dice()
        self.invalid_actions_count = 0  # Reset invalid action counter
        
        return self._get_obs(), {}

    def roll_dice(self):
        # Roll two dice (range 1 to 6)
        dice_sum = torch.randint(1, 7, (1,), device=self.device) + torch.randint(1, 7, (1,), device=self.device)
        return dice_sum.item()  # Convert tensor to Python number

    def get_valid_actions(self, dice_sum):
        # Generate valid actions based on current dice sum
        def combinations(idx, num, path):
            if num == 0:
                return [path]
            if idx >= len(self.tiles) or num < 0:
                return []
            if self.tiles[idx].item() == 0:  # If the tile is already shut, skip it
                return combinations(idx + 1, num, path)
            return combinations(idx + 1, num - (idx + 1), path + [idx]) + combinations(idx + 1, num, path)

        return combinations(0, dice_sum, [])

    def step(self, action):
        selected_tiles = [i + 1 for i in range(9) if action & (1 << i)]
        
        # Validate the action
        if sum(selected_tiles) != self.dice_sum or any(self.tiles[i - 1].item() == 0 for i in selected_tiles):
            # Invalid action: log and penalize
            self.invalid_actions_count += 1
            print(f"Invalid action: Trying to shut tiles {selected_tiles} for dice sum {self.dice_sum}")
            return self._get_obs(), -10, True, False, {}
        
        # Shut the selected tiles
        for tile in selected_tiles:
            self.tiles[tile - 1] = 0  # Shut the tile
        
        # Roll dice again
        self.dice_sum = self.roll_dice()

        # Check if the game is over (no valid moves left)
        valid_moves = self.get_valid_actions(self.dice_sum)
        if not valid_moves:
            return self._get_obs(), -sum(self.tiles).item(), True, False, {}

        return self._get_obs(), 1, False, False, {}  # Small reward for valid actions

    def _get_obs(self):
        # Return flattened observation (9 tiles + 1 dice_sum)
        return torch.cat([self.tiles, torch.tensor([self.dice_sum], device=self.device)]).cpu().numpy()

    def render(self):
        if self.render_mode == 'human':
            print(f"Tiles: {self.tiles.cpu().numpy()}, Dice Sum: {self.dice_sum}")
