from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from shut_the_box_environment import ShutTheBoxEnv
import torch

# Check if using GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Training on: {device}")

# Create the environment
env = ShutTheBoxEnv()
env = DummyVecEnv([lambda: env])

# Use "MlpPolicy" with GPU support
model = PPO("MlpPolicy", env, verbose=1, device=device)

# Train the agent for 500,000 steps
model.learn(total_timesteps=500000)

# Save the trained model
model.save("ppo_shut_the_box")
