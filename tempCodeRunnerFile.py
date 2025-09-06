import gymnasium as gym
from stable_baselines3 import PPO

# It's important to import your custom environment
from cat_brain.rl_model import CatEnv

# Create the environment instance
env = CatEnv(render_mode="human")

# --- The AI Brain ---
# We will use the PPO algorithm
# MlpPolicy is a type of neural network that will act as our agent
model = PPO("MlpPolicy", env, verbose=1)

# Start the training process
# The total_timesteps is how many steps the model will take to learn
print("Starting training...")
model.learn(total_timesteps=100000)

# Save the trained model to a file
model.save("cat_brain/ppo_cat_model")
print("Training complete! Model saved.")

# Clean up
env.close()