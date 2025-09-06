import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
import pygame # MODIFICATION: Import the pygame library

# It's important to import your custom environment
from cat_brain.rl_model import CatEnv

# --- Create the custom callback for rendering ---
class RenderCallback(BaseCallback):
    """
    A custom callback that renders the environment during training and keeps the window responsive.
    """
    def __init__(self, verbose=0):
        super(RenderCallback, self).__init__(verbose)

    def _on_step(self) -> bool:
        # Step 1: Call the environment's render method to draw the scene
        self.training_env.render()
        
        # --- MODIFICATION: Add display update and event handling ---
        # Step 2: Update the display to show what was drawn
        pygame.display.flip()
        # Step 3: Process Pygame events to keep the window from freezing
        pygame.event.pump()
        # --- END MODIFICATION ---

        return True

# Create the environment instance
env = CatEnv(render_mode="human")

# --- Define custom learning settings ---
learning_rate = 0.0001
policy_kwargs = dict(net_arch=dict(pi=[128, 128], vf=[128, 128]))

# --- The AI Brain ---
model = PPO(
    "MlpPolicy",
    env,
    verbose=1,
    learning_rate=learning_rate,
    policy_kwargs=policy_kwargs
)

# Create an instance of our custom callback
render_callback = RenderCallback()

# Start the training process
print("Starting training with live visualization...")
model.learn(total_timesteps=500000, callback=render_callback)

# Save the trained model to a file
model.save("cat_brain/ppo_cat_model")
print("Training complete! Model saved.")

# Clean up
env.close()