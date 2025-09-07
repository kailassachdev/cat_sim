import pygame
from stable_baselines3 import PPO
# MODIFICATION: Import wrappers
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from cat_brain.rl_model import CatEnv
from cat_brain.llm_interface import CatPersonality
import os
from dotenv import load_dotenv

def main():
    load_dotenv() 
    API_KEY = os.getenv("GOOGLE_API_KEY") 
    
    if not API_KEY:
        print("ERROR: GOOGLE_API_KEY not found.")
        return

    # --- MODIFICATION: Setup for VecNormalize ---
    # The environment must be wrapped in the same way it was during training
    def make_env():
        return CatEnv(render_mode="human")

    env = DummyVecEnv([make_env])
    # Load the saved statistics
    env = VecNormalize.load("cat_brain/vec_normalize_stationary.pkl", env)
    # Don't update the statistics during testing
    env.training = False
    # Don't normalize rewards during testing
    env.norm_reward = False
    
    # Load the trained model
    model = PPO.load("cat_brain/ppo_cat_model_maze_stationary", env=env)
    
    # --- Main Game Loop ---
    obs = env.reset()
    running = True
    while running:
        # Get action from the model
        action, _states = model.predict(obs, deterministic=True)
        # Apply the action to the environment
        obs, reward, done, info = env.step(action)

        # Render the unwrapped environment
        env.envs[0].render()
        pygame.display.flip()

        for event in pygame.event.get():
            if event.type == pygame.QUIT or (event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE):
                running = False
        
        # If an episode ends, reset it
        if done:
            print("Episode finished!")
            obs = env.reset()
    
    env.close()

if __name__ == "__main__":
    main()