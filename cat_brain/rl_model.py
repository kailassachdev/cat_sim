import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pygame
import os # MODIFICATION: Import os for path handling

# --- Environment Constants ---
SCREEN_SIZE = (800, 600) # Reverted to fixed size for consistency
CAT_SIZE = 50
MOUSE_SIZE = 30 # MODIFICATION: Renamed and adjusted size for mouse
MOVE_SPEED = 4
MOUSE_SPEED = 2 # MODIFICATION: New constant for mouse movement speed

class CatEnv(gym.Env):
    def __init__(self, render_mode=None):
        super(CatEnv, self).__init__()
        
        self.render_mode = render_mode
        self.screen = None

        if self.render_mode == "human":
            pygame.init()
            self.screen = pygame.display.set_mode(SCREEN_SIZE)
            pygame.display.set_caption("RL Cat Sim")
            
            # --- MODIFICATION: Load sprites ---
            script_dir = os.path.dirname(__file__) # Get directory of current script
            cat_path = os.path.join(script_dir, "cat_with_eyes.png")
            mouse_path = os.path.join(script_dir, "mouse.png")

            self.cat_image = pygame.image.load(cat_path).convert_alpha()
            self.cat_image = pygame.transform.scale(self.cat_image, (CAT_SIZE, CAT_SIZE))
            
            self.mouse_image = pygame.image.load(mouse_path).convert_alpha()
            self.mouse_image = pygame.transform.scale(self.mouse_image, (MOUSE_SIZE, MOUSE_SIZE))
            # --- END MODIFICATION ---

        self.cat_rect = pygame.Rect(0, 0, CAT_SIZE, CAT_SIZE)
        self.mouse_rect = pygame.Rect(0, 0, MOUSE_SIZE, MOUSE_SIZE) # MODIFICATION: Renamed food_rect to mouse_rect
        
        self.action_space = spaces.Discrete(4)
        
        self.observation_space = spaces.Box(low=0, high=max(SCREEN_SIZE), shape=(4,), dtype=np.float32)

    def _get_obs(self):
        # Observation still includes cat_x, cat_y, mouse_x, mouse_y
        return np.array([self.cat_rect.x, self.cat_rect.y, self.mouse_rect.x, self.mouse_rect.y])

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.cat_rect.x = np.random.randint(0, SCREEN_SIZE[0] - CAT_SIZE)
        self.cat_rect.y = np.random.randint(0, SCREEN_SIZE[1] - CAT_SIZE)
        self.mouse_rect.x = np.random.randint(0, SCREEN_SIZE[0] - MOUSE_SIZE)
        self.mouse_rect.y = np.random.randint(0, SCREEN_SIZE[1] - MOUSE_SIZE)
        
        observation = self._get_obs()
        info = {}
        return observation, info

    def step(self, action):
        # Cat movement based on RL agent
        if action == 0: self.cat_rect.y -= MOVE_SPEED
        elif action == 1: self.cat_rect.y += MOVE_SPEED
        elif action == 2: self.cat_rect.x -= MOVE_SPEED
        elif action == 3: self.cat_rect.x += MOVE_SPEED

        hit_wall = False
        if self.cat_rect.left < 0 or self.cat_rect.right > SCREEN_SIZE[0] or \
           self.cat_rect.top < 0 or self.cat_rect.bottom > SCREEN_SIZE[1]:
            hit_wall = True

        self.cat_rect.left = np.clip(self.cat_rect.left, 0, SCREEN_SIZE[0] - CAT_SIZE)
        self.cat_rect.top = np.clip(self.cat_rect.top, 0, SCREEN_SIZE[1] - CAT_SIZE)
            
        # --- MODIFICATION: Mouse random movement ---
        # Choose a random direction for the mouse
        mouse_action = np.random.randint(0, 4)
        if mouse_action == 0: self.mouse_rect.y -= MOUSE_SPEED # Up
        elif mouse_action == 1: self.mouse_rect.y += MOUSE_SPEED # Down
        elif mouse_action == 2: self.mouse_rect.x -= MOUSE_SPEED # Left
        elif mouse_action == 3: self.mouse_rect.x += MOUSE_SPEED # Right

        # Clamp mouse position to screen bounds
        self.mouse_rect.left = np.clip(self.mouse_rect.left, 0, SCREEN_SIZE[0] - MOUSE_SIZE)
        self.mouse_rect.top = np.clip(self.mouse_rect.top, 0, SCREEN_SIZE[1] - MOUSE_SIZE)
        # --- END MODIFICATION ---

        observation = self._get_obs()
        
        # Reward based on distance to the mouse
        distance_to_mouse = np.linalg.norm(np.array(self.cat_rect.center) - np.array(self.mouse_rect.center))
        reward = -distance_to_mouse / 1000.0 # Cat gets negative reward for being far from mouse
        
        if hit_wall: reward -= 5 

        # Check for collision with mouse
        if self.cat_rect.colliderect(self.mouse_rect):
            reward = 1000 # Large positive reward for catching the mouse
            done = True
        else:
            done = False
            
        truncated = False
        info = {}
        return observation, reward, done, truncated, info

    def render(self):
        if self.render_mode == "human":
            if self.screen is None:
                 pygame.init()
                 self.screen = pygame.display.set_mode(SCREEN_SIZE)
                 pygame.display.set_caption("RL Cat Sim")
                 # MODIFICATION: Re-load sprites if screen wasn't initialized before
                 script_dir = os.path.dirname(__file__)
                 cat_path = os.path.join(script_dir, "cat_with_eyes.png")
                 mouse_path = os.path.join(script_dir, "mouse.png")

                 self.cat_image = pygame.image.load(cat_path).convert_alpha()
                 self.cat_image = pygame.transform.scale(self.cat_image, (CAT_SIZE, CAT_SIZE))
                 
                 self.mouse_image = pygame.image.load(mouse_path).convert_alpha()
                 self.mouse_image = pygame.transform.scale(self.mouse_image, (MOUSE_SIZE, MOUSE_SIZE))

            self.screen.fill((0, 0, 0)) # Black background
            # --- MODIFICATION: Draw sprites instead of rectangles ---
            self.screen.blit(self.cat_image, self.cat_rect) 
            self.screen.blit(self.mouse_image, self.mouse_rect)
            # --- END MODIFICATION ---
        
    def close(self):
        if self.screen is not None:
            pygame.quit()