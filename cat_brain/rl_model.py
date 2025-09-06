import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pygame

# --- Environment Constants ---
# MODIFICATION: Reverted to a fixed 800x600 screen size
SCREEN_SIZE = (800, 600)
CAT_SIZE = 50
FOOD_SIZE = 30
MOVE_SPEED = 4 # Kept the subtle speed boost

class CatEnv(gym.Env):
    def __init__(self, render_mode=None):
        super(CatEnv, self).__init__()
        
        self.render_mode = render_mode
        self.screen = None

        # MODIFICATION: Reverted to use the fixed SCREEN_SIZE constant
        if self.render_mode == "human":
            pygame.init()
            self.screen = pygame.display.set_mode(SCREEN_SIZE)
            pygame.display.set_caption("RL Cat Sim")

        self.cat_rect = pygame.Rect(0, 0, CAT_SIZE, CAT_SIZE)
        self.food_rect = pygame.Rect(0, 0, FOOD_SIZE, FOOD_SIZE)
        
        self.action_space = spaces.Discrete(4)
        
        # MODIFICATION: Observation space is now fixed again
        self.observation_space = spaces.Box(low=0, high=max(SCREEN_SIZE), shape=(4,), dtype=np.float32)

    def _get_obs(self):
        return np.array([self.cat_rect.x, self.cat_rect.y, self.food_rect.x, self.food_rect.y])

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        # MODIFICATION: Use fixed SCREEN_SIZE for placement
        self.cat_rect.x = np.random.randint(0, SCREEN_SIZE[0] - CAT_SIZE)
        self.cat_rect.y = np.random.randint(0, SCREEN_SIZE[1] - CAT_SIZE)
        self.food_rect.x = np.random.randint(0, SCREEN_SIZE[0] - FOOD_SIZE)
        self.food_rect.y = np.random.randint(0, SCREEN_SIZE[1] - FOOD_SIZE)
        
        observation = self._get_obs()
        info = {}
        return observation, info

    def step(self, action):
        if action == 0: self.cat_rect.y -= MOVE_SPEED
        elif action == 1: self.cat_rect.y += MOVE_SPEED
        elif action == 2: self.cat_rect.x -= MOVE_SPEED
        elif action == 3: self.cat_rect.x += MOVE_SPEED

        hit_wall = False
        # MODIFICATION: Use fixed SCREEN_SIZE for boundary checks
        if self.cat_rect.left < 0 or self.cat_rect.right > SCREEN_SIZE[0] or \
           self.cat_rect.top < 0 or self.cat_rect.bottom > SCREEN_SIZE[1]:
            hit_wall = True

        self.cat_rect.left = np.clip(self.cat_rect.left, 0, SCREEN_SIZE[0] - CAT_SIZE)
        self.cat_rect.top = np.clip(self.cat_rect.top, 0, SCREEN_SIZE[1] - CAT_SIZE)
            
        observation = self._get_obs()
        
        distance_to_food = np.linalg.norm(np.array(self.cat_rect.center) - np.array(self.food_rect.center))
        reward = -distance_to_food / 1000.0
        
        if hit_wall: reward -= 5 

        if self.cat_rect.colliderect(self.food_rect):
            reward = 1000 
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

            self.screen.fill((0, 0, 0))
            pygame.draw.rect(self.screen, (255, 100, 0), self.cat_rect)
            pygame.draw.rect(self.screen, (0, 255, 0), self.food_rect)
        
    def close(self):
        if self.screen is not None:
            pygame.quit()