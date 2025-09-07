import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pygame
import os
import random
from collections import deque

# --- Environment Constants ---
SCREEN_SIZE = (800, 600)
CAT_SIZE = 38
MOUSE_SIZE = 38
MOVE_SPEED = 4
MOUSE_SPEED = 2
MAX_STEPS = 2000 
STUCK_RADIUS = 120
STUCK_PENALTY = -200
# MODIFICATION: Added the missing constant back
MAX_STUCK_STEPS = 10000

# --- Maze Constants ---
TILE_SIZE = 40
MAZE_WIDTH = SCREEN_SIZE[0] // TILE_SIZE
MAZE_HEIGHT = SCREEN_SIZE[1] // TILE_SIZE

def generate_maze(width, height):
    # ... (This function is unchanged)
    maze = np.ones((height, width), dtype=np.uint8)
    start_x = random.randint(0, (width // 2) - 1) * 2
    start_y = random.randint(0, (height // 2) - 1) * 2
    maze[start_y, start_x] = 0
    stack = [(start_x, start_y)]
    while stack:
        x, y = stack[-1]
        neighbors = []
        for dx, dy in [(0, 2), (0, -2), (2, 0), (-2, 0)]:
            nx, ny = x + dx, y + dy
            if 0 <= nx < width and 0 <= ny < height and maze[ny, nx] == 1:
                neighbors.append((nx, ny))
        if neighbors:
            nx, ny = random.choice(neighbors)
            maze[ny, nx] = 0
            maze[y + (ny - y) // 2, x + (nx - x) // 2] = 0
            stack.append((nx, ny))
        else:
            stack.pop()
    return maze

class CatEnv(gym.Env):
    metadata = {"render_modes": ["human"], "render_fps": 60}

    def __init__(self, render_mode=None, initial_complexity=0.1):
        super(CatEnv, self).__init__()
        
        self.render_mode = render_mode
        self.screen = None
        self.complexity = initial_complexity
        
        self.maze = None
        self.walls = []
        self.path_tiles = []
        self._regenerate_maze()

        if self.render_mode == "human":
            pygame.init()
            self.screen = pygame.display.set_mode(SCREEN_SIZE)
            pygame.display.set_caption("RL Cat Maze Sim")
            script_dir = os.path.dirname(__file__)
            cat_path = os.path.join(script_dir, "cat_with_eyes.png")
            mouse_path = os.path.join(script_dir, "mouse.png")
            self.cat_image = pygame.transform.scale(pygame.image.load(cat_path).convert_alpha(), (CAT_SIZE, CAT_SIZE))
            self.mouse_image = pygame.transform.scale(pygame.image.load(mouse_path).convert_alpha(), (MOUSE_SIZE, MOUSE_SIZE))

        self.cat_rect = pygame.Rect(0, 0, CAT_SIZE, CAT_SIZE)
        self.mouse_rect = pygame.Rect(0, 0, MOUSE_SIZE, MOUSE_SIZE)
        self.action_space = spaces.Discrete(4)
        
        low = np.array([-SCREEN_SIZE[0], -SCREEN_SIZE[1]] + [0] * 8)
        high = np.array([SCREEN_SIZE[0], SCREEN_SIZE[1]] + [1.0] * 8)
        self.observation_space = spaces.Box(low=low, high=high, dtype=np.float32)

        self.last_bfs_distance = 0
        self.stuck_counter = 0
        self.stuck_anchor_pos = None
        self.steps_taken = 0

    def _regenerate_maze(self):
        # ... (This method is unchanged)
        base_maze = generate_maze(MAZE_WIDTH, MAZE_HEIGHT)
        self.walls = []
        interior_walls = []
        for y in range(1, MAZE_HEIGHT - 1):
            for x in range(1, MAZE_WIDTH - 1):
                if base_maze[y, x] == 1:
                    interior_walls.append((x, y))
        if len(interior_walls) > 0:
            num_walls_to_remove = int(len(interior_walls) * (1.0 - self.complexity))
            if num_walls_to_remove > 0 and len(interior_walls) > num_walls_to_remove:
                 walls_to_remove = random.sample(interior_walls, num_walls_to_remove)
                 for x, y in walls_to_remove:
                     base_maze[y, x] = 0
        self.maze = base_maze
        for y, row in enumerate(self.maze):
            for x, tile in enumerate(row):
                if tile == 1:
                    self.walls.append(pygame.Rect(x * TILE_SIZE, y * TILE_SIZE, TILE_SIZE, TILE_SIZE))
        self.path_tiles = np.argwhere(self.maze == 0)

    def increase_complexity(self):
        self.complexity = min(self.complexity + 0.1, 1.0)
        print(f"Success! Increasing maze complexity to {self.complexity:.1f}")
        self._regenerate_maze()

    def _get_bfs_distance(self):
        # ... (This method is unchanged)
        cat_tile = (self.cat_rect.centerx // TILE_SIZE, self.cat_rect.centery // TILE_SIZE)
        mouse_tile = (self.mouse_rect.centerx // TILE_SIZE, self.mouse_rect.centery // TILE_SIZE)
        if cat_tile == mouse_tile: return 0
        queue = deque([(cat_tile, 0)])
        visited = {cat_tile}
        while queue:
            (x, y), dist = queue.popleft()
            for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                nx, ny = x + dx, y + dy
                if (nx, ny) == mouse_tile: return dist + 1
                if 0 <= nx < MAZE_WIDTH and 0 <= ny < MAZE_HEIGHT and \
                   self.maze[ny, nx] == 0 and (nx, ny) not in visited:
                    visited.add((nx, ny))
                    queue.append(((nx, ny), dist + 1))
        return MAZE_WIDTH * MAZE_HEIGHT

    def _get_lidar_readings(self):
        # ... (This method is unchanged)
        center = self.cat_rect.center
        readings = []
        max_dist = np.linalg.norm(SCREEN_SIZE)
        for angle in np.linspace(0, 2 * np.pi, 8, endpoint=False):
            ray_dir = np.array([np.cos(angle), np.sin(angle)])
            dist = 0
            while dist < max_dist:
                point = center + ray_dir * dist
                tile_x, tile_y = int(point[0] // TILE_SIZE), int(point[1] // TILE_SIZE)
                if not (0 <= tile_x < MAZE_WIDTH and 0 <= tile_y < MAZE_HEIGHT) or self.maze[tile_y, tile_x] == 1:
                    readings.append(dist / max_dist)
                    break
                dist += 1
            else:
                readings.append(1.0)
        return readings

    def _get_obs(self):
        relative_pos = [self.mouse_rect.x - self.cat_rect.x, self.mouse_rect.y - self.cat_rect.y]
        lidar = self._get_lidar_readings()
        return np.array(relative_pos + lidar, dtype=np.float32)

    def _place_on_random_path(self, rect_size):
        tile_y, tile_x = random.choice(self.path_tiles)
        return pygame.Rect(tile_x * TILE_SIZE + (TILE_SIZE - rect_size) // 2,
                           tile_y * TILE_SIZE + (TILE_SIZE - rect_size) // 2,
                           rect_size, rect_size)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
            
        self.cat_rect = self._place_on_random_path(CAT_SIZE)
        self.mouse_rect = self._place_on_random_path(MOUSE_SIZE)
        while self.cat_rect.colliderect(self.mouse_rect):
            self.mouse_rect = self._place_on_random_path(MOUSE_SIZE)
        self.last_bfs_distance = self._get_bfs_distance()
        self.stuck_counter = 0
        self.stuck_anchor_pos = self.cat_rect.center
        self.steps_taken = 0
        return self._get_obs(), {}

    def step(self, action):
        self.steps_taken += 1
        
        original_pos = self.cat_rect.copy()
        if action == 0: self.cat_rect.y -= MOVE_SPEED
        elif action == 1: self.cat_rect.y += MOVE_SPEED
        elif action == 2: self.cat_rect.x -= MOVE_SPEED
        elif action == 3: self.cat_rect.x += MOVE_SPEED

        hit_wall = False
        if self.cat_rect.collidelist(self.walls) != -1:
            self.cat_rect = original_pos
            hit_wall = True

        self.cat_rect.left = np.clip(self.cat_rect.left, 0, SCREEN_SIZE[0] - CAT_SIZE)
        self.cat_rect.top = np.clip(self.cat_rect.top, 0, SCREEN_SIZE[1] - CAT_SIZE)

        observation = self._get_obs()
        
        if self.cat_rect.colliderect(self.mouse_rect):
            reward = 1000 + (MAX_STEPS - self.steps_taken) * 2
            done = True
        else:
            done = False
            current_bfs_distance = self._get_bfs_distance()
            reward = (self.last_bfs_distance - current_bfs_distance) * 5.0
            reward -= 0.05
            reward += -0.01 * current_bfs_distance
            if hit_wall: 
                reward -= 2.0
            self.last_bfs_distance = current_bfs_distance
        
        if not done:
            distance_from_anchor = np.linalg.norm(np.array(self.cat_rect.center) - np.array(self.stuck_anchor_pos))
            if distance_from_anchor > STUCK_RADIUS:
                self.stuck_anchor_pos = self.cat_rect.center
                self.stuck_counter = 0
            else:
                self.stuck_counter += 1
            
            truncated = self.stuck_counter >= MAX_STUCK_STEPS
            if truncated:
                print("Agent stuck, applying penalty and resetting episode.")
                reward += STUCK_PENALTY
        else:
            truncated = False

        return observation, reward, done, truncated, {}

    def render(self):
        if self.render_mode == "human":
            self.screen.fill((20, 20, 20))
            for wall in self.walls:
                pygame.draw.rect(self.screen, (100, 100, 120), wall)
            self.screen.blit(self.cat_image, self.cat_rect) 
            self.screen.blit(self.mouse_image, self.mouse_rect)
            pygame.display.flip()
        
    def close(self):
        if self.screen is not None:
            pygame.quit()