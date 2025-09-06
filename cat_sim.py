import pygame

# --- Constants ---
SCREEN_WIDTH = 800
SCREEN_HEIGHT = 600
FPS = 60
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
ORANGE = (255, 100, 0)
GREEN = (0, 255, 0)

# --- Initialize Pygame ---
pygame.init()
screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
pygame.display.set_caption("Virtual Cat Sim")
clock = pygame.time.Clock()

class Cat(pygame.sprite.Sprite):
    def __init__(self, x, y):
        super().__init__()
        self.image = pygame.Surface([50, 50])
        self.image.fill(ORANGE)
        self.rect = self.image.get_rect(center=(x, y)) # Improved positioning
    def sense_food(self, food_rect):
        """
        Calculates the direction and distance to the food.
        Returns a tuple: (direction_x, direction_y)
        """
        direction_x = food_rect.x - self.rect.x
        direction_y = food_rect.y - self.rect.y
        return direction_x, direction_y
class Food(pygame.sprite.Sprite):
    def __init__(self, x, y):
        super().__init__()
        self.image = pygame.Surface([30, 30])
        self.image.fill(GREEN)
        self.rect = self.image.get_rect(center=(x, y)) # Improved positioning

# --- Main Game Logic ---
def main():
    # Create instances of the objects
    cat = Cat(SCREEN_WIDTH // 2, SCREEN_HEIGHT // 2)
    food = Food(100, 100)

    all_sprites = pygame.sprite.Group()
    all_sprites.add(cat, food) # Simplified adding to group

    running = True
    while running:
        # Event handling
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        # --- AI Brain Logic ---
        # 1. Sense the environment
        direction_x, direction_y = cat.sense_food(food.rect)

        # 2. Decide on an action (move towards the food)
        if direction_x > 0:
            cat.rect.x += 1
        elif direction_x < 0:
            cat.rect.x -= 1

        if direction_y > 0:
            cat.rect.y += 1
        elif direction_y < 0:
            cat.rect.y -= 1

        # Rendering
        screen.fill(BLACK)
        all_sprites.draw(screen)  # <-- This is the crucial line that was missing!
        pygame.display.flip()

        # Cap the frame rate
        clock.tick(FPS)
    pygame.quit()
# --- Run the game ---
if __name__ == "__main__":
    main()