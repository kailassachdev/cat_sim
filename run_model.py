import pygame
from stable_baselines3 import PPO
from cat_brain.rl_model import CatEnv
from cat_brain.llm_interface import CatPersonality
import os
from dotenv import load_dotenv

# --- Constants ---
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
GRAY = (200, 200, 200)
INPUT_BOX_HEIGHT = 40
CHAT_LOG_HEIGHT = 100
FPS = 30 

def main():
    # --- Setup ---
    load_dotenv() 
    API_KEY = os.getenv("GOOGLE_API_KEY") 
    
    if not API_KEY:
        print("ERROR: GOOGLE_API_KEY not found.")
        return

    pygame.init()
    pygame.font.init()
    font = pygame.font.SysFont('Arial', 20)
    small_font = pygame.font.SysFont('Arial', 16)
    clock = pygame.time.Clock()

    # --- Initialize AI Components ---
    print("Initializing environment and AI...")
    env = CatEnv(render_mode="human")
    
    # MODIFICATION: Reverted to the simpler constructor for CatPersonality
    cat_personality = CatPersonality(api_key=API_KEY)
    
    model = PPO.load("cat_brain/ppo_cat_model", env=env)
    print("AI components loaded.")

    # --- Game Variables ---
    obs, info = env.reset()
    running = True
    user_input = ""
    chat_history = ["Cat: Meow! (I'm ready for commands!)"]

    # --- Main Game Loop ---
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                if event.key == pygame.K_RETURN:
                    if user_input:
                        chat_history.append(f"You: {user_input}")
                        llm_output = cat_personality.interpret_command(user_input)
                        cat_response = llm_output.get("response", "...")
                        action_data = llm_output.get("action")
                        chat_history.append(f"Cat: {cat_response}")
                        
                        if action_data and action_data.get('type') == 'move':
                            target_x = action_data.get('x', env.food_rect.x)
                            target_y = action_data.get('y', env.food_rect.y)
                            env.food_rect.x = target_x
                            env.food_rect.y = target_y
                            print(f"New target set by LLM: ({target_x}, {target_y})")
                        user_input = "" 
                elif event.key == pygame.K_BACKSPACE:
                    user_input = user_input[:-1]
                else:
                    user_input += event.unicode

        action, _states = model.predict(obs, deterministic=True)
        obs, reward, done, truncated, info = env.step(action)

        if done:
            obs, info = env.reset() 
            chat_history.append("Cat: Reached my destination! Purrr.")

        # --- Rendering ---
        env.render()
        
        # Get screen dimensions for UI drawing
        screen_width, screen_height = env.screen.get_size()
        
        ui_panel_height = INPUT_BOX_HEIGHT + CHAT_LOG_HEIGHT
        ui_panel_rect = pygame.Rect(0, screen_height - ui_panel_height, screen_width, ui_panel_height)
        
        # Use a separate surface for transparency
        ui_panel = pygame.Surface((screen_width, ui_panel_height))
        ui_panel.set_alpha(200) 
        ui_panel.fill(GRAY)
        env.screen.blit(ui_panel, ui_panel_rect)

        for i, line in enumerate(reversed(chat_history[-4:])): 
            text_surface = small_font.render(line, True, BLACK)
            env.screen.blit(text_surface, (10, screen_height - ui_panel_height + 5 + i * 20))

        input_box_rect = pygame.Rect(0, screen_height - INPUT_BOX_HEIGHT, screen_width, INPUT_BOX_HEIGHT)
        pygame.draw.rect(env.screen, WHITE, input_box_rect)
        pygame.draw.rect(env.screen, BLACK, input_box_rect, 2)
        input_surface = font.render(f"> {user_input}", True, BLACK)
        env.screen.blit(input_surface, (input_box_rect.x + 10, input_box_rect.y + 10))
        
        pygame.display.flip()
        
        clock.tick(FPS)

    env.close()

if __name__ == "__main__":
    main()