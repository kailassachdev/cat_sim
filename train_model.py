import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
import pygame
import numpy as np

from cat_brain.rl_model import CatEnv

class CurriculumCallback(BaseCallback):
    def __init__(self, verbose=0):
        super(CurriculumCallback, self).__init__(verbose)

    def _on_rollout_end(self) -> None:
        """Check for success at the end of each rollout."""
        # --- MODIFICATION: New, correct logic to check for success ---
        # Check if the max reward in the buffer is the success reward
        if np.max(self.model.rollout_buffer.rewards) >= 1000:
            env = self.training_env.envs[0]
            if env.complexity < 1.0:
                env.increase_complexity()
                self.training_env.reset()
        # --- END MODIFICATION ---

    def _on_step(self) -> bool:
        # This part is just for rendering
        self.training_env.envs[0].render()
        pygame.display.flip()
        pygame.event.pump()
        return True

def make_env():
    # Start with the easiest maze (complexity 0.1)
    return CatEnv(render_mode="human", initial_complexity=0.1)

env = DummyVecEnv([make_env])
env = VecNormalize(env, norm_obs=True, norm_reward=False, clip_obs=10.)

learning_rate = 0.0001
policy_kwargs = dict(net_arch=dict(pi=[512, 512], vf=[512, 512]))
gamma = 0.999

model = PPO("MlpPolicy", env, verbose=1, learning_rate=learning_rate,
            policy_kwargs=policy_kwargs, gamma=gamma)

curriculum_callback = CurriculumCallback()

print("Starting curriculum training without timeouts...")
model.learn(total_timesteps=2_000_000, callback=curriculum_callback)

model.save("cat_brain/ppo_cat_model_curriculum_notimeout")
env.save("cat_brain/vec_normalize_curriculum_notimeout.pkl")
print("Training complete! Curriculum-trained model saved.")
env.close()