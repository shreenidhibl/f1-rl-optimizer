import gymnasium as gym
from stable_baselines3 import DDPG
from stable_baselines3.common.noise import NormalActionNoise
import numpy as np
from backend.envs.f1_gym_env import F1RacingEnv

def train_ddpg():
    # Initialize environment
    env = F1RacingEnv()

    # The Action space is continuous; setup action noise for exploration
    n_actions = env.action_space.shape[-1]
    action_noise = NormalActionNoise(mean=np.zeros(n_actions), sigma=0.1 * np.ones(n_actions))

    # Initialize DDPG model with MLP policy
    model = DDPG("MlpPolicy", env, action_noise=action_noise, verbose=1)

    # Train the model
    model.learn(total_timesteps=10000)  # Increase timesteps as you want

    # Save the trained model
    model.save("ddpg_f1_racing")

    print("Training complete, model saved as 'ddpg_f1_racing'.")

if __name__ == "__main__":
    train_ddpg()
