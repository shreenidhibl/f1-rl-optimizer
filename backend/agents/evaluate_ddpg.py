import gymnasium as gym
from stable_baselines3 import DDPG
from backend.envs.f1_gym_env import F1RacingEnv

def evaluate():
    env = F1RacingEnv()
    model = DDPG.load("ddpg_f1_racing")  # Load your saved model

    obs, info = env.reset()
    done = False
    total_reward = 0.0

    while not done:
        action, _states = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        total_reward += reward
        env.render()

    print(f"Evaluation finished with total reward: {total_reward}")

if __name__ == "__main__":
    evaluate()
