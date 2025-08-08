from f1_gym_env import F1RacingEnv
import time

env = F1RacingEnv()
obs, info = env.reset()
done = False
total_reward = 0.0

while not done:
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)
    total_reward += reward
    done = terminated or truncated
    time.sleep(0.01)

print(f"Episode finished with total reward: {total_reward}")
