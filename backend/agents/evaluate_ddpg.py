import fastf1
import numpy as np
from stable_baselines3 import DDPG
from backend.envs.f1_gym_env import F1RacingEnv
import matplotlib.pyplot as plt

def main():
    # Enable FastF1 cache for fast telemetry loads
    fastf1.Cache.enable_cache('./cache')

    session = fastf1.get_session(2023, 'Monza', 'Q')
    session.load()

    lap = session.laps.pick_fastest()
    telemetry = lap.get_telemetry()

    env = F1RacingEnv(telemetry)

    model = DDPG.load("ddpg_f1racing_telemetry_steering", env=env)

    obs, info = env.reset()
    done = False

    rewards = []
    actions = []
    speeds = []
    throttles = []
    brakes = []
    gears = []
    steerings = []

    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

        if obs is not None:
            rewards.append(reward)
            actions.append(action)
            speeds.append(obs[0])
            throttles.append(obs[1])
            brakes.append(obs[2])
            gears.append(obs[3])
            steerings.append(obs[4])

    total_reward = np.sum(rewards)
    print(f"Evaluation complete. Total reward: {total_reward:.2f}")

    timesteps = np.arange(len(rewards))

    plt.figure(figsize=(14, 10))

    plt.subplot(4, 1, 1)
    plt.plot(timesteps, speeds, label='Speed (km/h)')
    plt.ylabel('Speed')
    plt.legend()

    plt.subplot(4, 1, 2)
    plt.plot(timesteps, throttles, label='Throttle')
    plt.plot(timesteps, brakes, label='Brake')
    plt.ylabel('Controls')
    plt.legend()

    plt.subplot(4, 1, 3)
    plt.plot(timesteps, steerings, label='Steering')
    plt.ylabel('Steering')
    plt.legend()

    plt.subplot(4, 1, 4)
    actions = np.array(actions)
    plt.plot(timesteps, actions[:, 0], label='Agent Throttle Action')
    plt.plot(timesteps, actions[:, 1], label='Agent Steering Action')
    plt.xlabel('Timestep')
    plt.ylabel('Action Value')
    plt.legend()

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
