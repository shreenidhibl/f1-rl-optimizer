import fastf1
import torch
import numpy as np
import matplotlib.pyplot as plt
from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import DummyVecEnv
from backend.envs.f1_gym_env import F1RacingEnv


def plot_path(x, y, color_vals, title, cmap_name='viridis'):
    plt.figure(figsize=(14, 8))
    scatter = plt.scatter(x, y, c=color_vals, cmap=cmap_name, s=10)
    plt.colorbar(scatter)
    plt.xlabel("X position")
    plt.ylabel("Y position")
    plt.title(title)
    plt.axis('equal')
    plt.show()


def main():
    fastf1.Cache.enable_cache('./cache')

    # Load Monza Race 2023 session telemetry
    session = fastf1.get_session(2023, 'Monza', 'R')
    session.load()

    # Select the lap for agent evaluation (fastest lap for example)
    agent_lap = session.laps.pick_fastest()
    telemetry = agent_lap.get_telemetry()
    print("Agent telemetry columns:", telemetry.columns)

    # Create and wrap the environment with DummyVecEnv
    env = DummyVecEnv([lambda: F1RacingEnv(telemetry)])
    print("Wrapped env observation space:", env.observation_space)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Load the trained SAC model
    model = SAC.load("sac_f1racing_telemetry_steering_refined", env=env, device=device)

    obs = env.reset()
    done = False
    actions = []

    # Run the agent and collect actions
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        action = np.array(action).squeeze()
        actions.append(action)
        obs, reward, done, info = env.step([action])

    actions = np.array(actions)

    # Plot agent throttle and steering over time
    plt.figure(figsize=(12, 6))
    plt.plot(actions[:, 0], label="Throttle")
    plt.plot(actions[:, 1], label="Steering")
    plt.xlabel("Timestep")
    plt.ylabel("Action Value")
    plt.title("SAC Agent Actions Over Time")
    plt.legend()
    plt.show()

    # Visualize agent path colored by throttle
    plot_path(
        telemetry["X"],
        telemetry["Y"],
        actions[:, 0],
        "Agent Telemetry Path Colored by Throttle",
        cmap_name="Blues"
    )

    # Visualize agent path colored by steering
    plot_path(
        telemetry["X"],
        telemetry["Y"],
        actions[:, 1],
        "Agent Telemetry Path Colored by Steering",
        cmap_name="RdYlGn"
    )

    # Load baseline/pro lap telemetry for comparison (you can adjust lap selection)
    baseline_lap = session.laps.pick_fastest()
    baseline_telemetry = baseline_lap.get_telemetry()

    # Estimate missing steering in baseline lap
    if "Steering" not in baseline_telemetry.columns:
        dx = baseline_telemetry["X"].diff().fillna(0)
        dy = baseline_telemetry["Y"].diff().fillna(0)
        headings = np.arctan2(dy, dx)
        d_heading = np.unwrap(np.diff(headings, prepend=headings.iloc[0]))
        max_change = np.max(np.abs(d_heading)) or 1.0
        baseline_telemetry["Steering"] = (d_heading / max_change).clip(-1, 1)

    # Plot baseline lap steering on its own map
    plot_path(
        baseline_telemetry["X"],
        baseline_telemetry["Y"],
        baseline_telemetry["Steering"],
        "Baseline Human/Pro Lap Steering",
        cmap_name="RdYlGn"
    )

    # Direct comparison of agent & baseline steering on the same plot
    plt.figure(figsize=(14, 8))
    plt.scatter(
        telemetry["X"],
        telemetry["Y"],
        c=actions[:, 1],
        cmap="RdYlGn",
        s=15,
        label="Agent Steering",
        alpha=0.7
    )
    plt.scatter(
        baseline_telemetry["X"],
        baseline_telemetry["Y"],
        c=baseline_telemetry["Steering"],
        cmap="coolwarm",
        s=8,
        label="Baseline Steering",
        alpha=0.7
    )
    plt.colorbar(label="Steering Value")
    plt.xlabel("X position")
    plt.ylabel("Y position")
    plt.title("Agent vs Baseline Steering Comparison")
    plt.legend()
    plt.axis('equal')
    plt.show()


if __name__ == "__main__":
    main()
