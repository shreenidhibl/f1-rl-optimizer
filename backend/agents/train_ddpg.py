import fastf1
import numpy as np
import torch
from stable_baselines3 import DDPG
from stable_baselines3.common.noise import NormalActionNoise
from backend.envs.f1_gym_env import F1RacingEnv

def main():
    # Enable FastF1 cache for telemetry speedup
    fastf1.Cache.enable_cache('./cache')

    # Load qualifying session telemetry (make sure 'Steering' column is present)
    session = fastf1.get_session(2023, 'Monza', 'Q')
    session.load()

    lap = session.laps.pick_fastest()
    telemetry = lap.get_telemetry()

    if 'Steering' not in telemetry.columns:
        print("Warning: 'Steering' column missing in telemetry. Results may be affected.")

    env = F1RacingEnv(telemetry)

    n_actions = env.action_space.shape[-1]  # Should be 2 for throttle + steering

    # Exploration noise with moderate scale
    action_noise = NormalActionNoise(mean=np.zeros(n_actions), sigma=0.1 * np.ones(n_actions))

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Training device: {device}")

    # Smaller network for faster training, you can tune this as needed
    policy_kwargs = dict(net_arch=[64, 64])

    model = DDPG(
        "MlpPolicy",
        env,
        action_noise=action_noise,
        verbose=1,
        device=device,
        policy_kwargs=policy_kwargs,
        tensorboard_log="./tensorboard_ddpg"
    )

    print("Starting training...")
    model.learn(total_timesteps=200000)
    print("Training finished.")

    model.save("ddpg_f1racing_telemetry_steering")
    print("Model saved as 'ddpg_f1racing_telemetry_steering'.")

if __name__ == "__main__":
    main()
