import fastf1
import torch
from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import DummyVecEnv
from backend.envs.f1_gym_env import F1RacingEnv


def main(year=2023, track='Monza', session_name='R', lap_number=None, timesteps=200_000):
    fastf1.Cache.enable_cache('./cache')

    print(f"Loading session: {year} {track} {session_name}")
    session = fastf1.get_session(year, track, session_name)
    session.load()

    if lap_number is not None:
        lap = session.laps.pick(lap_number)
    else:
        lap = session.laps.pick_fastest()

    telemetry = lap.get_telemetry()
    print("Telemetry columns:", telemetry.columns)

    env = DummyVecEnv([lambda: F1RacingEnv(telemetry)])

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    model = SAC(
        "MlpPolicy",
        env,
        verbose=1,
        device=device,
        tensorboard_log="./tensorboard_sac",
        policy_kwargs=dict(net_arch=[64, 64])
    )

    print(f"Training SAC model for {timesteps} timesteps...")
    model.learn(total_timesteps=timesteps)
    print("Training complete.")

    model.save("sac_f1racing_telemetry_steering_refined")
    print("Model saved as 'sac_f1racing_telemetry_steering_refined'")


if __name__ == "__main__":
    main()
