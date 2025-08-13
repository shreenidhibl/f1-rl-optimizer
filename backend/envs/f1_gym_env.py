import numpy as np
import gymnasium as gym
from gymnasium import spaces

class F1RacingEnv(gym.Env):
    """
    Formula 1 RL environment for Stable Baselines3 (SAC).
    Synthetic steering handling, bounded observation space,
    and tuned reward for smoother steering.
    """

    metadata = {"render_modes": ["human"], "render_fps": 60}

    def __init__(self, telemetry):
        super().__init__()

        required_cols = ['Speed', 'Throttle', 'Brake', 'X', 'Y']
        missing = [c for c in required_cols if c not in telemetry.columns]
        if missing:
            raise ValueError(f"Missing required telemetry columns: {missing}")

        self.telemetry = telemetry.reset_index(drop=True)
        self.current_step = 0

        if 'Steering' not in self.telemetry.columns:
            print("[WARN] 'Steering' missing — estimating from X/Y data.")
            self.telemetry['Steering'] = self._estimate_steering_from_xy()

        # Action space: Throttle [0,1], Steering [-1,1]
        self.action_space = spaces.Box(
            low=np.array([0.0, -1.0], dtype=np.float32),
            high=np.array([1.0, 1.0], dtype=np.float32),
            dtype=np.float32,
        )

        # Observation space: Speed, Throttle, Brake, Gear, Steering
        self.observation_space = spaces.Box(
            low=np.array([0., 0., 0., 0., -1.], dtype=np.float32),
            high=np.array([400., 1., 1., 8., 1.], dtype=np.float32),
            dtype=np.float32,
        )

        self.prev_throttle = 0.0
        self.prev_steering = 0.0
        self.MAX_SPEED = 400.0

    def _estimate_steering_from_xy(self):
        dx = self.telemetry['X'].diff().fillna(0)
        dy = self.telemetry['Y'].diff().fillna(0)
        headings = np.arctan2(dy, dx)
        d_heading = np.unwrap(np.diff(headings, prepend=headings.iloc[0]))
        max_change = np.max(np.abs(d_heading)) or 1.0
        steering_norm = (d_heading / max_change).clip(-1, 1)
        return steering_norm.astype(np.float32)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.current_step = 0
        self.prev_throttle = 0.0
        self.prev_steering = 0.0
        return self._get_obs(), {}

    def _get_obs(self):
        row = self.telemetry.iloc[self.current_step]
        gear = float(row['nGear']) if 'nGear' in self.telemetry.columns else 0.0
        obs = np.array([
            row['Speed'],
            row['Throttle'],
            row['Brake'],
            gear,
            row['Steering']
        ], dtype=np.float32)
        return np.clip(obs, self.observation_space.low, self.observation_space.high)

    def step(self, action):
        throttle = np.clip(action[0], 0.0, 1.0)
        steering = np.clip(action[1], -1.0, 1.0)
        row = self.telemetry.iloc[self.current_step]
        lap_progress = self.current_step / (len(self.telemetry) - 1)

        reward = self.compute_reward(
            row['Speed'], throttle, row['Brake'], steering,
            self.prev_throttle, self.prev_steering, lap_progress
        )

        self.prev_throttle = throttle
        self.prev_steering = steering
        self.current_step += 1

        terminated = self.current_step >= len(self.telemetry) - 1
        truncated = False
        obs = self._get_obs() if not terminated else None

        return obs, reward, terminated, truncated, {}

    def compute_reward(self, speed, throttle, brake, steering,
                       prev_throttle, prev_steering, lap_progress):
        reward = 0.0
        # Encourage lap progress
        reward += lap_progress * 10.0
        # Speed reward normalized
        reward += np.clip(speed / self.MAX_SPEED, 0, 1) * 5.0
        # Penalize abrupt throttle changes to encourage smoothness
        reward -= abs(throttle - prev_throttle) * 1.5
        # Penalize abrupt steering changes more to smooth cornering
        reward -= abs(steering - prev_steering) * 3.0
        # Penalize large absolute steering to discourage excessive angle
        reward -= abs(steering) * 1.0
        # Penalize braking to favor smooth driving
        reward -= brake * 5.0
        return reward

    def render(self):
        pass
