import gym
import os
import sys
import imageio
import numpy as np
from stable_baselines3 import PPO
import argparse
import pickle

repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

from metaworld_envs import MetaworldDense


def unwrap_to_base_with_sim(e, max_depth=30):
    cur = e
    for _ in range(max_depth):
        if hasattr(cur, "sim"):
            return cur
        if hasattr(cur, "env"):
            cur = cur.env
        else:
            break
    return cur

def unwrap_to_env_with_get_obs(e, max_depth=30):
    cur = e
    for _ in range(max_depth):
        if hasattr(cur, "_get_obs"):
            return cur
        if hasattr(cur, "env"):
            cur = cur.env
        else:
            break
    return None


class InitialStateResetWrapper(gym.Wrapper):
    def __init__(self, env, init_states_path: str, seed: int = 0):
        super().__init__(env)
        self.rng = np.random.default_rng(seed)
        with open(init_states_path, "rb") as f:
            payload = pickle.load(f)
        self.states = payload.get("states", [])
        if len(self.states) == 0:
            raise ValueError(f"No states in {init_states_path}")

    def _get_base_obs(self):
        base = unwrap_to_env_with_get_obs(self.env)
        if base is None:
            raise AttributeError("Could not find env with _get_obs()")
        return base._get_obs()

    def reset(self, **kwargs):
        obs = self.env.reset(**kwargs)

        # reset time counter if your wrapper uses it
        if hasattr(self.env, "counter"):
            self.env.counter = 0

        sample = self.states[self.rng.integers(0, len(self.states))]
        mj_state = sample["mj_state"]

        base = unwrap_to_base_with_sim(self.env)
        sim = base.sim
        sim.set_state(mj_state)
        sim.forward()

        base_obs = self._get_base_obs()

        # append time channel if needed
        if isinstance(obs, np.ndarray) and obs.shape[-1] == base_obs.shape[-1] + 1:
            return np.concatenate([base_obs, np.array([0.0], dtype=base_obs.dtype)])
        return base_obs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--env-id", type=str, default="drawer-open-v2-goal-hidden")
    parser.add_argument("--max-steps", type=int, default=128)
    parser.add_argument("--outfile", type=str, default="rollout.gif")
    parser.add_argument("--rank", type=int, default=0)

    # NEW:
    parser.add_argument("--init-states", type=str, default=None,
                        help="Path to init_states .pkl; if set, start each rollout from a sampled saved state.")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    # --- Create environment matching training observation format ---
    env = MetaworldDense(env_id=args.env_id, time=True, rank=args.rank)

    if args.init_states is not None:
        env = InitialStateResetWrapper(env, init_states_path=args.init_states, seed=args.seed)

    model = PPO.load(args.model)

    frames = []
    obs = env.reset()

    for _ in range(args.max_steps):
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, info = env.step(action)
        frames.append(env.unwrapped.render())
        if done:
            break

    # --- Save GIF ---
    imageio.mimsave(args.outfile, frames, fps=30)
    print(f"Saved GIF to: {args.outfile}")


if __name__ == "__main__":
    main()
