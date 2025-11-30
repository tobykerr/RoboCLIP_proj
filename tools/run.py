import gym
import os
import sys
import imageio
import numpy as np
from stable_baselines3 import PPO
import argparse

repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

from metaworld_envs import MetaworldDense

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True,
                        help="Path to best_model.zip")
    parser.add_argument("--env-id", type=str, default="door-open-v2-goal-hidden",
                        help="MetaWorld environment ID")
    parser.add_argument("--max-steps", type=int, default=500,
                        help="Max episode length")
    parser.add_argument("--outfile", type=str, default="rollout.gif",
                        help="Output GIF filename")
    parser.add_argument("--rank", type=int, default=0,
                        help="Seed/rank passed to wrapper")
    args = parser.parse_args()

    # --- Create environment matching training observation format ---
    env = MetaworldDense(env_id=args.env_id, time=True, rank=args.rank)

    # --- Load trained model ---
    model = PPO.load(args.model)

    frames = []
    obs = env.reset()

    for _ in range(args.max_steps):
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, info = env.step(action)

        frame = env.render()        # (H,W,3) RGB image
        frames.append(frame)

        if done:
            break

    # --- Save GIF ---
    imageio.mimsave(args.outfile, frames, fps=30)
    print(f"Saved GIF to: {args.outfile}")


if __name__ == "__main__":
    main()
