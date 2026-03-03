import os, sys, imageio
import numpy as np

repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

from metaworld_envs import MetaworldDense
from metaworld.policies.sawyer_drawer_open_v2_policy import SawyerDrawerOpenV2Policy

def main(env_id="drawer-open-v2-goal-hidden", seed=0, max_steps=128,
         outfile="gifs/custom/drawer-open-robot/1.gif", rank=0):

    env = MetaworldDense(env_id=env_id, time=False, rank=rank)  # time flag as you like
    env.seed(seed)

    policy = SawyerDrawerOpenV2Policy()

    frames = []
    obs = env.reset()

    for t in range(max_steps):
        action = policy.get_action(obs)
        obs, rew, done, info = env.step(action)
        frames.append(env.unwrapped.render())   # uses your working render path
        if done:
            break

    imageio.mimsave(outfile, frames, fps=30)
    print(f"Saved GIF to {outfile} ({len(frames)} frames)")

if __name__ == "__main__":
    main()