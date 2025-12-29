import os
import imageio
import numpy as np

import metaworld
from metaworld.envs import ALL_V2_ENVIRONMENTS_GOAL_HIDDEN

# import the policy class (adjust import path to where your file lives)
from metaworld.policies.sawyer_drawer_open_v2_policy import SawyerDrawerOpenV2Policy

def main(
    env_id="drawer-open-v2-goal-hidden",
    seed=0,
    max_steps=128,
    outfile="gifs/custom/drawer-open-expert/1.gif",
    render_mode="rgb_array",
):
    env_cls = ALL_V2_ENVIRONMENTS_GOAL_HIDDEN[env_id]
    env = env_cls(seed=seed)

    # If your MetaWorld version supports it, prefer rgb_array:
    # Some versions use env.render() w/out args; others need mode="rgb_array".
    policy = SawyerDrawerOpenV2Policy()

    frames = []
    obs = env.reset()
    if isinstance(obs, tuple):  # gymnasium compat
        obs, _ = obs

    for t in range(max_steps):
        action = policy.get_action(obs)          # shape (4,)
        step_out = env.step(action)

        if len(step_out) == 4:
            obs, rew, done, info = step_out
        else:
            obs, rew, terminated, truncated, info = step_out
            done = terminated or truncated

        # --- render frame ---
        frame = None
        try:
            frame = env.render(mode=render_mode)
        except TypeError:
            frame = env.render()
        if frame is not None:
            frames.append(frame)

        if done:
            break

    if frames:
        imageio.mimsave(outfile, frames, fps=30)
        print(f"Saved GIF to {outfile} ({len(frames)} frames)")
    else:
        print("No frames captured (render returned None).")

if __name__ == "__main__":
    main()
