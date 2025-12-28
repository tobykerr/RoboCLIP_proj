#### leaving this in in case in future we want to debug DVD reward specifically in the Tabletop sim env
#### BUT NOTE: this is not used in main experiments, see debug_dvd_task_separation.py instead
#### AND NOTE: I couldnt get this to run due to various path / import issues with DVD repo structure, it looks like a total mess
#### Main issue is that DVD uses a different version of metaworld than roboclip / my main codebase, so Tabletop envs have completely different dependencies
#### So this is mostly for reference.

import os, sys
import csv
import numpy as np

sys.path.insert(0, os.path.abspath("third_party/dvd"))         # for multi_column, model3D_1, etc.
sys.path.insert(0, os.path.abspath("third_party/dvd/sim_envs"))# for sim_env.*

from rewards.dvd_reward import DVDReward, DVDConfig

# IMPORTANT: Tabletop is imported from the DVD repo path at runtime by DVDReward already,
# but we import it explicitly here too.
import sys
sys.path.insert(0, os.path.abspath("third_party/dvd"))
from sim_env.tabletop import Tabletop


# ---------------------------
# Config you will edit
# ---------------------------
XML_ENV = "env1"  # env1..env4 etc (the xml name)
DEMO_GIF = "third_party/dvd/demos/env1/task5_gifs/82.gif"  # replace with the demo you want
OUT_CSV = "logs/simenv_env1_dvd.csv"

N_RANDOM = 40
N_GENTLE = 10
N_THRASH = 10
MAX_STEPS = 50  # Tabletop default max_path_length is 50 unless you set it


# ---------------------------
# Helpers
# ---------------------------
def make_env(xml=XML_ENV, max_steps=MAX_STEPS, verbose=0):
    env = Tabletop(xml=xml, verbose=verbose, max_path_length=max_steps, log_freq=999999, filepath="logs/tmp")
    env.initialize()
    return env


def reset_env(env):
    obs, info = env.reset_model()
    return obs, info


def step_env(env, action):
    obs, r, done, info = env.step(action)
    return obs, done, info


def obs_to_uint8_frame(obs):
    """
    Tabletop.get_obs() returns float image in [0,1] with shape (H,W,3).
    Convert to uint8 RGB.
    """
    frame = (np.clip(obs, 0.0, 1.0) * 255.0).astype(np.uint8)
    return frame


def rollout(env, policy_fn, max_steps=MAX_STEPS):
    frames = []
    obs, info = reset_env(env)

    drawer_init = float(info.get("drawer", np.nan))
    drawer_final = drawer_init

    for t in range(max_steps):
        frames.append(obs_to_uint8_frame(obs))
        action = policy_fn(env)
        obs, done, info = step_env(env, action)
        drawer_final = float(info.get("drawer", drawer_final))
        if done:
            frames.append(obs_to_uint8_frame(obs))
            break

    return frames, drawer_init, drawer_final


def gentle_policy(env):
    # small random actions
    a = env.action_space.sample().astype(np.float32)
    return (0.1 * a).astype(np.float32)


def thrash_policy(env):
    # big random actions (clipped)
    a = env.action_space.sample().astype(np.float32)
    return np.clip(3.0 * a, env.action_space.low, env.action_space.high).astype(np.float32)


def random_policy(env):
    return env.action_space.sample().astype(np.float32)


def main():
    os.makedirs("logs", exist_ok=True)

    # DVD reward model (pretrained)
    dvd = DVDReward(
        dvd_repo_root="third_party/dvd",
        sim_discriminator_ckpt="third_party/dvd/pretrained/dvd_human_tasks_6_robot_tasks_3.pth.tar",
        video_encoder_ckpt="third_party/dvd/pretrained/video_encoder/model_best.pth.tar",
        demo_gif_path=DEMO_GIF,
        cfg=DVDConfig(
            clip_len=20,
            resize_h=120,
            resize_w=120,
            reward_mode="logit_diff",
            reward_scale=1.0,
        ),
    )

    env = make_env()

    rows = []

    def run_block(tag, n, pol):
        for k in range(n):
            frames, d0, d1 = rollout(env, pol)
            r = float(dvd.terminal_reward(frames))
            rows.append({
                "type": f"{tag}_{k}",
                "reward": r,
                "drawer_init": d0,
                "drawer_final": d1,
                "drawer_delta": d1 - d0,
                "n_frames": len(frames),
            })
            print(f"{tag}_{k:02d} | reward={r:.4f} | drawer_init={d0:.4f} drawer_final={d1:.4f} delta={d1-d0:.4f}")

    print("\n--- gentle ---")
    run_block("gentle", N_GENTLE, gentle_policy)

    print("\n--- thrash ---")
    run_block("thrash", N_THRASH, thrash_policy)

    print("\n--- random ---")
    run_block("random", N_RANDOM, random_policy)

    # "success vs fail" selection based on drawer_final
    # (since env reward is always 0.0)
    rows_sorted = sorted(rows, key=lambda x: x["drawer_final"])
    k = max(3, min(10, len(rows)//5))  # top/bottom ~20%, capped
    fail = rows_sorted[:k]
    succ = rows_sorted[-k:]

    fail_mean = float(np.mean([x["reward"] for x in fail]))
    succ_mean = float(np.mean([x["reward"] for x in succ]))
    print("\n=== SUCCESS/FAIL PROXY (by drawer_final) ===")
    print(f"Bottom-{k} drawer_final mean reward: {fail_mean:.4f}")
    print(f"Top-{k}    drawer_final mean reward: {succ_mean:.4f}")

    # also evaluate demo-vs-demo as an upper reference
    demo_frames = dvd._load_gif_clip(DEMO_GIF)  # returns torch clip, but we can just compute directly
    # easiest: just call terminal_reward on the gif path frames (DVDReward uses its own cached embedding)
    import imageio.v2 as imageio
    demo_list = [np.asarray(f)[..., :3].copy() for f in imageio.mimread(DEMO_GIF)]
    demo_r = float(dvd.terminal_reward(demo_list))
    print(f"Demo episode reward (gif vs itself-ish): {demo_r:.4f}")

    # write CSV
    with open(OUT_CSV, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print("\nWrote:", OUT_CSV)


if __name__ == "__main__":
    main()
