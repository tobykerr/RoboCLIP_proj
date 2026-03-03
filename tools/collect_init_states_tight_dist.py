# tools/collect_init_states.py
"""
Collect a *tight* distribution of reach-end states for relay-style grasp training.

Compared to your current version, this script:
  1) Uses deterministic policy actions (mu(s)) to avoid wide wandering
  2) Injects small Gaussian action noise (controlled)
  3) Optionally jitters the handoff timestep within a small window
  4) Optionally filters out obvious outliers by keeping only the best fraction
     according to a simple proxy: distance in qpos-space to the median state
     (works without needing handle/EEF positions).

Usage example:
python tools/collect_init_states.py \
  --model metaworld/drawer-open-v2-goal-hidden_max_reach_max25/best_model.zip \
  --env-id drawer-open-v2-goal-hidden \
  --out results/subtasks_max25/grasp/init_states_tight.pkl \
  --n-episodes 2000 \
  --max-steps 128 \
  --handoff-step 127 \
  --handoff-jitter 5 \
  --action-noise-std 0.01 \
  --keep-frac 0.9
"""

import os
import sys
import pickle
import argparse
import numpy as np
from stable_baselines3 import PPO

repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

from metaworld_envs import MetaworldDense


def unwrap_env(e, max_depth=20):
    cur = e
    for _ in range(max_depth):
        if hasattr(cur, "sim") or (hasattr(cur, "model") and hasattr(cur, "data")):
            return cur
        if hasattr(cur, "env"):
            cur = cur.env
        else:
            return cur
    return cur


def get_qpos_from_mj_state(mj_state):
    # mujoco_py MjSimState has .qpos
    if hasattr(mj_state, "qpos"):
        return mj_state.qpos.copy()
    # fallback if you ever save dict-based state
    if isinstance(mj_state, dict) and "qpos" in mj_state:
        return np.array(mj_state["qpos"]).copy()
    raise TypeError(f"Unrecognized mj_state type for qpos extraction: {type(mj_state)}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--env-id", type=str, required=True)
    parser.add_argument("--out", type=str, required=True)

    parser.add_argument("--n-episodes", type=int, default=1000)
    parser.add_argument("--max-steps", type=int, default=128)
    parser.add_argument("--handoff-step", type=int, default=127)
    parser.add_argument("--handoff-jitter", type=int, default=0,
                        help="If >0, actual snapshot step is sampled uniformly from "
                             "[handoff_step-handoff_jitter, handoff_step].")
    parser.add_argument("--action-noise-std", type=float, default=0.01,
                        help="Gaussian noise std added to actions while collecting.")
    parser.add_argument("--rank", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)

    parser.add_argument("--keep-frac", type=float, default=1.0,
                        help="Keep only this fraction of states after outlier filtering "
                             "(1.0 disables filtering). Typical: 0.8-0.95")
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)

    env = MetaworldDense(env_id=args.env_id, time=True, rank=args.rank)
    model = PPO.load(args.model)

    base_env = unwrap_env(env)
    if not hasattr(base_env, "sim"):
        raise AttributeError(f"Base env after unwrapping has no `.sim`: {type(base_env)}")

    saved = []
    qpos_saved = []

    meta = {
        "env_id": args.env_id,
        "max_steps": args.max_steps,
        "n_episodes": args.n_episodes,
        "handoff_step_nominal": args.handoff_step,
        "handoff_jitter": args.handoff_jitter,
        "action_noise_std": args.action_noise_std,
        "collection_policy_deterministic": True,
        "seed": args.seed,
        "rank": args.rank,
    }

    for ep in range(args.n_episodes):
        obs = env.reset()

        # Choose a per-episode snapshot step (tight jitter window near the end)
        if args.handoff_jitter > 0:
            snap_t = int(rng.integers(low=max(0, args.handoff_step - args.handoff_jitter),
                                      high=args.handoff_step + 1))
        else:
            snap_t = args.handoff_step

        for t in range(args.max_steps):
            # Deterministic action + controlled noise
            action, _ = model.predict(obs, deterministic=True)
            if args.action_noise_std > 0:
                action = action + rng.normal(0.0, args.action_noise_std, size=action.shape)

            # Clip to action bounds
            action = np.clip(action, env.action_space.low, env.action_space.high)

            obs, reward, done, info = env.step(action)

            if t == snap_t:
                sim = unwrap_env(env).sim
                mj_state = sim.get_state()

                extra = {}
                uw = env.unwrapped
                for k in ["_target_pos", "_goal", "_last_rand_vec", "obj_init_pos"]:
                    if hasattr(uw, k):
                        extra[k] = getattr(uw, k)

                saved.append({"mj_state": mj_state, "extra": extra, "t": t})
                qpos_saved.append(get_qpos_from_mj_state(mj_state))
                break

            if done:
                # If episode ends early, do NOT save here (keeps the distribution "near handoff")
                break

        if (ep + 1) % 50 == 0:
            print(f"Collected {len(saved)} states / {ep+1} episodes")

    if len(saved) == 0:
        raise RuntimeError("Collected 0 states. Increase max_steps, reduce handoff_step, or check done early.")

    # Optional outlier filtering in qpos space (proxy if you don't have EEF/handle distance)
    keep_frac = float(args.keep_frac)
    if keep_frac < 1.0:
        qpos_arr = np.stack(qpos_saved, axis=0)
        med = np.median(qpos_arr, axis=0)
        dists = np.linalg.norm(qpos_arr - med[None, :], axis=1)

        k = max(1, int(np.floor(keep_frac * len(saved))))
        keep_idx = np.argsort(dists)[:k]

        saved = [saved[i] for i in keep_idx]
        meta["filtered_keep_frac"] = keep_frac
        meta["filtered_kept"] = len(saved)
        meta["filtered_total_before"] = len(dists)
        meta["filtered_dist_mean"] = float(np.mean(dists))
        meta["filtered_dist_max"] = float(np.max(dists))

        print(f"Filtered: kept {len(saved)} / {len(dists)} states (keep_frac={keep_frac})")

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "wb") as f:
        pickle.dump({"meta": meta, "states": saved}, f)

    print(f"Saved {len(saved)} grasp-init states to {args.out}")


if __name__ == "__main__":
    main()