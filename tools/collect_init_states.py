import os
import sys
import pickle
import numpy as np
from stable_baselines3 import PPO

repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

from metaworld_envs import MetaworldDense


def unwrap_env(e, max_depth=20):
    cur = e
    for i in range(max_depth):
        if hasattr(cur, "sim") or (hasattr(cur, "model") and hasattr(cur, "data")):
            return cur
        if hasattr(cur, "env"):
            cur = cur.env
        else:
            return cur
    return cur


def collect_states(
    model_path: str,
    env_id: str,
    out_path: str,
    n_episodes: int = 500,
    max_steps: int = 128,
    handoff_step: int = 120,
    deterministic: bool = False,
    rank: int = 0,
):
    env = MetaworldDense(env_id=env_id, time=True, rank=rank)
    model = PPO.load(model_path)

    saved = []
    meta = {
        "env_id": env_id,
        "handoff_step": handoff_step,
        "max_steps": max_steps,
        "n_episodes": n_episodes,
    }

    for ep in range(n_episodes):
        obs = env.reset()

        for t in range(max_steps):
            action, _ = model.predict(obs, deterministic=deterministic)
            obs, reward, done, info = env.step(action)

            # Choose when to snapshot the handoff state
            if t == handoff_step:
                # sim = env.unwrapped.sim
                base_env = unwrap_env(env)

                sim = base_env.sim
                mj_state = sim.get_state()

                # Optional: store any extra env fields you might need to restore
                # (depends on your wrapper; keep these if they exist)
                extra = {}
                uw = env.unwrapped
                for k in ["_target_pos", "_goal", "_last_rand_vec", "obj_init_pos"]:
                    if hasattr(uw, k):
                        extra[k] = getattr(uw, k)

                saved.append({"mj_state": mj_state, "extra": extra})
                break

            if done:
                # If episode ends early, you can skip or store the final state instead
                break

        if (ep + 1) % 50 == 0:
            print(f"Collected {len(saved)} states / {ep+1} episodes")

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "wb") as f:
        pickle.dump({"meta": meta, "states": saved}, f)

    print(f"Saved {len(saved)} grasp-init states to {out_path}")

if __name__ == "__main__":
    collect_states(
        model_path="metaworld/drawer-open-v2-goal-hidden_max_reach_max25/best_model.zip",
        env_id="drawer-open-v2-goal-hidden",
        out_path="results/subtasks_max25/grasp/init_states_nondeterministic.pkl",
        n_episodes=1000,
        max_steps=128,
        handoff_step=127,
        deterministic=False,
    )