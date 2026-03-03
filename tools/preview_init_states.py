# tools/preview_init_states.py
"""
Sanity-check a relay initial-state dataset (.pkl) by:
1) Sampling N saved states
2) Resetting the env
3) Restoring the saved MuJoCo state
4) Rendering a single frame per state
5) Saving individual PNGs + a single grid PNG

Example:
python tools/preview_init_states.py \
  --env-id drawer-open-v2-goal-hidden \
  --pkl results/subtasks_max25/grasp/init_states.pkl \
  --outdir results/subtasks_max25/grasp/preview_states \
  --n 10 \
  --seed 0
"""

import os
import sys
import argparse
import pickle
import numpy as np
from PIL import Image

# Ensure repo root is on path (matches your pattern)
repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

from metaworld_envs import MetaworldDense  # your wrapper


def unwrap_env(e, max_depth=30):
    """
    Walk down the .env chain until we find something that looks like a MuJoCo env
    exposing `.sim` (mujoco_py). This matches what worked for your collector.
    """
    cur = e
    for _ in range(max_depth):
        if hasattr(cur, "sim"):
            return cur
        if hasattr(cur, "env"):
            cur = cur.env
        else:
            break
    return cur


def restore_state(env, mj_state, extra=None):
    """
    Restore mujoco_py state into base env and forward.

    NOTE: We only restore extra fields onto env/unwrapped if they exist.
    In your saved files `extra` was likely empty because MetaworldDense doesn't have those attrs,
    but the hook is here in case you extend it later.
    """
    base = unwrap_env(env)
    if not hasattr(base, "sim"):
        raise AttributeError(
            f"Could not find `.sim` on base env after unwrapping. Base type: {type(base)}"
        )

    sim = base.sim
    sim.set_state(mj_state)
    sim.forward()

    # Restore extra bookkeeping if present (optional)
    if extra:
        uw = env  # your wrapper
        for k, v in extra.items():
            if hasattr(uw, k):
                setattr(uw, k, v)


def make_grid(images, cols=5, pad=6, bg=(255, 255, 255)):
    """
    Create a simple image grid from a list of PIL Images.
    """
    if not images:
        raise ValueError("No images to grid")

    w, h = images[0].size
    rows = int(np.ceil(len(images) / cols))
    grid_w = cols * w + (cols + 1) * pad
    grid_h = rows * h + (rows + 1) * pad

    grid = Image.new("RGB", (grid_w, grid_h), color=bg)

    for idx, im in enumerate(images):
        r = idx // cols
        c = idx % cols
        x = pad + c * (w + pad)
        y = pad + r * (h + pad)
        grid.paste(im, (x, y))

    return grid


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--env-id", type=str, required=True)
    parser.add_argument("--pkl", type=str, required=True)
    parser.add_argument("--outdir", type=str, required=True)
    parser.add_argument("--n", type=int, default=10)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--rank", type=int, default=0, help="Seed/rank passed to env ctor")
    parser.add_argument("--cols", type=int, default=5, help="Columns for grid image")
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    # Load states
    with open(args.pkl, "rb") as f:
        payload = pickle.load(f)
    states = payload.get("states", [])
    if len(states) == 0:
        raise RuntimeError(f"No states found in {args.pkl}")

    n = min(args.n, len(states))
    idxs = rng.choice(len(states), size=n, replace=False)

    # Create env
    env = MetaworldDense(env_id=args.env_id, time=True, rank=args.rank)

    pil_images = []
    for j, idx in enumerate(idxs):
        s = states[idx]
        mj_state = s["mj_state"]
        extra = s.get("extra", None)

        # Reset first (so task-specific reset logic runs), then restore saved physics state
        env.reset()
        restore_state(env, mj_state, extra)

        frame = env.render()  # numpy (H,W,3)
        im = Image.fromarray(frame)
        pil_images.append(im)

        out_png = os.path.join(args.outdir, f"state_{j:02d}_idx_{idx}.png")
        im.save(out_png)
        print(f"Saved: {out_png}")

    grid = make_grid(pil_images, cols=args.cols)
    grid_path = os.path.join(args.outdir, f"grid_n{n}.png")
    grid.save(grid_path)
    print(f"Saved grid: {grid_path}")


if __name__ == "__main__":
    main()