import os
import sys
import argparse
import numpy as np
import imageio
import gym

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


def render_from_camera(sim, cam_name, width=640, height=480):
    frame = sim.render(
        width=width,
        height=height,
        camera_name=cam_name,
        depth=False
    )
    return frame


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument("--env-id", type=str,
                        default="drawer-open-v2-goal-hidden")

    parser.add_argument("--outdir", type=str,
                        required=True)

    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)

    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    print("Creating environment...")

    env = MetaworldDense(
        env_id=args.env_id,
        time=True,
        rank=0
    )

    env.reset()

    base = unwrap_to_base_with_sim(env)
    sim = base.sim

    ncam = sim.model.ncam

    print(f"\nFound {ncam} cameras\n")

    for cam_id in range(ncam):

        try:
            cam_name = sim.model.camera_id2name(cam_id)
        except Exception:
            cam_name = None

        pos = sim.model.cam_pos[cam_id]
        quat = sim.model.cam_quat[cam_id]

        print("-----")
        print(f"Camera ID: {cam_id}")
        print(f"Name: {cam_name}")
        print(f"Position: {pos}")
        print(f"Quaternion: {quat}")

        try:

            if cam_name is None:
                cam_name = f"camera_{cam_id}"

            frame = render_from_camera(
                sim,
                cam_name,
                width=args.width,
                height=args.height
            )

            outfile = os.path.join(
                args.outdir,
                f"cam_{cam_id}_{cam_name}.png"
            )

            imageio.imwrite(outfile, frame)

            print(f"Saved: {outfile}")

        except Exception as e:

            print(f"Failed to render camera {cam_id}: {e}")

    print("\nDone.")


if __name__ == "__main__":
    main()