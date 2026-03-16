import os
import sys
import csv
import argparse
import numpy as np
import imageio

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


def get_camera_id_and_name(sim, camera_name=None, camera_id=None):
    if camera_name is not None:
        cam_id = sim.model.camera_name2id(camera_name)
        name = camera_name
    elif camera_id is not None:
        cam_id = int(camera_id)
        try:
            name = sim.model.camera_id2name(cam_id)
        except Exception:
            name = f"camera_{cam_id}"
    else:
        raise ValueError("Provide camera_name or camera_id")
    return cam_id, name


def sanitize_name(name: str) -> str:
    return "".join(c if c.isalnum() or c in ("_", "-", ".") else "_" for c in str(name))


def render_camera(sim, camera_name, width, height):
    frame = sim.render(width=width, height=height, camera_name=camera_name, depth=False)
    if frame.dtype != np.uint8:
        frame = np.clip(frame, 0, 255).astype(np.uint8)
    return frame


def quat_to_rotmat(q):
    """
    MuJoCo camera quaternions are [w, x, y, z].
    Returns 3x3 rotation matrix.
    """
    w, x, y, z = q
    R = np.array([
        [1 - 2*y*y - 2*z*z,     2*x*y - 2*z*w,     2*x*z + 2*y*w],
        [    2*x*y + 2*z*w, 1 - 2*x*x - 2*z*z,     2*y*z - 2*x*w],
        [    2*x*z - 2*y*w,     2*y*z + 2*x*w, 1 - 2*x*x - 2*y*y]
    ], dtype=np.float32)
    return R


def parse_list_arg(text):
    return [float(x.strip()) for x in text.split(",") if x.strip()]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--env-id", type=str, default="drawer-open-v2-goal-hidden")
    parser.add_argument("--outdir", type=str, required=True)

    parser.add_argument("--camera-name", type=str, default="corner2")
    parser.add_argument("--camera-id", type=int, default=None)

    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)

    parser.add_argument(
        "--forward-distances",
        type=str,
        default="0.0,0.10,0.20,0.30,0.40,0.50,0.60",
        help="Comma-separated distances to move camera forward along its current view direction"
    )

    parser.add_argument(
        "--side-offsets",
        type=str,
        default="0.0",
        help="Comma-separated offsets along camera local right direction"
    )

    parser.add_argument(
        "--up-offsets",
        type=str,
        default="0.0",
        help="Comma-separated offsets along camera local up direction"
    )

    args = parser.parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    env = MetaworldDense(env_id=args.env_id, time=True, rank=0)
    env.reset()

    base = unwrap_to_base_with_sim(env)
    sim = base.sim

    cam_id, cam_name = get_camera_id_and_name(
        sim,
        camera_name=args.camera_name if args.camera_id is None else None,
        camera_id=args.camera_id,
    )

    base_pos = sim.model.cam_pos[cam_id].copy()
    base_quat = sim.model.cam_quat[cam_id].copy()

    R = quat_to_rotmat(base_quat)

    # Camera local axes in world coordinates.
    # Convention here: columns of R are local axes in world frame.
    right = R[:, 0]
    up = R[:, 1]
    forward = R[:, 2]

    # We want "toward what the camera is looking at".
    # If the sign is wrong, use -forward instead.
    look_dir = forward / (np.linalg.norm(forward) + 1e-8)

    forward_distances = parse_list_arg(args.forward_distances)
    forward_distances = [-fd for fd in forward_distances] # Because we want to move the camera forward, which is typically in the -Z direction in camera space.
    side_offsets = parse_list_arg(args.side_offsets)
    up_offsets = parse_list_arg(args.up_offsets)

    print(f"Using camera id={cam_id}, name={cam_name}")
    print(f"Base position:   {base_pos}")
    print(f"Base quaternion: {base_quat}")
    print(f"Look direction:  {look_dir}")
    print(f"Right direction: {right}")
    print(f"Up direction:    {up}")

    safe_name = sanitize_name(cam_name)
    manifest_path = os.path.join(args.outdir, "manifest.csv")

    with open(manifest_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "index",
            "filename",
            "camera_id",
            "camera_name",
            "base_x", "base_y", "base_z",
            "forward_d",
            "side_d",
            "up_d",
            "new_x", "new_y", "new_z",
            "quat_w", "quat_x", "quat_y", "quat_z"
        ])

        idx = 0

        sim.model.cam_pos[cam_id] = base_pos
        sim.model.cam_quat[cam_id] = base_quat
        sim.forward()
        frame = render_camera(sim, cam_name, args.width, args.height)
        filename = f"{idx:03d}_base_{safe_name}.png"
        imageio.imwrite(os.path.join(args.outdir, filename), frame)
        writer.writerow([
            idx, filename, cam_id, cam_name,
            float(base_pos[0]), float(base_pos[1]), float(base_pos[2]),
            0.0, 0.0, 0.0,
            float(base_pos[0]), float(base_pos[1]), float(base_pos[2]),
            float(base_quat[0]), float(base_quat[1]), float(base_quat[2]), float(base_quat[3]),
        ])
        idx += 1

        for fd in forward_distances:
            for sd in side_offsets:
                for ud in up_offsets:
                    # Move camera along its local axes
                    new_pos = base_pos + fd * look_dir + sd * right + ud * up

                    sim.model.cam_pos[cam_id] = new_pos
                    sim.model.cam_quat[cam_id] = base_quat
                    sim.forward()

                    try:
                        frame = render_camera(sim, cam_name, args.width, args.height)
                    except Exception as e:
                        print(f"Failed for fd={fd}, sd={sd}, ud={ud}: {e}")
                        continue

                    filename = (
                        f"{idx:03d}_{safe_name}"
                        f"_f{fd:+.3f}_s{sd:+.3f}_u{ud:+.3f}.png"
                    )
                    imageio.imwrite(os.path.join(args.outdir, filename), frame)

                    writer.writerow([
                        idx, filename, cam_id, cam_name,
                        float(base_pos[0]), float(base_pos[1]), float(base_pos[2]),
                        float(fd), float(sd), float(ud),
                        float(new_pos[0]), float(new_pos[1]), float(new_pos[2]),
                        float(base_quat[0]), float(base_quat[1]), float(base_quat[2]), float(base_quat[3]),
                    ])
                    idx += 1

    sim.model.cam_pos[cam_id] = base_pos
    sim.model.cam_quat[cam_id] = base_quat
    sim.forward()

    print(f"Saved PNGs to: {args.outdir}")
    print(f"Manifest written to: {manifest_path}")
    print("If images move the wrong way, rerun with negative forward distances.")
    print("Use manifest.csv to copy the chosen new_x,new_y,new_z into training.")
    

if __name__ == "__main__":
    main()