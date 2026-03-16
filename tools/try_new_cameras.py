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
        "--dx-list",
        type=str,
        default="-0.12,-0.08,-0.04,0.0,0.04",
        help="Comma-separated x offsets added to the base camera position"
    )
    parser.add_argument(
        "--dy-list",
        type=str,
        default="-0.12,-0.08,-0.04,0.0,0.04",
        help="Comma-separated y offsets added to the base camera position"
    )
    parser.add_argument(
        "--dz-list",
        type=str,
        default="-0.08,-0.04,0.0,0.04",
        help="Comma-separated z offsets added to the base camera position"
    )

    parser.add_argument(
        "--use-lookat-order",
        action="store_true",
        help=(
            "Sort outputs by distance-to-origin in offset space so smaller changes appear first. "
            "Useful for manual inspection."
        ),
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

    print(f"Using camera id={cam_id}, name={cam_name}")
    print(f"Base position:   {base_pos}")
    print(f"Base quaternion: {base_quat}")

    dx_list = parse_list_arg(args.dx_list)
    dy_list = parse_list_arg(args.dy_list)
    dz_list = parse_list_arg(args.dz_list)

    candidates = []
    for dx in dx_list:
        for dy in dy_list:
            for dz in dz_list:
                candidates.append((dx, dy, dz))

    if args.use_lookat_order:
        candidates.sort(key=lambda t: (t[0] ** 2 + t[1] ** 2 + t[2] ** 2, abs(t[2]), abs(t[1]), abs(t[0])))

    manifest_path = os.path.join(args.outdir, "manifest.csv")
    with open(manifest_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "index",
            "filename",
            "camera_id",
            "camera_name",
            "base_x", "base_y", "base_z",
            "dx", "dy", "dz",
            "new_x", "new_y", "new_z",
            "quat_w", "quat_x", "quat_y", "quat_z",
        ])

        saved = 0
        safe_name = sanitize_name(cam_name)

        # Save the original base view first
        sim.model.cam_pos[cam_id] = base_pos
        sim.model.cam_quat[cam_id] = base_quat
        sim.forward()
        base_frame = render_camera(sim, cam_name, args.width, args.height)
        base_filename = f"000_base_{safe_name}.png"
        imageio.imwrite(os.path.join(args.outdir, base_filename), base_frame)
        writer.writerow([
            0,
            base_filename,
            cam_id,
            cam_name,
            float(base_pos[0]), float(base_pos[1]), float(base_pos[2]),
            0.0, 0.0, 0.0,
            float(base_pos[0]), float(base_pos[1]), float(base_pos[2]),
            float(base_quat[0]), float(base_quat[1]), float(base_quat[2]), float(base_quat[3]),
        ])
        saved += 1

        for idx, (dx, dy, dz) in enumerate(candidates, start=1):
            new_pos = base_pos + np.array([dx, dy, dz], dtype=np.float32)

            sim.model.cam_pos[cam_id] = new_pos
            sim.model.cam_quat[cam_id] = base_quat
            sim.forward()

            try:
                frame = render_camera(sim, cam_name, args.width, args.height)
            except Exception as e:
                print(f"Failed at idx={idx}, offset=({dx},{dy},{dz}): {e}")
                continue

            filename = (
                f"{idx:03d}_{safe_name}"
                f"_dx{dx:+.3f}_dy{dy:+.3f}_dz{dz:+.3f}.png"
            )
            outpath = os.path.join(args.outdir, filename)
            imageio.imwrite(outpath, frame)

            writer.writerow([
                idx,
                filename,
                cam_id,
                cam_name,
                float(base_pos[0]), float(base_pos[1]), float(base_pos[2]),
                float(dx), float(dy), float(dz),
                float(new_pos[0]), float(new_pos[1]), float(new_pos[2]),
                float(base_quat[0]), float(base_quat[1]), float(base_quat[2]), float(base_quat[3]),
            ])
            saved += 1

    # Restore original pose
    sim.model.cam_pos[cam_id] = base_pos
    sim.model.cam_quat[cam_id] = base_quat
    sim.forward()

    print(f"Saved {saved} PNGs to: {args.outdir}")
    print(f"Manifest written to: {manifest_path}")
    print("Inspect 000_base_*.png first, then compare nearby offsets.")


if __name__ == "__main__":
    main()