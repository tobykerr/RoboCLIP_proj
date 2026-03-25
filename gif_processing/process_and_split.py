# process_and_split.py
"""
Interactive preprocessing + temporal splitting for drawer demos.

For each input video:
  1) Decode frames
  2) Select a square crop interactively (same as your existing script)
  3) Select TWO temporal cut points interactively:
        [0 .. cut1)      -> reach
        [cut1 .. cut2)   -> grasp
        [cut2 .. end)    -> pull
  4) For each segment, uniformly sample 32 frames, crop to square, resize to 250x250,
     and save as GIF.

Controls (crop selection):
  - Drag mouse to draw ROI rectangle (converted to centered square)
  - ENTER/SPACE: accept ROI
  - 'd': discard this demo (skip video)
  - 'r' or ESC: redo crop

Controls (cut selection):
  - Trackbar to scrub frames
  - '1': set cut1 (reach->grasp) at current frame
  - '2': set cut2 (grasp->pull) at current frame
  - ENTER/SPACE: accept cuts
  - 'r' or ESC: redo cuts
  - 'd': discard this demo

Output:
  output_dir/
    reach/1.gif, 2.gif, ...
    grasp/1.gif, 2.gif, ...
    pull/1.gif, 2.gif, ...

Usage:
  python process_and_split.py --input ./drawer-open-demos --output ./gifs/split_demos
"""

import os
import cv2
import numpy as np
from PIL import Image
import argparse

TARGET_FRAMES = 32
OUT_SIZE = 250
PREVIEW_MAX_SIDE = 800
VIDEO_EXTS = (".mp4", ".mov", ".m4v", ".avi", ".mkv")


def uniform_sample_indices(n: int, k: int) -> np.ndarray:
    if n <= 0:
        raise ValueError("Segment has no frames")
    if n >= k:
        return np.linspace(0, n - 1, k).round().astype(int)
    idx = list(range(n))
    idx += [n - 1] * (k - n)
    return np.array(idx, dtype=int)


def clamp(v, lo, hi):
    return max(lo, min(v, hi))


def resize_for_preview(frame_rgb, max_side):
    h, w = frame_rgb.shape[:2]
    scale = min(1.0, max_side / max(h, w))
    new_w = int(round(w * scale))
    new_h = int(round(h * scale))
    preview = cv2.resize(frame_rgb, (new_w, new_h), interpolation=cv2.INTER_AREA)
    return preview, scale


def roi_to_square_centered(x, y, w, h, img_w, img_h):
    """Convert arbitrary ROI to a centered square ROI (side=min(w,h)) clamped to bounds."""
    side = int(min(w, h))
    cx = x + w / 2.0
    cy = y + h / 2.0
    x0 = int(round(cx - side / 2.0))
    y0 = int(round(cy - side / 2.0))
    x0 = clamp(x0, 0, img_w - side)
    y0 = clamp(y0, 0, img_h - side)
    return x0, y0, side, side


def _roi_select_with_discard(preview_bgr, win_name):
    """
    OpenCV selectROI wrapper with accept/discard/redo.
    Returns:
      (x, y, w, h)  accepted
      None          discard
      "cancel"      redo
    """
    roi = cv2.selectROI(win_name, preview_bgr, fromCenter=False, showCrosshair=True)
    x, y, w, h = roi

    if w == 0 or h == 0:
        return "cancel"

    overlay = preview_bgr.copy()
    cv2.rectangle(overlay, (x, y), (x + w, y + h), (0, 255, 0), 2)
    cv2.imshow(win_name, overlay)

    while True:
        key = cv2.waitKey(0) & 0xFF
        if key in (13, 32):  # ENTER or SPACE
            return (x, y, w, h)
        if key in (ord("d"), ord("D")):
            return None
        if key in (ord("r"), ord("R")):
            return "cancel"
        if key == 27:  # ESC
            return "cancel"


def select_square_crop_on_preview(rep_rgb):
    """
    Returns either:
      - (ox0, oy0, oside)  -> accepted crop in ORIGINAL coords
      - None               -> discard demo
    """
    preview, scale = resize_for_preview(rep_rgb, PREVIEW_MAX_SIDE)
    preview_bgr = cv2.cvtColor(preview, cv2.COLOR_RGB2BGR)

    win = "Crop: draw box | ENTER accept | d discard | r/ESC redo"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(win, preview_bgr.shape[1], preview_bgr.shape[0])
    cv2.imshow(win, preview_bgr)

    while True:
        roi = _roi_select_with_discard(preview_bgr, win)
        if roi is None:
            cv2.destroyWindow(win)
            return None  # discard
        if roi == "cancel":
            cv2.imshow(win, preview_bgr)
            continue

        x, y, w, h = roi
        ph, pw = preview.shape[:2]
        px0, py0, pside, _ = roi_to_square_centered(x, y, w, h, pw, ph)

        # Map square ROI back to original coordinates
        img_h, img_w = rep_rgb.shape[:2]
        ox0 = int(round(px0 / scale))
        oy0 = int(round(py0 / scale))
        oside = int(round(pside / scale))

        # Clamp to original bounds
        oside = max(1, min(oside, img_w, img_h))
        ox0 = clamp(ox0, 0, img_w - oside)
        oy0 = clamp(oy0, 0, img_h - oside)

        cv2.destroyWindow(win)
        return ox0, oy0, oside


def draw_hud(frame_bgr, idx, n, cut1, cut2):
    hud = frame_bgr.copy()
    txt1 = f"Frame {idx+1}/{n}"
    txt2 = f"cut1(reach->grasp): {cut1+1 if cut1 is not None else 'unset'}   |   cut2(grasp->pull): {cut2+1 if cut2 is not None else 'unset'}"
    txt3 = "Keys: 1=set cut1  2=set cut2  ENTER/SPACE=accept  r/ESC=redo  d=discard"

    # Put text with outline for readability
    def put(line, y):
        cv2.putText(hud, line, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 4, cv2.LINE_AA)
        cv2.putText(hud, line, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)

    put(txt1, 25)
    put(txt2, 50)
    put(txt3, 75)

    # Visual markers if cuts set
    if cut1 is not None:
        put("Reach ends here (cut1)", 105)
    if cut2 is not None:
        put("Grasp ends / Pull starts here (cut2)", 130)

    return hud


def select_two_cuts_interactive(frames_rgb):
    """
    Returns:
      (cut1, cut2) as integers (0-based indices), where 0 < cut1 < cut2 < n
      None if discard
      "redo" if redo requested
    """
    n = len(frames_rgb)
    if n < 3:
        return "redo"

    win = "Cuts: scrub w/ slider | 1=set cut1 | 2=set cut2 | ENTER accept | r/ESC redo | d discard"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)

    # Trackbar callback needs mutable state
    state = {"pos": 0, "cut1": None, "cut2": None}

    def on_trackbar(v):
        state["pos"] = v

    cv2.createTrackbar("frame", win, 0, n - 1, on_trackbar)

    # Start at middle for convenience
    mid = n // 2
    cv2.setTrackbarPos("frame", win, mid)
    state["pos"] = mid

    while True:
        idx = state["pos"]
        frame_rgb = frames_rgb[idx]
        frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
        hud = draw_hud(frame_bgr, idx, n, state["cut1"], state["cut2"])
        cv2.imshow(win, hud)

        key = cv2.waitKey(30) & 0xFF

        if key in (ord("1"),):
            state["cut1"] = idx
        elif key in (ord("2"),):
            state["cut2"] = idx
        elif key in (13, 32):  # ENTER / SPACE
            cut1, cut2 = state["cut1"], state["cut2"]
            if cut1 is None or cut2 is None:
                print("Please set both cut1 and cut2 (press 1 and 2).")
                continue
            if not (0 < cut1 < cut2 < n):
                print("Cuts must satisfy 0 < cut1 < cut2 < n. Adjust and try again.")
                continue
            cv2.destroyWindow(win)
            return (cut1, cut2)
        elif key in (ord("r"), ord("R"), 27):  # redo or ESC
            cv2.destroyWindow(win)
            return "redo"
        elif key in (ord("d"), ord("D")):
            cv2.destroyWindow(win)
            return None


def segment_frames(frames, cut1, cut2):
    # Define segments as:
    # reach:  [0, cut1)
    # grasp:  [cut1, cut2)
    # pull:   [cut2, end)
    reach = frames[:cut1]
    grasp = frames[cut1:cut2]
    pull = frames[cut2:]
    return reach, grasp, pull


def process_segment_to_gif(frames_rgb, crop, out_gif_path):
    x0, y0, side = crop
    idx = uniform_sample_indices(len(frames_rgb), TARGET_FRAMES)
    processed = []
    for i in idx:
        f = frames_rgb[i]
        crop_f = f[y0:y0 + side, x0:x0 + side]
        crop_f = cv2.resize(crop_f, (OUT_SIZE, OUT_SIZE), interpolation=cv2.INTER_AREA)
        processed.append(Image.fromarray(crop_f))

    os.makedirs(os.path.dirname(out_gif_path), exist_ok=True)
    processed[0].save(
        out_gif_path,
        save_all=True,
        append_images=processed[1:],
        duration=50,
        loop=0,
    )


def decode_video(video_path):
    cap = cv2.VideoCapture(video_path)
    frames = []
    while True:
        ok, bgr = cap.read()
        if not ok:
            break
        frames.append(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
    cap.release()
    return frames


def preprocess_and_split_video(video_path, out_dir_reach, out_dir_grasp, out_dir_pull, out_idx):
    frames = decode_video(video_path)
    if not frames:
        raise RuntimeError(f"No frames decoded from {video_path}")

    # Representative frame for cropping: use last frame (like your original script)
    rep = frames[-1]

    print(f"\n=== {os.path.basename(video_path)} ===")
    print("Step 1: Select crop (square). ENTER accept, d discard, r/ESC redo.\n")
    crop = select_square_crop_on_preview(rep)
    if crop is None:
        return False  # discarded

    print("Step 2: Select cut points for reach/grasp/pull.")
    print("Scrub with slider; press '1' to set cut1, '2' to set cut2; ENTER to accept.\n")

    while True:
        cuts = select_two_cuts_interactive(frames)
        if cuts is None:
            return False  # discarded
        if cuts == "redo":
            continue
        cut1, cut2 = cuts
        break

    reach_frames, grasp_frames, pull_frames = segment_frames(frames, cut1, cut2)

    # Guard against pathological tiny segments (still padded by uniform_sample_indices, but warn)
    def warn(seg_name, seg):
        if len(seg) < 8:
            print(f"Warning: {seg_name} segment has only {len(seg)} frames; will pad last frame.")

    warn("reach", reach_frames)
    warn("grasp", grasp_frames)
    warn("pull", pull_frames)

    out_path_reach = os.path.join(out_dir_reach, f"{out_idx}.gif")
    out_path_grasp = os.path.join(out_dir_grasp, f"{out_idx}.gif")
    out_path_pull = os.path.join(out_dir_pull, f"{out_idx}.gif")

    process_segment_to_gif(reach_frames, crop, out_path_reach)
    process_segment_to_gif(grasp_frames, crop, out_path_grasp)
    process_segment_to_gif(pull_frames, crop, out_path_pull)

    print(f"Saved reach: {out_path_reach}")
    print(f"Saved grasp: {out_path_grasp}")
    print(f"Saved pull : {out_path_pull}")
    return True


def batch_preprocess_and_split(input_dir, output_dir):
    videos = sorted([f for f in os.listdir(input_dir) if f.lower().endswith(VIDEO_EXTS)])
    if not videos:
        raise RuntimeError(f"No videos found in {input_dir}")

    out_dir_reach = os.path.join(output_dir, "reach")
    out_dir_grasp = os.path.join(output_dir, "grasp")
    out_dir_pull = os.path.join(output_dir, "pull")
    os.makedirs(out_dir_reach, exist_ok=True)
    os.makedirs(out_dir_grasp, exist_ok=True)
    os.makedirs(out_dir_pull, exist_ok=True)

    out_idx = 1  # increments only when we keep a demo

    for in_idx, fname in enumerate(videos, start=1):
        in_path = os.path.join(input_dir, fname)
        print(f"\nInput [{in_idx}/{len(videos)}]: {in_path}")
        print(f"Output index (if kept): {out_idx}")

        kept = preprocess_and_split_video(
            video_path=in_path,
            out_dir_reach=out_dir_reach,
            out_dir_grasp=out_dir_grasp,
            out_dir_pull=out_dir_pull,
            out_idx=out_idx,
        )
        if kept:
            out_idx += 1
        else:
            print("Discarded demo. Output numbering unchanged.")

    print(f"\nDone. Kept {out_idx - 1} demos.")
    print(f"Output dir: {output_dir}")


def get_args():
    p = argparse.ArgumentParser()
    p.add_argument("--input", type=str, required=True, help="Input directory containing videos")
    p.add_argument("--output", type=str, required=True, help="Output directory for split gifs")
    return p.parse_args()


if __name__ == "__main__":
    args = get_args()
    batch_preprocess_and_split(args.input, args.output)
