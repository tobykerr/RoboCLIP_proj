import os
import cv2
import numpy as np
from PIL import Image

TARGET_FRAMES = 32
OUT_SIZE = 250
PREVIEW_MAX_SIDE = 800
VIDEO_EXTS = (".mp4", ".mov", ".m4v", ".avi", ".mkv")


def uniform_sample_indices(n, k):
    if n <= 0:
        raise ValueError("Video has no frames")
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
    """Convert an arbitrary ROI to a square ROI (side=min(w,h)) centered on it, clamped to bounds."""
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
    Interactive ROI select with ability to discard.

    Controls:
      - Drag mouse to draw ROI rectangle.
      - ENTER or SPACE: accept ROI
      - 'd': discard this demo (returns None)
      - 'r': reset ROI drawing
      - ESC: cancel selection (returns "cancel" so caller can re-prompt)

    Returns:
      (x, y, w, h) or None for discard or "cancel" to retry.
    """
    # Use OpenCV's built-in ROI selector, but we wrap it to allow key handling.
    # Approach: use selectROI for drawing, then ask user whether to accept/discard.
    roi = cv2.selectROI(win_name, preview_bgr, fromCenter=False, showCrosshair=True)
    x, y, w, h = roi

    # If user hit ESC in selectROI, OpenCV returns (0,0,0,0)
    if w == 0 or h == 0:
        return "cancel"

    # Show the selected ROI overlay and prompt for accept/discard
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

    win = "Draw crop  |  ENTER accept  |  d discard  |  r redo"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(win, preview_bgr.shape[1], preview_bgr.shape[0])
    cv2.imshow(win, preview_bgr)

    while True:
        roi = _roi_select_with_discard(preview_bgr, win)
        if roi is None:
            cv2.destroyWindow(win)
            return None  # discard
        if roi == "cancel":
            # retry
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


def preprocess_video_to_gif_interactive(video_path, out_gif_path):
    cap = cv2.VideoCapture(video_path)
    frames = []
    while True:
        ok, bgr = cap.read()
        if not ok:
            break
        frames.append(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
    cap.release()

    if not frames:
        raise RuntimeError(f"No frames decoded from {video_path}")

    rep = frames[-1]

    print(f"\nCrop selection for: {os.path.basename(video_path)}")
    print("Draw a box around the drawer. It will be converted to a square crop then resized to 250×250.")
    print("ENTER/SPACE = accept, d = discard this demo, r/ESC = redo crop\n")

    crop = select_square_crop_on_preview(rep)
    if crop is None:
        # Discard this demo
        return False

    x0, y0, side = crop

    idx = uniform_sample_indices(len(frames), TARGET_FRAMES)

    processed = []
    for i in idx:
        f = frames[i]
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
    return True


def batch_preprocess(input_dir, output_dir):
    videos = sorted([f for f in os.listdir(input_dir) if f.lower().endswith(VIDEO_EXTS)])
    if not videos:
        raise RuntimeError(f"No videos found in {input_dir}")

    os.makedirs(output_dir, exist_ok=True)

    out_idx = 1  # only increments when we KEEP a demo

    for in_idx, fname in enumerate(videos, start=1):
        in_path = os.path.join(input_dir, fname)
        out_path = os.path.join(output_dir, f"{out_idx}.gif")

        print(f"\nInput [{in_idx}/{len(videos)}]: {in_path}")
        print(f"Output (if kept): {out_path}")

        kept = preprocess_video_to_gif_interactive(in_path, out_path)
        if kept:
            print(f"Saved: {out_path}")
            out_idx += 1
        else:
            print("Discarded demo (no file written). Output numbering unchanged.")

    print(f"\nDone. Kept {out_idx - 1} demos. Output dir: {output_dir}")


if __name__ == "__main__":
    batch_preprocess("./drawer-open-demos", "./gifs/human_opening_door")
