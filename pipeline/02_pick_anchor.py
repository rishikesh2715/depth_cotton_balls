"""
Stage 2 — Pick the best "anchor" frame to start annotation from
================================================================
Scrub through the extracted frames and pick the frame where the most
numbered tags / bolls are clearly visible. SAM 2 propagates BOTH
forward and backward from the anchor frame, so the best anchor is
usually somewhere in the middle of the recording, not at the start.

Controls:
    a / d        — back / forward 1 frame
    j / l        — back / forward 10 frames
    h / ;        — back / forward 50 frames
    g            — jump to frame number (typed in terminal)
    SPACE        — mark current frame as anchor and quit
    q            — quit without saving

The selected anchor index is written to <work_dir>/anchor.json.

Usage:
    python 02_pick_anchor.py --work work/plant_a
"""

import argparse
import json
import os
import sys

import cv2


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--work", required=True, help="Working dir from stage 1")
    args = p.parse_args()

    frames_dir = os.path.join(args.work, "frames")
    if not os.path.isdir(frames_dir):
        sys.exit(f"ERROR: {frames_dir} not found. Run 01_extract_frames.py first.")

    files = sorted(f for f in os.listdir(frames_dir) if f.endswith(".jpg"))
    if not files:
        sys.exit(f"ERROR: No frames in {frames_dir}")

    n = len(files)
    idx = n // 2  # start in the middle

    print(f"[INFO] {n} frames loaded.  Start index: {idx}")
    print("[KEYS] a/d=±1  j/l=±10  h/;=±50  g=goto  SPACE=select  q=quit")

    win = "Pick anchor frame"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)

    while True:
        img = cv2.imread(os.path.join(frames_dir, files[idx]))
        disp = img.copy()
        cv2.putText(disp, f"Frame {idx} / {n - 1}",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv2.putText(disp, "SPACE = select anchor",
                    (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        cv2.imshow(win, disp)

        k = cv2.waitKey(0) & 0xFF
        if k == ord("q"):
            print("[INFO] Cancelled.")
            cv2.destroyAllWindows()
            return
        elif k == ord(" "):
            anchor_path = os.path.join(args.work, "anchor.json")
            with open(anchor_path, "w") as f:
                json.dump({"anchor_frame": idx, "total_frames": n}, f, indent=2)
            print(f"[DONE] Anchor = frame {idx}.  Saved to {anchor_path}")
            cv2.destroyAllWindows()
            return
        elif k == ord("a"):
            idx = max(0, idx - 1)
        elif k == ord("d"):
            idx = min(n - 1, idx + 1)
        elif k == ord("j"):
            idx = max(0, idx - 10)
        elif k == ord("l"):
            idx = min(n - 1, idx + 10)
        elif k == ord("h"):
            idx = max(0, idx - 50)
        elif k == ord(";"):
            idx = min(n - 1, idx + 50)
        elif k == ord("g"):
            try:
                target = int(input(f"Jump to frame [0-{n-1}]: "))
                idx = max(0, min(n - 1, target))
            except ValueError:
                print("  invalid")


if __name__ == "__main__":
    main()
