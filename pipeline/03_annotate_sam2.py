"""
Stage 3 — Click bolls on ANY frame, propagate masks across video (MULTI-FRAME)
===============================================================================
Handheld recordings often show only 2-3 bolls per frame, so a single anchor
doesn't work.  This version lets you scrub to any frame, click whichever
bolls are visible there, then scrub to another frame for the next batch.
Each boll gets its prompt planted on whatever frame you were on when you
clicked it.

SAM 2 supports this natively: every call to `add_new_points_or_box` takes
its own `frame_idx`, and the model propagates each object independently
from whichever frame it was prompted on.

Workflow:
    1. Scrub through the video with a/d/j/l/h/;  until you see bolls
    2. Press 'n', type the tag number for the boll you want to click
    3. Left-click on the boll (add more clicks to refine if needed)
    4. Press 'n' again to commit this boll
       -> this pushes the prompt to SAM 2 for THIS frame
    5. Continue scrubbing (committed bolls do NOT disappear as you scrub
       — they're stored with the frame they were clicked on)
    6. Repeat for every boll in the recording
    7. Press 'p' to propagate and save masks

Keys:
    a / d        — back / forward 1 frame
    j / l        — back / forward 10 frames
    h / ;        — back / forward 50 frames
    g            — jump to a specific frame number
    LEFT CLICK   — add positive point to current boll
    RIGHT CLICK  — add negative point (refine)
    n            — start new boll / commit current boll
    u            — undo last click on current boll
    c            — clear current boll's points (start over on this one)
    r            — remove current boll entirely
    L            — list all committed bolls
    p            — finished annotating, propagate through video
    q            — quit without saving

Outputs:
    <work_dir>/masks/<frame_idx>/<boll_id>.png    (binary masks, 0/255)
    <work_dir>/annotations.json                   (frame + clicks per boll)

Usage:
    python 03_annotate_sam2.py --work work/handheld_1 \\
        --sam2-checkpoint /path/to/sam2.1_hiera_large.pt \\
        --sam2-config configs/sam2.1/sam2.1_hiera_l.yaml
"""

import argparse
import json
import os
import sys

import cv2
import numpy as np

try:
    import torch
except ImportError:
    sys.exit("ERROR: torch not found. Install PyTorch with CUDA support.")

try:
    from sam2.build_sam import build_sam2_video_predictor
except ImportError:
    sys.exit(
        "ERROR: sam2 not found. Install with:\n"
        "  git clone https://github.com/facebookresearch/sam2.git\n"
        "  cd sam2 && pip install -e ."
    )


# ── Annotation UI ────────────────────────────────────────────────────────


class Annotator:
    """
    Click-based UI for collecting points/labels per boll, across
    multiple frames.  Each committed boll records the frame it was
    clicked on, so stage 5 can remember where the prompt lived.
    """

    def __init__(self, frames_dir: str, files: list[str]):
        self.frames_dir = frames_dir
        self.files = files
        self.n = len(files)
        self.frame_idx = self.n // 2  # start in the middle

        # committed: tag_id -> {"frame_idx": int, "points": [...], "labels": [...]}
        self.bolls = {}
        self.current_id = None      # tag being clicked right now
        self.current_frame_idx = None  # frame the current boll was started on
        self.current_points = []
        self.current_labels = []
        self.message = "Scrub to a boll, then press 'n' to start clicking"

        self._image_cache = {}

    def get_frame_image(self, idx: int) -> np.ndarray:
        if idx not in self._image_cache:
            path = os.path.join(self.frames_dir, self.files[idx])
            self._image_cache[idx] = cv2.imread(path)
            # keep cache bounded
            if len(self._image_cache) > 8:
                # drop oldest entry
                oldest = next(iter(self._image_cache))
                if oldest != idx:
                    del self._image_cache[oldest]
        return self._image_cache[idx]

    # ── Scrub ────────────────────────────────────────────────────────────

    def scrub(self, delta: int):
        new_idx = max(0, min(self.n - 1, self.frame_idx + delta))
        if new_idx == self.frame_idx:
            return
        # If currently mid-click on a boll, warn — clicks are tied to
        # the frame they were placed on, so you can't mix frames in one boll
        if self.current_id is not None and self.current_points:
            self.message = (f"Can't scrub while clicking boll #{self.current_id}. "
                            f"Commit (n), clear (c), or remove (r) first.")
            return
        self.frame_idx = new_idx
        # current_frame_idx stays None until first click
        if self.current_id is not None:
            self.current_frame_idx = self.frame_idx

    def goto(self, target: int):
        if self.current_id is not None and self.current_points:
            self.message = "Commit or clear current boll before jumping"
            return
        self.frame_idx = max(0, min(self.n - 1, target))
        if self.current_id is not None:
            self.current_frame_idx = self.frame_idx

    # ── Click / boll management ──────────────────────────────────────────

    def set_current_id(self, tag_id: int):
        self.current_id = tag_id
        self.current_frame_idx = self.frame_idx
        self.current_points = []
        self.current_labels = []
        self.message = (f"Clicking boll #{tag_id} on frame {self.frame_idx}. "
                        f"L=positive R=negative  n=commit  c=clear  r=remove")

    def on_mouse(self, event, x, y, flags, param):
        if self.current_id is None:
            return
        if event == cv2.EVENT_LBUTTONDOWN:
            self.current_points.append((x, y))
            self.current_labels.append(1)
        elif event == cv2.EVENT_RBUTTONDOWN:
            self.current_points.append((x, y))
            self.current_labels.append(0)

    def undo(self):
        if self.current_points:
            self.current_points.pop()
            self.current_labels.pop()

    def clear_current(self):
        self.current_points = []
        self.current_labels = []

    def remove_current(self):
        if self.current_id is not None and self.current_id in self.bolls:
            del self.bolls[self.current_id]
        self.current_id = None
        self.current_frame_idx = None
        self.current_points = []
        self.current_labels = []

    def commit(self) -> tuple[int, int, np.ndarray, np.ndarray] | None:
        """
        Finalize the current boll.  Returns (tag_id, frame_idx, points_arr,
        labels_arr) for the caller to push to SAM 2.  None on failure.
        """
        if self.current_id is None or not self.current_points:
            self.message = "No points to commit"
            return None
        tag_id = self.current_id
        fr = self.current_frame_idx
        pts = np.array(self.current_points, dtype=np.float32)
        labs = np.array(self.current_labels, dtype=np.int32)
        self.bolls[tag_id] = {
            "frame_idx": int(fr),
            "points": [list(p) for p in self.current_points],
            "labels": list(self.current_labels),
        }
        self.message = (f"Committed boll #{tag_id} on frame {fr} "
                        f"({len(self.current_points)} pts)")
        self.current_id = None
        self.current_frame_idx = None
        self.current_points = []
        self.current_labels = []
        return (tag_id, int(fr), pts, labs)

    # ── Render ───────────────────────────────────────────────────────────

    def render(self) -> np.ndarray:
        disp = self.get_frame_image(self.frame_idx).copy()

        # Show committed bolls: if a committed boll was clicked on THIS
        # frame, draw its clicks.  Otherwise just show a small badge in
        # the corner listing them.
        same_frame_bolls = [(tid, d) for tid, d in self.bolls.items()
                            if d["frame_idx"] == self.frame_idx]
        for tid, data in same_frame_bolls:
            for (x, y), lab in zip(data["points"], data["labels"]):
                color = (0, 255, 0) if lab == 1 else (0, 0, 255)
                cv2.circle(disp, (int(x), int(y)), 5, color, -1)
                cv2.circle(disp, (int(x), int(y)), 6, (255, 255, 255), 1)
            if data["points"]:
                x, y = data["points"][0]
                cv2.putText(disp, f"#{tid}", (int(x) + 8, int(y) - 8),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

        # Current in-progress boll — yellow points, but ONLY if we're
        # on the same frame we started clicking on
        if self.current_id is not None and self.current_frame_idx == self.frame_idx:
            for (x, y), lab in zip(self.current_points, self.current_labels):
                color = (0, 255, 255) if lab == 1 else (0, 100, 255)
                cv2.circle(disp, (int(x), int(y)), 6, color, -1)
                cv2.circle(disp, (int(x), int(y)), 8, (255, 255, 255), 2)

        # HUD
        h, w = disp.shape[:2]
        bar_top = h - 85
        cv2.rectangle(disp, (0, bar_top), (w, h), (0, 0, 0), -1)

        line1 = (f"Frame {self.frame_idx}/{self.n-1}   "
                 f"Bolls committed: {len(self.bolls)}")
        if self.current_id is not None:
            line1 += (f"   NOW: #{self.current_id} "
                      f"({len(self.current_points)} pts on frame "
                      f"{self.current_frame_idx})")
        cv2.putText(disp, line1, (10, bar_top + 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)

        keys = "a/d=1  j/l=10  h/;=50  g=goto  n=new/commit  p=propagate  q=quit"
        cv2.putText(disp, keys, (10, bar_top + 45),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180, 180, 180), 1)

        cv2.putText(disp, self.message, (10, bar_top + 70),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 255, 200), 1)

        # Top-left: frame counter big
        cv2.putText(disp, f"F{self.frame_idx}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        return disp


def prompt_tag_id(existing_ids: set) -> int | None:
    """Read a tag number from stdin. Returns None on cancel."""
    while True:
        try:
            s = input("  Boll tag number (or 'cancel'): ").strip()
        except EOFError:
            return None
        if s.lower() in ("cancel", "c", ""):
            return None
        if not s.isdigit():
            print("    must be a number")
            continue
        tag = int(s)
        if tag in existing_ids:
            print(f"    boll #{tag} already exists. Press 'r' to remove first.")
            continue
        return tag


# ── Main ─────────────────────────────────────────────────────────────────


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--work", required=True, help="Working dir from stage 1")
    p.add_argument("--sam2-checkpoint", required=True,
                   help="Path to SAM 2 checkpoint .pt file")
    p.add_argument("--sam2-config", default="configs/sam2.1/sam2.1_hiera_l.yaml",
                   help="SAM 2 model config (default: sam2.1_hiera_l.yaml)")
    args = p.parse_args()

    frames_dir = os.path.join(args.work, "frames")
    if not os.path.isdir(frames_dir):
        sys.exit(f"ERROR: {frames_dir} not found. Run 01_extract_frames.py first.")

    files = sorted(f for f in os.listdir(frames_dir) if f.endswith(".jpg"))
    n_frames = len(files)
    if n_frames == 0:
        sys.exit("ERROR: no frames")

    print(f"[INFO] {n_frames} frames loaded from {frames_dir}")

    # ── Load SAM 2 ───────────────────────────────────────────────────────
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cpu":
        print("[WARN] CUDA not available — propagation will be SLOW.")
    print(f"[INFO] Loading SAM 2 from {args.sam2_checkpoint} on {device}...")
    predictor = build_sam2_video_predictor(args.sam2_config, args.sam2_checkpoint,
                                           device=device)
    print("[INFO] Initializing inference state (loads frames into VRAM)...")
    with torch.inference_mode(), torch.autocast(device, dtype=torch.bfloat16):
        state = predictor.init_state(video_path=frames_dir)
    print("[INFO] Ready.")

    # ── Annotation UI ────────────────────────────────────────────────────
    ui = Annotator(frames_dir, files)
    win = "SAM 2 Annotator (multi-frame)"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(win, ui.on_mouse)

    print()
    print("[KEYS]")
    print("  a/d = ±1 frame    j/l = ±10    h/; = ±50    g = goto frame")
    print("  L click  = positive point   R click  = negative point")
    print("  n = new boll (then tag num) / commit current boll")
    print("  u = undo last click   c = clear current boll's points")
    print("  r = remove current boll   L = list committed")
    print("  p = propagate & save   q = quit")
    print()
    print("Scrub to a frame where you can see a boll, press 'n', type the")
    print("tag number, click the boll, press 'n' again to commit.  Repeat.")

    propagate_now = False
    while True:
        cv2.imshow(win, ui.render())
        k = cv2.waitKey(20) & 0xFF

        if k == 255:
            continue

        # Scrub keys
        if k == ord("a"):
            ui.scrub(-1)
        elif k == ord("d"):
            ui.scrub(+1)
        elif k == ord("j"):
            ui.scrub(-10)
        elif k == ord("l"):
            ui.scrub(+10)
        elif k == ord("h"):
            ui.scrub(-50)
        elif k == ord(";"):
            ui.scrub(+50)
        elif k == ord("g"):
            try:
                target = int(input(f"  Jump to frame [0-{n_frames-1}]: "))
                ui.goto(target)
            except ValueError:
                print("    invalid")

        # Quit
        elif k == ord("q"):
            print("[INFO] Quit without saving.")
            cv2.destroyAllWindows()
            return

        # Undo / clear / remove
        elif k == ord("u"):
            ui.undo()
        elif k == ord("c"):
            ui.clear_current()
            ui.message = "Cleared current boll's points"
        elif k == ord("r"):
            ui.remove_current()
            ui.message = "Removed current boll"

        # List committed bolls
        elif k == ord("L"):
            print(f"[LIST] {len(ui.bolls)} committed bolls:")
            for tid in sorted(ui.bolls.keys()):
                d = ui.bolls[tid]
                print(f"  #{tid:>4d}  frame {d['frame_idx']:>4d}  "
                      f"{len(d['points'])} pts")

        # New / commit boll
        elif k == ord("n"):
            if ui.current_id is None:
                tag = prompt_tag_id(set(ui.bolls.keys()))
                if tag is not None:
                    ui.set_current_id(tag)
            else:
                result = ui.commit()
                if result is not None:
                    tag_id, fr, pts, labs = result
                    # Push prompt to SAM 2 immediately on the correct frame
                    with torch.inference_mode(), torch.autocast(device, dtype=torch.bfloat16):
                        predictor.add_new_points_or_box(
                            inference_state=state,
                            frame_idx=fr,
                            obj_id=int(tag_id),
                            points=pts,
                            labels=labs,
                        )
                    print(f"[SAM2] Added prompt for boll #{tag_id} on frame {fr}")

        # Propagate
        elif k == ord("p"):
            if not ui.bolls:
                ui.message = "No bolls committed yet"
                continue
            propagate_now = True
            break

    cv2.destroyAllWindows()

    if not propagate_now:
        return

    # ── Save annotations.json ────────────────────────────────────────────
    ann_path = os.path.join(args.work, "annotations.json")
    with open(ann_path, "w") as f:
        json.dump({
            "bolls": {str(k): v for k, v in ui.bolls.items()},
        }, f, indent=2)
    print(f"[INFO] Annotations saved: {ann_path}")

    # ── Propagate ────────────────────────────────────────────────────────
    masks_root = os.path.join(args.work, "masks")
    os.makedirs(masks_root, exist_ok=True)

    print(f"\n[INFO] Propagating {len(ui.bolls)} bolls through {n_frames} frames...")
    print("       (forward then reverse, each from its own prompt frame)")

    # Track which frames each boll appears in (for reporting)
    presence = {bid: 0 for bid in ui.bolls.keys()}

    def save_frame_masks(frame_idx, obj_ids, mask_logits):
        frame_dir = os.path.join(masks_root, f"{frame_idx:05d}")
        masks = (mask_logits > 0.0).cpu().numpy()
        if masks.ndim == 4:
            masks = masks[:, 0]
        for obj_id, mask in zip(obj_ids, masks):
            if not mask.any():
                continue
            os.makedirs(frame_dir, exist_ok=True)
            out_path = os.path.join(frame_dir, f"{int(obj_id)}.png")
            cv2.imwrite(out_path, (mask.astype(np.uint8) * 255))
            if int(obj_id) in presence:
                presence[int(obj_id)] += 1

    with torch.inference_mode(), torch.autocast(device, dtype=torch.bfloat16):
        # Single forward pass — SAM 2 handles prompts at different frames
        print("  forward pass...")
        for frame_idx, obj_ids, mask_logits in predictor.propagate_in_video(state):
            save_frame_masks(frame_idx, obj_ids, mask_logits)
            if frame_idx % 25 == 0:
                print(f"    ...frame {frame_idx}")

        # Reverse pass for frames before each object's prompt frame
        print("  reverse pass...")
        for frame_idx, obj_ids, mask_logits in predictor.propagate_in_video(
            state, reverse=True
        ):
            save_frame_masks(frame_idx, obj_ids, mask_logits)
            if frame_idx % 25 == 0:
                print(f"    ...frame {frame_idx}")

    print("\n[DONE] Propagation finished.")
    print("[REPORT] Frames each boll appears in:")
    for bid in sorted(presence.keys()):
        fr = ui.bolls[bid]["frame_idx"]
        print(f"  boll #{bid:>4d}  prompted@f{fr:>4d}  tracked in {presence[bid]:>4d} frames")
    print(f"\n[DONE] Masks saved under: {masks_root}")


if __name__ == "__main__":
    main()