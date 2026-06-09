"""
infer_aux.py — Combined inference: YOLO seg/pose + AuxHead.

Given an image (path, numpy BGR array, or PIL Image), run YOLOv11-seg
to get bolls, then IoU-match each detection to a YOLOv11-pose
prediction (for tip/base keypoints) and run AuxHead on each bbox crop
(for occlusion class + visibility fraction). Optional ByteTrack /
BoT-SORT tracking on the seg stream gives every boll a stable
`track_id` across frames.

Each per-boll record:
    bbox          : [x, y, w, h] in pixels (original image space)
    det_conf      : float (YOLO seg objectness)
    mask_poly     : list of [x, y] vertices (original image space) or None
    kpts          : [[x, y, kpt_conf], [x, y, kpt_conf]] in original
                    image space, or None. kpt_conf is a per-keypoint
                    confidence in [0, 1] from YOLOv11-pose (NOT the
                    {0,1,2} visibility flag from training labels — at
                    inference, the model emits a continuous score).
    occ_class     : str in {none, leaf, stem, other} or None
    occ_class_conf: float
    visfrac       : float in [0, 1] or None
    track_id      : int or None (only present when tracking is enabled)
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Optional, Union

import cv2
import numpy as np
from PIL import Image
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from aux_head import AuxHead, OCC_CLASSES  # noqa: E402

ImageSource = Union[str, Path, np.ndarray, Image.Image]


def _load_aux(aux_pt, device):
    ckpt = torch.load(str(aux_pt), map_location=device, weights_only=False)
    classes = ckpt.get("occ_classes", list(OCC_CLASSES))
    model = AuxHead(num_classes=len(classes), pretrained=False).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    return model


def _normalize_to_bgr(source):
    """Return a uint8 HxWx3 BGR ndarray. Single decode shared by YOLO + aux."""
    if isinstance(source, (str, Path)):
        img = cv2.imread(str(source), cv2.IMREAD_COLOR)
        if img is None:
            raise FileNotFoundError(source)
        return img
    if isinstance(source, np.ndarray):
        if source.ndim != 3 or source.shape[2] != 3:
            raise ValueError(f"expected HxWx3, got {source.shape}")
        return source
    if isinstance(source, Image.Image):
        rgb = np.asarray(source.convert("RGB"))
        return np.ascontiguousarray(rgb[:, :, ::-1])
    raise TypeError(f"Unsupported source type: {type(source)}")


class _Timer:
    """Cheap per-stage timer. Synchronizes CUDA when device is cuda."""
    def __init__(self, enabled, device):
        self.enabled = enabled
        self.device = device
        self.records = {}
        self.last = {}

    def tic(self, name):
        if not self.enabled:
            return
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        self.last[name] = time.perf_counter()

    def toc(self, name):
        if not self.enabled:
            return
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        dt = time.perf_counter() - self.last.pop(name)
        self.records[name] = self.records.get(name, 0.0) + dt

    def dump(self, n_frames):
        if not self.enabled:
            return ""
        if n_frames <= 0:
            n_frames = 1
        parts = [f"avg over {n_frames} frame(s):"]
        for k, v in sorted(self.records.items(), key=lambda x: -x[1]):
            parts.append(f"  {k:<22s} {v*1000/n_frames:7.2f} ms/frame")
        return "\n".join(parts)


class CombinedInfer:
    """Wraps YOLO seg, YOLO pose, and AuxHead under one call."""

    def __init__(self, seg_pt, pose_pt=None, aux_pt=None,
                 device="cuda:0", imgsz=1280, aux_imgsz=224,
                 aux_pad_frac=0.30, conf=0.25, iou_match=0.5,
                 half=True, track=False, tracker="bytetrack.yaml",
                 profile=False):
        try:
            from ultralytics import YOLO
        except ImportError as e:
            raise ImportError("pip install ultralytics") from e

        self.device = torch.device(device) if isinstance(device, str) else device
        self.imgsz = imgsz
        self.aux_imgsz = aux_imgsz
        self.aux_pad_frac = aux_pad_frac
        self.conf = conf
        self.iou_match = iou_match
        self.half = bool(half) and self.device.type == "cuda"
        self.track = bool(track)
        self.tracker = tracker
        self._n_frames = 0
        self.timer = _Timer(enabled=profile, device=self.device)

        self.seg = YOLO(str(seg_pt))
        self.pose = YOLO(str(pose_pt)) if pose_pt else None
        self.aux = _load_aux(aux_pt, self.device) if aux_pt else None

        # Conv+BN fusion (small but free win).
        try:
            self.seg.fuse()
        except Exception:
            pass
        if self.pose is not None:
            try:
                self.pose.fuse()
            except Exception:
                pass

        # Promote aux head to FP16 if requested.
        if self.aux is not None and self.half:
            self.aux = self.aux.half()

        # ImageNet normalize for the aux head.
        self._mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        self._std = np.array([0.229, 0.224, 0.225], dtype=np.float32)

    def reset_tracker(self):
        """Reset tracker state between separate sources (videos)."""
        try:
            if hasattr(self.seg, "predictor") and self.seg.predictor is not None:
                trackers = getattr(self.seg.predictor, "trackers", None)
                if trackers:
                    for tr in trackers:
                        if hasattr(tr, "reset"):
                            tr.reset()
                else:
                    self.seg.predictor = None
        except Exception:
            self.seg.predictor = None

    def _build_aux_batch_np(self, img_bgr, bboxes_xywh):
        H, W = img_bgr.shape[:2]
        crops = []
        for x, y, w, h in bboxes_xywh:
            pad_x = w * self.aux_pad_frac
            pad_y = h * self.aux_pad_frac
            l = int(max(0, round(x - pad_x)))
            t = int(max(0, round(y - pad_y)))
            r = int(min(W, round(x + w + pad_x)))
            b = int(min(H, round(y + h + pad_y)))
            if r <= l or b <= t:
                crops.append(None)
                continue
            cr = img_bgr[t:b, l:r]
            cr = cv2.cvtColor(cr, cv2.COLOR_BGR2RGB)
            cr = cv2.resize(cr, (self.aux_imgsz, self.aux_imgsz),
                            interpolation=cv2.INTER_LINEAR)
            arr = cr.astype(np.float32) * (1.0 / 255.0)
            arr = (arr - self._mean) / self._std
            crops.append(arr.transpose(2, 0, 1))
        valid_idxs = [i for i, c in enumerate(crops) if c is not None]
        if not valid_idxs:
            return torch.zeros(0, 3, self.aux_imgsz, self.aux_imgsz,
                               device=self.device), []
        stacked = np.stack([crops[i] for i in valid_idxs])
        t = torch.from_numpy(stacked).to(self.device, non_blocking=True)
        if self.half:
            t = t.half()
        return t, valid_idxs

    @staticmethod
    def _iou_xyxy(a, b):
        ax1, ay1, ax2, ay2 = a
        bx1, by1, bx2, by2 = b
        x1 = max(ax1, bx1); y1 = max(ay1, by1)
        x2 = min(ax2, bx2); y2 = min(ay2, by2)
        iw = max(0.0, x2 - x1); ih = max(0.0, y2 - y1)
        inter = iw * ih
        if inter == 0:
            return 0.0
        a_area = (ax2 - ax1) * (ay2 - ay1)
        b_area = (bx2 - bx1) * (by2 - by1)
        return inter / (a_area + b_area - inter + 1e-9)

    @staticmethod
    def _polygon_from_seg_result(seg_res, idx):
        if seg_res.masks is None:
            return None
        try:
            polys = seg_res.masks.xy
            if polys is None or idx >= len(polys):
                return None
            poly = polys[idx]
            if poly is None or len(poly) < 3:
                return None
            return [[int(round(float(x))), int(round(float(y)))] for x, y in poly]
        except (AttributeError, IndexError):
            return None

    def __call__(self, source):
        return self.predict(source)

    def predict(self, source):
        self._n_frames += 1
        T = self.timer
        T.tic("decode")
        img_bgr = _normalize_to_bgr(source)
        T.toc("decode")

        # --- YOLO seg (track or predict) ---
        T.tic("yolo_seg")
        if self.track:
            seg_res = self.seg.track(img_bgr, imgsz=self.imgsz, conf=self.conf,
                                     persist=True, tracker=self.tracker,
                                     half=self.half, verbose=False)[0]
        else:
            seg_res = self.seg.predict(img_bgr, imgsz=self.imgsz, conf=self.conf,
                                       half=self.half, verbose=False)[0]
        T.toc("yolo_seg")

        if seg_res.boxes is None or len(seg_res.boxes) == 0:
            return []

        boxes_xyxy = seg_res.boxes.xyxy.cpu().numpy()
        confs = seg_res.boxes.conf.cpu().numpy()
        n_dets = len(boxes_xyxy)

        track_ids = [None] * n_dets
        if self.track and getattr(seg_res.boxes, "id", None) is not None:
            try:
                ids = seg_res.boxes.id.cpu().numpy().astype(int)
                for i, tid in enumerate(ids):
                    track_ids[i] = int(tid)
            except Exception:
                pass

        # --- YOLO pose (optional) ---
        T.tic("yolo_pose")
        kpts_per_seg_box = [None] * n_dets
        if self.pose is not None:
            pose_res = self.pose.predict(img_bgr, imgsz=self.imgsz, conf=self.conf,
                                         half=self.half, verbose=False)[0]
            if (pose_res.boxes is not None and len(pose_res.boxes) > 0
                    and pose_res.keypoints is not None):
                pose_boxes = pose_res.boxes.xyxy.cpu().numpy()
                # keypoints.data is (N, K, 3) where channel 2 is per-kpt confidence
                pose_kpts = pose_res.keypoints.data.cpu().numpy()
                for i, sb in enumerate(boxes_xyxy):
                    best_iou = 0.0
                    best_j = -1
                    for j, pb in enumerate(pose_boxes):
                        iou = self._iou_xyxy(sb, pb)
                        if iou > best_iou:
                            best_iou = iou
                            best_j = j
                    if best_iou >= self.iou_match and best_j >= 0:
                        kpts_per_seg_box[i] = pose_kpts[best_j]
        T.toc("yolo_pose")

        # --- AuxHead (optional) ---
        T.tic("aux_head")
        bboxes_xywh = [
            (float(x1), float(y1), float(x2 - x1), float(y2 - y1))
            for x1, y1, x2, y2 in boxes_xyxy
        ]
        occ_class = [None] * n_dets
        occ_class_conf = [0.0] * n_dets
        visfrac = [None] * n_dets
        if self.aux is not None and n_dets:
            batch, valid_idxs = self._build_aux_batch_np(img_bgr, bboxes_xywh)
            if batch.shape[0] > 0:
                with torch.no_grad():
                    out = self.aux(batch)
                cls_probs = torch.softmax(out["cls_logits"].float(), dim=-1).cpu().numpy()
                cls_ids = cls_probs.argmax(axis=-1)
                visf = out["visfrac"].float().cpu().numpy()
                for k, i in enumerate(valid_idxs):
                    occ_class[i] = OCC_CLASSES[int(cls_ids[k])]
                    occ_class_conf[i] = float(cls_probs[k, int(cls_ids[k])])
                    visfrac[i] = float(visf[k])
        T.toc("aux_head")

        # --- assemble output ---
        T.tic("assemble")
        results = []
        for i in range(n_dets):
            x1, y1, x2, y2 = (float(v) for v in boxes_xyxy[i])
            kpts = (None if kpts_per_seg_box[i] is None
                    else [[float(x), float(y), float(c)]
                          for x, y, c in kpts_per_seg_box[i]])
            entry = {
                "bbox": [x1, y1, x2 - x1, y2 - y1],
                "det_conf": float(confs[i]),
                "mask_poly": self._polygon_from_seg_result(seg_res, i),
                "kpts": kpts,
                "occ_class": occ_class[i],
                "occ_class_conf": occ_class_conf[i],
                "visfrac": visfrac[i],
                "track_id": track_ids[i],
            }
            results.append(entry)
        T.toc("assemble")
        return results

    def profile_dump(self):
        return self.timer.dump(self._n_frames)


def _serialize_for_json(results):
    return [dict(r) for r in results]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seg", required=True)
    ap.add_argument("--pose", default=None)
    ap.add_argument("--aux", default=None)
    ap.add_argument("--image", required=True)
    ap.add_argument("--out", default=None)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--imgsz", type=int, default=1280)
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--no-half", dest="half", action="store_false",
                    help="Force FP32 (slower; default FP16 on CUDA).")
    ap.set_defaults(half=True)
    ap.add_argument("--profile", action="store_true")
    args = ap.parse_args()

    infer = CombinedInfer(
        seg_pt=args.seg, pose_pt=args.pose, aux_pt=args.aux,
        device=args.device, imgsz=args.imgsz, conf=args.conf,
        half=args.half, profile=args.profile,
    )
    results = infer(args.image)
    print(f"Detected {len(results)} bolls in {args.image}")
    for i, r in enumerate(results):
        kpt_str = "no-pose"
        if r["kpts"] is not None:
            kpt_str = " | ".join(f"({x:.0f},{y:.0f},c={c:.2f})"
                                  for x, y, c in r["kpts"])
        occ_str = "no-aux"
        if r["occ_class"] is not None:
            occ_str = (f"{r['occ_class']}@{r['occ_class_conf']:.2f}  "
                       f"vis={r['visfrac']:.2f}")
        bx = r["bbox"]
        n_pts = 0 if r["mask_poly"] is None else len(r["mask_poly"])
        tid = "" if r.get("track_id") is None else f"  id={r['track_id']}"
        print(f"  #{i}  bbox=[{bx[0]:.0f},{bx[1]:.0f},{bx[2]:.0f},{bx[3]:.0f}]"
              f"{tid}  det={r['det_conf']:.2f}  poly={n_pts}pts  "
              f"kpts=[{kpt_str}]  {occ_str}")

    if args.profile:
        print("\n[profile] " + infer.profile_dump())

    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(_serialize_for_json(results), indent=2))
        print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
