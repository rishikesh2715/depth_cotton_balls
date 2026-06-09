"""
aux_head.py — Auxiliary head for the cotton-boll loss function.

Covers the two components of the six-term loss the YOLO seg/pose models do
not natively predict:

  L_occ      — occluded_by classifier   (4-way: none / leaf / stem / other)
  L_visfrac  — visibility_fraction      (Huber regression on 1 - occlusion_pct)

Architecture (small by design — 274 training instances):

    Input crop (224x224 RGB, ImageNet-normalized)
        |
        V
    ResNet-18 (ImageNet pre-trained, fc removed)
        |
        +--> classifier_head (Linear 512 -> 4)        -> CrossEntropy
        |
        +--> visfrac_head    (Linear 512 -> 1, sigm)  -> SmoothL1 (Huber)

The two heads share the backbone. Total loss is

    L = w_occ * CE(cls_logits, cls_label) + w_vis * Huber(visfrac_pred, 1 - occ_pct)

Inference adapter (see infer_aux.py) takes a YOLO seg detection, crops the
boll, and runs this model to fill in the per-boll occlusion / visibility
fields the YOLO model can't produce.

Module exports:
    AuxHead                 — nn.Module
    BollCropDataset         — torch Dataset that reads merged_dataset_v3
    OCC_CLASSES             — ('none', 'leaf', 'stem', 'other') in label-id order
    coco_attrs_to_targets() — pure helper; converts a COCO `attributes` dict
                              into (cls_id, visfrac) for any caller that wants
                              to materialize labels without instantiating the
                              dataset.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
from torchvision import models, transforms


# ---------------------------------------------------------------------------
# Class scheme
# ---------------------------------------------------------------------------
OCC_CLASSES: Tuple[str, ...] = ("none", "leaf", "stem", "other")
OCC_TO_ID = {c: i for i, c in enumerate(OCC_CLASSES)}

# Map raw `occluded_by` strings (v3 + v4) to our 4-class scheme.
# v3 used "frame_edge_cutoff"; v4 (combined_all_sequential) uses "frame_edge".
_OCC_RAW_MAP = {
    "none": "none",
    "leaf": "leaf",
    "stem": "stem",
    "frame_edge_cutoff": "other",   # v3
    "frame_edge":        "other",   # v4
    "other_boll":        "other",
}


def coco_attrs_to_targets(attrs_or_ann: dict) -> Optional[Tuple[int, float]]:
    """Convert a COCO annotation (either v3's `attributes` dict OR a v4
    annotation dict where occlusion fields are top-level) into (cls_id, visfrac).

    Returns None if the required fields aren't present.

    visfrac is in [0, 1]; we compute it as 1 - occlusion_pct so the head's
    sigmoid output represents a meaningful "fraction of boll visible to
    the camera".
    """
    if not isinstance(attrs_or_ann, dict):
        return None
    occ_by_raw = attrs_or_ann.get("occluded_by")
    occ_pct = attrs_or_ann.get("occlusion_pct")
    is_occ = attrs_or_ann.get("is_occluded")

    if occ_by_raw is None or occ_pct is None:
        return None

    # Resolve label noise: is_occluded=True with occluded_by="none"
    # (observed in 1 instance) -> treat as "other".
    if is_occ is True and occ_by_raw == "none":
        cls_name = "other"
    else:
        cls_name = _OCC_RAW_MAP.get(occ_by_raw, "other")

    try:
        visfrac = float(1.0 - float(occ_pct))
    except (TypeError, ValueError):
        return None
    visfrac = max(0.0, min(1.0, visfrac))
    return OCC_TO_ID[cls_name], visfrac


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------
@dataclass
class _Record:
    image_path: Path
    bbox: Tuple[float, float, float, float]   # x, y, w, h in pixels
    cls_id: int
    visfrac: float
    image_wh: Tuple[int, int]                  # full-image (W, H)
    boll_id: Optional[int] = None              # for debugging only


class BollCropDataset(Dataset):
    """Per-instance cotton-boll crop dataset built from merged_dataset_v3.

    Each item is one annotation. The crop is the bbox padded by `pad_frac`
    on each side, clipped to image bounds, resized to `imgsz` x `imgsz`,
    and ImageNet-normalized.

    Args:
        v3_root: Path to merged_dataset_v3 (containing backdown/, frontdown/).
        split:   "train" or "val". Selects which COCO files to read
                 (backdown/<split>.json + frontdown/<split>.json).
        imgsz:   Output crop size in pixels.
        pad_frac: Fraction of bbox to pad on each side (0.30 = 30%).
        augment: Enable training augmentation (color jitter + h-flip).
    """
    def __init__(
        self,
        v3_root: Path | str,
        split: str = "train",
        imgsz: int = 224,
        pad_frac: float = 0.30,
        augment: bool = False,
    ) -> None:
        self.v3_root = Path(v3_root)
        self.split = split
        self.imgsz = imgsz
        self.pad_frac = pad_frac
        self.augment = augment

        self.records: List[_Record] = self._build_records()

        # Standard ImageNet normalization for a ResNet-18 trunk.
        self._normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        )
        # Color jitter / hflip applied at the crop level for train only.
        if augment:
            self._aug = transforms.Compose([
                transforms.ColorJitter(brightness=0.2, contrast=0.2,
                                       saturation=0.15, hue=0.02),
                transforms.RandomHorizontalFlip(p=0.5),
            ])
        else:
            self._aug = None

    # ----- record assembly -----
    def _build_records(self) -> List[_Record]:
        out: List[_Record] = []
        skipped_no_attrs = 0
        skipped_no_image = 0
        for rec_dir in ("backdown", "frontdown"):
            coco_path = self.v3_root / rec_dir / f"{self.split}.json"
            if not coco_path.exists():
                continue
            data = json.loads(coco_path.read_text())
            images_by_id = {im["id"]: im for im in data["images"]}
            img_dir = self.v3_root / rec_dir / "images"
            for ann in data.get("annotations", []):
                tgt = coco_attrs_to_targets(ann.get("attributes") or {})
                if tgt is None:
                    skipped_no_attrs += 1
                    continue
                cls_id, visfrac = tgt
                im = images_by_id.get(ann["image_id"])
                if im is None:
                    continue
                # Use original_name to bypass the src1_ phantom prefix.
                fname = im.get("original_name") or im["file_name"]
                if fname.startswith("src1_"):
                    fname = fname[len("src1_"):]
                img_path = img_dir / fname
                if not img_path.exists():
                    skipped_no_image += 1
                    continue
                bbox = tuple(float(v) for v in ann["bbox"])  # (x,y,w,h)
                out.append(_Record(
                    image_path=img_path,
                    bbox=bbox,
                    cls_id=cls_id,
                    visfrac=visfrac,
                    image_wh=(im["width"], im["height"]),
                    boll_id=(ann.get("attributes") or {}).get("boll_id"),
                ))
        if skipped_no_attrs:
            print(f"[BollCropDataset/{self.split}] skipped {skipped_no_attrs} anns "
                  f"missing attributes")
        if skipped_no_image:
            print(f"[BollCropDataset/{self.split}] skipped {skipped_no_image} anns "
                  f"with missing image files")
        return out

    # ----- per-item crop -----
    def _crop_padded(self, img: Image.Image, bbox) -> Image.Image:
        x, y, w, h = bbox
        pad_x = w * self.pad_frac
        pad_y = h * self.pad_frac
        W, H = img.size
        l = max(0.0, x - pad_x)
        t = max(0.0, y - pad_y)
        r = min(float(W), x + w + pad_x)
        b = min(float(H), y + h + pad_y)
        # PIL crop wants (left, top, right, bottom) ints
        return img.crop((int(round(l)), int(round(t)),
                         int(round(r)), int(round(b))))

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int):
        r = self.records[idx]
        img = Image.open(r.image_path).convert("RGB")
        crop = self._crop_padded(img, r.bbox)
        if self._aug is not None:
            crop = self._aug(crop)
        crop = crop.resize((self.imgsz, self.imgsz), Image.BILINEAR)
        # PIL -> tensor [0,1] CHW
        arr = np.asarray(crop, dtype=np.float32) / 255.0
        ten = torch.from_numpy(arr).permute(2, 0, 1).contiguous()
        ten = self._normalize(ten)
        return {
            "crop": ten,
            "cls_id": torch.tensor(r.cls_id, dtype=torch.long),
            "visfrac": torch.tensor(r.visfrac, dtype=torch.float32),
            "boll_id": r.boll_id if r.boll_id is not None else -1,
        }

    # ----- diagnostics -----
    def class_counts(self) -> dict:
        from collections import Counter
        c = Counter(r.cls_id for r in self.records)
        return {OCC_CLASSES[i]: c.get(i, 0) for i in range(len(OCC_CLASSES))}


# ---------------------------------------------------------------------------
# v4 dataset (combined_all_sequential)
# ---------------------------------------------------------------------------
class BollCropDatasetV4(BollCropDataset):
    """Same crop pipeline as BollCropDataset, but reads the v4 schema:

      - Single COCO JSON per split (output_coco_dir/train.json + val.json)
      - file_name carries the camera subfolder ("track/frame_xxx.jpg")
      - Occlusion fields are top-level on each annotation (no `attributes` wrap)
      - `occluded_by` uses "frame_edge" instead of v3's "frame_edge_cutoff"

    Args:
        images_root: directory containing back/, front/, track/ image subfolders.
        coco_dir:    directory containing train.json + val.json.
        split:       "train" or "val".
        imgsz, pad_frac, augment: same as v3.
    """
    def __init__(self,
                 images_root: Path | str,
                 coco_dir: Path | str,
                 split: str = "train",
                 imgsz: int = 224,
                 pad_frac: float = 0.30,
                 augment: bool = False) -> None:
        # Skip the v3-style __init__ that derives v3_root + builds records;
        # we override _build_records() with the v4 layout.
        self.images_root = Path(images_root)
        self.coco_dir = Path(coco_dir)
        self.split = split
        self.imgsz = imgsz
        self.pad_frac = pad_frac
        self.augment = augment

        self.records: List[_Record] = self._build_records_v4()

        self._normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        if augment:
            self._aug = transforms.Compose([
                transforms.ColorJitter(brightness=0.2, contrast=0.2,
                                       saturation=0.15, hue=0.02),
                transforms.RandomHorizontalFlip(p=0.5),
            ])
        else:
            self._aug = None

    def _build_records_v4(self) -> List[_Record]:
        out: List[_Record] = []
        skipped_no_attrs = 0
        skipped_no_image = 0
        coco_path = self.coco_dir / f"{self.split}.json"
        if not coco_path.exists():
            return out
        data = json.loads(coco_path.read_text())
        images_by_id = {im["id"]: im for im in data["images"]}
        for ann in data.get("annotations", []):
            # v4: occlusion fields are top-level on the annotation.
            tgt = coco_attrs_to_targets(ann)
            if tgt is None:
                skipped_no_attrs += 1
                continue
            cls_id, visfrac = tgt
            im = images_by_id.get(ann["image_id"])
            if im is None:
                continue
            img_path = self.images_root / im["file_name"]
            if not img_path.exists():
                skipped_no_image += 1
                continue
            bbox = tuple(float(v) for v in ann["bbox"])
            out.append(_Record(
                image_path=img_path,
                bbox=bbox,
                cls_id=cls_id,
                visfrac=visfrac,
                image_wh=(im["width"], im["height"]),
                boll_id=ann.get("boll_id"),
            ))
        if skipped_no_attrs:
            print(f"[BollCropDatasetV4/{self.split}] skipped {skipped_no_attrs} "
                  f"anns missing occlusion fields")
        if skipped_no_image:
            print(f"[BollCropDatasetV4/{self.split}] skipped {skipped_no_image} "
                  f"anns with missing image files")
        return out


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------
class AuxHead(nn.Module):
    """ResNet-18 backbone + (occlusion classifier, visfrac regressor) heads.

    Designed to be loaded from a single checkpoint and evaluated alongside
    the YOLO seg/pose models in infer_aux.py.
    """
    def __init__(self, num_classes: int = len(OCC_CLASSES),
                 pretrained: bool = True, dropout: float = 0.2) -> None:
        super().__init__()
        # Use the post-2022 weights API — falls back gracefully if the
        # specific enum name changes.
        if pretrained:
            try:
                weights = models.ResNet18_Weights.IMAGENET1K_V1
            except AttributeError:
                weights = "DEFAULT"
            backbone = models.resnet18(weights=weights)
        else:
            backbone = models.resnet18(weights=None)
        feat_dim = backbone.fc.in_features  # 512 for ResNet-18
        backbone.fc = nn.Identity()
        self.backbone = backbone

        self.cls_head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(feat_dim, num_classes),
        )
        self.visfrac_head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(feat_dim, 1),
        )

    def forward(self, x: torch.Tensor):
        feat = self.backbone(x)                                # (B, 512)
        cls_logits = self.cls_head(feat)                       # (B, K)
        visfrac = torch.sigmoid(self.visfrac_head(feat)).squeeze(-1)  # (B,)
        return {"cls_logits": cls_logits, "visfrac": visfrac, "feat": feat}


# ---------------------------------------------------------------------------
# Loss
# ---------------------------------------------------------------------------
class AuxLoss(nn.Module):
    """Weighted CE + Huber loss matching the six-term scheme.

    Class weights are derived from training-set frequencies if `class_counts`
    is supplied; otherwise uniform.
    """
    def __init__(self, w_occ: float = 1.0, w_vis: float = 1.0,
                 class_counts: Optional[dict] = None,
                 huber_beta: float = 0.05) -> None:
        super().__init__()
        self.w_occ = w_occ
        self.w_vis = w_vis
        if class_counts is None:
            self.class_weights = None
        else:
            counts = np.array([class_counts.get(c, 0)
                               for c in OCC_CLASSES], dtype=np.float64)
            counts = np.maximum(counts, 1.0)         # avoid div by zero
            inv = counts.sum() / counts
            inv = inv / inv.mean()                   # normalize to mean 1
            self.class_weights = torch.tensor(inv, dtype=torch.float32)
        self.huber_beta = huber_beta

    def forward(self, out: dict, batch: dict) -> dict:
        cls_logits = out["cls_logits"]
        visfrac_pred = out["visfrac"]
        cls_target = batch["cls_id"].to(cls_logits.device)
        visfrac_target = batch["visfrac"].to(visfrac_pred.device)

        cw = self.class_weights.to(cls_logits.device) if self.class_weights is not None else None
        l_occ = F.cross_entropy(cls_logits, cls_target, weight=cw)
        l_vis = F.smooth_l1_loss(visfrac_pred, visfrac_target, beta=self.huber_beta)
        total = self.w_occ * l_occ + self.w_vis * l_vis
        return {"loss": total, "l_occ": l_occ.detach(), "l_vis": l_vis.detach()}
