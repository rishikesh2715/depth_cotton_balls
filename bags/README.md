# bags/ — raw RealSense recordings (not in git)

`.bag` files recorded with `capture/record_rgbd.py` (synchronized color +
16-bit depth + intrinsics). Multi-GB each, so they are gitignored — keep
them backed up elsewhere (lab NAS / external drive).

Naming: `rgbd_recording_YYYYMMDD_HHMMSS.bag`.

These are the single source of truth: every `work/<name>/` folder can be
regenerated from a bag via `pipeline/01_extract_frames.py`.
