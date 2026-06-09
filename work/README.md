# work/ — SAM 2 pipeline outputs (not in git)

One folder per recording, produced by the 5-stage pipeline in
`pipeline/` (see its README). Typical contents of `work/<name>/`:

- `frames/` — extracted color JPEGs
- `depth/` — per-frame 16-bit depth as `.npy`
- `metadata.json` — intrinsics + depth scale from the bag
- `anchor.json` — chosen anchor frame + boll click points
- `masks/<frame_idx>/<boll_id>.png` — SAM 2 masks
- `measurements_per_frame.csv` — per-(frame, boll) height/width
- `annotations.json`, `report/` — aggregated results

Everything here is regenerable from the corresponding `.bag` in `bags/`,
so only this README is tracked.
