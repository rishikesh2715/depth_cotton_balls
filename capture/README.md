# capture/ — RealSense recording & analysis

- `record_rgbd.py` — records synchronized color + 16-bit depth to a `.bag`
  file with intrinsics preserved. Run with `--help` for options; in the
  preview window: `r` record, `s` snapshot, `q` quit. Move finished
  recordings into `bags/`.
- `analyze_rgbd.py` — replays a `.bag` and runs a segmentation model on
  each frame to measure boll dimensions from depth
  (`--bag <file> --model <weights.pt>`).
- `metrics.csv` — early distance-estimation accuracy check (known
  distances vs estimates from depth).

```bash
pip install pyrealsense2 opencv-python numpy ultralytics
```
