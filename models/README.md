# Model checkpoints

Place the trained YOLO weights in this directory using these exact filenames:

```
yolov8n.pt
yolov9n.pt
yolov10n.pt
yolov11n.pt
yolov12n.pt
```

The dashboard reads the registry in `utils/config.py`. Any checkpoint listed
there but missing from this folder is shown as "not installed" and excluded from
the model selector; the rest of the app keeps working.

To register a sixth model, add one `ModelSpec` entry to `MODEL_REGISTRY` in
`utils/config.py` and drop the file here. No UI code changes are needed.

Checkpoints are not tracked in version control. On a free-tier host, either
commit them with Git LFS or download them on first boot.
