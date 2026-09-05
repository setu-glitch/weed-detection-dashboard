# Deployment

Checked against platform documentation in September 2026. Free tiers change
often — confirm the current limits before you commit to one.

## Which platform

| Platform | Free? | Fit for this app |
| --- | --- | --- |
| Streamlit Community Cloud | Yes | Best first choice. Purpose-built for Streamlit, no card, deploys from GitHub. |
| Google Cloud Run | Free monthly allowance | Best if you want a container. Scales to zero, so a demo costs nothing. Card required. |
| Oracle Cloud Always Free | Yes | Most headroom (2 OCPU / 12 GB Arm since June 2026). Card required, capacity can be scarce, manual setup. |
| Hugging Face Spaces | No longer | Creating a Space that runs on compute now needs a PRO plan. Only static Spaces are free. |
| Render free web service | Technically | 512 MB RAM does not hold PyTorch plus a loaded model. Expect out-of-memory kills. |

The whole memory question is PyTorch. Importing torch costs roughly 300–500 MB
of RSS before a single weight is read; a nano checkpoint adds about 10–60 MB.
Streamlit itself sits around 150 MB. Budget 700 MB–1 GB per running app.

---

## Option 1 — Streamlit Community Cloud

Resource limits, as documented: 0.078–2 CPU cores, 690 MB–2.7 GB memory, up to
50 GB storage. Apps sleep when idle and wake on the next visit.

**1. Push to a public GitHub repository.**

The nano checkpoints are a few megabytes each, so commit them directly — no Git
LFS needed. `.gitignore` ships with the `models/*.pt` line commented out for
exactly this reason.

```bash
git init
git add .
git commit -m "Weed detection dashboard"
git remote add origin https://github.com/<you>/weed-detection-dashboard.git
git push -u origin main
```

Confirm the weights actually made it:

```bash
git ls-files models/
```

**2. Deploy.** At <https://share.streamlit.io>, sign in with GitHub, click
*Create app*, select the repository, set the main file to `app.py`, and deploy.
The first build takes several minutes because PyTorch is large.

**3. What the two config files do.**

- `requirements.txt` starts with `--extra-index-url https://download.pytorch.org/whl/cpu`.
  Without it, pip installs the CUDA build of torch and the build runs out of
  disk. If the resolver still picks the CUDA wheel, pin explicitly instead:
  `torch==2.8.0+cpu` and `torchvision==0.23.0+cpu`.
- `packages.txt` installs `libgl1` and `libglib2.0-0`. Ultralytics imports
  OpenCV, which fails on Debian slim images without them. The symptom is
  `ImportError: libGL.so.1: cannot open shared object file`.

**4. Keep memory down.** The app already loads models lazily and caches one
instance per file, so only the models you actually run are resident. If you hit
the limit anyway, trim `MODEL_REGISTRY` in `utils/config.py` to two or three
models for the public demo, and lower `MAX_INPUT_EDGE` from 1600 to 1024.

Streamlit grants extra resources to educational and research apps — worth an
email to support with the TUM affiliation if the default allocation is tight.

---

## Option 2 — Google Cloud Run

Free allowance covers roughly 2 million requests and 180,000 vCPU-seconds a
month, and the service scales to zero between visits. A card is required but a
demo of this size stays inside the allowance.

```bash
gcloud run deploy weed-dashboard \
  --source . \
  --region europe-west3 \
  --memory 2Gi \
  --cpu 2 \
  --timeout 300 \
  --allow-unauthenticated
```

Cloud Run builds the included `Dockerfile` and injects `PORT`, which the
`CMD` already reads. Two settings matter: `--memory 2Gi` (the 512 MB default
kills the container during torch import) and `--cpu 2` (nano inference on one
shared vCPU is slow).

Cold starts are noticeable — the container has to import torch before serving.
Setting `--min-instances 1` removes that, but leaves the free allowance.

---

## Option 3 — Oracle Cloud Always Free

A full VM, so nothing is constrained but your own setup. Always Free Ampere A1
was reduced to 2 OCPU / 12 GB in June 2026; still far more than this app needs.
The Arm architecture is fine — PyTorch publishes aarch64 wheels.

```bash
sudo apt update && sudo apt install -y python3-venv libgl1 libglib2.0-0
git clone https://github.com/<you>/weed-detection-dashboard.git
cd weed-detection-dashboard
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# Keep it running across reboots and SSH disconnects.
sudo tee /etc/systemd/system/weed-dashboard.service > /dev/null << 'UNIT'
[Unit]
Description=Weed detection dashboard
After=network.target

[Service]
User=ubuntu
WorkingDirectory=/home/ubuntu/weed-detection-dashboard
Environment=YOLO_CONFIG_DIR=/tmp
ExecStart=/home/ubuntu/weed-detection-dashboard/.venv/bin/streamlit run app.py \
  --server.port=8501 --server.address=0.0.0.0 --server.headless=true
Restart=always

[Install]
WantedBy=multi-user.target
UNIT

sudo systemctl enable --now weed-dashboard
```

Open port 8501 in both the OCI security list and the instance firewall
(`sudo iptables -I INPUT -p tcp --dport 8501 -j ACCEPT` on Oracle's Ubuntu
images, which ship with restrictive rules). For a public demonstration, put
Caddy or nginx in front for HTTPS on port 443.

---

## If the image is still too large

Export the checkpoints to ONNX and drop PyTorch entirely. `onnxruntime` is about
50 MB against roughly 800 MB for torch plus torchvision, and CPU inference is
usually faster.

```python
from ultralytics import YOLO
YOLO("models/yolov11n.pt").export(format="onnx", imgsz=640, simplify=True)
```

This needs a second inference backend in `utils/model_manager.py` that runs the
ONNX session and applies non-maximum suppression itself, since Ultralytics'
post-processing goes away with torch. Worth it only if the platform's build
limits leave no alternative.

---

## Before you show it to anyone

- Replace `assets/dummy_monitoring.jpg` with a real FarmBot capture.
- Fill in the remaining values in `benchmarks.json`, or leave them null so they
  render as dashes rather than as invented numbers.
- Check that `data.yaml` matches the checkpoints — the Dataset page shows a
  warning if the class names disagree.
- Run one detection end to end on the deployed instance. A missing system
  library only shows up when a model is actually loaded, not at startup.
