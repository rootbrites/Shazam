#!/bin/sh
# Fetches runtime data (kept out of git — see data/.gitignore) and launches
# the server on the port provided by the platform ($PORT on HF Spaces/Render).
set -e

BASE="https://github.com/rootbrites/Shazam/releases/download/data-v1"
mkdir -p data

for f in windows.npy siamese.pt fusion.pt fusion_config.json; do
  if [ ! -f "data/$f" ]; then
    echo "[start] fetching $f ..."
    curl -fsSL --retry 3 -o "data/$f" "$BASE/$f"
  fi
done

echo "[start] data ready; launching on port ${PORT:-7860}"
exec uvicorn main:app --host 0.0.0.0 --port "${PORT:-7860}"
