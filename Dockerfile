# Deploy image for Hugging Face Spaces (free CPU basic: 2 vCPU / 16 GB RAM).
# Runtime data (windows.npy, siamese.pt, fusion.pt) is fetched by start.sh
# from GitHub Releases at startup — kept out of git, so the build stays small.
FROM python:3.10-slim

WORKDIR /app

# curl is needed by start.sh to fetch runtime data
RUN apt-get update \
    && apt-get install -y --no-install-recommends curl \
    && rm -rf /var/lib/apt/lists/*

# Install CPU-only PyTorch first (smaller than the CUDA default), then the rest
COPY requirements.txt .
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu \
    && pip install --no-cache-dir -r requirements.txt

COPY . .

RUN chmod +x start.sh

# Hugging Face Spaces exposes the app on port 7860
EXPOSE 7860
CMD ["sh", "start.sh"]
