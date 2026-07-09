"""
Flask web server for GANomaly inference.
Serves the frontend and provides /api/predict endpoint.
"""

import os, sys, io, base64, time
import torch
import numpy as np
from flask import Flask, request, jsonify, send_from_directory
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import config
from models.ganomaly import Generator
from data.preprocessing import load_and_preprocess, get_test_transforms, denormalize
from utils import anomaly_score

app = Flask(__name__, static_folder="frontend", static_url_path="")

# ── Load model once at startup ────────────────────────────────
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
ckpt_path = os.path.join(config.CHECKPOINT_DIR, "best.pth")

G = Generator(config.CHANNELS, config.LATENT_DIM, config.FEATURE_MAPS).to(device)
checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)
G.load_state_dict(checkpoint['G'])
G.eval()

# Load calibrated threshold
thr_path = os.path.join(config.CHECKPOINT_DIR, "threshold.txt")
THRESHOLD = float(open(thr_path).read().strip()) if os.path.exists(thr_path) else config.ANOMALY_THRESHOLD

print(f"[Server] Model loaded on {device} | Threshold: {THRESHOLD:.4f}")


def tensor_to_base64(tensor_img):
    """Convert a [C,H,W] tensor in [0,1] range to base64 PNG string."""
    img_np = tensor_img.permute(1, 2, 0).numpy()
    img_np = (np.clip(img_np, 0, 1) * 255).astype(np.uint8)
    pil_img = Image.fromarray(img_np)
    # Upscale for better visibility in the UI
    pil_img = pil_img.resize((256, 256), Image.NEAREST)
    buf = io.BytesIO()
    pil_img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def error_map_to_base64(orig_tensor, recon_tensor):
    """Generate a heatmap error map and return as base64 PNG."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    error = torch.abs(orig_tensor - recon_tensor).mean(dim=0).numpy()  # [H, W]
    fig, ax = plt.subplots(1, 1, figsize=(2.56, 2.56), dpi=100)
    ax.imshow(error, cmap='hot', interpolation='bilinear')
    ax.axis('off')
    fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
    buf = io.BytesIO()
    fig.savefig(buf, format='PNG', bbox_inches='tight', pad_inches=0)
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.getvalue()).decode("utf-8")


@app.route("/")
def index():
    return send_from_directory("frontend", "index.html")


@app.route("/api/predict", methods=["POST"])
def predict():
    if "image" not in request.files:
        return jsonify({"error": "No image file provided"}), 400

    file = request.files["image"]
    if file.filename == "":
        return jsonify({"error": "Empty filename"}), 400

    # Save temp file for preprocessing (OpenCV needs file path)
    tmp_dir = os.path.join(config.OUTPUT_DIR, "tmp")
    os.makedirs(tmp_dir, exist_ok=True)
    tmp_path = os.path.join(tmp_dir, f"upload_{int(time.time()*1000)}.png")

    try:
        file.save(tmp_path)

        # Preprocess and run inference
        img = load_and_preprocess(tmp_path, config.IMAGE_SIZE)
        tensor = get_test_transforms(config.IMAGE_SIZE)(img).unsqueeze(0).to(device)

        with torch.no_grad():
            recon, z, z_hat = G(tensor)
            score = anomaly_score(z, z_hat).item()

        is_anomalous = score >= THRESHOLD

        # Compute confidence: how far the score is from the threshold
        # Normalized to 0-100% range using a sigmoid-like mapping
        distance = abs(score - THRESHOLD)
        raw_confidence = 1.0 - np.exp(-distance / (THRESHOLD * 0.5 + 1e-8))
        confidence = min(raw_confidence * 100, 99.9)

        # Generate images for display
        orig_display = denormalize(tensor[0].cpu())
        recon_display = denormalize(recon[0].cpu())

        result = {
            "score": round(score, 6),
            "threshold": round(THRESHOLD, 6),
            "is_anomalous": bool(is_anomalous),
            "verdict": "DR SUSPECTED" if is_anomalous else "HEALTHY",
            "action": "Refer to Stage 2 for grading" if is_anomalous else "Routine follow-up in 12 months",
            "confidence": round(confidence, 2),
            "original_b64": tensor_to_base64(orig_display),
            "reconstructed_b64": tensor_to_base64(recon_display),
            "error_map_b64": error_map_to_base64(orig_display, recon_display),
            "device": str(device),
        }

        return jsonify(result)

    except Exception as e:
        return jsonify({"error": str(e)}), 500

    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)
