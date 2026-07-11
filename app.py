"""Retail Shelf Monitor — detects products on shelf photos with YOLO
and flags empty shelf regions. Falls back to a mock detector if the
YOLO model can't be loaded (e.g. no internet to download weights)."""

import io
import random

import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image, ImageDraw

st.set_page_config(page_title="Shelf Monitor", page_icon="🛒", layout="wide")


@st.cache_resource
def load_model():
    """Try to load YOLOv8; return (model, mode)."""
    try:
        from ultralytics import YOLO
        model = YOLO("yolov8n.pt")  # downloads weights on first run
        return model, "yolo"
    except Exception as e:
        st.session_state["model_error"] = str(e)
        return None, "mock"


def detect_yolo(model, image: Image.Image):
    results = model.predict(np.array(image), verbose=False)
    rows = []
    for r in results:
        names = r.names
        for box in r.boxes:
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            rows.append({
                "label": names[int(box.cls[0])],
                "confidence": round(float(box.conf[0]), 2),
                "x1": int(x1), "y1": int(y1), "x2": int(x2), "y2": int(y2),
            })
    return rows


def detect_mock(image: Image.Image):
    """Deterministic fake detections so the demo works without a model."""
    rng = random.Random(image.size[0] * 31 + image.size[1])
    w, h = image.size
    labels = ["bottle", "box", "can", "jar", "carton"]
    rows = []
    for i in range(rng.randint(5, 9)):
        bw, bh = rng.randint(w // 12, w // 6), rng.randint(h // 8, h // 4)
        x1 = rng.randint(0, max(1, w - bw))
        y1 = rng.randint(0, max(1, h - bh))
        rows.append({
            "label": rng.choice(labels),
            "confidence": round(rng.uniform(0.55, 0.97), 2),
            "x1": x1, "y1": y1, "x2": x1 + bw, "y2": y1 + bh,
        })
    return rows


def find_empty_regions(image: Image.Image, detections, cols=4, rows=3):
    """Split the shelf into a grid; cells with no detections are 'empty'."""
    w, h = image.size
    empty = []
    for gy in range(rows):
        for gx in range(cols):
            cx1, cy1 = gx * w // cols, gy * h // rows
            cx2, cy2 = (gx + 1) * w // cols, (gy + 1) * h // rows
            covered = any(
                d["x1"] < cx2 and d["x2"] > cx1 and d["y1"] < cy2 and d["y2"] > cy1
                for d in detections
            )
            if not covered:
                empty.append((cx1, cy1, cx2, cy2))
    return empty


def draw_overlay(image: Image.Image, detections, empty_regions):
    img = image.convert("RGB").copy()
    draw = ImageDraw.Draw(img)
    for d in detections:
        draw.rectangle([d["x1"], d["y1"], d["x2"], d["y2"]], outline="#22c55e", width=3)
        draw.text((d["x1"] + 4, d["y1"] + 4), f'{d["label"]} {d["confidence"]}', fill="#22c55e")
    for (x1, y1, x2, y2) in empty_regions:
        draw.rectangle([x1, y1, x2, y2], outline="#ef4444", width=3)
        draw.text((x1 + 4, y1 + 4), "EMPTY", fill="#ef4444")
    return img


st.title("🛒 Retail Shelf Monitor")
st.write("Upload a shelf photo — products are boxed in green, empty shelf regions flagged in red.")

model, mode = load_model()
if mode == "yolo":
    st.success("Running in **YOLO mode** (real object detection).")
else:
    st.warning("YOLO model could not be loaded — running in **MOCK mode** (simulated detections). "
               f"Reason: {st.session_state.get('model_error', 'unknown')}")

uploaded = st.file_uploader("Shelf photo", type=["jpg", "jpeg", "png"])

if uploaded:
    image = Image.open(io.BytesIO(uploaded.read()))
    detections = detect_yolo(model, image) if mode == "yolo" else detect_mock(image)
    empty_regions = find_empty_regions(image, detections)

    col1, col2 = st.columns(2)
    col1.image(image, caption="Original", use_container_width=True)
    col2.image(draw_overlay(image, detections, empty_regions),
               caption="Detections + empty regions", use_container_width=True)

    m1, m2, m3 = st.columns(3)
    m1.metric("Products detected", len(detections))
    m2.metric("Empty regions", len(empty_regions))
    m3.metric("Mode", mode.upper())

    if detections:
        st.subheader("Detections")
        st.dataframe(pd.DataFrame(detections), use_container_width=True)
else:
    st.info("👆 Upload a photo to start. Any shelf or pantry photo works.")
