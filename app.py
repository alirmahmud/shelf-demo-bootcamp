"""Retail Shelf Monitor — detects products on shelf photos with YOLO,
scores planogram compliance, flags replenishment gaps, and reconciles
shelf depletion against POS sales.

Falls back to a mock detector if the YOLO model can't be loaded
(e.g. no internet to download weights)."""

import io
import random

import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image, ImageDraw, ImageOps

st.set_page_config(page_title="Decathlon Shelf Intelligence", page_icon="🏅", layout="wide")


@st.cache_resource
def load_model():
    """Try to load YOLOv8; return (model, mode, error_message)."""
    try:
        from ultralytics import YOLO
        model = YOLO("yolov8n.pt")  # downloads weights on first run
        return model, "yolo", ""
    except Exception as e:
        # Return the error (session_state would be lost across cached reruns).
        return None, "mock", f"{type(e).__name__}: {e}"


def detect_yolo(model, image: Image.Image, conf=0.25):
    results = model.predict(np.array(image), verbose=False, conf=conf)
    w, h = image.size
    rows = []
    for r in results:
        names = r.names
        for box in r.boxes:
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            # Ignore oversized boxes (e.g. the whole shelf frame misread as
            # one object) — anything covering more than half the image.
            if (x2 - x1) * (y2 - y1) > 0.5 * w * h:
                continue
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
    for i in range(rng.randint(12, 20)):
        bw, bh = rng.randint(w // 12, w // 6), rng.randint(h // 8, h // 4)
        x1 = rng.randint(0, max(1, w - bw))
        y1 = rng.randint(0, max(1, h - bh))
        rows.append({
            "label": rng.choice(labels),
            "confidence": round(rng.uniform(0.55, 0.97), 2),
            "x1": x1, "y1": y1, "x2": x1 + bw, "y2": y1 + bh,
        })
    return rows


def region_center(d):
    """Center point of a detection box, used to assign it to one grid cell."""
    return ((d["x1"] + d["x2"]) / 2, (d["y1"] + d["y2"]) / 2)


def score_regions(image: Image.Image, detections, rows, cols, expected, threshold_pct):
    """Assign each detection to a grid cell by its center, count per cell,
    and compute a fill percentage vs. the expected count.

    Returns (region_rows, compliance_score) where region_rows is a list of
    dicts describing every cell, and compliance_score is the share of cells
    at/above the fill threshold."""
    w, h = image.size
    counts = [[0] * cols for _ in range(rows)]
    for d in detections:
        cx, cy = region_center(d)
        gx = min(cols - 1, int(cx / w * cols))
        gy = min(rows - 1, int(cy / h * rows))
        counts[gy][gx] += 1

    region_rows = []
    compliant_cells = 0
    for gy in range(rows):
        for gx in range(cols):
            detected = counts[gy][gx]
            fill = (detected / expected * 100) if expected > 0 else 0
            is_gap = fill < threshold_pct
            if not is_gap:
                compliant_cells += 1
            region_rows.append({
                "gy": gy, "gx": gx,
                "Row": gy + 1,
                "Region": f"R{gy + 1}C{gx + 1}",
                "Detected": detected,
                "Expected": expected,
                "Fill %": round(fill, 0),
                "is_gap": is_gap,
                "box": (gx * w // cols, gy * h // rows,
                        (gx + 1) * w // cols, (gy + 1) * h // rows),
            })

    total_cells = rows * cols
    compliance = round(compliant_cells / total_cells * 100) if total_cells else 0
    return region_rows, compliance


def draw_overlay(image: Image.Image, detections, region_rows, rows, cols,
                 show_labels=False):
    """Green boxes on detected items; grid lines; red fill on gap regions."""
    img = image.convert("RGB").copy()
    overlay = Image.new("RGBA", img.size, (0, 0, 0, 0))
    odraw = ImageDraw.Draw(overlay)

    # Shade gap regions red
    for r in region_rows:
        if r["is_gap"]:
            odraw.rectangle(r["box"], fill=(239, 68, 68, 90))
    img = Image.alpha_composite(img.convert("RGBA"), overlay).convert("RGB")

    draw = ImageDraw.Draw(img)
    w, h = img.size
    # Grid lines
    for gx in range(1, cols):
        draw.line([(gx * w // cols, 0), (gx * w // cols, h)], fill="#94a3b8", width=1)
    for gy in range(1, rows):
        draw.line([(0, gy * h // rows), (w, gy * h // rows)], fill="#94a3b8", width=1)

    # Detected items
    for d in detections:
        draw.rectangle([d["x1"], d["y1"], d["x2"], d["y2"]], outline="#22c55e", width=3)
        if show_labels:
            draw.text((d["x1"] + 4, d["y1"] + 4), f'{d["label"]} {d["confidence"]}', fill="#22c55e")

    # Label gap regions
    for r in region_rows:
        if r["is_gap"]:
            x1, y1, _, _ = r["box"]
            draw.text((x1 + 4, y1 + 4), "GAP", fill="#ef4444")
    return img


# ── Sidebar controls (collapsed — defaults are demo-calibrated) ────
with st.sidebar.expander("⚙️ Advanced settings", expanded=False):
    grid_rows = st.number_input("Grid rows", min_value=1, max_value=10, value=1)
    grid_cols = st.number_input("Regions per row (columns)", min_value=1, max_value=10, value=2)
    expected_items = st.number_input("Expected items per region", min_value=1, max_value=100, value=2)
    threshold = st.slider("Compliance threshold (fill %)", min_value=0, max_value=100, value=50)
    confidence = st.slider("Detection confidence", min_value=0.0, max_value=1.0, value=0.20, step=0.05)
    show_labels = st.checkbox("Show detection labels", value=False,
                              help="Raw model class names (COCO) — often "
                                   "wrong for retail products; off keeps "
                                   "the focus on detection boxes.")
    st.markdown("**Section names** (one per line, top-left to bottom-right)")
    section_text = st.text_area(
        "Section names", label_visibility="collapsed",
        value="Soft Rollers\nFirm Rollers")
section_names = [s.strip() for s in section_text.splitlines() if s.strip()]


def section_for(gy, gx):
    """Name a grid cell using the sidebar list (row-major), cycling if short."""
    if not section_names:
        return ""
    return section_names[(gy * grid_cols + gx) % len(section_names)]


# ── Header ──────────────────────────────────────────────────────────
st.title("🏅 Decathlon Shelf Intelligence")
st.write("Real-time shelf monitoring for Decathlon stores — detect gaps, "
         "score compliance, restock before the sale is lost.")

model, mode, model_error = load_model()
if mode == "yolo":
    st.success("Running in **YOLO mode** (real object detection).")
else:
    st.warning("YOLO model could not be loaded — running in **MOCK mode** (simulated detections). "
               f"Reason: {model_error or 'unknown'}")

st.caption("Upload a shelf photo — or try the sample images in the repo's "
           "`test_images/` folder (start with `massage_before.jpg`, then `massage_after.jpg`).")
uploaded = st.file_uploader("Shelf photo", type=["jpg", "jpeg", "png"])

if uploaded:
    try:
        image = Image.open(io.BytesIO(uploaded.read()))
        image.verify()  # validate it's a real image
        uploaded.seek(0)
        image = Image.open(io.BytesIO(uploaded.read()))
        # Normalize: apply EXIF rotation (phone photos) and cap resolution —
        # detection behavior is calibrated at 1280px.
        image = ImageOps.exif_transpose(image).convert("RGB")
        image.thumbnail((1280, 1280))
    except Exception:
        st.error("⚠️ That file doesn't look like a valid image. "
                 "Please upload a JPG or PNG photo of a shelf.")
        st.stop()

    detections = (detect_yolo(model, image, conf=confidence)
                  if mode == "yolo" else detect_mock(image))
    region_rows, compliance = score_regions(
        image, detections, grid_rows, grid_cols, expected_items, threshold)
    gaps = [r for r in region_rows if r["is_gap"]]

    # ── Images ──────────────────────────────────────────────────────
    col1, col2 = st.columns(2)
    col1.image(image, caption="Original", use_container_width=True)
    col2.image(draw_overlay(image, detections, region_rows, grid_rows, grid_cols,
                            show_labels=show_labels),
               caption="Detections + planogram grid (red = gap)", use_container_width=True)

    # ── Top-line metrics ────────────────────────────────────────────
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("📊 Planogram Compliance", f"{compliance}%")
    m2.metric("Products detected", len(detections))
    m3.metric("Gap regions", len(gaps))
    m4.metric("Mode", mode.upper())

    # ── 2. Gap alert table ──────────────────────────────────────────
    st.subheader("🚨 Replenishment gaps")
    if gaps:
        gap_df = pd.DataFrame([{
            "Section": section_for(r["gy"], r["gx"]),
            "Region": r["Region"],
            "Detected": r["Detected"],
            "Expected": r["Expected"],
            "Fill %": f'{int(r["Fill %"])}%',
            "Action": "REPLENISH NOW",
        } for r in gaps])
        st.dataframe(gap_df, use_container_width=True, hide_index=True)
    else:
        st.success("✅ Shelf fully compliant — every region is stocked at or above target.")

    # ── 3. Event-Day Monitor — Shelf vs. POS ────────────────────────
    st.subheader("🔄 Event-Day Monitor — Shelf vs. POS")
    rc1, rc2 = st.columns(2)
    shelf_removed = rc1.number_input("Items removed from shelf (camera, last hour)",
                                     min_value=0, value=5)
    pos_sold = rc2.number_input("Items sold at POS (last hour)",
                                min_value=0, value=3)
    diff = shelf_removed - pos_sold
    if diff > 0:
        st.warning(f"⚠️ **{diff} units** left the shelf but haven't hit POS — "
                   "during store events, shelves empty faster than sales data shows. "
                   "Early replenishment triggered.")
    else:
        st.success("✅ Shelf and POS in sync.")

    # ── Detection detail ────────────────────────────────────────────
    if detections:
        with st.expander("See raw detections"):
            st.dataframe(pd.DataFrame(detections), use_container_width=True)
else:
    st.info("👆 Upload a shelf photo to start — a bottle wall, ball bin, or backpack section works best.")
