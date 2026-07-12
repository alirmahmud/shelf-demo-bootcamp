# Decathlon Shelf Intelligence

Real-time shelf monitoring demo for Decathlon stores — detects products with
YOLOv8, scores planogram compliance, flags replenishment gaps, and reconciles
shelf depletion against POS sales.

Live app: deployed on Streamlit Community Cloud.
Run locally: `python -m streamlit run app.py`

## Demo images & recommended slider settings

The app's default sliders are tuned for the **bike** hero image. Other images
detect best with different settings — adjust them live in the sidebar.

| Image (`test_images/`) | Grid | Expected/region | Confidence | Result |
|------------------------|------|-----------------|------------|--------|
| **bikes_row.jpg** (default hero) | 1 × 4 | 3 | 0.25 | ~75%, 1 gap on "City Bikes" |
| bottles_row_cc0.jpg (fallback)   | 2 × 3 | 2 | 0.15 | ~83%, 1 gap on "Hydration" |

For the bottle fallback, also set section names to e.g.
`Hydration / Team Sports / Backpacks / Footwear / Fitness / Cycling`.

## Notes

- If the YOLO model can't load, the app falls back to a MOCK detector so the
  UI still works. The banner shows which mode is active.
- Generic YOLOv8 (COCO classes) reliably detects **bicycles, bottles, sports
  balls, backpacks**; it does NOT recognize shoes, apparel, or specialized
  gear. Demo images are chosen accordingly.
- Image credits and licenses: see `test_images/CREDITS.txt`.
