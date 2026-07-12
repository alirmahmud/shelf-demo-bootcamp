# Decathlon Shelf Intelligence

Real-time shelf monitoring demo for Decathlon stores — detects products with
YOLOv8, scores planogram compliance, flags replenishment gaps, and reconciles
shelf depletion against POS sales.

Live app: deployed on Streamlit Community Cloud.
Run locally: `python -m streamlit run app.py`

## Demo images & recommended slider settings

Default sliders are tuned for the **massage-shelf before/after pair** (real
photos from the Decathlon BGC branch). Other images detect best with
different settings — adjust live in the sidebar.

| Image (`test_images/`) | Grid | Expected/region | Confidence | Result |
|------------------------|------|-----------------|------------|--------|
| **massage_before.jpg** (hero, default settings) | 1 × 3 | 2 | 0.15 | **100%**, no gaps |
| **massage_after.jpg** (hero, default settings)  | 1 × 3 | 2 | 0.15 | **67%**, gap on left region ("Soft Rollers") |
| bikes_row.jpg (backup) | 1 × 4 | 3 | 0.25 | ~75%, 1 gap on first section |
| bottles_row_cc0.jpg (fallback) | 2 × 3 | 2 | 0.15 | ~83%, 1 gap on first section |

Section names for each demo (sidebar text box, one per line):
- Massage pair: `Soft Rollers / Firm Rollers / Massage Kits` (default)
  — IMPORTANT: upload the committed `massage_before.jpg` / `massage_after.jpg`
  files; the raw phone originals shift one marginal detection at these settings.
- Bikes: `City Bikes / Mountain Bikes / Kids' Bikes / Road Bikes`
- Bottles: `Hydration / Team Sports / Backpacks / Footwear / Fitness / Cycling`

## The before/after story (massage pair)

`massage_before.jpg` is the fully stocked shelf → 100% compliant.
`massage_after.jpg` is the same shelf minutes later with two rollers removed
from the top-left → detection count drops in that region and the app flags
"Soft Rollers — REPLENISH NOW" at 67% compliance.

Uploads are normalized in-app (EXIF rotation + resize to 1280px), so the
original phone photos and the committed copies produce identical results.
This calibration was verified against both.

## Notes

- If the YOLO model can't load, the app falls back to a MOCK detector so the
  UI still works. The banner shows which mode is active.
- Generic YOLOv8 (COCO classes) has no vocabulary for most sporting goods.
  It still *detects* strong shapes (roller ends, bicycles, distinct bottles)
  but labels them with its own classes — the "Show detection labels" toggle
  is off by default for this reason. Detections larger than half the image
  are ignored (shelf frames sometimes register as one giant object).
- Dense same-product walls (packed bottle shelves, ball walls, racket racks)
  detect poorly — chosen demo images avoid these.
- Image credits and licenses: see `test_images/CREDITS.txt`.
