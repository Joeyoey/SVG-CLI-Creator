# SVG CLI Creator

Convert raster images (PNG/JPG) into clean, editable SVGs using region-growing segmentation, optional gradient fills, geometric smoothing, and gap-closing strategies.

This tool is designed to produce attractive, vector-friendly results on logos, icons, illustrations, and smooth imagery, with controls to avoid alpha holes, reduce speckling, and smooth edges.

---

## Features

- Region-growing segmentation with 4/8-connectivity and color tolerance.
- Per-region average color fill; optional linear gradient fill along each region's principal axis using PCA and channel-wise linear fits gated by R² and min-area.
- Draw large regions first to minimize visible seams.
- Robust hole handling via `RETR_CCOMP` + `fill-rule="evenodd"` and `--hole_min_area` to ignore tiny interior holes.
- Morphological closing (`--dilate_px`, `--close_iterations`) and optional per-region growth (`--grow_px`) to close hairline gaps.
- Path simplification with `epsilon_ratio` (relative to contour perimeter).
- Chaikin smoothing of contours and holes (`--smooth_iter`) to produce visually smoother edges.
- Optional anti-seam stroke on each region using its own fill color (`--stroke_px`, `--stroke_round`).
- Optional background rectangle (`--add_background`) to ensure no transparency.
- Pre-filters: median blur (`--median_ksize`) to reduce speckles; Gaussian blur (`--blur_sigma`) to gently smooth color fields.
- Progress bars via `tqdm`.

---

## Requirements

- Python 3.9+ (tested on Windows with Python 3.11)
- Install dependencies:

```bash
pip install -r requirements.txt
```

Dependencies (pinned in `requirements.txt`): `numpy`, `opencv-python`, `svgwrite`, `tqdm`, `colorama`.

---

## Install as a global CLI (add to PATH)

You can install this project as a command-line tool that you can run from any terminal as `svgify`.

Option A — pipx (recommended)
- Ensure pipx is installed and PATH is set:
  - `python -m pip install --user pipx`
  - `python -m pipx ensurepath`
  - Close and reopen your terminal so PATH changes take effect.
- From the project root, install:
  - `pipx install .`
- Verify:
  - `svgify --help`

Option B — Developer editable install (local virtualenv)
- Create and activate a venv:
  - `python -m venv .venv`
  - Windows PowerShell: `.\.venv\Scripts\Activate.ps1`
  - CMD: `.\.venv\Scripts\activate.bat`
- Install in editable mode:
  - `python -m pip install -e .`
- Use while venv is active:
  - `svgify input.png output.svg --tolerance 16 ...`

Option C — User install (adds scripts to your user Scripts folder)
- `python -m pip install --user .`
- Ensure the User Scripts folder is on PATH:
  - Print it: `python -c "import site, os; print(os.path.join(site.getuserbase(), 'Scripts'))"`
  - Add that folder to your Windows PATH (then restart the terminal).

Once installed, you can replace `venv\\Scripts\\python.exe vectorize_image.py ...` with just `svgify ...`.

---

## Quick Start

From the project root:

```bash
# Balanced, good starting point
venv\Scripts\python.exe vectorize_image.py input.png output.svg \
  --use_gradients true \
  --tolerance 16 \
  --median_ksize 3 --blur_sigma 0.4 \
  --epsilon_ratio 0.0032 \
  --min_area 24 --hole_min_area 256 \
  --dilate_px 2 --close_iterations 2 --grow_px 1 --smooth_iter 1 \
  --stroke_px 0.3 --stroke_round true \
  --gradient_r2 0.6 --min_gradient_area 450 \
  --add_background true
```

Open `output.svg` in your browser or vector editor.

---

## Full CLI

Positional:
- `input_path` (str): Input raster file (e.g., `input.png`).
- `output_path` (str): Output SVG file (e.g., `output.svg`).

Parameters (with defaults):
- `--tolerance` (float, default: `10.0`): Color distance threshold for region-growing (BGR Euclidean).
- `--blur_sigma` (float, default: `0.0`): Gaussian blur sigma; softens color regions.
- `--use_gradients` (bool, default: `false`): Enable per-region linear gradient fills.
- `--connectivity` (`4` or `8`, default: `8`): Neighbor connectivity for growing.
- `--epsilon_ratio` (float, default: `0.003`): Polygon simplification ratio of perimeter.
- `--min_area` (int, default: `9`): Skip contours smaller than this area in pixels.
- `--dilate_px` (int, default: `1`): Morphological closing kernel radius (ellipse). Use with `--close_iterations`.
- `--close_iterations` (int, default: `1`): How many closing iterations to perform on the mask.
- `--grow_px` (int, default: `0`): Extra dilation after closing to eliminate hairline gaps.
- `--smooth_iter` (int, default: `0`): Chaikin smoothing iterations applied to contours and holes.
- `--hole_min_area` (int, default: `64`): Minimum hole area to subtract; smaller holes ignored to avoid pinholes.
- `--add_background` (bool, default: `true`): Draw a background rect (largest-region color fallback to white).
- `--median_ksize` (int, default: `0`): Median blur kernel size (odd >=3). 0 disables.
- `--gradient_r2` (float, default: `0.55`): Min average R² across B,G,R linear fits required to use gradient.
- `--min_gradient_area` (int, default: `400`): Minimum region size (pixels) eligible for gradients.
- `--stroke_px` (float, default: `0.0`): Optional anti-seam stroke width on each path using same fill.
- `--stroke_round` (bool, default: `true`): Round caps/joins for path stroke.

Boolean flags accept: `true/false`, `yes/no`, `1/0`, `y/n`.

---

## Recommended Presets

Balanced smooth (good coverage, fewer speckles)
```bash
venv\Scripts\python.exe vectorize_image.py input.png output_smooth_best.svg \
  --use_gradients true --tolerance 16 \
  --median_ksize 5 --blur_sigma 0.4 \
  --epsilon_ratio 0.003 \
  --min_area 24 --hole_min_area 256 \
  --dilate_px 2 --close_iterations 2 --grow_px 1 --smooth_iter 1 \
  --gradient_r2 0.6 --min_gradient_area 450 \
  --add_background true --stroke_px 0.3 --stroke_round true
```

Very clean (min speckle, stronger smoothing and gap closing)
```bash
venv\Scripts\python.exe vectorize_image.py input.png output_smooth_clean.svg \
  --use_gradients true --tolerance 18 \
  --median_ksize 5 --blur_sigma 0.5 \
  --epsilon_ratio 0.0032 \
  --min_area 36 --hole_min_area 400 \
  --dilate_px 3 --close_iterations 2 --grow_px 2 --smooth_iter 2 \
  --gradient_r2 0.65 --min_gradient_area 500 \
  --add_background true --stroke_px 0.6 --stroke_round true
```

More detail (keeps more micro features)
```bash
venv\Scripts\python.exe vectorize_image.py input.png output_smooth_detail.svg \
  --use_gradients true --tolerance 14 \
  --median_ksize 3 --blur_sigma 0.3 \
  --epsilon_ratio 0.0025 \
  --min_area 16 --hole_min_area 128 \
  --dilate_px 1 --close_iterations 1 --grow_px 1 --smooth_iter 1 \
  --gradient_r2 0.6 --min_gradient_area 400 \
  --add_background true --stroke_px 0.3 --stroke_round true
```

Seam-fix balanced (focused on closing thin gaps)
```bash
venv\Scripts\python.exe vectorize_image.py input.png output_seamfix_balanced.svg \
  --use_gradients true --tolerance 16 \
  --median_ksize 3 --blur_sigma 0.4 \
  --epsilon_ratio 0.0032 \
  --min_area 24 --hole_min_area 256 \
  --dilate_px 2 --close_iterations 2 --grow_px 1 --smooth_iter 1 \
  --stroke_px 0.3 --stroke_round true \
  --gradient_r2 0.6 --min_gradient_area 450 \
  --add_background true
```

---

## Tuning Guide

- **Reduce thin gaps (separation lines)**
  - Increase `--close_iterations` (e.g., 2 → 3) and/or `--grow_px` (0 → 1 → 2).
  - Enable a small `--stroke_px` (0.3–0.6) with `--stroke_round true`.

- **Edges look jagged**
  - Increase `--smooth_iter` (e.g., 0 → 1 → 2).
  - Slightly raise `--epsilon_ratio` (e.g., 0.003 → 0.0032).

- **Speckles or tiny islands**
  - Increase `--median_ksize` (3 → 5).
  - Raise `--min_area` (e.g., 16 → 24 → 36).
  - Increase `--tolerance` modestly (e.g., 14 → 16).

- **Too many gradients or noisy gradients**
  - Increase `--gradient_r2` and/or `--min_gradient_area`.

- **Not enough gradients (flat looking)**
  - Decrease `--gradient_r2` (e.g., 0.6 → 0.5) and lower `--min_gradient_area`.

- **Interior holes missing or incorrect**
  - Decrease `--hole_min_area` to allow more interior voids.

---

## How It Works (High-Level)

1. Load image and optionally apply median/Gaussian smoothing.
2. Region-growing with 4/8 connectivity merges pixels into regions if their color is within `--tolerance` from a running region average.
3. For each region, compute average color (RGB) for solid fills.
4. If `--use_gradients` is enabled, compute the region's principal axis via PCA and fit a linear model per B,G,R along that axis. If mean R² ≥ `--gradient_r2` and size ≥ `--min_gradient_area`, add a `<linearGradient>` in SVG defs and use it as fill.
5. Build masks and contours per region using `RETR_CCOMP` with morphological closing and optional post-close dilation to remove seams.
6. Simplify polygons by `epsilon_ratio` and optionally smooth with Chaikin `--smooth_iter`.
7. Draw larger regions first to hide seams, with `fill-rule="evenodd"` to handle holes. Optionally add a same-color stroke to cover any remaining micro gaps.
8. Save the final SVG.

---

## Performance Notes

- Large images and/or fine parameters can be compute-heavy. The script parallelizes gradient stats and contour assembly across processes when beneficial.
- Start with moderate parameters (see presets), then iterate small changes.

---

## Troubleshooting

- "Hairline gaps between touching shapes": Increase `--close_iterations` and `--grow_px`, enable a small `--stroke_px`.
- "Too jagged": Use `--smooth_iter 1` (or 2) and slightly raise `--epsilon_ratio`.
- "Speckled regions": Use `--median_ksize 5`, increase `--min_area`, and/or raise `--tolerance` a bit.
- "Unexpected holes": Raise or lower `--hole_min_area` depending on whether you want to keep/remove small interior voids.

---

## Acknowledgements

- Built with `OpenCV`, `NumPy`, `svgwrite`, and `tqdm`.

---
