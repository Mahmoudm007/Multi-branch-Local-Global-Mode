# CLAHE Thermal Model Implementation

This project does not contain a model named `CLACHE`. The implemented thermal model is
`CLAHE_Inferno`, which uses CLAHE: Contrast Limited Adaptive Histogram Equalization.
The requested `CLACH.md` file documents that implementation.

## What the model is

`CLAHE_Inferno` is a deterministic pseudo-thermal image transform. It is not a trained
thermal neural network and it does not estimate calibrated temperature in Celsius,
Fahrenheit, or Kelvin. It converts a normal RGB image into a relative heat-style map by:

1. Converting RGB input to grayscale intensity.
2. Applying OpenCV CLAHE to improve local contrast.
3. Applying OpenCV's Inferno colormap.
4. Converting OpenCV's BGR output back to RGB.
5. Returning the result as a PIL image.

The numeric output range is image intensity based: `0` to `255`.

## Source files

- Main implementation: `Depth_Thermal_Estimation.py`
- User-facing model list and outputs: `README.md`
- Requirements: `requirements.txt`
- Existing generated outputs: `outputs/`

## Required packages

The project dependencies are:

```text
numpy
pillow
matplotlib
opencv-python
tqdm
torch
torchvision
```

For the `CLAHE_Inferno` thermal model itself, the active packages are:

- `numpy`: converts PIL images to arrays and handles numeric operations.
- `PIL.Image`: reads, returns, blends, and saves images.
- `opencv-python` / `cv2`: grayscale conversion, CLAHE, and Inferno colormap.
- `matplotlib`: saves output images with colorbars.

## Model registration

`CLAHE_Inferno` is the first default thermal model.

```python
THERMAL_DEFAULT_MODELS = [
    "CLAHE_Inferno",
    "Log_Magma",
    "HistEq_Plasma",
    "Gamma_Plasma",
    "Percentile_Turbo",
    "Bilateral_Inferno",
    "Unsharp_Magma",
    "EdgeBoost_Viridis",
    "LocalContrast_Cividis",
]
```

The runtime registry maps the model name to the implementation function:

```python
def load_thermal_models(model_names: list[str]) -> list[ThermalModel]:
    registry: dict[str, Callable[[Image.Image], Image.Image]] = {
        "CLAHE_Inferno": thermal_clahe_inferno,
        "Log_Magma": thermal_log_magma,
        "HistEq_Plasma": thermal_histeq_plasma,
        "Gamma_Plasma": thermal_gamma_plasma,
        "Percentile_Turbo": thermal_percentile_turbo,
        "Bilateral_Inferno": thermal_bilateral_inferno,
        "Unsharp_Magma": thermal_unsharp_magma,
        "EdgeBoost_Viridis": thermal_edgeboost_viridis,
        "LocalContrast_Cividis": thermal_local_contrast_cividis,
    }
    models: list[ThermalModel] = []
    for name in model_names:
        if name not in registry:
            print(f"Unknown thermal model '{name}', skipping.")
            continue
        models.append(ThermalModel(name=name, fn=registry[name]))
    if not models:
        raise SystemExit("No thermal models could be loaded.")
    return models
```

The model wrapper is:

```python
@dataclass
class ThermalModel:
    name: str
    fn: Callable[[Image.Image], Image.Image]

    def predict(self, image: Image.Image) -> Image.Image:
        return self.fn(image)
```

## Core CLAHE thermal function

Exact function used to create the main `CLAHE_Inferno` output:

```python
def thermal_clahe_inferno(image: Image.Image) -> Image.Image:
    rgb = np.array(image)
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)
    colored = cv2.applyColorMap(enhanced, cv2.COLORMAP_INFERNO)
    colored = cv2.cvtColor(colored, cv2.COLOR_BGR2RGB)
    return Image.fromarray(colored)
```

### Numeric values used

| Value | Where used | Meaning |
| --- | --- | --- |
| `clipLimit=2.0` | `cv2.createCLAHE` | Limits contrast amplification inside each local tile. |
| `tileGridSize=(8, 8)` | `cv2.createCLAHE` | Splits the image into an 8 by 8 tile grid for local histogram equalization. |
| `cv2.COLOR_RGB2GRAY` | grayscale conversion | Converts input RGB image into one intensity channel. |
| `cv2.COLORMAP_INFERNO` | color mapping | Applies the Inferno heat-style colormap. |
| `cv2.COLOR_BGR2RGB` | output conversion | Converts OpenCV BGR colormap output to normal RGB order. |
| `0` to `255` | grayscale and enhanced image range | Relative intensity/thermal level range. |

## Threshold overlay function

The project also creates a special CLAHE threshold overlay only for `CLAHE_Inferno`.
This is separate from the main thermal image.

```python
def thermal_clahe_threshold_overlay(
    image: Image.Image,
    threshold: int = 180,
    alpha: float = OVERLAY_ALPHA,
) -> Image.Image:
    rgb = np.array(image)
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)
    colored = cv2.applyColorMap(enhanced, cv2.COLORMAP_INFERNO)
    colored = cv2.cvtColor(colored, cv2.COLOR_BGR2RGB)
    overlay = rgb.astype(np.float32)
    colored_f = colored.astype(np.float32)
    mask = enhanced > threshold
    overlay[mask] = (1.0 - alpha) * overlay[mask] + alpha * colored_f[mask]
    return Image.fromarray(overlay.astype(np.uint8))
```

### Threshold overlay values

| Value | Meaning |
| --- | --- |
| `threshold=180` | Only pixels with CLAHE-enhanced intensity greater than `180` are heat-colored. |
| `mask = enhanced > threshold` | The threshold comparison is strict: values must be greater than `180`, not equal. |
| `OVERLAY_ALPHA = 0.7` | Heat-colored pixels use 70 percent thermal color and 30 percent original RGB. |
| `(1.0 - alpha) * original + alpha * colored` | Overlay blend formula. |
| `np.float32` | Temporary type used to avoid integer truncation during blending. |
| `np.uint8` | Final saved image type. |

The global overlay alpha is:

```python
OVERLAY_ALPHA = 0.7
```

## Standard thermal overlay

Every thermal model, including `CLAHE_Inferno`, also gets a full-image overlay:

```python
def blend_overlay(
    original: Image.Image,
    overlay: Image.Image,
    alpha: float = OVERLAY_ALPHA,
) -> Image.Image:
    if overlay.size != original.size:
        overlay = overlay.resize(original.size, Image.BILINEAR)
    return Image.blend(original, overlay, alpha=alpha)
```

For `CLAHE_Inferno`, this uses:

- `alpha=0.7`
- 70 percent CLAHE Inferno thermal image
- 30 percent original RGB image
- Bilinear resize only if the thermal image size differs from the original

## Colorbar and thermal level values

The model name is inspected to choose the output colormap:

```python
def infer_cmap_from_name(name: str, fallback: str) -> str:
    lowered = name.lower()
    for key in ("inferno", "magma", "plasma", "viridis", "cividis", "turbo"):
        if key in lowered:
            return key
    return fallback
```

For `CLAHE_Inferno`, `infer_cmap_from_name` returns `inferno`.

Thermal outputs use the colorbar range `0.0` to `255.0`:

```python
def thermal_level_spec(model_name: str) -> tuple[str, str, float, float]:
    cmap_name = infer_cmap_from_name(model_name, "inferno")
    if model_name.startswith("ML_"):
        return cmap_name, f"Thermal level (ML normalized)", 0.0, 1.0
    return cmap_name, f"Thermal level ({model_name})", 0.0, 255.0
```

Saved thermal images with colorbars use:

```python
def save_image_with_colorbar(
    image: Image.Image,
    path: Path,
    cmap_name: str,
    label: str,
    vmin: float,
    vmax: float,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    dpi = 100
    extra_width = 1.0
    fig_width = image.width / dpi + extra_width
    fig_height = image.height / dpi
    fig, ax = plt.subplots(figsize=(fig_width, fig_height), dpi=dpi)
    ax.imshow(image)
    ax.axis("off")
    norm = colors.Normalize(vmin=vmin, vmax=vmax)
    mappable = cm.ScalarMappable(norm=norm, cmap=cmap_name)
    mappable.set_array([])
    cbar = fig.colorbar(mappable, ax=ax, fraction=0.05, pad=0.04)
    cbar.set_label(label, fontsize=8)
    cbar.ax.tick_params(labelsize=7)
    fig.savefig(path, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)
```

Colorbar numeric values:

| Value | Meaning |
| --- | --- |
| `vmin=0.0` | Minimum thermal level shown on the colorbar. |
| `vmax=255.0` | Maximum thermal level shown on the colorbar. |
| `dpi=100` | Matplotlib figure DPI for individual saved images. |
| `extra_width=1.0` | Extra figure width in inches to make room for the colorbar. |
| `fraction=0.05` | Colorbar width fraction for individual image saves. |
| `pad=0.04` | Colorbar padding for individual image saves. |
| `fontsize=8` | Individual colorbar label font size. |
| `labelsize=7` | Individual colorbar tick font size. |
| `pad_inches=0.02` | Tight bounding-box padding when saving the figure. |

## Processing path in `main`

For each input image, the thermal processing path is:

```python
if thermal_models:
    for model in thermal_models:
        start = time.perf_counter()
        safe_model = safe_name(model.name)
        out_path = output_dir / "thermal" / safe_model / f"{img_path.stem}.png"
        try:
            thermal_img = model.predict(original)
            cmap_name, label, vmin, vmax = thermal_level_spec(model.name)
            save_image_with_colorbar(thermal_img, out_path, cmap_name, label, vmin, vmax)
            overlay_img = blend_overlay(original, thermal_img, alpha=OVERLAY_ALPHA)
            overlay_path = (
                output_dir / "overlays" / "thermal" / safe_model / f"{img_path.stem}.png"
            )
            save_image(overlay_img, overlay_path)
            if model.name == "CLAHE_Inferno":
                threshold_overlay = thermal_clahe_threshold_overlay(original, threshold=180)
                threshold_path = (
                    output_dir
                    / "overlays"
                    / "thermal_clahe_threshold180"
                    / f"{img_path.stem}.png"
                )
                save_image(threshold_overlay, threshold_path)
            thermal_outputs[model.name] = thermal_img
            status = "ok"
        except Exception as exc:  # noqa: BLE001
            thermal_outputs[model.name] = None
            status = f"error: {exc}"
        elapsed = time.perf_counter() - start
        records.append(
            {
                "task": "thermal",
                "model": model.name,
                "image": img_path.name,
                "seconds": round(elapsed, 4),
                "status": status,
            }
        )
```

Output paths for `CLAHE_Inferno`:

- Main thermal image with colorbar:
  `outputs/thermal/CLAHE_Inferno/<image>.png`
- Full-image thermal overlay:
  `outputs/overlays/thermal/CLAHE_Inferno/<image>.png`
- Threshold overlay:
  `outputs/overlays/thermal_clahe_threshold180/<image>.png`
- Per-image comparison figure:
  `outputs/figures/per_image/<image>_comparison.png`
- Dataset thermal overview:
  `outputs/figures/overview/thermal_overview.png`
- Timing/status metrics:
  `outputs/run_metrics.csv`
- Run arguments:
  `outputs/run_config.json`

## CLI defaults relevant to CLAHE

```python
parser.add_argument("--input-dir", default="Data", help="Folder containing input images.")
parser.add_argument("--output-dir", default="outputs", help="Folder to store outputs.")
parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"], help="Device.")
parser.add_argument("--thermal-models", nargs="*", default=THERMAL_DEFAULT_MODELS)
parser.add_argument("--skip-thermal", action="store_true", help="Skip thermal generation.")
parser.add_argument("--recursive", action="store_true", help="Recursively scan input folder.")
parser.add_argument("--max-images", type=int, default=None, help="Limit number of images.")
parser.add_argument("--stride", type=int, default=1, help="Use every Nth image.")
parser.add_argument("--overview-count", type=int, default=6, help="Images per overview grid.")
```

Important CLI/default values:

| Setting | Default/current value |
| --- | --- |
| Input directory | `Data` |
| Output directory | `outputs` |
| Default thermal model list includes | `CLAHE_Inferno` |
| Recursive input scan | `False` by default |
| Max images | `None` by default |
| Stride | `1` |
| Overview count | `6` |
| Thermal ML paths | `[]` in current run |
| Thermal ML colormap | `inferno` |

The saved `outputs/run_config.json` shows the existing run used:

```json
{
  "input_dir": "Data",
  "output_dir": "outputs",
  "device": "cuda",
  "thermal_ml_paths": [],
  "thermal_ml_cmap": "inferno",
  "skip_depth": false,
  "skip_thermal": false,
  "recursive": false,
  "max_images": null,
  "stride": 1,
  "overview_count": 6
}
```

## Current generated-output numbers

From the existing `outputs/` folder:

| Item | Value |
| --- | --- |
| Input images in `Data/` | `91` |
| `CLAHE_Inferno` thermal output PNGs | `91` |
| `CLAHE_Inferno` full thermal overlay PNGs | `91` |
| `thermal_clahe_threshold180` overlay PNGs | `91` |
| Per-image comparison figures | `91` |
| CLAHE metric rows in `outputs/run_metrics.csv` | `91` |
| CLAHE successful rows | `91` |
| CLAHE failed rows | `0` |
| Average CLAHE processing time | `0.277168131868132` seconds/image |
| Minimum CLAHE processing time | `0.2242` seconds |
| Maximum CLAHE processing time | `0.4623` seconds |
| Fastest recorded CLAHE image | `IMG_2025-12-16_11-49-40_UofAw.jpg` |
| Slowest recorded CLAHE image | `IMG_2025-12-16_11-50-21_UofAw.jpg` |

Sample dimensions observed from existing outputs:

| File type | Sample dimensions | Mode |
| --- | --- | --- |
| Input image | `800 x 600` | `RGB` |
| Full thermal overlay | `800 x 600` | `RGB` |
| Threshold overlay | `800 x 600` | `RGB` |
| Saved thermal image with colorbar | `715 x 472` | `RGBA` |

The core `thermal_clahe_inferno` function returns an image the same size as the input.
The saved thermal PNG under `outputs/thermal/CLAHE_Inferno/` is a Matplotlib figure
with a colorbar, so its file dimensions can differ from the original image dimensions.

## End-to-end implementation summary

1. `main()` parses CLI arguments.
2. Images are loaded from `Data/` and converted to RGB:
   `Image.open(img_path).convert("RGB")`.
3. `load_thermal_models()` creates a `ThermalModel` named `CLAHE_Inferno`.
4. `ThermalModel.predict()` calls `thermal_clahe_inferno(original)`.
5. `thermal_clahe_inferno()` builds the pseudo-thermal image:
   RGB -> grayscale -> CLAHE -> Inferno colormap -> RGB PIL image.
6. `thermal_level_spec()` assigns the `inferno` colorbar and range `0.0` to `255.0`.
7. `save_image_with_colorbar()` saves the main thermal PNG.
8. `blend_overlay()` saves the full-image thermal overlay with `alpha=0.7`.
9. Because the model name is exactly `CLAHE_Inferno`,
   `thermal_clahe_threshold_overlay(original, threshold=180)` also runs.
10. Runtime metrics are appended to `outputs/run_metrics.csv`.
11. Per-image and overview comparison figures are generated.

## Minimal standalone reproduction

This is the smallest equivalent implementation of the project model:

```python
import cv2
import numpy as np
from PIL import Image


def clahe_inferno(image: Image.Image) -> Image.Image:
    rgb = np.array(image.convert("RGB"))
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)
    colored = cv2.applyColorMap(enhanced, cv2.COLORMAP_INFERNO)
    colored = cv2.cvtColor(colored, cv2.COLOR_BGR2RGB)
    return Image.fromarray(colored)
```

## Limitations

- The model produces relative thermal-looking intensity maps, not physical temperature.
- It depends on visible RGB brightness and contrast, so dark objects can appear cool even
  if they would be physically hot.
- The CLAHE threshold overlay highlights pixels with enhanced intensity greater than
  `180`; that threshold is heuristic and not tied to a calibrated temperature.
- The colorbar range `0.0` to `255.0` describes image intensity, not measured heat.
