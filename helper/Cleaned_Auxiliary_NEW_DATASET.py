"""Python script version of Cleaned_Auxiliary_NEW_DATASET.ipynb."""

# %%
import os
import random
import re
import sys
import importlib.util
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from PIL import Image

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, default_collate
from torchvision import transforms
from torch.amp import autocast, GradScaler

import matplotlib.pyplot as plt

from tqdm.auto import tqdm

from sklearn.metrics import (
    accuracy_score, f1_score, recall_score,
    confusion_matrix, classification_report,
    roc_curve, auc
)
from sklearn.preprocessing import label_binarize


_DLL_HANDLES = []


def repair_ssl_import() -> None:
    try:
        import ssl  # noqa: F401
        return
    except Exception:
        pass

    python_exe = Path(sys.executable).resolve()
    candidate_env_roots = [
        python_exe.parent,
        Path.home() / "anaconda3" / "envs" / "eye_contact_py10",
        Path.home() / "anaconda3",
    ]

    seen = set()
    for env_root in candidate_env_roots:
        env_root = Path(env_root)
        env_key = str(env_root).lower()
        if env_key in seen:
            continue
        seen.add(env_key)

        ssl_py_candidates = [
            env_root / "lib" / "ssl.py",
            env_root / "Lib" / "ssl.py",
        ]
        ssl_py = next((path for path in ssl_py_candidates if path.exists()), None)
        if ssl_py is None:
            continue

        for dll_dir in [env_root / "DLLs", env_root / "Library" / "bin"]:
            if dll_dir.exists():
                try:
                    _DLL_HANDLES.append(os.add_dll_directory(str(dll_dir)))
                except Exception:
                    pass

        sys.modules.pop("ssl", None)
        spec = importlib.util.spec_from_file_location("ssl", ssl_py)
        if spec is None or spec.loader is None:
            continue

        module = importlib.util.module_from_spec(spec)
        sys.modules["ssl"] = module
        try:
            spec.loader.exec_module(module)
            _ = module.SSLContext
            print(f"[env-fix] Using ssl from {ssl_py}")
            return
        except Exception:
            sys.modules.pop("ssl", None)
            continue

    raise RuntimeError(
        "Could not import a working ssl module. The current environment is missing OpenSSL dependencies."
    )


def repair_urllib3_import() -> None:
    try:
        import urllib3.exceptions  # noqa: F401
        return
    except Exception:
        pass

    python_exe = Path(sys.executable).resolve()
    candidate_package_dirs = [
        Path(sys.prefix) / "Lib" / "site-packages" / "urllib3",
        Path(sys.base_prefix) / "Lib" / "site-packages" / "urllib3",
        python_exe.parents[2] / "Lib" / "site-packages" / "urllib3" if len(python_exe.parents) >= 3 else None,
        Path.home() / "anaconda3" / "Lib" / "site-packages" / "urllib3",
        Path.home() / "anaconda3" / "envs" / "eye_contact_py10" / "Lib" / "site-packages" / "urllib3",
        Path.home() / "AppData" / "Local" / "Programs" / "Python" / "Python311" / "Lib" / "site-packages" / "urllib3",
        Path.home() / "AppData" / "Local" / "Programs" / "ArcGIS" / "Pro" / "bin" / "Python" / "envs" / "arcgispro-py3" / "Lib" / "site-packages" / "urllib3",
    ]

    seen = set()
    for package_dir in candidate_package_dirs:
        if package_dir is None:
            continue
        package_dir = Path(package_dir)
        init_py = package_dir / "__init__.py"
        exceptions_py = package_dir / "exceptions.py"
        if not init_py.exists() or not exceptions_py.exists():
            continue

        package_key = str(package_dir).lower()
        if package_key in seen:
            continue
        seen.add(package_key)

        sys.modules.pop("urllib3", None)
        spec = importlib.util.spec_from_file_location(
            "urllib3",
            init_py,
            submodule_search_locations=[str(package_dir)],
        )
        if spec is None or spec.loader is None:
            continue

        module = importlib.util.module_from_spec(spec)
        sys.modules["urllib3"] = module
        try:
            spec.loader.exec_module(module)
            import urllib3.exceptions  # noqa: F401
            print(f"[env-fix] Using urllib3 from {package_dir}")
            return
        except Exception:
            sys.modules.pop("urllib3", None)
            continue

    raise RuntimeError(
        "Could not import urllib3.exceptions. The current environment has a broken urllib3 installation "
        "and no compatible fallback location was found."
    )


repair_ssl_import()
repair_urllib3_import()


try:
    import timm
except Exception as e:
    raise RuntimeError("timm is required. Install it with: pip install timm") from e

GRADCAM_AVAILABLE = True
try:
    from pytorch_grad_cam import GradCAM
    from pytorch_grad_cam.utils.image import show_cam_on_image
except Exception:
    GRADCAM_AVAILABLE = False

print("Torch:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
print("Grad-CAM available:", GRADCAM_AVAILABLE)

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

# %%
@dataclass
class Config:
    train_root: str = "./Data/1 Defined"
    test_root: str = "./New Labeled Data"
    output_dir: str = "./Output/cleaned_auxiliary_external_test"
    img_size: int = 224
    batch_size: int = 16
    epochs: int = 25
    lr: float = 3e-4
    weight_decay: float = 1e-4
    num_workers: int = 0
    seed: int = 42
    amp: bool = True                        # mixed precision (recommended on GPU)
    pin_memory: bool = True
    fully_class_boost: float = 2.0          # extra loss weight multiplier for class 'Fully'

CFG = Config()

LABEL_TO_CLASS = {
    0: "Bare",
    1: "Centre_Partly",
    2: "TwoTrack_Partly",
    3: "OneTrack_Partly",
    4: "Fully",
}
CLASS_NAMES = [LABEL_TO_CLASS[i] for i in sorted(LABEL_TO_CLASS)]

CLASS_FOLDER_ALIASES = {
    0: ["Bare", "0 Bare"],
    1: ["Centre_Partly", "Centre - Partly", "1 Centre - Partly"],
    2: ["TwoTrack_Partly", "Two_Track_Partly", "Two Track - Partly", "2 Two Track - Partly"],
    3: ["OneTrack_Partly", "One_Track_Partly", "One Track - Partly", "3 One Track - Partly"],
    4: ["Fully", "4 Fully"],
}
UNSUPPORTED_TEST_FOLDERS = ["Undefined"]

if "Fully" not in CLASS_NAMES:
    raise ValueError("'Fully' must exist in CLASS_NAMES to apply class weighting.")

print(f"Using label mapping: {LABEL_TO_CLASS}")
print(f"Training root: {Path(CFG.train_root).resolve()}")
print(f"Testing root: {Path(CFG.test_root).resolve()}")

Path(CFG.output_dir).mkdir(parents=True, exist_ok=True)

def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True

set_seed(CFG.seed)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device

# %%
HF_CACHE_ROOT = Path.home() / ".cache" / "huggingface" / "hub"


def find_hf_snapshot_dir(model_id: str, required_files: Optional[List[str]] = None) -> Optional[Path]:
    repo_dir = HF_CACHE_ROOT / f"models--{model_id.replace('/', '--')}"
    snapshots_dir = repo_dir / "snapshots"
    if not snapshots_dir.exists():
        return None

    snapshot_dirs = sorted(
        [path for path in snapshots_dir.iterdir() if path.is_dir()],
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    if not snapshot_dirs:
        return None

    if not required_files:
        return snapshot_dirs[0]

    for snapshot_dir in snapshot_dirs:
        if all((snapshot_dir / rel_path).exists() for rel_path in required_files):
            return snapshot_dir
    return None


def find_local_timm_weight_file(timm_name: str) -> Optional[Path]:
    snapshot_dir = find_hf_snapshot_dir(f"timm/{timm_name}")
    if snapshot_dir is None:
        return None

    for filename in ("model.safetensors", "pytorch_model.bin", "model.bin"):
        candidate = snapshot_dir / filename
        if candidate.exists():
            return candidate

    pth_files = sorted(snapshot_dir.glob("*.pth"))
    if pth_files:
        return pth_files[0]
    return None


def create_timm_model_cached(
    timm_name: str,
    *,
    pretrained: bool,
    **kwargs,
) -> nn.Module:
    if not pretrained:
        return timm.create_model(timm_name, pretrained=False, **kwargs)

    local_weight_file = find_local_timm_weight_file(timm_name)
    if local_weight_file is not None:
        print(f"[local-cache] Using cached timm weights for {timm_name}: {local_weight_file}")
        return timm.create_model(
            timm_name,
            pretrained=True,
            pretrained_cfg_overlay={"file": str(local_weight_file)},
            **kwargs,
        )

    try:
        return timm.create_model(timm_name, pretrained=True, **kwargs)
    except Exception as e:
        message = str(e)
        if "Hugging Face hub model specified" in message or "urllib3.exceptions" in message:
            print(f"[pretrained-fallback] Could not load pretrained weights for {timm_name}: {e}")
            print("[pretrained-fallback] Continuing with random initialization.")
            return timm.create_model(timm_name, pretrained=False, **kwargs)
        raise


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def normalize_class_token(text: str) -> str:
    token = str(text).strip().lower()
    token = token.replace("&", "and")
    token = token.replace("-", " ")
    token = token.replace("/", " ")
    token = re.sub(r"[^a-z0-9]+", "_", token)
    return token.strip("_")


NORMALIZED_CLASS_ALIAS_TO_INDEX = {}
for label_idx, aliases in CLASS_FOLDER_ALIASES.items():
    for alias in aliases:
        NORMALIZED_CLASS_ALIAS_TO_INDEX[normalize_class_token(alias)] = label_idx

UNSUPPORTED_TEST_TOKENS = {normalize_class_token(name) for name in UNSUPPORTED_TEST_FOLDERS}


def is_image_file(path: Path) -> bool:
    return path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS


def resolve_label_from_folder(folder_name: str, *, allow_unsupported: bool) -> int:
    token = normalize_class_token(folder_name)
    if token in NORMALIZED_CLASS_ALIAS_TO_INDEX:
        return NORMALIZED_CLASS_ALIAS_TO_INDEX[token]
    if allow_unsupported and token in UNSUPPORTED_TEST_TOKENS:
        return -1
    raise ValueError(f"Unsupported folder name: {folder_name}")


def collect_image_items_from_root(root_dir: str, *, allow_unsupported: bool) -> List[Tuple[str, int]]:
    root = Path(root_dir)
    if not root.exists():
        raise FileNotFoundError(f"Dataset root not found: {root.resolve()}")

    items: List[Tuple[str, int]] = []
    skipped_folders: List[str] = []

    for child in sorted(root.iterdir(), key=lambda p: p.name.lower()):
        if not child.is_dir():
            continue

        try:
            label_idx = resolve_label_from_folder(child.name, allow_unsupported=allow_unsupported)
        except ValueError:
            skipped_folders.append(child.name)
            continue

        image_files = sorted(
            (p.resolve() for p in child.rglob("*") if is_image_file(p)),
            key=lambda p: str(p).lower(),
        )
        items.extend((str(path), label_idx) for path in image_files)

    if skipped_folders:
        print(f"[dataset] Skipped unsupported folders under {root}: {skipped_folders}")

    if not items:
        raise ValueError(f"No images found under {root.resolve()}")

    return items


train_items = collect_image_items_from_root(CFG.train_root, allow_unsupported=False)
test_items = collect_image_items_from_root(CFG.test_root, allow_unsupported=True)

splits = {
    "train": train_items,
    "val": [],
    "test": test_items,
}

# %%
def is_known_label(label: int) -> bool:
    label = int(label)
    return 0 <= label < len(CLASS_NAMES)


def build_eval_mask(labels: np.ndarray) -> np.ndarray:
    labels = np.asarray(labels)
    return (labels >= 0) & (labels < len(CLASS_NAMES))


def load_rgb_image(path: str) -> Image.Image:
    with Image.open(path) as img:
        if getattr(img, "is_animated", False):
            try:
                img.seek(0)
            except EOFError:
                pass
        img.load()
        rgb = img.convert("RGB")
        rgb.load()
        return rgb


PREDICTION_EXPORT_MAX_SIDE = 1024
PIL_RESAMPLING = Image.Resampling if hasattr(Image, "Resampling") else Image


def prepare_prediction_preview(img: Image.Image, max_side: int = PREDICTION_EXPORT_MAX_SIDE) -> Image.Image:
    if max(img.size) <= max_side:
        return img

    preview = img.copy()
    preview.thumbnail((max_side, max_side), PIL_RESAMPLING.LANCZOS)
    return preview


UNREADABLE_IMAGE_RECORDS: List[Dict[str, Union[str, int]]] = []
UNREADABLE_IMAGE_KEYS = set()


def record_unreadable_image(split_name: str, path: str, label: int, error: Exception) -> None:
    key = (split_name, path)
    if key in UNREADABLE_IMAGE_KEYS:
        return
    UNREADABLE_IMAGE_KEYS.add(key)

    record = {
        "split_name": split_name,
        "image_path": path,
        "label_idx": int(label),
        "label_name": label_to_display_name(label, path),
        "error_type": type(error).__name__,
        "error_message": str(error),
    }
    UNREADABLE_IMAGE_RECORDS.append(record)
    print(
        f"[dataset][{split_name}] Skipping unreadable image: {path} "
        f"({record['error_type']}: {record['error_message']})"
    )


def save_unreadable_image_report() -> Optional[Path]:
    if not UNREADABLE_IMAGE_RECORDS:
        return None

    report_dir = Path(CFG.output_dir) / "_dataset_reports"
    report_dir.mkdir(parents=True, exist_ok=True)
    report_path = report_dir / "unreadable_images.csv"
    pd.DataFrame(UNREADABLE_IMAGE_RECORDS).to_csv(report_path, index=False)
    return report_path


def collate_skip_none(batch):
    batch = [item for item in batch if item is not None]
    if not batch:
        return None
    return default_collate(batch)


def label_to_display_name(label: int, image_path: str = "") -> str:
    label = int(label)
    if is_known_label(label):
        return CLASS_NAMES[label]
    if image_path:
        folder_name = Path(image_path).parent.name.strip()
        if folder_name:
            return folder_name.replace(" ", "_")
    return "Unknown"


def split_counts(split_list: List[Tuple[str, int]]) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    for path, label in split_list:
        name = label_to_display_name(label, path)
        counts[name] = counts.get(name, 0) + 1
    return dict(sorted(counts.items(), key=lambda kv: kv[0].lower()))


for split_name, items in splits.items():
    print(split_name, len(items))

print("Train counts:", split_counts(splits["train"]))
print("Test counts:", split_counts(splits["test"]))

unsupported_test_count = sum(not is_known_label(label) for _, label in splits["test"])
if unsupported_test_count:
    print(
        f"Test samples without supported ground-truth labels: {unsupported_test_count} "
        "(retained for predictions and Grad-CAM, excluded from metrics)."
    )

# %%
# Plot the train/test distribution
split_names = [split_name for split_name, items in splits.items() if len(items) > 0]
fig, axes = plt.subplots(1, len(split_names), figsize=(6 * len(split_names), 5))
if len(split_names) == 1:
    axes = [axes]

for ax, split_name in zip(axes, split_names):
    counts = split_counts(splits[split_name])
    ax.bar(list(counts.keys()), list(counts.values()))
    ax.set_title(f"{split_name.capitalize()} Distribution")
    ax.set_ylabel("Number of images")
    ax.tick_params(axis="x", rotation=30)

plt.tight_layout()
plt.show()

# %%
class RscImageDataset(Dataset):
    def __init__(self, items: List[Tuple[str, int]], split_name: str, transform=None):
        self.items = items
        self.split_name = split_name
        self.transform = transform

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        path, label = self.items[idx]
        try:
            img = load_rgb_image(path)
        except Exception as e:
            record_unreadable_image(self.split_name, path, label, e)
            return None
        if self.transform:
            img = self.transform(img)
        return img, label, path

train_tf = transforms.Compose([
    transforms.Resize((CFG.img_size, CFG.img_size)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomApply([transforms.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.15)], p=0.3),
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
])

eval_tf = transforms.Compose([
    transforms.Resize((CFG.img_size, CFG.img_size)),
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
])

train_ds = RscImageDataset(splits["train"], split_name="train", transform=train_tf)
val_ds = RscImageDataset(splits["val"], split_name="val", transform=eval_tf)
test_ds = RscImageDataset(splits["test"], split_name="test", transform=eval_tf)

train_loader = DataLoader(train_ds, batch_size=CFG.batch_size, shuffle=True,
                          num_workers=CFG.num_workers, pin_memory=CFG.pin_memory, collate_fn=collate_skip_none)
val_loader = DataLoader(val_ds, batch_size=CFG.batch_size, shuffle=False,
                        num_workers=CFG.num_workers, pin_memory=CFG.pin_memory, collate_fn=collate_skip_none)
test_loader = DataLoader(test_ds, batch_size=CFG.batch_size, shuffle=False,
                         num_workers=CFG.num_workers, pin_memory=CFG.pin_memory, collate_fn=collate_skip_none)

{
    "train_batches": len(train_loader),
    "val_batches": len(val_loader),
    "test_batches": len(test_loader),
}

# %%
# Visual sanity-check: show a few training samples
inv_norm = transforms.Normalize(
    mean=[-0.485/0.229, -0.456/0.224, -0.406/0.225],
    std=[1/0.229, 1/0.224, 1/0.225]
)

# %%
NUM_CLASSES = len(CLASS_NAMES)

AVAILABLE_TIMM_PRETRAINED = set(timm.list_models(pretrained=True))

MODEL_SPECS = {
    "convnext": {
        "timm_name": ["convnext_base", "convnext_base_in22k", "convnext_tiny"],
        "notes": "ConvNeXt"
    },
}

def _rank_pretrained_variant(name: str) -> Tuple[int, int]:
    score = 0

    # Prefer supervised ImageNet variants.
    if "ft_in1k" in name:
        score += 60
    if name.endswith("in1k"):
        score += 50
    if ".in1k" in name or "_in1k" in name:
        score += 30

    # Prefer lighter/default input-size variants.
    if any(tag in name for tag in ["_320", "_384", "_512"]):
        score -= 20

    # Avoid CLIP / self-supervised / multimodal checkpoints unless no alternative exists.
    if any(tag in name for tag in ["clip", "openai", "dino", "mae", "mclip", "siglip"]):
        score -= 80

    # Tie-breaker: shorter names are usually more canonical.
    return score, -len(name)

def resolve_pretrained_timm_name(name_or_list: Union[str, List[str]]) -> str:
    candidates = [name_or_list] if isinstance(name_or_list, str) else list(name_or_list)

    # 1) Exact pretrained name.
    for name in candidates:
        if name in AVAILABLE_TIMM_PRETRAINED:
            return name

    # 2) Prefix match to handle timm variant suffixes like '.fb_in1k'.
    # Keep the candidate order so model family preference is preserved.
    for name in candidates:
        matches = sorted(set(timm.list_models(f"{name}*", pretrained=True)))
        if matches:
            best = sorted(matches, key=_rank_pretrained_variant, reverse=True)[0]
            print(f"[pretrained-resolve] {name} -> {best}")
            return best

    raise ValueError(
        "No pretrained timm weights found for candidates: "
        + ", ".join(candidates)
        + ". Update MODEL_SPECS to pretrained-capable names."
    )

def create_model(model_key: str) -> Tuple[nn.Module, str]:
    spec = MODEL_SPECS[model_key]
    timm_name = resolve_pretrained_timm_name(spec["timm_name"])
    model = create_timm_model_cached(timm_name, pretrained=True, num_classes=NUM_CLASSES)
    return model, timm_name

for k in MODEL_SPECS:
    resolved_name = resolve_pretrained_timm_name(MODEL_SPECS[k]["timm_name"])
    # Build without pretrained weights here to avoid downloading all checkpoints in this validation cell.
    m = timm.create_model(resolved_name, pretrained=False, num_classes=NUM_CLASSES)
    print(k, "=>", resolved_name, "params(M):", sum(p.numel() for p in m.parameters()) / 1e6)

# %%
def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    return {
        "acc": accuracy_score(y_true, y_pred),
        "f1": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "recall": recall_score(y_true, y_pred, average="macro", zero_division=0),
    }

def get_model_output_dirs(model_key: str) -> Dict[str, Path]:
    root = Path(CFG.output_dir) / model_key
    dirs = {
        "root": root,
        "checkpoints": root / "checkpoints",
        "plots": root / "plots",
        "predictions": root / "predictions",
        "gradcam": root / "gradcam",
        "reports": root / "reports",
    }
    for d in dirs.values():
        d.mkdir(parents=True, exist_ok=True)
    return dirs

def save_training_curves(hist_df: pd.DataFrame, model_key: str) -> Path:
    model_dirs = get_model_output_dirs(model_key)
    out_path = model_dirs["plots"] / "training_curves.png"

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].plot(hist_df["epoch"], hist_df["train_loss"], label="train_loss")
    if "val_loss" in hist_df.columns and hist_df["val_loss"].notna().any():
        axes[0].plot(hist_df["epoch"], hist_df["val_loss"], label="val_loss")
    axes[0].set_title(f"Loss Curves - {model_key}")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].legend()

    axes[1].plot(hist_df["epoch"], hist_df["train_acc"], label="train_acc")
    if "val_acc" in hist_df.columns and hist_df["val_acc"].notna().any():
        axes[1].plot(hist_df["epoch"], hist_df["val_acc"], label="val_acc")
    axes[1].set_title(f"Accuracy Curves - {model_key}")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy")
    axes[1].legend()

    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close(fig)
    print(f"Saved training curves: {out_path}")
    return out_path

def build_class_weights(train_items: List[Tuple[str, int]]) -> torch.Tensor:
    counts = np.zeros(NUM_CLASSES, dtype=np.int64)
    for _, label_idx in train_items:
        counts[label_idx] += 1

    if np.any(counts == 0):
        missing = [CLASS_NAMES[i] for i, c in enumerate(counts) if c == 0]
        raise ValueError(f"Classes with zero training samples: {missing}")

    weights = counts.sum() / (NUM_CLASSES * counts.astype(np.float32))
    fully_idx = CLASS_NAMES.index("Fully")
    weights[fully_idx] *= CFG.fully_class_boost
    weights = weights / weights.mean()

    print("Training class counts:", {CLASS_NAMES[i]: int(counts[i]) for i in range(NUM_CLASSES)})
    print("Loss class weights:", {CLASS_NAMES[i]: float(np.round(weights[i], 4)) for i in range(NUM_CLASSES)})

    return torch.tensor(weights, dtype=torch.float32, device=device)

CLASS_WEIGHTS = build_class_weights(splits["train"])

def empty_eval_metrics() -> Dict[str, float]:
    return {
        "loss": float("nan"),
        "acc": float("nan"),
        "f1": float("nan"),
        "recall": float("nan"),
    }

@torch.no_grad()
def run_eval(model, loader, criterion) -> Dict[str, float]:
    if len(loader.dataset) == 0:
        return empty_eval_metrics()

    model.eval()
    all_true, all_pred = [], []
    losses = []
    for batch in loader:
        if batch is None:
            continue
        imgs, labels, _ = batch
        imgs = imgs.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        logits = model(imgs)
        loss = criterion(logits, labels)
        losses.append(loss.item())
        preds = torch.argmax(logits, dim=1)
        all_true.append(labels.detach().cpu().numpy())
        all_pred.append(preds.detach().cpu().numpy())
    if not all_true:
        return empty_eval_metrics()

    y_true = np.concatenate(all_true)
    y_pred = np.concatenate(all_pred)
    metrics = compute_metrics(y_true, y_pred)
    metrics["loss"] = float(np.mean(losses)) if losses else float("nan")
    return metrics

def train_one_model(model_key: str):
    model_dirs = get_model_output_dirs(model_key)

    model, timm_name = create_model(model_key)
    model = model.to(device)

    criterion = nn.CrossEntropyLoss(weight=CLASS_WEIGHTS)
    optimizer = optim.AdamW(model.parameters(), lr=CFG.lr, weight_decay=CFG.weight_decay)
    scaler = GradScaler(enabled=(CFG.amp and device.type == "cuda"))

    has_val_split = len(val_ds) > 0
    monitor_name = "val_acc" if has_val_split else "train_acc"
    best_monitor_acc = -float("inf")
    ckpt_path = model_dirs["checkpoints"] / "best.pth"
    history = []

    for epoch in range(1, CFG.epochs + 1):
        model.train()
        epoch_losses = []
        y_true_epoch, y_pred_epoch = [], []

        pbar = tqdm(train_loader, desc=f"[{model_key}] Epoch {epoch:02d}/{CFG.epochs}", leave=False)
        for batch in pbar:
            if batch is None:
                continue
            imgs, labels, _ = batch
            imgs = imgs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)

            with autocast(device_type="cuda", enabled=(CFG.amp and device.type == "cuda")):
                logits = model(imgs)
                loss = criterion(logits, labels)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            epoch_losses.append(loss.item())
            preds = torch.argmax(logits, dim=1)
            y_true_epoch.append(labels.detach().cpu().numpy())
            y_pred_epoch.append(preds.detach().cpu().numpy())

            y_true_tmp = np.concatenate(y_true_epoch)
            y_pred_tmp = np.concatenate(y_pred_epoch)
            m = compute_metrics(y_true_tmp, y_pred_tmp)
            pbar.set_postfix({"loss": float(np.mean(epoch_losses)), "acc": m["acc"], "f1": m["f1"], "recall": m["recall"]})

        if not y_true_epoch:
            raise RuntimeError(f"No readable training images were available for split 'train' in model {model_key}.")

        y_true_tr = np.concatenate(y_true_epoch)
        y_pred_tr = np.concatenate(y_pred_epoch)
        train_metrics = compute_metrics(y_true_tr, y_pred_tr)
        train_metrics["loss"] = float(np.mean(epoch_losses)) if epoch_losses else float("nan")

        val_metrics = run_eval(model, val_loader, criterion) if has_val_split else empty_eval_metrics()

        row = {
            "epoch": epoch,
            "train_loss": train_metrics["loss"],
            "train_acc": train_metrics["acc"],
            "train_f1": train_metrics["f1"],
            "train_recall": train_metrics["recall"],
            "val_loss": val_metrics["loss"],
            "val_acc": val_metrics["acc"],
            "val_f1": val_metrics["f1"],
            "val_recall": val_metrics["recall"],
        }
        history.append(row)

        if has_val_split:
            print(
                f"[{model_key}] Epoch {epoch:02d} | "
                f"train loss {row['train_loss']:.4f} acc {row['train_acc']:.4f} f1 {row['train_f1']:.4f} rec {row['train_recall']:.4f} || "
                f"val loss {row['val_loss']:.4f} acc {row['val_acc']:.4f} f1 {row['val_f1']:.4f} rec {row['val_recall']:.4f}"
            )
        else:
            print(
                f"[{model_key}] Epoch {epoch:02d} | "
                f"train loss {row['train_loss']:.4f} acc {row['train_acc']:.4f} "
                f"f1 {row['train_f1']:.4f} rec {row['train_recall']:.4f}"
            )

        monitor_value = row[monitor_name]
        if monitor_value >= best_monitor_acc:
            best_monitor_acc = float(monitor_value)
            torch.save({
                "model_key": model_key,
                "timm_name": timm_name,
                "num_classes": NUM_CLASSES,
                "class_names": CLASS_NAMES,
                "state_dict": model.state_dict(),
                "monitor_metric": monitor_name,
                "monitor_value": best_monitor_acc,
                "epoch": epoch,
            }, ckpt_path)
            print(f"Saved checkpoint to {ckpt_path} ({monitor_name}={best_monitor_acc:.4f})")

    hist_df = pd.DataFrame(history)
    hist_path = model_dirs["root"] / "history.csv"
    hist_df.to_csv(hist_path, index=False)
    print(f"History saved: {hist_path}")
    save_training_curves(hist_df, model_key)

    return model, hist_df, ckpt_path

# %%
@torch.no_grad()
def predict_proba(model, loader):
    model.eval()
    probs_list = []
    y_true_list = []
    paths_list = []
    for batch in tqdm(loader, desc="Predict", leave=False):
        if batch is None:
            continue
        imgs, labels, paths = batch
        imgs = imgs.to(device, non_blocking=True)
        logits = model(imgs)
        probs = torch.softmax(logits, dim=1).detach().cpu().numpy()
        probs_list.append(probs)
        y_true_list.append(labels.numpy())
        paths_list += list(paths)
    if not probs_list:
        return (
            np.empty((0, NUM_CLASSES), dtype=np.float32),
            np.asarray([], dtype=np.int64),
            np.asarray([], dtype=np.int64),
            np.asarray([], dtype=np.float32),
            [],
        )
    probs = np.vstack(probs_list)
    y_true = np.concatenate(y_true_list)
    y_pred = np.argmax(probs, axis=1)
    pred_conf = probs[np.arange(len(y_pred)), y_pred]
    return probs, y_true, y_pred, pred_conf, paths_list

def prepare_labeled_eval_subset(
    y_proba: np.ndarray,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    pred_conf: np.ndarray,
    paths: List[str],
    split_name: str,
) -> Dict[str, Union[np.ndarray, List[str]]]:
    eval_mask = build_eval_mask(y_true)
    labeled_count = int(eval_mask.sum())
    unsupported_count = int((~eval_mask).sum())

    if unsupported_count:
        print(
            f"[{split_name}] Metrics use {labeled_count} supported-label samples; "
            f"{unsupported_count} unsupported samples are still exported for predictions and Grad-CAM."
        )

    return {
        "mask": eval_mask,
        "y_proba": y_proba[eval_mask],
        "y_true": y_true[eval_mask],
        "y_pred": y_pred[eval_mask],
        "pred_conf": pred_conf[eval_mask],
        "paths": [path for path, keep in zip(paths, eval_mask) if bool(keep)],
    }

def save_confusion_matrix_image(cm: np.ndarray, title: str, out_path: Path):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(cm, interpolation="nearest")
    ax.set_title(title)
    fig.colorbar(im, ax=ax)

    tick_marks = np.arange(NUM_CLASSES)
    ax.set_xticks(tick_marks)
    ax.set_xticklabels(CLASS_NAMES, rotation=30, ha="right")
    ax.set_yticks(tick_marks)
    ax.set_yticklabels(CLASS_NAMES)

    thresh = cm.max() / 2.0 if cm.size else 0.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(
                j, i, format(cm[i, j], "d"),
                ha="center", va="center",
                color="white" if cm[i, j] > thresh else "black",
                fontsize=10,
                fontweight="bold",
            )

    ax.set_ylabel("True label")
    ax.set_xlabel("Predicted label")
    fig.tight_layout()
    fig.savefig(out_path, dpi=220)
    plt.close(fig)
    print(f"Saved confusion matrix image: {out_path}")

def save_multiclass_roc_image(y_true: np.ndarray, y_proba: np.ndarray, title: str, out_path: Path):
    if y_true.size == 0:
        print(f"Skipping ROC image for {out_path.name}: no labeled samples available.")
        return

    out_path.parent.mkdir(parents=True, exist_ok=True)

    y_true_bin = label_binarize(y_true, classes=list(range(NUM_CLASSES)))
    fig, ax = plt.subplots(figsize=(7, 6))

    plotted = 0
    for i, cls in enumerate(CLASS_NAMES):
        positives = int(y_true_bin[:, i].sum())
        negatives = int(y_true_bin.shape[0] - positives)
        if positives == 0 or negatives == 0:
            print(f"Skipping ROC for class '{cls}' (positives={positives}, negatives={negatives}).")
            continue
        fpr, tpr, _ = roc_curve(y_true_bin[:, i], y_proba[:, i])
        roc_auc = auc(fpr, tpr)
        ax.plot(fpr, tpr, label=f"{cls} (AUC={roc_auc:.3f})")
        plotted += 1

    ax.plot([0, 1], [0, 1], linestyle="--")
    ax.set_title(title)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    if plotted > 0:
        ax.legend(fontsize=8)

    fig.tight_layout()
    fig.savefig(out_path, dpi=220)
    plt.close(fig)
    print(f"Saved ROC image: {out_path}")

def save_split_predictions_csv(model_key: str, split_name: str, paths, y_true, y_pred, y_proba, pred_conf):
    model_dirs = get_model_output_dirs(model_key)
    out_path = model_dirs["predictions"] / f"{split_name}_predictions.csv"

    known_mask = build_eval_mask(np.asarray(y_true))
    df = pd.DataFrame({
        "image_path": paths,
        "source_folder": [Path(path).parent.name for path in paths],
        "actual_label": [label_to_display_name(int(label), path) for label, path in zip(y_true, paths)],
        "actual_label_supported": known_mask,
        "predicted_label": [CLASS_NAMES[i] for i in y_pred],
        "prediction_confidence": pred_conf,
        "is_correct": [
            bool(int(y_true[i]) == int(y_pred[i])) if bool(known_mask[i]) else None
            for i in range(len(paths))
        ],
    })
    for j, cls in enumerate(CLASS_NAMES):
        df[f"proba_{cls}"] = y_proba[:, j]

    df.to_csv(out_path, index=False)
    print(f"Saved prediction CSV: {out_path}")
    return out_path

def save_prediction_images(paths, y_true, y_pred, pred_conf, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    saved = 0

    for i, path in enumerate(tqdm(paths, desc=f"Prediction images -> {out_dir.name}", leave=False)):
        img = None
        preview = None
        fig = None
        try:
            img = load_rgb_image(path)
            preview = prepare_prediction_preview(img)

            actual = label_to_display_name(int(y_true[i]), path)
            pred = CLASS_NAMES[int(y_pred[i])]
            conf = float(pred_conf[i])
            supported_label = is_known_label(int(y_true[i]))

            fig, ax = plt.subplots(figsize=(5, 5))
            ax.imshow(preview)
            ax.axis("off")
            if supported_label:
                title_color = "green" if actual == pred else "red"
            else:
                title_color = "black"
            ax.set_title(
                f"Actual: {actual} | Predicted: {pred} | Confidence: {conf:.3f}",
                color=title_color,
                fontsize=10,
            )

            fname = f"{i:06d}_{Path(path).stem}_pred.jpg"
            save_path = out_dir / fname
            fig.savefig(save_path, bbox_inches="tight", dpi=160)
            saved += 1
        except Exception as e:
            print(f"Skipping prediction image for {path}: {type(e).__name__}: {e}")
        finally:
            if fig is not None:
                plt.close(fig)
            if preview is not None and preview is not img:
                preview.close()
            if img is not None:
                img.close()

    print(f"Saved {saved} prediction images to {out_dir}")

# %%
def denormalize(img_tensor: torch.Tensor) -> np.ndarray:
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std  = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    x = img_tensor.detach().cpu().numpy().transpose(1, 2, 0)
    x = (x * std) + mean
    return np.clip(x, 0, 1)

def find_last_conv2d_layer(model: nn.Module) -> nn.Module:
    last_conv = None
    for module in model.modules():
        if isinstance(module, nn.Conv2d):
            last_conv = module
    if last_conv is None:
        raise ValueError("No Conv2d layer found for Grad-CAM target layer selection.")
    return last_conv

def _reshape_transform_tokens(tensor: torch.Tensor, model: nn.Module) -> torch.Tensor:
    if tensor.ndim == 4:
        if tensor.shape[-1] > 16 and tensor.shape[1] <= 64 and tensor.shape[2] <= 64:
            return tensor.permute(0, 3, 1, 2)
        return tensor

    if tensor.ndim != 3:
        raise ValueError(f"Unsupported token tensor shape for Grad-CAM: {tuple(tensor.shape)}")

    b, n, c = tensor.shape
    grid = None
    patch_embed = getattr(model, "patch_embed", None)
    if patch_embed is not None and hasattr(patch_embed, "grid_size"):
        try:
            gh, gw = patch_embed.grid_size
            grid = (int(gh), int(gw))
        except Exception:
            grid = None

    if grid is not None:
        gh, gw = grid
        if n == gh * gw + 1:
            tensor = tensor[:, 1:, :]
            n = tensor.shape[1]
        elif n != gh * gw:
            grid = None

    if grid is None:
        if int(np.sqrt(n - 1)) ** 2 == (n - 1):
            tensor = tensor[:, 1:, :]
            n = tensor.shape[1]
        side = int(np.sqrt(n))
        if side * side != n:
            raise ValueError(f"Cannot infer spatial grid from token count n={n}.")
        gh, gw = side, side

    tensor = tensor.reshape(b, gh, gw, c)
    tensor = tensor.permute(0, 3, 1, 2)
    return tensor

def get_gradcam_components(model_key: str, model: nn.Module):
    if model_key in {"vit_base", "swin_base"}:
        if model_key == "vit_base":
            target_layer = model.blocks[-1].norm1
        else:
            target_layer = model.layers[-1].blocks[-1].norm1
        reshape_transform = lambda t: _reshape_transform_tokens(t, model)
        return target_layer, reshape_transform

    target_layer = find_last_conv2d_layer(model)
    return target_layer, None

def gradcam_group_name(label: int, image_path: str) -> str:
    if is_known_label(label):
        return CLASS_NAMES[int(label)]
    folder_name = Path(image_path).parent.name.strip()
    return folder_name.replace(" ", "_") if folder_name else "Unknown"

def save_gradcam_for_loader(model_key: str, model: nn.Module, loader: DataLoader, out_dir: Path, split_name: str):
    if not GRADCAM_AVAILABLE:
        print("Grad-CAM not available. Install with: pip install grad-cam")
        return

    out_dir.mkdir(parents=True, exist_ok=True)
    model.eval()

    try:
        target_layer, reshape_transform = get_gradcam_components(model_key, model)
    except Exception as e:
        print(f"Grad-CAM target layer selection failed for {model_key}: {e}")
        return

    cam = GradCAM(model=model, target_layers=[target_layer], reshape_transform=reshape_transform)
    saved = 0

    for batch in tqdm(loader, desc=f"Grad-CAM [{model_key}:{split_name}]", leave=False):
        if batch is None:
            continue
        imgs, labels, paths = batch
        imgs = imgs.to(device)
        for i in range(imgs.size(0)):
            img_t = imgs[i].detach().cpu()
            label = int(labels[i])
            path_str = paths[i]
            input_tensor = imgs[i].unsqueeze(0)
            try:
                grayscale_cam = cam(input_tensor=input_tensor, targets=None)[0]
            except Exception as e:
                print(f"Grad-CAM failed for {path_str}: {e}")
                continue

            rgb = denormalize(img_t)
            cam_img = show_cam_on_image(rgb, grayscale_cam, use_rgb=True)
            group_name = gradcam_group_name(label, path_str)
            fname = f"{saved:06d}_{Path(path_str).stem}_cam.jpg"
            save_path = out_dir / group_name / fname
            save_path.parent.mkdir(parents=True, exist_ok=True)
            Image.fromarray(cam_img).save(save_path)
            saved += 1

    print(f"Saved {saved} Grad-CAM images to {out_dir}")

# %%
def empty_split_metric_fields(split_name: str) -> Dict[str, float]:
    return {
        f"{split_name}_acc": float("nan"),
        f"{split_name}_f1_macro": float("nan"),
        f"{split_name}_recall_macro": float("nan"),
    }

def evaluate_prediction_artifacts(
    model_key: str,
    model: nn.Module,
    split_name: str,
    loader_for_gradcam: DataLoader,
    paths: List[str],
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_proba: np.ndarray,
    pred_conf: np.ndarray,
) -> Dict[str, float]:
    subset = prepare_labeled_eval_subset(y_proba, y_true, y_pred, pred_conf, paths, split_name)
    y_true_eval = subset["y_true"]
    y_pred_eval = subset["y_pred"]
    y_proba_eval = subset["y_proba"]

    model_dirs = get_model_output_dirs(model_key)

    if y_true_eval.size > 0:
        acc = accuracy_score(y_true_eval, y_pred_eval)
        f1 = f1_score(y_true_eval, y_pred_eval, average="macro", zero_division=0)
        rec = recall_score(y_true_eval, y_pred_eval, average="macro", zero_division=0)

        print(f"[{model_key}][{split_name}] Acc={acc:.4f} | F1(macro)={f1:.4f} | Recall(macro)={rec:.4f}")

        report_text = classification_report(
            y_true_eval,
            y_pred_eval,
            labels=list(range(NUM_CLASSES)),
            target_names=CLASS_NAMES,
            zero_division=0,
        )
        cm = confusion_matrix(y_true_eval, y_pred_eval, labels=list(range(NUM_CLASSES)))

        save_confusion_matrix_image(
            cm,
            title=f"Confusion Matrix ({split_name}) - {model_key}",
            out_path=model_dirs["plots"] / f"{split_name}_confusion_matrix.png",
        )
        save_multiclass_roc_image(
            y_true_eval,
            y_proba_eval,
            title=f"ROC Curves ({split_name}) - {model_key}",
            out_path=model_dirs["plots"] / f"{split_name}_roc_curve.png",
        )
    else:
        acc = f1 = rec = float("nan")
        report_text = (
            f"No supported ground-truth labels were available for split '{split_name}'.\n"
            f"Predictions and Grad-CAM were still exported for {len(paths)} samples.\n"
        )
        print(f"[{model_key}][{split_name}] No supported labels available for metrics.")

    report_path = model_dirs["reports"] / f"{split_name}_classification_report.txt"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report_text)
    print(f"Saved classification report: {report_path}")

    save_split_predictions_csv(model_key, split_name, paths, y_true, y_pred, y_proba, pred_conf)
    save_prediction_images(paths, y_true, y_pred, pred_conf, model_dirs["predictions"] / split_name)
    save_gradcam_for_loader(
        model_key,
        model,
        loader_for_gradcam,
        out_dir=model_dirs["gradcam"] / split_name,
        split_name=split_name,
    )

    return {
        f"{split_name}_acc": acc,
        f"{split_name}_f1_macro": f1,
        f"{split_name}_recall_macro": rec,
    }

def evaluate_model_on_split(model_key: str, model: nn.Module, loader: DataLoader, split_name: str) -> Dict[str, float]:
    y_proba, y_true, y_pred, pred_conf, paths = predict_proba(model, loader)
    return evaluate_prediction_artifacts(
        model_key=model_key,
        model=model,
        split_name=split_name,
        loader_for_gradcam=loader,
        paths=paths,
        y_true=y_true,
        y_pred=y_pred,
        y_proba=y_proba,
        pred_conf=pred_conf,
    )

# %%
def evaluate_model_artifacts(model_key: str, model: nn.Module, ckpt_path: Path) -> Dict[str, Union[str, float]]:
    print()
    print("-" * 80)
    print(f"EVALUATING: {model_key} | {MODEL_SPECS[model_key]['notes']}")
    print("-" * 80)

    result = {
        "model_key": model_key,
        "best_ckpt": str(ckpt_path),
    }

    if len(val_ds) > 0:
        result.update(evaluate_model_on_split(model_key, model, val_loader, split_name="val"))
    else:
        print(f"[{model_key}] No validation split configured. All images under {Path(CFG.train_root).resolve()} were used for training.")
        result.update(empty_split_metric_fields("val"))

    result.update(evaluate_model_on_split(model_key, model, test_loader, split_name="test"))
    unreadable_report = save_unreadable_image_report()
    if unreadable_report is not None:
        print(f"[dataset] Unreadable image report: {unreadable_report}")
    return result

def upsert_summary(results_list: List[Dict[str, Union[str, float]]], result: Dict[str, Union[str, float]]):
    for i, row in enumerate(results_list):
        if row["model_key"] == result["model_key"]:
            results_list[i] = result
            return
    results_list.append(result)

# %%
trained_models = {}
histories = {}
checkpoints = {}
results_summary = []

# Train the baseline pretrained ConvNeXt model on all images under Data/1 Defined
# and evaluate it on every image under New Labeled Data.
TRAIN_MODEL_KEYS = ["convnext"]
# To skip baseline training on a rerun:
# TRAIN_MODEL_KEYS = []

for model_key in TRAIN_MODEL_KEYS:
    print()
    print("=" * 80)
    print(f"Training model: {model_key} | {MODEL_SPECS[model_key]['notes']}")
    print("=" * 80)

    model, hist_df, ckpt_path = train_one_model(model_key)
    histories[model_key] = hist_df
    checkpoints[model_key] = ckpt_path

    ckpt = torch.load(ckpt_path, map_location="cpu")
    model.load_state_dict(ckpt["state_dict"])
    model.to(device)
    trained_models[model_key] = model

    model_result = evaluate_model_artifacts(model_key, model, ckpt_path)
    upsert_summary(results_summary, model_result)

summary_df = pd.DataFrame(results_summary)
if not summary_df.empty:
    summary_df = summary_df.sort_values("test_acc", ascending=False).reset_index(drop=True)

summary_path = Path(CFG.output_dir) / "all_models_test_summary.csv"
summary_df.to_csv(summary_path, index=False)
print(f"Saved summary: {summary_path}")

summary_df

# %%
# -----------------------------------------------------------------------------
# Auxiliary learning with VLM context (JSON -> text features -> ConvNeXt fusion)
# -----------------------------------------------------------------------------
import json
from collections import Counter
from typing import Any, Optional

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import Normalizer
from sklearn.pipeline import make_pipeline

import torch.nn.functional as F

AUX_MODEL_KEY = "convnext_aux_context"
CONTEXT_JSON_PATH = Path("./gemini_road_context.json")
CONTEXT_IGNORE_FIELDS = {"id", "Image_name", "Class", "VLM_Raw_Output"}
CONTEXT_MAX_TFIDF_FEATURES = 4000
CONTEXT_NGRAM_RANGE = (1, 2)
CONTEXT_SVD_DIM = 128
CONTEXT_ALIGN_LOSS_WEIGHT = 0.15


def normalize_image_name(name: str) -> str:
    return Path(str(name)).name.lower().strip()


def row_to_context_text(row: Dict[str, Any]) -> str:
    chunks = []
    for key, value in row.items():
        if key in CONTEXT_IGNORE_FIELDS:
            continue
        if value is None:
            continue
        text = str(value).strip()
        if not text:
            continue
        chunks.append(f"{key}: {text}")
    return " | ".join(chunks)


def load_context_lookup(json_path: Path) -> Dict[str, Dict[str, str]]:
    if not json_path.exists():
        raise FileNotFoundError(f"Context JSON not found: {json_path.resolve()}")

    rows = json.loads(json_path.read_text(encoding="utf-8"))
    if not isinstance(rows, list):
        raise ValueError("Context JSON must be a list of records.")

    lookup: Dict[str, Dict[str, str]] = {}
    duplicate_keys = []
    for row in rows:
        image_name = str(row.get("Image_name", "")).strip()
        if not image_name:
            continue

        key = normalize_image_name(image_name)
        if key in lookup:
            duplicate_keys.append(key)

        lookup[key] = {
            "class_name": str(row.get("Class", "")).strip(),
            "context_text": row_to_context_text(row),
        }

    if duplicate_keys:
        print(f"[context] Warning: {len(duplicate_keys)} duplicate Image_name keys found in JSON.")

    return lookup


def split_context_coverage(split_items: List[Tuple[str, int]], lookup: Dict[str, Dict[str, str]]) -> Dict[str, int]:
    available = 0
    for path, _ in split_items:
        key = normalize_image_name(path)
        if key in lookup and lookup[key].get("context_text", "").strip():
            available += 1
    total = len(split_items)
    return {"available": available, "missing": total - available, "total": total}


CONTEXT_LOOKUP = load_context_lookup(CONTEXT_JSON_PATH)

all_split_items = splits["train"] + splits["val"] + splits["test"]
name_to_dataset_class = {
    normalize_image_name(path): label_to_display_name(label_idx, path)
    for path, label_idx in all_split_items
    if is_known_label(label_idx)
}

context_class_mismatch = 0
for key, payload in CONTEXT_LOOKUP.items():
    if key not in name_to_dataset_class:
        continue
    json_class = payload.get("class_name", "")
    dataset_class = name_to_dataset_class[key]
    if json_class and json_class != dataset_class:
        context_class_mismatch += 1

print(f"[context] Loaded JSON records: {len(CONTEXT_LOOKUP)}")
print(f"[context] Class mismatches vs dataset folder labels: {context_class_mismatch}")

for split_name in ["train", "val", "test"]:
    cov = split_context_coverage(splits[split_name], CONTEXT_LOOKUP)
    pct = (100.0 * cov["available"] / max(cov["total"], 1))
    print(
        f"[context][{split_name}] available={cov['available']} "
        f"missing={cov['missing']} total={cov['total']} ({pct:.2f}% with context)"
    )

# %%
def get_context_text_and_mask(image_path: str) -> Tuple[str, float]:
    payload = CONTEXT_LOOKUP.get(normalize_image_name(image_path))
    if payload is None:
        return "", 0.0
    context_text = payload.get("context_text", "").strip()
    return context_text, float(bool(context_text))


def collect_split_context(split_items: List[Tuple[str, int]]) -> Tuple[List[str], np.ndarray]:
    texts: List[str] = []
    masks: List[float] = []
    for path, _ in split_items:
        txt, m = get_context_text_and_mask(path)
        texts.append(txt)
        masks.append(m)
    return texts, np.asarray(masks, dtype=np.float32)


train_context_texts, train_context_masks = collect_split_context(splits["train"])

if int((train_context_masks > 0).sum()) == 0:
    raise ValueError("No non-empty context text found in training split. Cannot build text features.")

context_vectorizer = TfidfVectorizer(
    max_features=CONTEXT_MAX_TFIDF_FEATURES,
    ngram_range=CONTEXT_NGRAM_RANGE,
    lowercase=True,
    strip_accents="unicode",
)

train_context_sparse = context_vectorizer.fit_transform(train_context_texts)
max_rank = int(min(train_context_sparse.shape[0] - 1, train_context_sparse.shape[1] - 1))

if max_rank >= 2:
    context_components = int(min(CONTEXT_SVD_DIM, max_rank))
    context_reducer = make_pipeline(
        TruncatedSVD(n_components=context_components, random_state=CFG.seed),
        Normalizer(copy=False),
    )
    train_context_features = context_reducer.fit_transform(train_context_sparse).astype(np.float32)
else:
    context_reducer = None
    train_context_features = train_context_sparse.toarray().astype(np.float32)

CONTEXT_FEATURE_DIM = int(train_context_features.shape[1])
print(f"[context] TF-IDF vocabulary size: {len(context_vectorizer.vocabulary_)}")
print(f"[context] Context feature dimension after reduction: {CONTEXT_FEATURE_DIM}")


def build_context_feature_map(
    split_items: List[Tuple[str, int]],
    features: np.ndarray,
    masks: np.ndarray,
) -> Dict[str, Tuple[np.ndarray, float]]:
    output: Dict[str, Tuple[np.ndarray, float]] = {}
    for (path, _), feature_vec, mask in zip(split_items, features, masks):
        output[path] = (feature_vec.astype(np.float32), float(mask))
    return output


train_context_feature_map = build_context_feature_map(splits["train"], train_context_features, train_context_masks)

# %%
class RscImageContextDataset(Dataset):
    def __init__(
        self,
        items: List[Tuple[str, int]],
        split_name: str,
        context_feature_map: Dict[str, Tuple[np.ndarray, float]],
        context_dim: int,
        transform=None,
    ):
        self.items = items
        self.split_name = split_name
        self.context_feature_map = context_feature_map
        self.context_dim = context_dim
        self.transform = transform

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        path, label = self.items[idx]

        try:
            img = load_rgb_image(path)
        except Exception as e:
            record_unreadable_image(self.split_name, path, label, e)
            return None
        if self.transform:
            img = self.transform(img)

        context_feature, context_mask = self.context_feature_map.get(path, (None, 0.0))
        if context_feature is None:
            context_feature = np.zeros(self.context_dim, dtype=np.float32)

        context_tensor = torch.tensor(context_feature, dtype=torch.float32)
        mask_tensor = torch.tensor(context_mask, dtype=torch.float32)

        return img, label, path, context_tensor, mask_tensor


aux_train_ds = RscImageContextDataset(
    items=splits["train"],
    split_name="train",
    context_feature_map=train_context_feature_map,
    context_dim=CONTEXT_FEATURE_DIM,
    transform=train_tf,
)

aux_train_loader = DataLoader(
    aux_train_ds,
    batch_size=CFG.batch_size,
    shuffle=True,
    num_workers=CFG.num_workers,
    pin_memory=CFG.pin_memory,
    collate_fn=collate_skip_none,
)
print(f"[context] Auxiliary train loader batches: {len(aux_train_loader)}")

# %%
class ConvNeXtContextAuxModel(nn.Module):
    def __init__(
        self,
        timm_name: str,
        num_classes: int,
        context_dim: int,
        fusion_hidden_dim: int = 512,
        align_dim: int = 256,
        dropout_p: float = 0.2,
    ):
        super().__init__()
        self.context_dim = int(context_dim)

        self.image_backbone = create_timm_model_cached(
            timm_name,
            pretrained=True,
            num_classes=0,
            global_pool="avg",
        )
        image_feat_dim = int(self.image_backbone.num_features)

        self.context_encoder = nn.Sequential(
            nn.LayerNorm(self.context_dim),
            nn.Linear(self.context_dim, fusion_hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout_p),
            nn.Linear(fusion_hidden_dim, fusion_hidden_dim),
            nn.GELU(),
        )

        self.image_align_head = nn.Sequential(
            nn.Linear(image_feat_dim, align_dim),
            nn.GELU(),
            nn.Linear(align_dim, align_dim),
        )
        self.context_align_head = nn.Sequential(
            nn.Linear(fusion_hidden_dim, align_dim),
            nn.GELU(),
            nn.Linear(align_dim, align_dim),
        )

        self.classifier = nn.Sequential(
            nn.Linear(image_feat_dim, fusion_hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout_p),
            nn.Linear(fusion_hidden_dim, num_classes),
        )

    def _encode_image(self, imgs: torch.Tensor) -> torch.Tensor:
        image_feat = self.image_backbone(imgs)
        if isinstance(image_feat, (list, tuple)):
            image_feat = image_feat[-1]
        if image_feat.ndim > 2:
            image_feat = torch.flatten(F.adaptive_avg_pool2d(image_feat, 1), 1)
        return image_feat

    def forward(
        self,
        imgs: torch.Tensor,
        context_vec: Optional[torch.Tensor] = None,
        context_mask: Optional[torch.Tensor] = None,
        return_aux: bool = False,
    ):
        image_feat = self._encode_image(imgs)
        logits = self.classifier(image_feat)

        if not return_aux:
            return logits

        batch_size = image_feat.size(0)
        if context_vec is None:
            context_vec = torch.zeros((batch_size, self.context_dim), device=image_feat.device, dtype=image_feat.dtype)
        else:
            context_vec = context_vec.to(device=image_feat.device, dtype=image_feat.dtype)

        if context_mask is None:
            context_mask = torch.zeros(batch_size, device=image_feat.device, dtype=image_feat.dtype)
        else:
            context_mask = context_mask.to(device=image_feat.device, dtype=image_feat.dtype).view(-1)

        valid = context_mask > 0.5
        if bool(valid.any()):
            img_emb = F.normalize(self.image_align_head(image_feat[valid]), dim=1)
            context_feat = self.context_encoder(context_vec[valid])
            ctx_emb = F.normalize(self.context_align_head(context_feat), dim=1)
            align_loss = 1.0 - (img_emb * ctx_emb).sum(dim=1).mean()
        else:
            align_loss = torch.tensor(0.0, device=image_feat.device, dtype=image_feat.dtype)

        aux = {
            "align_loss": align_loss,
            "context_ratio": context_mask.mean().detach(),
        }
        return logits, aux


def create_aux_convnext_model() -> Tuple[nn.Module, str]:
    timm_name = resolve_pretrained_timm_name(MODEL_SPECS["convnext"]["timm_name"])
    model = ConvNeXtContextAuxModel(timm_name=timm_name, num_classes=NUM_CLASSES, context_dim=CONTEXT_FEATURE_DIM)
    return model, timm_name


def empty_aux_eval_metrics() -> Dict[str, float]:
    metrics = empty_eval_metrics()
    metrics["align_loss"] = float("nan")
    return metrics


def save_alignment_curve(hist_df: pd.DataFrame, model_key: str) -> Path:
    model_dirs = get_model_output_dirs(model_key)
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(hist_df["epoch"], hist_df["train_align_loss"], label="train_align_loss")
    if "val_align_loss" in hist_df.columns and hist_df["val_align_loss"].notna().any():
        ax.plot(hist_df["epoch"], hist_df["val_align_loss"], label="val_align_loss")
    ax.set_title(f"Alignment Loss - {model_key}")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.legend()
    fig.tight_layout()
    align_curve_path = model_dirs["plots"] / "alignment_loss_curve.png"
    fig.savefig(align_curve_path, dpi=200)
    plt.close(fig)
    print(f"Saved alignment curve: {align_curve_path}")
    return align_curve_path


@torch.no_grad()
def run_eval_aux(model: nn.Module, loader: DataLoader, criterion: nn.Module) -> Dict[str, float]:
    metrics = run_eval(model, loader, criterion)
    metrics["align_loss"] = float("nan")
    return metrics


def train_aux_convnext_model(model_key: str = AUX_MODEL_KEY):
    model_dirs = get_model_output_dirs(model_key)

    model, timm_name = create_aux_convnext_model()
    model = model.to(device)

    criterion = nn.CrossEntropyLoss(weight=CLASS_WEIGHTS)
    optimizer = optim.AdamW(model.parameters(), lr=CFG.lr, weight_decay=CFG.weight_decay)
    scaler = GradScaler(enabled=(CFG.amp and device.type == "cuda"))

    has_val_split = len(val_ds) > 0
    monitor_name = "val_acc" if has_val_split else "train_acc"
    best_monitor_acc = -float("inf")
    ckpt_path = model_dirs["checkpoints"] / "best.pth"
    history = []

    for epoch in range(1, CFG.epochs + 1):
        model.train()
        epoch_losses, epoch_align_losses = [], []
        y_true_epoch, y_pred_epoch = [], []

        pbar = tqdm(aux_train_loader, desc=f"[{model_key}] Epoch {epoch:02d}/{CFG.epochs}", leave=False)
        for batch in pbar:
            if batch is None:
                continue
            imgs, labels, _, context_vec, context_mask = batch
            imgs = imgs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            context_vec = context_vec.to(device, non_blocking=True)
            context_mask = context_mask.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)

            with autocast(device_type="cuda", enabled=(CFG.amp and device.type == "cuda")):
                logits, aux = model(imgs, context_vec, context_mask, return_aux=True)
                cls_loss = criterion(logits, labels)
                align_loss = aux["align_loss"]
                loss = cls_loss + (CONTEXT_ALIGN_LOSS_WEIGHT * align_loss)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            epoch_losses.append(float(loss.item()))
            epoch_align_losses.append(float(align_loss.detach().cpu().item()))
            preds = torch.argmax(logits, dim=1)
            y_true_epoch.append(labels.detach().cpu().numpy())
            y_pred_epoch.append(preds.detach().cpu().numpy())

            y_true_tmp = np.concatenate(y_true_epoch)
            y_pred_tmp = np.concatenate(y_pred_epoch)
            m = compute_metrics(y_true_tmp, y_pred_tmp)
            pbar.set_postfix({
                "loss": float(np.mean(epoch_losses)),
                "align": float(np.mean(epoch_align_losses)),
                "acc": m["acc"],
                "f1": m["f1"],
            })

        if not y_true_epoch:
            raise RuntimeError(f"No readable training images were available for split 'train' in model {model_key}.")

        y_true_tr = np.concatenate(y_true_epoch)
        y_pred_tr = np.concatenate(y_pred_epoch)
        train_metrics = compute_metrics(y_true_tr, y_pred_tr)
        train_metrics["loss"] = float(np.mean(epoch_losses)) if epoch_losses else float("nan")
        train_metrics["align_loss"] = float(np.mean(epoch_align_losses)) if epoch_align_losses else float("nan")

        val_metrics = run_eval_aux(model, val_loader, criterion) if has_val_split else empty_aux_eval_metrics()

        row = {
            "epoch": epoch,
            "train_loss": train_metrics["loss"],
            "train_acc": train_metrics["acc"],
            "train_f1": train_metrics["f1"],
            "train_recall": train_metrics["recall"],
            "train_align_loss": train_metrics["align_loss"],
            "val_loss": val_metrics["loss"],
            "val_acc": val_metrics["acc"],
            "val_f1": val_metrics["f1"],
            "val_recall": val_metrics["recall"],
            "val_align_loss": val_metrics["align_loss"],
        }
        history.append(row)

        if has_val_split:
            print(
                f"[{model_key}] Epoch {epoch:02d} | "
                f"train loss {row['train_loss']:.4f} acc {row['train_acc']:.4f} f1 {row['train_f1']:.4f} "
                f"align {row['train_align_loss']:.4f} || "
                f"val loss {row['val_loss']:.4f} acc {row['val_acc']:.4f} f1 {row['val_f1']:.4f} "
                f"align {row['val_align_loss']:.4f}"
            )
        else:
            print(
                f"[{model_key}] Epoch {epoch:02d} | "
                f"train loss {row['train_loss']:.4f} acc {row['train_acc']:.4f} "
                f"f1 {row['train_f1']:.4f} align {row['train_align_loss']:.4f}"
            )

        monitor_value = row[monitor_name]
        if monitor_value >= best_monitor_acc:
            best_monitor_acc = float(monitor_value)
            torch.save(
                {
                    "model_key": model_key,
                    "timm_name": timm_name,
                    "num_classes": NUM_CLASSES,
                    "class_names": CLASS_NAMES,
                    "context_feature_dim": CONTEXT_FEATURE_DIM,
                    "context_align_loss_weight": CONTEXT_ALIGN_LOSS_WEIGHT,
                    "context_used_for_inference": False,
                    "state_dict": model.state_dict(),
                    "monitor_metric": monitor_name,
                    "monitor_value": best_monitor_acc,
                    "epoch": epoch,
                },
                ckpt_path,
            )
            print(f"Saved checkpoint to {ckpt_path} ({monitor_name}={best_monitor_acc:.4f})")

    hist_df = pd.DataFrame(history)
    hist_path = model_dirs["root"] / "history.csv"
    hist_df.to_csv(hist_path, index=False)
    print(f"History saved: {hist_path}")

    save_training_curves(hist_df, model_key)
    save_alignment_curve(hist_df, model_key)

    return model, hist_df, ckpt_path


@torch.no_grad()
def predict_proba_aux(model: nn.Module, loader: DataLoader):
    return predict_proba(model, loader)


def evaluate_aux_model_on_split(model_key: str, model: nn.Module, loader: DataLoader, split_name: str) -> Dict[str, float]:
    y_proba, y_true, y_pred, pred_conf, paths = predict_proba_aux(model, loader)

    return evaluate_prediction_artifacts(
        model_key=model_key,
        model=model,
        split_name=split_name,
        loader_for_gradcam=loader,
        paths=paths,
        y_true=y_true,
        y_pred=y_pred,
        y_proba=y_proba,
        pred_conf=pred_conf,
    )


def evaluate_aux_model_artifacts(model_key: str, model: nn.Module, ckpt_path: Path) -> Dict[str, Union[str, float]]:
    print()
    print("-" * 80)
    print(f"EVALUATING: {model_key} | ConvNeXt + JSON context auxiliary training (image-only inference)")
    print("-" * 80)

    result = {
        "model_key": model_key,
        "best_ckpt": str(ckpt_path),
    }

    if len(val_ds) > 0:
        result.update(evaluate_aux_model_on_split(model_key, model, val_loader, split_name="val"))
    else:
        print(f"[{model_key}] No validation split configured. All images under {Path(CFG.train_root).resolve()} were used for training.")
        result.update(empty_split_metric_fields("val"))

    result.update(evaluate_aux_model_on_split(model_key, model, test_loader, split_name="test"))
    unreadable_report = save_unreadable_image_report()
    if unreadable_report is not None:
        print(f"[dataset] Unreadable image report: {unreadable_report}")
    return result

# %%
# Train and evaluate ConvNeXt with auxiliary context learning.
aux_model, aux_hist_df, aux_ckpt_path = train_aux_convnext_model(AUX_MODEL_KEY)

aux_ckpt = torch.load(aux_ckpt_path, map_location="cpu")
aux_model.load_state_dict(aux_ckpt["state_dict"])
aux_model.to(device)

aux_result = evaluate_aux_model_artifacts(AUX_MODEL_KEY, aux_model, aux_ckpt_path)

if "results_summary" not in globals():
    results_summary = []
upsert_summary(results_summary, aux_result)

summary_df = pd.DataFrame(results_summary)
if not summary_df.empty:
    summary_df = summary_df.sort_values("test_acc", ascending=False).reset_index(drop=True)

summary_path = Path(CFG.output_dir) / "all_models_test_summary.csv"
summary_df.to_csv(summary_path, index=False)
print(f"Saved summary: {summary_path}")

summary_df

# %% [markdown]
# # Feature extraction using CLIP

# %%
# CLIP-based text feature extraction for auxiliary learning.
CLIP_AUX_MODEL_KEY = "convnext_aux_clip_context"
CLIP_TEXT_MODEL_NAME = "openai/clip-vit-base-patch32"
CLIP_TEXT_BATCH_SIZE = 32
CLIP_ALIGN_LOSS_WEIGHT = CONTEXT_ALIGN_LOSS_WEIGHT

try:
    from transformers import AutoTokenizer, CLIPTextModel
except Exception as e:
    raise RuntimeError(
        "transformers is required for CLIP text encoding. Install with: pip install transformers"
    ) from e

clip_local_snapshot = find_hf_snapshot_dir(
    CLIP_TEXT_MODEL_NAME,
    required_files=["config.json", "tokenizer.json", "tokenizer_config.json", "pytorch_model.bin"],
)
if clip_local_snapshot is not None:
    print(f"[local-cache] Using cached CLIP assets from {clip_local_snapshot}")
    clip_tokenizer = AutoTokenizer.from_pretrained(str(clip_local_snapshot), local_files_only=True)
    clip_text_encoder = CLIPTextModel.from_pretrained(str(clip_local_snapshot), local_files_only=True)
else:
    clip_tokenizer = AutoTokenizer.from_pretrained(CLIP_TEXT_MODEL_NAME)
    clip_text_encoder = CLIPTextModel.from_pretrained(CLIP_TEXT_MODEL_NAME)

clip_text_encoder = clip_text_encoder.to(device)
clip_text_encoder.eval()
for p in clip_text_encoder.parameters():
    p.requires_grad_(False)

CLIP_MAX_LEN = int(min(clip_tokenizer.model_max_length, 77))
CLIP_TEXT_HIDDEN_SIZE = int(clip_text_encoder.config.hidden_size)
print(f"[clip] text model: {CLIP_TEXT_MODEL_NAME} | max_len={CLIP_MAX_LEN}")


def encode_context_texts_with_clip(texts: List[str], batch_size: int = CLIP_TEXT_BATCH_SIZE) -> np.ndarray:
    if not texts:
        return np.zeros((0, CLIP_TEXT_HIDDEN_SIZE), dtype=np.float32)

    feats = []
    for start in tqdm(range(0, len(texts), batch_size), desc="CLIP text encode", leave=False):
        batch_texts = [t if str(t).strip() else "" for t in texts[start:start + batch_size]]
        tokenized = clip_tokenizer(
            batch_texts,
            padding=True,
            truncation=True,
            max_length=CLIP_MAX_LEN,
            return_tensors="pt",
        )
        tokenized = {k: v.to(device) for k, v in tokenized.items()}

        with torch.no_grad():
            outputs = clip_text_encoder(**tokenized)
            emb = outputs.pooler_output
            emb = F.normalize(emb, dim=1)

        feats.append(emb.detach().cpu().numpy().astype(np.float32))

    return np.concatenate(feats, axis=0)


clip_train_context_features = encode_context_texts_with_clip(train_context_texts)

clip_train_context_features[train_context_masks == 0] = 0.0

CLIP_CONTEXT_FEATURE_DIM = int(clip_train_context_features.shape[1])
print(f"[clip] context feature dim: {CLIP_CONTEXT_FEATURE_DIM}")

clip_train_context_feature_map = build_context_feature_map(
    splits["train"], clip_train_context_features, train_context_masks
)

# %%
# CLIP-context dataloaders (same image transforms as existing pipelines).
clip_aux_train_ds = RscImageContextDataset(
    items=splits["train"],
    split_name="train",
    context_feature_map=clip_train_context_feature_map,
    context_dim=CLIP_CONTEXT_FEATURE_DIM,
    transform=train_tf,
)

clip_aux_train_loader = DataLoader(
    clip_aux_train_ds,
    batch_size=CFG.batch_size,
    shuffle=True,
    num_workers=CFG.num_workers,
    pin_memory=CFG.pin_memory,
    collate_fn=collate_skip_none,
)
print(f"[clip] auxiliary train loader batches: {len(clip_aux_train_loader)}")

# %%
# ConvNeXt + CLIP-text auxiliary model training/evaluation.

def create_clip_aux_convnext_model() -> Tuple[nn.Module, str]:
    timm_name = resolve_pretrained_timm_name(MODEL_SPECS["convnext"]["timm_name"])
    model = ConvNeXtContextAuxModel(
        timm_name=timm_name,
        num_classes=NUM_CLASSES,
        context_dim=CLIP_CONTEXT_FEATURE_DIM,
    )
    return model, timm_name


def train_clip_aux_convnext_model(model_key: str = CLIP_AUX_MODEL_KEY):
    model_dirs = get_model_output_dirs(model_key)

    model, timm_name = create_clip_aux_convnext_model()
    model = model.to(device)

    criterion = nn.CrossEntropyLoss(weight=CLASS_WEIGHTS)
    optimizer = optim.AdamW(model.parameters(), lr=CFG.lr, weight_decay=CFG.weight_decay)
    scaler = GradScaler(enabled=(CFG.amp and device.type == "cuda"))

    has_val_split = len(val_ds) > 0
    monitor_name = "val_acc" if has_val_split else "train_acc"
    best_monitor_acc = -float("inf")
    ckpt_path = model_dirs["checkpoints"] / "best.pth"
    history = []

    for epoch in range(1, CFG.epochs + 1):
        model.train()
        epoch_losses, epoch_align_losses = [], []
        y_true_epoch, y_pred_epoch = [], []

        pbar = tqdm(clip_aux_train_loader, desc=f"[{model_key}] Epoch {epoch:02d}/{CFG.epochs}", leave=False)
        for batch in pbar:
            if batch is None:
                continue
            imgs, labels, _, context_vec, context_mask = batch
            imgs = imgs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            context_vec = context_vec.to(device, non_blocking=True)
            context_mask = context_mask.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)

            with autocast(device_type="cuda", enabled=(CFG.amp and device.type == "cuda")):
                logits, aux = model(imgs, context_vec, context_mask, return_aux=True)
                cls_loss = criterion(logits, labels)
                align_loss = aux["align_loss"]
                loss = cls_loss + (CLIP_ALIGN_LOSS_WEIGHT * align_loss)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            epoch_losses.append(float(loss.item()))
            epoch_align_losses.append(float(align_loss.detach().cpu().item()))
            preds = torch.argmax(logits, dim=1)
            y_true_epoch.append(labels.detach().cpu().numpy())
            y_pred_epoch.append(preds.detach().cpu().numpy())

            y_true_tmp = np.concatenate(y_true_epoch)
            y_pred_tmp = np.concatenate(y_pred_epoch)
            m = compute_metrics(y_true_tmp, y_pred_tmp)
            pbar.set_postfix({
                "loss": float(np.mean(epoch_losses)),
                "align": float(np.mean(epoch_align_losses)),
                "acc": m["acc"],
                "f1": m["f1"],
            })

        if not y_true_epoch:
            raise RuntimeError(f"No readable training images were available for split 'train' in model {model_key}.")

        y_true_tr = np.concatenate(y_true_epoch)
        y_pred_tr = np.concatenate(y_pred_epoch)
        train_metrics = compute_metrics(y_true_tr, y_pred_tr)
        train_metrics["loss"] = float(np.mean(epoch_losses)) if epoch_losses else float("nan")
        train_metrics["align_loss"] = float(np.mean(epoch_align_losses)) if epoch_align_losses else float("nan")

        val_metrics = run_eval_aux(model, val_loader, criterion) if has_val_split else empty_aux_eval_metrics()

        row = {
            "epoch": epoch,
            "train_loss": train_metrics["loss"],
            "train_acc": train_metrics["acc"],
            "train_f1": train_metrics["f1"],
            "train_recall": train_metrics["recall"],
            "train_align_loss": train_metrics["align_loss"],
            "val_loss": val_metrics["loss"],
            "val_acc": val_metrics["acc"],
            "val_f1": val_metrics["f1"],
            "val_recall": val_metrics["recall"],
            "val_align_loss": val_metrics["align_loss"],
        }
        history.append(row)

        if has_val_split:
            print(
                f"[{model_key}] Epoch {epoch:02d} | "
                f"train loss {row['train_loss']:.4f} acc {row['train_acc']:.4f} f1 {row['train_f1']:.4f} "
                f"align {row['train_align_loss']:.4f} || "
                f"val loss {row['val_loss']:.4f} acc {row['val_acc']:.4f} f1 {row['val_f1']:.4f} "
                f"align {row['val_align_loss']:.4f}"
            )
        else:
            print(
                f"[{model_key}] Epoch {epoch:02d} | "
                f"train loss {row['train_loss']:.4f} acc {row['train_acc']:.4f} "
                f"f1 {row['train_f1']:.4f} align {row['train_align_loss']:.4f}"
            )

        monitor_value = row[monitor_name]
        if monitor_value >= best_monitor_acc:
            best_monitor_acc = float(monitor_value)
            torch.save(
                {
                    "model_key": model_key,
                    "timm_name": timm_name,
                    "num_classes": NUM_CLASSES,
                    "class_names": CLASS_NAMES,
                    "context_feature_dim": CLIP_CONTEXT_FEATURE_DIM,
                    "context_align_loss_weight": CLIP_ALIGN_LOSS_WEIGHT,
                    "context_used_for_inference": False,
                    "state_dict": model.state_dict(),
                    "monitor_metric": monitor_name,
                    "monitor_value": best_monitor_acc,
                    "epoch": epoch,
                    "clip_text_model_name": CLIP_TEXT_MODEL_NAME,
                },
                ckpt_path,
            )
            print(f"Saved checkpoint to {ckpt_path} ({monitor_name}={best_monitor_acc:.4f})")

    hist_df = pd.DataFrame(history)
    hist_path = model_dirs["root"] / "history.csv"
    hist_df.to_csv(hist_path, index=False)
    print(f"History saved: {hist_path}")

    save_training_curves(hist_df, model_key)
    save_alignment_curve(hist_df, model_key)

    return model, hist_df, ckpt_path


def evaluate_clip_aux_model_artifacts(model_key: str, model: nn.Module, ckpt_path: Path) -> Dict[str, Union[str, float]]:
    print()
    print("-" * 80)
    print(f"EVALUATING: {model_key} | ConvNeXt + CLIP text context auxiliary training (image-only inference)")
    print("-" * 80)

    result = {
        "model_key": model_key,
        "best_ckpt": str(ckpt_path),
    }

    if len(val_ds) > 0:
        result.update(evaluate_aux_model_on_split(model_key, model, val_loader, split_name="val"))
    else:
        print(f"[{model_key}] No validation split configured. All images under {Path(CFG.train_root).resolve()} were used for training.")
        result.update(empty_split_metric_fields("val"))

    result.update(evaluate_aux_model_on_split(model_key, model, test_loader, split_name="test"))
    unreadable_report = save_unreadable_image_report()
    if unreadable_report is not None:
        print(f"[dataset] Unreadable image report: {unreadable_report}")
    return result

# %%
# Train and evaluate ConvNeXt + CLIP-text auxiliary learning.
clip_aux_model, clip_aux_hist_df, clip_aux_ckpt_path = train_clip_aux_convnext_model(CLIP_AUX_MODEL_KEY)

clip_aux_ckpt = torch.load(clip_aux_ckpt_path, map_location="cpu")
clip_aux_model.load_state_dict(clip_aux_ckpt["state_dict"])
clip_aux_model.to(device)

clip_aux_result = evaluate_clip_aux_model_artifacts(CLIP_AUX_MODEL_KEY, clip_aux_model, clip_aux_ckpt_path)

if "results_summary" not in globals():
    results_summary = []
upsert_summary(results_summary, clip_aux_result)

summary_df = pd.DataFrame(results_summary)
if not summary_df.empty:
    summary_df = summary_df.sort_values("test_acc", ascending=False).reset_index(drop=True)

summary_path = Path(CFG.output_dir) / "all_models_test_summary.csv"
summary_df.to_csv(summary_path, index=False)
print(f"Saved summary: {summary_path}")

summary_df

# %%
# Optional: ensure all three tracks are present in summary (baseline ConvNeXt, TF-IDF aux, CLIP aux).
required_keys = ["convnext", AUX_MODEL_KEY, CLIP_AUX_MODEL_KEY]
existing_keys = set(pd.DataFrame(results_summary).get("model_key", pd.Series(dtype=str)).tolist()) if "results_summary" in globals() else set()

if "results_summary" not in globals():
    results_summary = []

if "convnext" not in existing_keys:
    print("[run-all] convnext baseline missing -> training/evaluating now")
    base_model, base_hist_df, base_ckpt_path = train_one_model("convnext")
    base_ckpt = torch.load(base_ckpt_path, map_location="cpu")
    base_model.load_state_dict(base_ckpt["state_dict"])
    base_model.to(device)
    base_result = evaluate_model_artifacts("convnext", base_model, base_ckpt_path)
    upsert_summary(results_summary, base_result)

if AUX_MODEL_KEY not in existing_keys:
    print(f"[run-all] {AUX_MODEL_KEY} missing -> training/evaluating now")
    aux_model_all, aux_hist_all, aux_ckpt_all = train_aux_convnext_model(AUX_MODEL_KEY)
    aux_ckpt_loaded = torch.load(aux_ckpt_all, map_location="cpu")
    aux_model_all.load_state_dict(aux_ckpt_loaded["state_dict"])
    aux_model_all.to(device)
    aux_result_all = evaluate_aux_model_artifacts(AUX_MODEL_KEY, aux_model_all, aux_ckpt_all)
    upsert_summary(results_summary, aux_result_all)

if CLIP_AUX_MODEL_KEY not in existing_keys:
    print(f"[run-all] {CLIP_AUX_MODEL_KEY} missing -> training/evaluating now")
    clip_model_all, clip_hist_all, clip_ckpt_all = train_clip_aux_convnext_model(CLIP_AUX_MODEL_KEY)
    clip_ckpt_loaded = torch.load(clip_ckpt_all, map_location="cpu")
    clip_model_all.load_state_dict(clip_ckpt_loaded["state_dict"])
    clip_model_all.to(device)
    clip_result_all = evaluate_clip_aux_model_artifacts(CLIP_AUX_MODEL_KEY, clip_model_all, clip_ckpt_all)
    upsert_summary(results_summary, clip_result_all)

summary_df = pd.DataFrame(results_summary)
if not summary_df.empty:
    summary_df = summary_df.sort_values("test_acc", ascending=False).reset_index(drop=True)

summary_path = Path(CFG.output_dir) / "all_models_test_summary.csv"
summary_df.to_csv(summary_path, index=False)
print(f"Saved summary: {summary_path}")
summary_df

