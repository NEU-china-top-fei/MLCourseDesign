# -*- coding: utf-8 -*-
from __future__ import annotations

import os
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
import re
import gc
import sys
import json
import math
import random
import hashlib
import warnings
import platform
from pathlib import Path
from dataclasses import dataclass, asdict, replace
from collections import defaultdict
from contextlib import nullcontext

import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import GroupKFold, StratifiedGroupKFold


@dataclass
class Config:
    # Usually leave DATA_DIR empty: locate the attached competition automatically.
    INPUT_ROOT: str = "/kaggle/input"
    DATA_DIR: str | None = None
    LABEL_CSV: str | None = None
    SAMPLE_CSV: str | None = None
    OUTPUT_DIR: str = "/kaggle/working/fovea_run"
    SUBMISSION_PATH: str = "/kaggle/working/submission.csv"
    GROUP_COLUMN: str | None = None  # Optional real patient/subject column in labels.
    COORDINATE_ORIGIN: int = 0  # Change to 1 only if annotation documentation confirms it.
    DUPLICATE_LABEL_TOLERANCE_PX: float = 5.0  # Preserve small label differences; keep duplicates in one fold.
    SEED: int = 2026
    N_FOLDS: int = 5
    INPUT_SIZE: int = 512
    HEATMAP_SIZE: int = 128
    HEATMAP_SIGMA: float = 2.5
    BATCH_SIZE: int = 8
    NUM_WORKERS: int = 2
    EPOCHS: int = 80
    PATIENCE: int = 12
    LR: float = 2e-4
    WEIGHT_DECAY: float = 1e-4
    WARMUP_EPOCHS: int = 3
    AMP: bool = True
    GRAD_CLIP: float = 1.0
    FREEZE_BN: bool = True
    COORD_SCALE: float = 512.0
    HEATMAP_LOSS_WEIGHT: float = 1.0
    COORD_LOSS_WEIGHT: float = 10.0
    VIS_LOSS_WEIGHT: float = 1.0
    PRETRAINED: bool = True
    ALLOW_WEIGHT_DOWNLOAD: bool = True
    PRETRAINED_WEIGHTS_PATH: str | None = None
    VISIBILITY_THRESHOLD: float = 0.5
    # False avoids reporting a threshold fitted on OOF labels as unbiased CV.
    TUNE_VISIBILITY_THRESHOLD: bool = False
    TTA: bool = True  # Original + horizontal flip; both decoded in original pixels.
    SHOW_PLOTS: bool = True
    REUSE_CHECKPOINTS: bool = False  # True only with identical data/config/code.
    SMOKE_TEST: bool = False  # 2 folds, 2 epochs, low resolution; NOT a scored model.


CFG = Config()


@dataclass
class ResearchConfig(Config):
    OUTPUT_DIR: str = "/kaggle/working/fovea_optimized"
    SUBMISSION_PATH: str = "/kaggle/working/submission.csv"
    PRETRAINED: bool = True
    ALLOW_WEIGHT_DOWNLOAD: bool = False
    PRETRAINED_WEIGHTS_PATH: str | None = None
    N_FOLDS: int = 5
    CALIBRATION_FRACTION: float = 0.25
    EARLY_STOP_FRACTION: float = 0.20
    INNER_CROP_FOLDS: int = 3
    FINE_TRAIN_CROPS: str = "oof"  # "jitter" is a cheaper, explicitly different ablation.
    COARSE_EPOCHS: int = 50
    FINE_EPOCHS: int = 40
    PATIENCE: int = 8
    INPUT_SIZE: int = 512
    HEATMAP_SIZE: int = 128
    FINE_INPUT_SIZE: int = 384
    FINE_HEATMAP_SIZE: int = 96
    BATCH_SIZE: int = 8
    NUM_WORKERS: int = 2
    BACKBONE_LR_MULTIPLIER: float = 1.0  # 0.25 is an optional candidate, not a proven improvement.
    USE_EMA: bool = False  # Enable only after an equal-budget development comparison.
    EMA_DECAY: float = 0.99
    EMA_WARMUP_UPDATES: int = 10
    ROI_FRACTIONS: tuple = (0.15, 0.20, 0.30, 0.40, 0.60)
    FINE_MULTISCALE_TRAIN: bool = True  # Preserve OOF centre; sample the training field of view.
    SELECT_FINE_CROP_POLICY: bool = True  # Compare fixed/multiscale only on each fold's inner stopping split.
    FINE_MULTISCALE_EARLY_STOP: bool = True
    FINE_VALIDATION_ROI_FRACTIONS: tuple = (0.20, 0.30, 0.40)  # Prespecified inner-only mean RMSE.
    OOF_CENTER_JITTER: float = 0.08
    TRAIN_CENTER_JITTER: float = 0.20
    FIXED_ROI_FRACTION: float = 0.30
    ROI_MIN_FRACTION: float = 0.15
    ROI_MAX_FRACTION: float = 0.60
    ROI_MARGIN_PX: float = 16.0
    ROI_COVERAGE: float = 0.90
    RAW_RADIUS_MULTIPLIER: float = 2.0
    UNCERTAINTY_FLOOR_PX: float = 1.0
    FALLBACK_RULE_CANDIDATES: tuple = ((0.0, 2.0), (0.03, 4.0), (0.08, 1e9))
    FALLBACK_MIN_GAIN_PX: float = 1.0
    FALLBACK_MIN_RELATIVE_GAIN: float = 0.01
    FALLBACK_MIN_GROUPS: int = 12
    EXPERIMENTS: tuple = ("B0", "B1", "B2", "B3", "B4", "C1")
    SUBMISSION_EXPERIMENT: str = "B4"  # Preselected; never auto-pick the best outer OOF score.
    BOOTSTRAP_REPEATS: int = 1000
    REUSE_CHECKPOINTS: bool = True  # Reuse matching completed stages/folds after integrity checks.
    SHOW_PLOTS: bool = False
    SMOKE_TEST: bool = False
    SMOKE_FOLDS: int = 2
    SMOKE_INPUT_SIZE: int = 64
    SMOKE_EPOCHS: int = 1
    REQUIRE_GPU: bool = True
    MODEL_SEED_OFFSET: int = 0  # Changes model/augmentation RNG only; all data splits remain unchanged.


CFG = ResearchConfig()
RESEARCH_VERSION = "3.0.0"
EXPERIMENT_DESCRIPTIONS = {
    "B0": "Single-stage coarse heatmap + DSNT",
    "B1": "Coarse-to-fine, fixed ROI",
    "B2": "Coarse-to-fine, uncalibrated uncertainty ROI",
    "B3": "Coarse-to-fine, empirically calibrated uncertainty ROI",
    "B4": "B3 plus calibration-selected coarse fallback",
    "C1": "Fixed ROI matched to B3 mean calibration crop area",
}


# Official pretrained weights: resolve local Kaggle inputs without internet.
# The bundle contains ImageNet-1K initialization, not fovea-trained weights.
RESEARCH_WEIGHT_FILENAME = "resnet18-f37072fd.pth"
RESEARCH_WEIGHT_SHA256 = "f37072fd47e89c5e827621c5baffa7500819f7896bbacec160b1a16c560e07ec"
RESEARCH_WEIGHT_BYTES = 46830571
RESEARCH_WEIGHT_URL = "https://download.pytorch.org/models/resnet18-f37072fd.pth"


def resolve_research_weights(cfg):
    """Return a SHA256-verified .pth path, or None only if PRETRAINED=False.

    PRETRAINED_WEIGHTS_PATH accepts a .pth, the supplied ZIP, or a directory.
    With no explicit path, recursively discover the official filename or
    fovea_pretrained_weights.zip under /kaggle/input. Kaggle may unpack uploaded
    ZIPs automatically; both layouts work. All matching copies must have the
    expected complete SHA256. A ZIP is streamed to a fixed filename under
    OUTPUT_DIR/pretrained_cache; input datasets are never modified.
    """
    from pathlib import Path, PurePosixPath
    import hashlib
    import os
    import urllib.request
    import zipfile

    if not bool(getattr(cfg, "PRETRAINED", True)):
        print("PRETRAINED=False: explicitly using random encoder initialization.")
        return None

    def checked_stream(stream, label, destination=None):
        checksum = hashlib.sha256()
        total = 0
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            total += len(block)
            if total > RESEARCH_WEIGHT_BYTES:
                raise ValueError(f"Pretrained weight is larger than expected: {label}")
            checksum.update(block)
            if destination is not None:
                destination.write(block)
        actual = checksum.hexdigest()
        if total != RESEARCH_WEIGHT_BYTES or actual != RESEARCH_WEIGHT_SHA256:
            raise ValueError(
                f"Pretrained weight verification failed: {label}\n"
                f"Expected {RESEARCH_WEIGHT_BYTES} bytes / SHA256 {RESEARCH_WEIGHT_SHA256}; "
                f"got {total} bytes / SHA256 {actual}. "
                "Remove the corrupt or incompatible input and attach the supplied official bundle."
            )

    def check_file(path):
        with path.open("rb") as stream:
            checked_stream(stream, str(path))

    explicit = str(getattr(cfg, "PRETRAINED_WEIGHTS_PATH", "") or "").strip()
    root = Path(explicit).expanduser() if explicit else Path("/kaggle/input")
    if explicit and not root.exists():
        raise FileNotFoundError(f"PRETRAINED_WEIGHTS_PATH does not exist: {root}")
    if root.is_file():
        inputs = [root]
    elif root.is_dir():
        inputs = sorted(set(root.rglob(RESEARCH_WEIGHT_FILENAME)) |
                        set(root.rglob("fovea_pretrained_weights.zip")))
    else:
        inputs = []

    # Verify every matching source before choosing one: conflicting copies fail.
    candidates = []
    for path in inputs:
        if path.suffix.lower() == ".zip":
            with zipfile.ZipFile(path, "r") as archive:
                members = [info for info in archive.infolist()
                           if not info.is_dir() and
                           PurePosixPath(info.filename.replace("\\", "/")).name == RESEARCH_WEIGHT_FILENAME]
                if not members:
                    raise ValueError(f"No {RESEARCH_WEIGHT_FILENAME} found inside {path}")
                for info in members:
                    entry = PurePosixPath(info.filename.replace("\\", "/"))
                    if entry.is_absolute() or ".." in entry.parts:
                        raise ValueError(f"Unsafe ZIP member name: {info.filename}")
                    if info.file_size != RESEARCH_WEIGHT_BYTES:
                        raise ValueError(f"Wrong pretrained member size: {path}!{info.filename}")
                    with archive.open(info, "r") as stream:
                        checked_stream(stream, f"{path}!{info.filename}")
                    candidates.append((path, info.filename))
        else:
            check_file(path)
            candidates.append((path, None))

    if candidates:
        # Prefer already-unpacked input when duplicate copies are identical.
        plain = [path for path, member in candidates if member is None]
        if plain:
            selected = plain[0].resolve()
            print(f"Verified official ImageNet-1K ResNet18 weights: {selected}")
            return str(selected)

    if not candidates and explicit:
        raise FileNotFoundError(
            f"No {RESEARCH_WEIGHT_FILENAME} or fovea_pretrained_weights.zip found under {root}."
        )

    cache_dir = (Path(cfg.OUTPUT_DIR).expanduser() / "pretrained_cache").resolve()
    kaggle_input = Path("/kaggle/input").resolve()
    if cache_dir == kaggle_input or kaggle_input in cache_dir.parents:
        raise ValueError("OUTPUT_DIR must be writable and outside /kaggle/input.")
    cache = cache_dir / RESEARCH_WEIGHT_FILENAME
    if cache.exists():
        check_file(cache)
        print(f"Verified pretrained weight cache: {cache}")
        return str(cache)

    if not candidates and not bool(getattr(cfg, "ALLOW_WEIGHT_DOWNLOAD", False)):
        raise FileNotFoundError(
            "Pretrained weights were not found. Upload fovea_pretrained_weights.zip as a Kaggle "
            "Dataset and attach it with Add Input, or set PRETRAINED_WEIGHTS_PATH to its .pth, "
            "ZIP, or folder. For an online download, explicitly set ALLOW_WEIGHT_DOWNLOAD=True "
            "and enable Notebook Internet. No random-initialization fallback is performed."
        )

    cache_dir.mkdir(parents=True, exist_ok=True)
    temporary = cache_dir / f"{RESEARCH_WEIGHT_FILENAME}.{os.getpid()}.partial"
    try:
        if candidates:
            archive_path, member = candidates[0]
            with zipfile.ZipFile(archive_path, "r") as archive:
                with archive.open(member, "r") as stream, temporary.open("wb") as output:
                    checked_stream(stream, f"{archive_path}!{member}", output)
        else:
            print(f"Downloading official ImageNet-1K weights: {RESEARCH_WEIGHT_URL}")
            request = urllib.request.Request(RESEARCH_WEIGHT_URL, headers={"User-Agent": "fovea-research-notebook/1.0"})
            with urllib.request.urlopen(request, timeout=30) as stream, temporary.open("wb") as output:
                checked_stream(stream, RESEARCH_WEIGHT_URL, output)
        temporary.replace(cache)
    finally:
        if temporary.exists():
            temporary.unlink()
    print(f"Verified pretrained weights available at {cache}")
    return str(cache)


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    # warn_only: some CUDA kernels do not have deterministic implementations.
    torch.use_deterministic_algorithms(True, warn_only=True)


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % (2**32)
    random.seed(worker_seed)
    np.random.seed(worker_seed)
    cv2.setNumThreads(0)


def validate_config(cfg):
    assert cfg.INPUT_SIZE >= 64 and cfg.INPUT_SIZE % 32 == 0
    assert cfg.HEATMAP_SIZE >= 8 and cfg.HEATMAP_SIGMA > 0
    assert cfg.N_FOLDS >= 2 and cfg.EPOCHS >= 1 and cfg.BATCH_SIZE >= 1
    assert cfg.COORDINATE_ORIGIN in (0, 1)
    assert cfg.COORD_SCALE > 0
    assert cfg.DUPLICATE_LABEL_TOLERANCE_PX >= 0
    assert 0 <= cfg.VISIBILITY_THRESHOLD <= 1
    return cfg


def csv_read(path):
    # Strings preserve leading zeroes in image names and exact submission IDs.
    frame = pd.read_csv(path, dtype=str, keep_default_na=False, encoding="utf-8-sig")
    frame.columns = frame.columns.str.strip()
    if frame.columns.duplicated().any():
        raise ValueError(f"Duplicate CSV columns: {path}")
    return frame


def column_name(frame, expected):
    matches = [c for c in frame.columns if c.casefold() == expected.casefold()]
    if len(matches) != 1:
        raise ValueError(f"Expected column {expected!r}; found {list(frame.columns)}")
    return matches[0]


def discover_data(cfg):
    base = Path(cfg.DATA_DIR or cfg.INPUT_ROOT)
    if not base.exists():
        raise FileNotFoundError(
            f"{base} does not exist. In Kaggle, Add Input -> Competition -> 2026task3, "
            "then set CFG.DATA_DIR if automatic discovery is ambiguous."
        )
    csvs = sorted(base.rglob("*.csv"))
    if cfg.LABEL_CSV:
        label_path = Path(cfg.LABEL_CSV)
    else:
        candidates = [p for p in csvs if p.name.casefold() == "fovea_localization_train_gt.csv"]
        if not candidates:
            candidates = []
            for p in csvs:
                try:
                    cols = set(pd.read_csv(p, nrows=0).columns.str.strip().str.casefold())
                    if {"data", "fovea_x", "fovea_y"} <= cols:
                        candidates.append(p)
                except Exception:
                    pass
        if len(candidates) != 1:
            raise ValueError(
                "Cannot uniquely locate labels. Set CFG.DATA_DIR or CFG.LABEL_CSV. "
                f"Candidates: {[str(p) for p in candidates]}"
            )
        label_path = candidates[0]
    if not label_path.is_file():
        raise FileNotFoundError(label_path)

    if cfg.SAMPLE_CSV:
        sample_path = Path(cfg.SAMPLE_CSV)
        root = base if cfg.DATA_DIR else Path(os.path.commonpath([label_path.parent, sample_path.parent]))
    else:
        # Search nearest enclosing directory first, avoiding other attached datasets.
        root = label_path.parent
        while True:
            candidates = [p for p in root.rglob("*.csv") if p.name.casefold() == "sample_submission.csv"]
            if candidates:
                if len(candidates) != 1:
                    raise ValueError(f"Several sample submissions under {root}; set CFG.SAMPLE_CSV.")
                sample_path = candidates[0]
                break
            if root == base or root.parent == root or base not in root.parents:
                raise FileNotFoundError("sample_submission.csv missing. Set CFG.SAMPLE_CSV explicitly.")
            root = root.parent
        if cfg.DATA_DIR:
            root = base
    if not sample_path.is_file():
        raise FileNotFoundError(sample_path)
    images = sorted(p for p in root.rglob("*") if p.suffix.casefold() in IMAGE_EXTENSIONS)
    if not images:
        raise FileNotFoundError(
            f"No extracted images under {root}. Add the competition data in Kaggle; "
            "if your own upload contains only a ZIP, extract it and set CFG.DATA_DIR."
        )
    print(f"Data root: {root}\nLabels: {label_path}\nSample: {sample_path}\nImage files: {len(images)}")
    print("Image path examples:", [str(p.relative_to(root)) for p in images[:12]])
    return root, label_path, sample_path, images


def canonical_image_id(value):
    name = str(value).strip().replace("\\", "/").split("/")[-1]
    return Path(name).stem.casefold() if Path(name).suffix.casefold() in IMAGE_EXTENSIONS else name.casefold()


def parse_submission_id(value):
    text = str(value)
    # Remove only a recognized suffix; retain underscores and leading zeroes in image names.
    match = re.search(r"(?i)(?:[_\-\s]+)?fovea[_\-\s]*([xy])\s*$", text)
    if match is None:
        match = re.search(r"(?i)[_\-\s]+([xy])\s*$", text)
    if match is None:
        raise ValueError(f"Unrecognized ImageID {text!r}; expected image name + Fovea_X/Y.")
    name = text[:match.start()].rstrip("_- ")
    if not name:
        raise ValueError(f"Empty image name in ImageID {text!r}")
    return canonical_image_id(name), match.group(1).lower()


class ImageIndex(defaultdict):
    def __init__(self, root=None):
        super().__init__(list)
        self.root = Path(root) if root is not None else None


def build_image_index(paths, root=None):
    index = ImageIndex(root)
    for path in paths:
        path = Path(path)
        index[canonical_image_id(path.name)].append(path)
    return index


def image_split_role(path, root=None):
    """Use the nearest explicit split directory, limited to the dataset root."""
    path = Path(path)
    if root is not None:
        try:
            parts = path.relative_to(root).parts[:-1]
        except ValueError:
            raise ValueError(f"Image path {path} is outside the selected dataset root {root}.")
    else:
        parts = path.parts[:-1]
    for part in reversed(parts):
        tokens = set(re.split(r"[^a-z0-9]+", part.casefold()))
        is_train = bool(tokens & {"train", "training"})
        is_test = bool(tokens & {"test", "testing"})
        if is_train or is_test:
            return "ambiguous" if is_train and is_test else ("train" if is_train else "test")
    filename_role = re.match(r"^(train(?:ing)?|test(?:ing)?)[_\-\s.]", path.stem.casefold())
    if filename_role:
        return "train" if filename_role.group(1).startswith("train") else "test"
    return None


def numeric_image_signature(value, csv_identifier=False):
    """Single numeric block only: never extract a patient/date's trailing number."""
    key = canonical_image_id(value)
    if csv_identifier and re.fullmatch(r"[0-9]+\.0+", key):
        key = key.split(".")[0]  # Spreadsheet-exported integer, e.g. '1.0'.
    match = re.fullmatch(r"([^0-9]*)([0-9]+)([^0-9]*)", key)
    if match is None:
        return None
    prefix, number, suffix = match.groups()
    return prefix, number.lstrip("0") or "0", suffix


def resolve_image(index, image_id, role):
    if role not in {"train", "test"}:
        raise ValueError(f"Unknown image role: {role}")
    root = getattr(index, "root", None)
    all_paths = list(dict.fromkeys(path for paths in index.values() for path in paths))
    roles = {path: image_split_role(path, root) for path in all_paths}

    def scope(paths):
        # Never use an explicitly opposite-split image, even if its name is exact.
        eligible = [path for path in paths if roles[path] in (None, role)]
        explicit = [path for path in eligible if roles[path] == role]
        return explicit if explicit else eligible

    key = canonical_image_id(image_id)
    matches = scope(index.get(key, []))
    stage = "exact filename"
    if not matches:
        signature = numeric_image_signature(image_id, csv_identifier=True)
        if signature is not None:
            prefix, number, suffix = signature
            eligible = scope(all_paths)
            signatures = {path: numeric_image_signature(path.name) for path in eligible}
            # Prefer the same textual prefix/suffix: '01' -> '001', 'eye_1' -> 'eye_001'.
            matches = [path for path, item in signatures.items() if item == (prefix, number, suffix)]
            stage = "same prefix/suffix with normalized numeric ID"
            if not matches and prefix == "" and suffix == "":
                # CSV '1' may identify 'image_001.jpg'. Require a unique single-number
                # filename in this split; 'patient12_eye1' is deliberately not guessed.
                matches = [path for path, item in signatures.items() if item is not None and item[1] == number]
                stage = "unique numeric ID in a single-number filename"
    if len(matches) != 1:
        examples = [str(path.relative_to(root) if root is not None else path) for path in scope(all_paths)[:12]]
        raise ValueError(
            f"{role} image {image_id!r}: expected one unambiguous file at {stage}, "
            f"found {[str(p) for p in matches]}. Available {role} examples: {examples}. "
            "Check the label/file naming convention; no positional or substring matching is used."
        )
    return matches[0]


def read_rgb(path):
    # PIL supports non-ASCII local paths; do not apply EXIF rotations without changing labels.
    with Image.open(path) as image:
        return np.asarray(image.convert("RGB"))


def prepare_tables(cfg):
    root, label_path, sample_path, images = discover_data(cfg)
    labels, sample = csv_read(label_path), csv_read(sample_path)
    dc, xc, yc = [column_name(labels, n) for n in ("data", "Fovea_X", "Fovea_Y")]
    ic, vc = [column_name(sample, n) for n in ("ImageID", "value")]
    if list(sample.columns) != [ic, vc]:
        raise ValueError("Expected sample_submission to contain exactly ImageID and value.")
    index = build_image_index(images, root=root)
    train = pd.DataFrame({
        "image_id": labels[dc].map(canonical_image_id),
        "source_id": labels[dc],
        "x": pd.to_numeric(labels[xc], errors="raise"),
        "y": pd.to_numeric(labels[yc], errors="raise"),
    })
    if not len(train) or train.image_id.duplicated().any():
        raise ValueError("Training labels are empty or contain repeated image IDs.")
    if not np.isfinite(train[["x", "y"]].to_numpy(float)).all():
        raise ValueError("Training coordinates contain NaN or infinity.")
    train["visible"] = ~((train.x == 0) & (train.y == 0))
    # Internally all image coordinates are zero-based pixel centres. Preserve sentinel.
    train.loc[train.visible, ["x", "y"]] -= cfg.COORDINATE_ORIGIN
    train["path"] = [str(resolve_image(index, name, "train")) for name in train.source_id]
    if train.path.duplicated().any():
        raise ValueError(f"Different training IDs map to the same file:\n{train.loc[train.path.duplicated(False), ['source_id', 'path']]}")

    if sample[ic].duplicated().any():
        raise ValueError("Duplicate submission ImageIDs.")
    parsed = [parse_submission_id(v) for v in sample[ic]]
    layout = pd.DataFrame(parsed, columns=["image_id", "axis"])
    counts = layout.groupby(["image_id", "axis"]).size().unstack(fill_value=0)
    if set(counts.columns) != {"x", "y"} or not (counts[["x", "y"]] == 1).all().all():
        raise ValueError("Each test image must have exactly one X row and one Y row.")
    test_ids = list(dict.fromkeys(layout.image_id))
    test = pd.DataFrame({"image_id": test_ids})
    test["path"] = [str(resolve_image(index, name, "test")) for name in test_ids]
    if test.path.duplicated().any():
        raise ValueError(f"Different test IDs map to the same file:\n{test.loc[test.path.duplicated(False), ['image_id', 'path']]}")
    if set(train.path) & set(test.path):
        raise ValueError("A physical image is referenced by both train and test. Check ID mapping.")
    print("Training ID -> file examples:", list(zip(train.source_id.head(8), train.path.map(lambda p: Path(p).name).head(8))))
    print("Test ID -> file examples:", list(zip(test.image_id.head(8), test.path.map(lambda p: Path(p).name).head(8))))

    for frame, role in ((train, "train"), (test, "test")):
        sizes = []
        hashes = []
        for path in frame.path:
            rgb = read_rgb(path)
            h, w = rgb.shape[:2]
            if min(w, h) < 2:
                raise ValueError(f"Invalid image dimensions: {path}")
            sizes.append((w, h))
            hashes.append(hashlib.sha256(str(rgb.shape).encode() + rgb.tobytes()).hexdigest())
        frame[["width", "height"]] = pd.DataFrame(sizes, index=frame.index)
        frame["pixel_hash"] = hashes
        if role == "train":
            bad = frame.visible & ((frame.x < 0) | (frame.y < 0) |
                                   (frame.x > frame.width - 1) | (frame.y > frame.height - 1))
            if bad.any():
                raise ValueError(f"Visible labels outside zero-based image bounds:\n{frame.loc[bad]}")
    if set(train.pixel_hash) & set(test.pixel_hash):
        warnings.warn("Exact pixel duplicates exist between train and test; inspect dataset provenance.")
    if not train.visible.any():
        raise ValueError("All training labels are (0,0): cannot train a supervised localization model.")

    group_column = cfg.GROUP_COLUMN
    if group_column is None:
        group_column = next((c for c in labels if c.casefold() in {"patient_id", "subject_id", "patient", "subject"}), None)
    if group_column is not None:
        if group_column not in labels:
            raise ValueError(f"GROUP_COLUMN {group_column!r} missing from label CSV.")
        raw_groups = labels[group_column].str.strip()
        if (raw_groups == "").any():
            raise ValueError("Patient group column contains empty values.")
        train["patient_group"] = raw_groups
    else:
        train["patient_group"] = train.image_id
        warnings.warn("No patient IDs available: grouping exact duplicates only; patient independence is unverified.")
    train["group"] = connected_groups(train)
    duplicate_audit = []
    for pixel_hash, group in train.groupby("pixel_hash"):
        if len(group) < 2:
            continue
        points = group[["x", "y"]].to_numpy(float)
        maximum_distance = float(np.linalg.norm(points[:, None, :] - points[None, :, :], axis=2).max())
        consistent_visibility = group.visible.nunique() == 1
        ids = group.image_id.tolist()
        duplicate_audit.append({
            "pixel_hash": pixel_hash,
            "image_ids": ",".join(ids),
            "count": len(group),
            "max_label_distance_px": maximum_distance,
            "visibility_consistent": consistent_visibility,
            "policy": "keep original rows and coordinates; force same validation fold",
        })
        if not consistent_visibility:
            raise ValueError(f"Identical pixels have conflicting visible/invisible labels: {ids}")
        if maximum_distance > cfg.DUPLICATE_LABEL_TOLERANCE_PX:
            raise ValueError(
                f"Identical pixels {ids} have label differences of {maximum_distance:.3f}px, "
                f"above DUPLICATE_LABEL_TOLERANCE_PX={cfg.DUPLICATE_LABEL_TOLERANCE_PX}. "
                "Inspect the annotations before changing the tolerance."
            )
        if maximum_distance > 0:
            warnings.warn(
                f"Duplicate images {ids} differ in point annotation by {maximum_distance:.3f}px. "
                "Keeping both ORIGINAL labels and placing them in the SAME fold; see duplicate_label_audit.csv."
            )
    print(f"Train={len(train)}, test={len(test)}, submission rows={len(sample)}")
    print(f"Visible={int(train.visible.sum())}, invisible={int((~train.visible).sum())}, groups={train.group.nunique()}")
    if len(train) != 80 or len(test) != 20:
        warnings.warn("Counts differ from the published 80 train / 20 test. Verify the selected data.")
    return train, test, sample, layout, {"root": str(root), "label_csv": str(label_path),
                                      "sample_csv": str(sample_path), "id_column": ic, "value_column": vc,
                                      "duplicate_label_audit": duplicate_audit}


def connected_groups(frame):
    # Union both patient and duplicate relationships; neither may cross a fold.
    parent = list(range(len(frame)))
    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a
    def union(a, b):
        parent[find(a)] = find(b)
    for column in ("patient_group", "pixel_hash"):
        first = {}
        for i, key in enumerate(frame[column]):
            if key in first:
                union(i, first[key])
            else:
                first[key] = i
    return [f"group_{find(i):04d}" for i in range(len(frame))]


def assign_folds(train, cfg):
    frame = train.copy()
    n_folds = min(cfg.N_FOLDS, frame.group.nunique())
    if n_folds < 2:
        raise ValueError("At least two independent groups are required for cross-validation.")
    class_groups = frame.groupby("visible").group.nunique()
    if len(class_groups) == 2 and class_groups.min() >= 2:
        n_folds = min(n_folds, int(class_groups.min()))
        splitter = StratifiedGroupKFold(n_splits=n_folds, shuffle=True, random_state=cfg.SEED)
        splits = splitter.split(frame, frame.visible.astype(int), frame.group)
    else:
        # Randomize group labels before deterministic GroupKFold assignment.
        names = frame.group.unique().copy()
        np.random.default_rng(cfg.SEED).shuffle(names)
        group_numbers = frame.group.map({name: i for i, name in enumerate(names)})
        splits = GroupKFold(n_splits=n_folds).split(frame, groups=group_numbers)
        if len(class_groups) == 2:
            warnings.warn("Too few independent invisible/visible groups for stratification; visibility CV is unreliable.")
    frame["fold"] = -1
    for fold, (tr, va) in enumerate(splits):
        if set(frame.iloc[tr].group) & set(frame.iloc[va].group):
            raise AssertionError("Group leakage in fold assignment.")
        if not frame.iloc[tr].visible.any():
            raise ValueError("A training fold has no visible examples; reduce folds or inspect group labels.")
        frame.loc[frame.index[va], "fold"] = fold
    assert (frame.fold >= 0).all()
    print(frame.groupby("fold").agg(images=("image_id", "size"), visible=("visible", "sum")))
    return frame


def letterbox_geometry(width, height, size):
    factor = min(size / width, size / height)
    new_width = min(size, max(1, int(round(width * factor))))
    new_height = min(size, max(1, int(round(height * factor))))
    left, top = (size - new_width) // 2, (size - new_height) // 2
    sx, sy = new_width / width, new_height / height
    # OpenCV resize maps pixel centres: dst = scale*(src+0.5)-0.5.
    transform = np.array([[sx, 0, left + (sx - 1) / 2],
                          [0, sy, top + (sy - 1) / 2],
                          [0, 0, 1]], dtype=np.float64)
    return new_width, new_height, left, top, transform


def letterbox(image, size):
    height, width = image.shape[:2]
    nw, nh, left, top, transform = letterbox_geometry(width, height, size)
    resized = cv2.resize(image, (nw, nh), interpolation=cv2.INTER_AREA if nw < width else cv2.INTER_LINEAR)
    canvas = np.zeros((size, size, 3), dtype=np.uint8)
    canvas[top:top + nh, left:left + nw] = resized
    return canvas, transform


def gaussian_heatmap(xy, size, sigma):
    x, y = np.asarray(xy, dtype=np.float32) * (size - 1)
    axis = np.arange(size, dtype=np.float32)
    values = np.exp(-((axis[None, :] - x)**2 + (axis[:, None] - y)**2) / (2 * sigma**2))
    values /= max(float(values.sum()), 1e-12)
    return values[None].astype(np.float32)


class FoveaDataset(Dataset):
    def __init__(self, frame, cfg, training=False):
        self.frame = frame.reset_index(drop=True)
        self.cfg = cfg
        self.training = training

    def __len__(self):
        return len(self.frame)

    def __getitem__(self, index):
        row = self.frame.iloc[index]
        image = read_rgb(row.path)
        height, width = image.shape[:2]
        size = self.cfg.INPUT_SIZE
        image, transform = letterbox(image, size)
        has_label = "visible" in self.frame.columns
        visible = bool(row.visible) if has_label else False
        original_xy = np.array([row.x, row.y], dtype=np.float64) if has_label else np.zeros(2)
        point = (transform @ np.r_[original_xy, 1.0])[:2] if visible else np.zeros(2)

        if self.training:
            # Retry if a visible point leaves the input; never relabel it invisible.
            for _ in range(8):
                rotate = cv2.getRotationMatrix2D(((size - 1) / 2, (size - 1) / 2),
                                                np.random.uniform(-15, 15),
                                                np.random.uniform(0.93, 1.07))
                rotate[:, 2] += np.random.uniform(-0.03, 0.03, 2) * size
                aug = np.vstack([rotate, [0, 0, 1]])
                if np.random.random() < 0.5:
                    flip = np.array([[-1, 0, size - 1], [0, 1, 0], [0, 0, 1]], dtype=float)
                    aug = flip @ aug
                candidate = (aug @ np.r_[point, 1.0])[:2] if visible else point
                if not visible or np.all((candidate >= 0) & (candidate <= size - 1)):
                    image = cv2.warpAffine(image, aug[:2], (size, size), flags=cv2.INTER_LINEAR,
                                           borderMode=cv2.BORDER_CONSTANT, borderValue=0)
                    transform = aug @ transform
                    point = candidate
                    break
            gain = np.random.uniform(0.85, 1.15)
            bias = np.random.uniform(-10, 10)
            image = np.clip(image.astype(np.float32) * gain + bias, 0, 255).astype(np.uint8)

        xy = (point / (size - 1)).astype(np.float32) if visible else np.zeros(2, dtype=np.float32)
        target = gaussian_heatmap(xy, self.cfg.HEATMAP_SIZE, self.cfg.HEATMAP_SIGMA) if visible else \
            np.zeros((1, self.cfg.HEATMAP_SIZE, self.cfg.HEATMAP_SIZE), dtype=np.float32)
        image = image.astype(np.float32) / 255
        image = (image - np.array([0.485, 0.456, 0.406], np.float32)) / np.array([0.229, 0.224, 0.225], np.float32)
        item = {
            "image": torch.from_numpy(np.ascontiguousarray(image.transpose(2, 0, 1))),
            "target_heatmap": torch.from_numpy(target),
            "target_xy": torch.from_numpy(xy),
            "visible": torch.tensor(float(visible), dtype=torch.float32),
            "orig_wh": torch.tensor([width, height], dtype=torch.float32),
            "input_to_orig": torch.from_numpy(np.linalg.inv(transform)[:2].astype(np.float32)),
            "image_id": str(row.image_id),
        }
        if has_label:
            item["x_true"] = torch.tensor(float(row.x), dtype=torch.float32)
            item["y_true"] = torch.tensor(float(row.y), dtype=torch.float32)
        return item


def make_loader(frame, cfg, training=False, seed_offset=0):
    generator = torch.Generator().manual_seed(cfg.SEED + seed_offset)
    return DataLoader(FoveaDataset(frame, cfg, training), batch_size=cfg.BATCH_SIZE,
                      shuffle=training, num_workers=cfg.NUM_WORKERS,
                      pin_memory=torch.cuda.is_available(), drop_last=False,
                      worker_init_fn=seed_worker, generator=generator,
                      persistent_workers=False)


"""Model and training helpers. Concatenate after the notebook config/data cells."""
import math
import socket
import time
import warnings
from pathlib import Path
from urllib.parse import urlparse

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


_RESNET18_STATE = None
_RESNET18_DOWNLOAD_FAILED = False


def _load_torch_file(path, map_location="cpu"):
    # Our checkpoints contain tensors and primitive values, not pickled modules.
    try:
        return torch.load(path, map_location=map_location, weights_only=True)
    except TypeError:  # Compatibility with older PyTorch installations.
        return torch.load(path, map_location=map_location)


def build_resnet18(pretrained=True, allow_download=True, weights_path=None):
    """Use explicit local weights/cache first; Internet is optional.

    Kaggle Internet=Off users can attach a dataset containing the official
    resnet18-f37072fd.pth and supply its path as weights_path. A failed download
    produces a clearly reported random-initialization fallback.
    """
    global _RESNET18_STATE, _RESNET18_DOWNLOAD_FAILED
    from torchvision.models import ResNet18_Weights, resnet18

    encoder = resnet18(weights=None)
    if not pretrained:
        return encoder
    if weights_path:
        explicit_path = Path(weights_path)
        if not explicit_path.is_file():
            raise FileNotFoundError(f"Pretrained weights do not exist: {explicit_path}")
        state = _load_torch_file(explicit_path)
        encoder.load_state_dict(state, strict=True)
        return encoder
    if _RESNET18_STATE is not None:
        encoder.load_state_dict(_RESNET18_STATE, strict=True)
        return encoder

    weights = ResNet18_Weights.DEFAULT
    cache_path = (
        Path(torch.hub.get_dir()) / "checkpoints" / Path(urlparse(weights.url).path).name
    )
    try:
        if cache_path.is_file():
            state = _load_torch_file(cache_path)
        elif allow_download and not _RESNET18_DOWNLOAD_FAILED:
            # torchvision delegates to urllib without exposing its timeout.
            # Restrict only this synchronous download; restore the caller's
            # socket default even when a timeout or checksum error is raised.
            previous_timeout = socket.getdefaulttimeout()
            try:
                socket.setdefaulttimeout(20.0)
                state = weights.get_state_dict(progress=True, check_hash=True)
            finally:
                socket.setdefaulttimeout(previous_timeout)
        else:
            reason = "downloads disabled" if not allow_download else "previous download failed"
            warnings.warn(
                f"ResNet18 pretrained weights unavailable ({reason}); "
                "using random initialization. This may reduce accuracy on 80 images."
            )
            return encoder
        encoder.load_state_dict(state, strict=True)
        _RESNET18_STATE = state
    except Exception as exc:
        _RESNET18_DOWNLOAD_FAILED = True
        warnings.warn(
            f"Could not load ResNet18 pretrained weights: {type(exc).__name__}: {exc}. "
            "Using random initialization. Attach official weights for offline use."
        )
    return encoder


class ConvGNAct(nn.Sequential):
    def __init__(self, in_channels, out_channels):
        super().__init__(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(8, out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(8, out_channels),
            nn.ReLU(inplace=True),
        )


class DecoderBlock(nn.Module):
    def __init__(self, in_channels, skip_channels, out_channels):
        super().__init__()
        self.conv = ConvGNAct(in_channels + skip_channels, out_channels)

    def forward(self, x, skip):
        x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        return self.conv(torch.cat([x, skip], dim=1))


def heatmap_to_coords(logits):
    """Spatial softmax expectation in [0,1], retaining subpixel coordinates.

    A multi-peak heatmap can place its expectation between peaks; inspect OOF
    errors before changing the decoder or introducing a refinement model.
    """
    batch_size, _, height, width = logits.shape
    probabilities = logits.float().flatten(1).softmax(dim=1).reshape(
        batch_size, height, width
    )
    xs = torch.linspace(0, 1, width, device=logits.device, dtype=torch.float32)
    ys = torch.linspace(0, 1, height, device=logits.device, dtype=torch.float32)
    x = (probabilities.sum(dim=1) * xs).sum(dim=1)
    y = (probabilities.sum(dim=2) * ys).sum(dim=1)
    return torch.stack([x, y], dim=1)


class FoveaNet(nn.Module):
    def __init__(
        self,
        pretrained=True,
        allow_download=True,
        use_visibility=True,
        heatmap_size=128,
        freeze_bn=True,
        pretrained_weights_path=None,
    ):
        super().__init__()
        self.encoder = build_resnet18(
            pretrained=pretrained,
            allow_download=allow_download,
            weights_path=pretrained_weights_path,
        )
        self.encoder.fc = nn.Identity()
        self.decoder3 = DecoderBlock(512, 256, 256)
        self.decoder2 = DecoderBlock(256, 128, 128)
        self.decoder1 = DecoderBlock(128, 64, 64)
        self.heatmap_head = nn.Conv2d(64, 1, kernel_size=1)
        self.visibility_head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Dropout(0.1),
            nn.Linear(512, 1),
        )
        self.use_visibility = bool(use_visibility)
        self.freeze_bn = bool(freeze_bn)
        self.heatmap_size = int(heatmap_size)
        if not self.use_visibility:
            for parameter in self.visibility_head.parameters():
                parameter.requires_grad = False

    def train(self, mode=True):
        super().train(mode)
        if mode and self.freeze_bn:
            # Small folds/batches give poor running-statistic estimates.
            # Affine BN weights still receive gradients.
            for module in self.encoder.modules():
                if isinstance(module, nn.BatchNorm2d):
                    module.eval()
        return self

    def forward(self, image):
        e = self.encoder
        x = e.relu(e.bn1(e.conv1(image)))
        c1 = e.layer1(e.maxpool(x))
        c2 = e.layer2(c1)
        c3 = e.layer3(c2)
        c4 = e.layer4(c3)
        decoded = self.decoder3(c4, c3)
        decoded = self.decoder2(decoded, c2)
        decoded = self.decoder1(decoded, c1)
        logits = self.heatmap_head(decoded)
        if logits.shape[-2:] != (self.heatmap_size, self.heatmap_size):
            logits = F.interpolate(
                logits,
                size=(self.heatmap_size, self.heatmap_size),
                mode="bilinear",
                align_corners=False,
            )
        if self.use_visibility:
            visibility_logits = self.visibility_head(c4).squeeze(1)
        else:
            # Disabled only when the training set is all visible. The data
            # validation/orchestrator rejects an all-invisible training fold.
            visibility_logits = logits.new_zeros(image.shape[0])
        return {
            "logits": logits,
            "coords": heatmap_to_coords(logits),
            "visibility_logits": visibility_logits,
        }


def _to_device(batch, device):
    return {
        key: value.to(device, non_blocking=True) if torch.is_tensor(value) else value
        for key, value in batch.items()
    }


def fovea_loss(output, batch, cfg, use_visibility):
    """Mean KL + original-pixel coordinate MSE + optional visibility BCE."""
    target_heatmap = batch["target_heatmap"].float()
    if target_heatmap.ndim == 3:
        target_heatmap = target_heatmap.unsqueeze(1)
    if target_heatmap.shape != output["logits"].shape:
        raise ValueError(
            f"Heatmap shape mismatch: target={target_heatmap.shape}, "
            f"prediction={output['logits'].shape}"
        )
    visible = batch["visible"].float().reshape(-1)
    n_visible = visible.sum().clamp_min(1.0)
    target_prob = target_heatmap.flatten(1)
    target_prob = target_prob / target_prob.sum(dim=1, keepdim=True).clamp_min(1e-12)
    log_prob = output["logits"].float().flatten(1).log_softmax(dim=1)
    per_image_kl = (
        target_prob * (target_prob.clamp_min(1e-12).log() - log_prob)
    ).sum(dim=1)
    heatmap_loss = (per_image_kl * visible).sum() / n_visible

    # Convert input-coordinate differences to original pixels using the same
    # affine geometry as inference. No division by each image's own dimensions:
    # a 20-pixel error has the same loss on small and large originals.
    image_height, image_width = batch["image"].shape[-2:]
    input_extent = output["coords"].new_tensor([image_width - 1, image_height - 1])
    delta_input = (output["coords"] - batch["target_xy"].float()) * input_extent
    inverse_linear = batch["input_to_orig"].float()[:, :, :2]
    delta_original = torch.bmm(inverse_linear, delta_input.unsqueeze(-1)).squeeze(-1)
    per_image_coordinate = (
        delta_original / float(getattr(cfg, "COORD_SCALE", 512.0))
    ).square().mean(dim=1)
    coordinate_loss = (per_image_coordinate * visible).sum() / n_visible

    if use_visibility:
        visibility_loss = F.binary_cross_entropy_with_logits(
            output["visibility_logits"].float(), visible
        )
    else:
        visibility_loss = output["logits"].float().sum() * 0.0
    total = (
        float(getattr(cfg, "HEATMAP_LOSS_WEIGHT", 1.0)) * heatmap_loss
        + float(getattr(cfg, "COORD_LOSS_WEIGHT", 10.0)) * coordinate_loss
        + float(getattr(cfg, "VIS_LOSS_WEIGHT", 1.0)) * visibility_loss
    )
    return total, {
        "heatmap": float(heatmap_loss.detach()),
        "coordinate": float(coordinate_loss.detach()),
        "visibility": float(visibility_loss.detach()),
    }


def normalized_to_original(coords, batch, clamp=True):
    """Invert the recorded input-pixel transform, including resize offsets."""
    image_height, image_width = batch["image"].shape[-2:]
    extent = coords.new_tensor([image_width - 1, image_height - 1])
    input_pixels = coords.float() * extent
    homogeneous = torch.cat([input_pixels, torch.ones_like(input_pixels[:, :1])], dim=1)
    original = torch.bmm(
        batch["input_to_orig"].float(), homogeneous.unsqueeze(-1)
    ).squeeze(-1)
    if clamp:
        upper = (batch["orig_wh"].float() - 1.0).clamp_min(0.0)
        original = torch.minimum(original.clamp_min(0.0), upper)
    return original


def _autocast_context(device, cfg):
    return torch.autocast(
        device_type=device.type,
        dtype=torch.float16 if device.type == "cuda" else torch.bfloat16,
        enabled=bool(getattr(cfg, "AMP", True)) and device.type == "cuda",
    )


def _prediction_rows(batch, original_coords, visibility_probability):
    coordinates = original_coords.detach().cpu().numpy()
    probabilities = visibility_probability.detach().cpu().numpy()
    ids = batch["image_id"]
    rows = []
    for i in range(len(coordinates)):
        row = {
            "image_id": str(ids[i]),
            "x": float(coordinates[i, 0]),
            "y": float(coordinates[i, 1]),
            "visibility_prob": float(probabilities[i]),
        }
        for field, destination in (
            ("visible", "visible_true"),
            ("x_true", "x_true"),
            ("y_true", "y_true"),
        ):
            if field in batch:
                value = batch[field][i]
                row[destination] = float(value.item() if torch.is_tensor(value) else value)
        rows.append(row)
    return rows


@torch.inference_mode()
def predict_loader(model, loader, device, cfg, tta=False):
    """Return raw original-coordinate predictions; threshold only after fusion.

    Horizontal TTA flips the full input canvas. Undo that flip in input space,
    then invert the affine transform for each view before averaging.
    """
    device = torch.device(device)
    model.eval()
    rows = []
    for cpu_batch in loader:
        batch = _to_device(cpu_batch, device)
        with _autocast_context(device, cfg):
            output = model(batch["image"])
        original_views = [normalized_to_original(output["coords"], batch, clamp=False)]
        probability_views = [
            output["visibility_logits"].float().sigmoid()
            if model.use_visibility
            else torch.ones(batch["image"].shape[0], device=device)
        ]
        if tta:
            with _autocast_context(device, cfg):
                flipped = model(torch.flip(batch["image"], dims=[-1]))
            corrected_coords = flipped["coords"].clone()
            corrected_coords[:, 0] = 1.0 - corrected_coords[:, 0]
            original_views.append(normalized_to_original(corrected_coords, batch, clamp=False))
            probability_views.append(
                flipped["visibility_logits"].float().sigmoid()
                if model.use_visibility
                else torch.ones(batch["image"].shape[0], device=device)
            )
        original = torch.stack(original_views).mean(dim=0)
        upper = (batch["orig_wh"].float() - 1.0).clamp_min(0.0)
        original = torch.minimum(original.clamp_min(0.0), upper)
        probability = torch.stack(probability_views).mean(dim=0)
        rows.extend(_prediction_rows(batch, original, probability))
    return rows


def prediction_rmse(rows, visibility_threshold=0.5, coordinate_origin=0):
    prediction = np.asarray([[row["x"], row["y"]] for row in rows], dtype=np.float64)
    truth = np.asarray([[row["x_true"], row["y_true"]] for row in rows], dtype=np.float64)
    probability = np.asarray([row["visibility_prob"] for row in rows], dtype=np.float64)
    predicted_visible = probability >= float(visibility_threshold)
    if coordinate_origin:
        if any("visible_true" not in row for row in rows):
            raise ValueError("coordinate_origin=1 requires each row's visible_true label.")
        true_visible = np.asarray([row["visible_true"] for row in rows], dtype=bool)
        # All internal coordinates are zero-based. Convert visible coordinates
        # only; the invisible sentinel remains (0, 0) in either convention.
        truth[true_visible] += float(coordinate_origin)
        prediction[predicted_visible] += float(coordinate_origin)
    prediction[~predicted_visible] = 0.0
    # Mean over all 2*N scalar coordinates, matching the long submission table.
    return float(np.sqrt(np.mean(np.square(prediction - truth))))


@torch.inference_mode()
def evaluate_model(model, loader, device, cfg):
    model.eval()
    loss_sum, image_count, rows = 0.0, 0, []
    for cpu_batch in loader:
        batch = _to_device(cpu_batch, device)
        with _autocast_context(device, cfg):
            output = model(batch["image"])
        loss, _ = fovea_loss(output, batch, cfg, model.use_visibility)
        batch_size = batch["image"].shape[0]
        loss_sum += float(loss) * batch_size
        image_count += batch_size
        original = normalized_to_original(output["coords"], batch)
        probability = (
            output["visibility_logits"].float().sigmoid()
            if model.use_visibility
            else torch.ones(batch_size, device=device)
        )
        rows.extend(_prediction_rows(batch, original, probability))
    if image_count == 0:
        raise ValueError("Validation loader is empty.")
    score = prediction_rmse(
        rows,
        visibility_threshold=getattr(cfg, "VISIBILITY_THRESHOLD", 0.5),
        coordinate_origin=getattr(cfg, "COORDINATE_ORIGIN", 0),
    )
    return loss_sum / image_count, score


def load_model_checkpoint(model, checkpoint_path, device):
    checkpoint = _load_torch_file(checkpoint_path, map_location="cpu")
    if bool(checkpoint.get("use_visibility", model.use_visibility)) != model.use_visibility:
        raise ValueError("Checkpoint and model use_visibility settings differ.")
    model.load_state_dict(checkpoint["model"], strict=True)
    model.to(device)
    model.eval()
    return checkpoint


def fit_fold(model, train_loader, val_loader, cfg, device, fold, checkpoint_path):
    """Train one fold, checkpoint minimum fixed-threshold RMSE, restore best."""
    device = torch.device(device)
    model.to(device)
    if len(train_loader) == 0:
        raise ValueError("Training loader is empty; do not use drop_last=True on tiny folds.")
    epochs = int(getattr(cfg, "EPOCHS", 80))
    if epochs < 1:
        raise ValueError("EPOCHS must be >= 1.")
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=float(getattr(cfg, "LR", 2e-4)),
        weight_decay=float(getattr(cfg, "WEIGHT_DECAY", 1e-4)),
    )
    total_steps = epochs * len(train_loader)
    warmup_steps = min(
        int(getattr(cfg, "WARMUP_EPOCHS", 3)) * len(train_loader),
        max(0, total_steps - 1),
    )

    def learning_rate_factor(step):
        if warmup_steps and step < warmup_steps:
            return float(step + 1) / warmup_steps
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        progress = min(1.0, max(0.0, progress))
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, learning_rate_factor)
    amp_enabled = bool(getattr(cfg, "AMP", True)) and device.type == "cuda"
    try:
        scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)
    except (AttributeError, TypeError):
        scaler = torch.cuda.amp.GradScaler(enabled=amp_enabled)
    checkpoint_path = Path(checkpoint_path)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    patience = int(getattr(cfg, "PATIENCE", 12))
    best_score, best_epoch, bad_epochs = math.inf, -1, 0
    history = []
    for epoch in range(epochs):
        start_time = time.perf_counter()
        model.train()
        total_loss, image_count = 0.0, 0
        component_sums = {"heatmap": 0.0, "coordinate": 0.0, "visibility": 0.0}
        epoch_learning_rate = optimizer.param_groups[0]["lr"]
        for cpu_batch in train_loader:
            batch = _to_device(cpu_batch, device)
            optimizer.zero_grad(set_to_none=True)
            with _autocast_context(device, cfg):
                output = model(batch["image"])
            # Spatial probabilities and losses are explicitly float32.
            loss, components = fovea_loss(output, batch, cfg, model.use_visibility)
            if not torch.isfinite(loss):
                raise FloatingPointError(f"Non-finite loss in fold={fold}, epoch={epoch + 1}.")
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), max_norm=float(getattr(cfg, "GRAD_CLIP", 1.0))
            )
            previous_scale = scaler.get_scale()
            scaler.step(optimizer)
            scaler.update()
            # A scale reduction means AMP skipped the optimizer update.
            if scaler.get_scale() >= previous_scale:
                scheduler.step()
            batch_size = batch["image"].shape[0]
            total_loss += float(loss.detach()) * batch_size
            image_count += batch_size
            for key, value in components.items():
                component_sums[key] += value * batch_size

        validation_loss, validation_rmse = evaluate_model(model, val_loader, device, cfg)
        if not math.isfinite(validation_rmse):
            raise FloatingPointError(f"Non-finite validation RMSE in fold={fold}.")
        improved = validation_rmse < best_score
        if improved:
            best_score, best_epoch, bad_epochs = validation_rmse, epoch + 1, 0
            torch.save(
                {
                    "model": {key: value.detach().cpu() for key, value in model.state_dict().items()},
                    "epoch": best_epoch,
                    "rmse": float(best_score),
                    "use_visibility": bool(model.use_visibility),
                    "heatmap_size": int(model.heatmap_size),
                    "input_size": int(getattr(cfg, "INPUT_SIZE", 512)),
                    "fold": int(fold),
                },
                checkpoint_path,
            )
        else:
            bad_epochs += 1
        record = {
            "fold": int(fold),
            "epoch": epoch + 1,
            "train_loss": total_loss / image_count,
            "val_loss": validation_loss,
            "val_rmse": validation_rmse,
            "learning_rate": epoch_learning_rate,
            "seconds": time.perf_counter() - start_time,
            **{f"train_{key}": value / image_count for key, value in component_sums.items()},
        }
        history.append(record)
        print(
            f"fold {fold} | epoch {epoch + 1:03d}/{epochs} | "
            f"train {record['train_loss']:.4f} | val {validation_loss:.4f} | "
            f"RMSE {validation_rmse:.3f}px | best {best_score:.3f}px | "
            f"{record['seconds']:.1f}s" + (" *" if improved else ""),
            flush=True,
        )
        if patience > 0 and bad_epochs >= patience:
            print(f"Early stopping: no improvement for {patience} epochs.", flush=True)
            break
    load_model_checkpoint(model, checkpoint_path, device)
    return {
        "history": history,
        "best_epoch": best_epoch,
        "best_rmse": best_score,
        "checkpoint_path": str(checkpoint_path),
    }


"""Research dataset, strict pretrained model training, and distribution inference.

Concatenate after data_part.py and model_training_part.py. Inference explicitly
removes label columns. The uncertainty below is spatial heatmap dispersion, not
an automatically calibrated probability or an epistemic uncertainty estimate.
"""
import copy


def research_crop_transform(center_x, center_y, side, input_size):
    """Map original pixel centres into a square crop with pixel-edge side `side`.

    The crop covers [cx-side/2, cx+side/2] in pixel-edge coordinates relative
    to its centre. Output pixel centres are symmetrically located around
    (input_size-1)/2. cv2.warpAffine receives this forward transform and performs
    inverse sampling itself. Crops may extend beyond the original image.
    """
    side = float(side)
    if not np.isfinite([center_x, center_y, side]).all() or side <= 0:
        raise ValueError("Crop centre and side must be finite; side must be positive.")
    scale = float(input_size) / side
    centre = (float(input_size) - 1.0) / 2.0
    return np.array([[scale, 0.0, centre - scale * float(center_x)],
                     [0.0, scale, centre - scale * float(center_y)],
                     [0.0, 0.0, 1.0]], dtype=np.float64)


def research_roi_side(short_side, fraction, cfg):
    """Same base-fraction + 2*margin and clamp used by make_roi_spec."""
    short_side = np.asarray(short_side, dtype=np.float64)
    fraction = float(fraction)
    low, high = float(cfg.ROI_MIN_FRACTION), float(cfg.ROI_MAX_FRACTION)
    margin = float(cfg.ROI_MARGIN_PX)
    if not np.isfinite(short_side).all() or (short_side <= 1).any():
        raise ValueError("Image short sides must be finite and greater than one pixel.")
    if not math.isfinite(fraction) or fraction <= 0 or not 0 < low <= high <= 1:
        raise ValueError("Invalid ROI sampling fraction or fraction bounds.")
    if not math.isfinite(margin) or margin < 0:
        raise ValueError("ROI margin must be finite and nonnegative.")
    return np.clip(fraction * short_side + 2 * margin, low * short_side, high * short_side)


class ResearchDataset(Dataset):
    """All-visible localization with optional prediction-centred fine crops.

    Fine training without crop_spec uses random GT-centred perturbations. This
    is a declared synthetic localization-error augmentation, not inner OOF
    predictions. Validation/inference never repairs a crop using the label.
    """

    def __init__(self, frame, cfg, training=False, stage="coarse", crop_spec=None):
        if stage not in {"coarse", "fine"}:
            raise ValueError(f"Unknown stage: {stage}")
        self.frame = frame.reset_index(drop=True).copy()
        self.cfg, self.training, self.stage = cfg, bool(training), stage
        self.has_labels = {"x", "y", "visible"}.issubset(self.frame.columns)
        if self.training and not self.has_labels:
            raise ValueError("Training requires x, y and visible labels.")
        if self.has_labels and not self.frame.visible.astype(bool).all():
            raise ValueError("Research model supports this all-visible dataset only.")
        self.size = int(cfg.INPUT_SIZE if stage == "coarse" else cfg.FINE_INPUT_SIZE)
        self.hm_size = int(cfg.HEATMAP_SIZE if stage == "coarse" else cfg.FINE_HEATMAP_SIZE)
        self.crops = None
        if crop_spec is not None:
            required = ["image_id", "crop_x", "crop_y", "crop_side"]
            if not set(required).issubset(crop_spec.columns):
                raise ValueError(f"crop_spec must contain {required}")
            self.crops = crop_spec[required].copy()
            self.crops["image_id"] = self.crops.image_id.astype(str)
            if self.crops.image_id.duplicated().any():
                raise ValueError("crop_spec contains duplicate image IDs.")
            self.crops = self.crops.set_index("image_id")
            missing = set(self.frame.image_id.astype(str)) - set(self.crops.index)
            if missing:
                raise ValueError(f"Missing crops for image IDs: {sorted(missing)}")
            values = self.crops[["crop_x", "crop_y", "crop_side"]].to_numpy(float)
            if not np.isfinite(values).all() or (values[:, 2] <= 0).any():
                raise ValueError("Non-finite crop coordinates or non-positive crop side.")
        if stage == "fine" and not self.training and self.crops is None:
            raise ValueError("Fine validation/inference requires prediction-derived crop_spec.")
        self.roi_fractions = np.asarray(getattr(cfg, "ROI_FRACTIONS", (0.2, 0.3, 0.4)), float)
        if self.roi_fractions.size == 0 or not np.isfinite(self.roi_fractions).all() or (self.roi_fractions <= 0).any():
            raise ValueError("ROI_FRACTIONS must be finite positive numbers.")

    def __len__(self):
        return len(self.frame)

    def __getitem__(self, index):
        row = self.frame.iloc[index]
        image = read_rgb(row.path)
        height, width = image.shape[:2]
        if int(row.width) != width or int(row.height) != height:
            raise ValueError(f"Image dimensions changed: {row.path}")
        original = np.array([float(row.x), float(row.y)], np.float64) if self.has_labels else None
        crop_was_enlarged, initial_crop_miss = False, False
        sampled_fraction = -1.0  # Negative means no random scale was selected.
        crop_from_prediction = self.stage == "fine" and self.crops is not None
        if self.stage == "coarse":
            image, transform = letterbox(image, self.size)
            crop = np.array([(width - 1) / 2, (height - 1) / 2, max(width, height)], float)
        else:
            if self.crops is not None:
                crop = self.crops.loc[str(row.image_id), ["crop_x", "crop_y", "crop_side"]].to_numpy(float).copy()
                if self.training:
                    if bool(getattr(self.cfg, "FINE_MULTISCALE_TRAIN", True)):
                        sampled_fraction = float(np.random.choice(self.roi_fractions))
                        crop[2] = float(research_roi_side(min(width, height), sampled_fraction, self.cfg))
                    initial_crop_miss = bool(np.abs(original - crop[:2]).max() >= crop[2] / 2)
                    jitter = float(getattr(self.cfg, "OOF_CENTER_JITTER", 0.08))
                    if jitter < 0:
                        raise ValueError("OOF_CENTER_JITTER must be non-negative.")
                    crop[:2] += np.clip(np.random.normal(0, jitter, 2), -0.20, 0.20) * crop[2]
            else:
                sampled_fraction = float(np.random.choice(self.roi_fractions))
                side = float(research_roi_side(min(width, height), sampled_fraction, self.cfg))
                jitter = float(getattr(self.cfg, "TRAIN_CENTER_JITTER", 0.30))
                if jitter < 0:
                    raise ValueError("TRAIN_CENTER_JITTER must be non-negative.")
                # Truncated normal errors cover off-centre targets without
                # training on crops that cannot contain the annotated point.
                offset = np.clip(np.random.normal(0, jitter, 2), -0.42, 0.42) * side
                crop = np.r_[original + offset, side]
            sampled_side = float(crop[2])
            if self.training:
                # When optional prediction crops miss the point, increase the
                # *training* field of view; preserve the original coordinate.
                required_side = 2.5 * float(np.abs(original - crop[:2]).max()) + 2
                crop_was_enlarged = required_side > crop[2]
                crop[2] = max(crop[2], required_side)
            transform = research_crop_transform(*crop, self.size)
            image = cv2.warpAffine(image, transform[:2], (self.size, self.size),
                                   flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT,
                                   borderValue=0)
        point = (transform @ np.r_[original, 1.0])[:2] if original is not None else np.zeros(2)
        if self.training:
            for _ in range(8):
                affine = cv2.getRotationMatrix2D(((self.size - 1) / 2,) * 2,
                                                np.random.uniform(-12, 12),
                                                np.random.uniform(0.95, 1.05))
                affine[:, 2] += np.random.uniform(-0.025, 0.025, 2) * self.size
                augmentation = np.vstack([affine, [0, 0, 1]])
                if np.random.random() < 0.5:
                    augmentation = np.array([[-1, 0, self.size - 1], [0, 1, 0], [0, 0, 1]]) @ augmentation
                candidate = (augmentation @ np.r_[point, 1.0])[:2]
                if np.all((candidate >= 0) & (candidate <= self.size - 1)):
                    image = cv2.warpAffine(image, augmentation[:2], (self.size, self.size),
                                           flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT,
                                           borderValue=0)
                    transform = augmentation @ transform
                    point = candidate
                    break
            image = np.clip(image.astype(np.float32) * np.random.uniform(0.85, 1.15)
                            + np.random.uniform(-8, 8), 0, 255).astype(np.uint8)
        xy = (point / (self.size - 1)).astype(np.float32)
        # Labels outside validation ROIs stay outside; no clipping or crop repair.
        # Validation uses original-pixel errors, not this possibly tiny heatmap.
        target = gaussian_heatmap(xy, self.hm_size, self.cfg.HEATMAP_SIGMA) if self.has_labels else np.zeros((1, self.hm_size, self.hm_size), np.float32)
        image = image.astype(np.float32) / 255.0
        image = (image - np.array([0.485, 0.456, 0.406], np.float32)) / np.array([0.229, 0.224, 0.225], np.float32)
        item = {
            "image": torch.from_numpy(np.ascontiguousarray(image.transpose(2, 0, 1))),
            "target_heatmap": torch.from_numpy(target),
            "target_xy": torch.from_numpy(xy),
            "visible": torch.tensor(float(self.has_labels), dtype=torch.float32),
            "orig_wh": torch.tensor([width, height], dtype=torch.float32),
            "input_to_orig": torch.from_numpy(np.linalg.inv(transform)[:2].astype(np.float32)),
            "crop": torch.tensor(crop, dtype=torch.float32),
            "crop_from_prediction": torch.tensor(bool(crop_from_prediction), dtype=torch.bool),
            "crop_was_enlarged": torch.tensor(bool(crop_was_enlarged), dtype=torch.bool),
            "initial_crop_miss": torch.tensor(bool(initial_crop_miss), dtype=torch.bool),
            "crop_sampled_fraction": torch.tensor(sampled_fraction, dtype=torch.float32),
            "crop_sampled_side": torch.tensor(sampled_side if self.stage == "fine" else crop[2], dtype=torch.float32),
            "crop_final_side": torch.tensor(crop[2], dtype=torch.float32),
            "image_id": str(row.image_id),
        }
        if self.has_labels:
            item["x_true"] = torch.tensor(float(row.x), dtype=torch.float32)
            item["y_true"] = torch.tensor(float(row.y), dtype=torch.float32)
        return item


def make_research_loader(frame, cfg, training=False, stage="coarse", crop_spec=None, seed=0):
    return DataLoader(ResearchDataset(frame, cfg, training, stage, crop_spec),
                      batch_size=int(cfg.BATCH_SIZE), shuffle=training,
                      num_workers=int(cfg.NUM_WORKERS), pin_memory=torch.cuda.is_available(),
                      drop_last=False, worker_init_fn=seed_worker,
                      generator=torch.Generator().manual_seed(int(seed)), persistent_workers=False)


def research_build_model(cfg, stage, device):
    """Require the explicitly resolved official weight file; no silent fallback."""
    path = getattr(cfg, "PRETRAINED_WEIGHTS_PATH", None)
    if cfg.PRETRAINED and (not path or not Path(path).is_file()):
        raise FileNotFoundError("Research training requires the attached official ResNet18 weight file. Resolve PRETRAINED_WEIGHTS_PATH first.")
    return FoveaNet(pretrained=bool(cfg.PRETRAINED), allow_download=False,
                    use_visibility=False, freeze_bn=bool(cfg.FREEZE_BN),
                    heatmap_size=int(cfg.HEATMAP_SIZE if stage == "coarse" else cfg.FINE_HEATMAP_SIZE),
                    pretrained_weights_path=path if cfg.PRETRAINED else None).to(device)


@torch.inference_mode()
def research_validation_rmse(model, loader, cfg, device):
    """Use precisely the same probability-mixture TTA decoder as inference."""
    model.eval()
    squared_error, coordinate_count = 0.0, 0
    for cpu_batch in loader:
        batch = _to_device(cpu_batch, device)
        prediction, _ = research_predict_batch(model, batch, cfg, device)
        truth = torch.stack([batch["x_true"], batch["y_true"]], dim=1)
        squared_error += float((prediction.double() - truth.double()).square().sum())
        coordinate_count += int(truth.numel())
    if not coordinate_count:
        raise ValueError("Validation set is empty.")
    return math.sqrt(squared_error / coordinate_count)


def research_schedule_factor(step, steps_per_epoch, horizon_epochs, warmup_epochs):
    """Epoch-normalized schedule: a shorter refit keeps the selection-run prefix.

    `horizon_epochs` is the configured stage budget, never the selected refit
    length. Different data sizes have different steps per epoch but the same
    epoch-level warmup/cosine trajectory.
    """
    if int(steps_per_epoch) < 1 or int(horizon_epochs) < 1 or int(step) < 0:
        raise ValueError("Schedule requires positive horizon/steps and nonnegative step.")
    if int(warmup_epochs) < 0:
        raise ValueError("WARMUP_EPOCHS must be nonnegative.")
    steps = int(horizon_epochs) * int(steps_per_epoch)
    warmup = min(int(warmup_epochs) * int(steps_per_epoch), max(0, steps - 1))
    if warmup and step < warmup:
        return (float(step) + 1.0) / warmup
    progress = min(1.0, max(0.0, (float(step) - warmup) / max(1, steps - warmup)))
    return 0.5 * (1 + math.cos(math.pi * progress))


def research_validation_loaders(validation, cfg, stage, val_crop, seed):
    """Prespecified fields of view on the supplied inner stopping split only.

    Centres are prediction-derived and stay fixed. Neither field-of-view
    selection nor resizing reads ground-truth coordinates.
    """
    if stage != "fine" or not bool(getattr(cfg, "FINE_MULTISCALE_EARLY_STOP", True)):
        return [("provided", make_research_loader(validation, cfg, False, stage, val_crop, seed))]
    if val_crop is None:
        raise ValueError("Multiscale fine early stopping requires prediction-derived centres.")
    crops = val_crop[["image_id", "crop_x", "crop_y", "crop_side"]].copy()
    crops["image_id"] = crops.image_id.astype(str)
    if crops.image_id.duplicated().any():
        raise ValueError("Validation crop IDs must be unique.")
    crops = crops.set_index("image_id").loc[validation.image_id.astype(str)].reset_index()
    short = validation[["width", "height"]].to_numpy(float).min(axis=1)
    fractions = tuple(float(x) for x in cfg.FINE_VALIDATION_ROI_FRACTIONS)
    if not fractions or len(set(fractions)) != len(fractions):
        raise ValueError("Fine validation scale list must be nonempty and unique.")
    loaders = []
    for fraction in fractions:
        spec = crops.copy()
        spec["crop_side"] = research_roi_side(short, fraction, cfg)
        loaders.append((f"fraction_{fraction:g}", make_research_loader(validation, cfg, False, "fine", spec, seed)))
    return loaders


def research_optimizer_groups(model, cfg):
    """Separate pretrained encoder/head rates, with explicit coverage checks."""
    multiplier = float(getattr(cfg, "BACKBONE_LR_MULTIPLIER", 1.0))
    base_lr = float(cfg.LR)
    if not math.isfinite(multiplier) or multiplier <= 0 or not math.isfinite(base_lr) or base_lr <= 0:
        raise ValueError("Learning rates and backbone multiplier must be finite and positive.")
    named = [(name, parameter) for name, parameter in model.named_parameters() if parameter.requires_grad]
    encoder = [parameter for name, parameter in named if name.startswith("encoder.")]
    decoder = [parameter for name, parameter in named if not name.startswith("encoder.")]
    groups = [{"params": encoder, "lr": base_lr * multiplier, "name": "encoder"},
              {"params": decoder, "lr": base_lr, "name": "decoder"}]
    groups = [group for group in groups if group["params"]]
    ids = [id(parameter) for group in groups for parameter in group["params"]]
    expected = {id(parameter) for _, parameter in named}
    if not ids or len(ids) != len(set(ids)) or set(ids) != expected:
        raise RuntimeError("Optimizer parameter groups omit or duplicate trainable parameters.")
    return groups


class ResearchEMA:
    """EMA copy with a short-run warm start; buffers always copy the online model.

    Only successful optimizer steps count. First update copies parameters;
    subsequent decay=min(max_decay,(1+updates)/(warmup+updates)). BN buffers
    remain synchronized, preserving the default frozen pretrained statistics.
    Validation and checkpoints use this same copy whenever EMA is enabled.
    """

    def __init__(self, model, decay=0.99, warmup_updates=10):
        self.max_decay = float(decay)
        self.warmup_updates = int(warmup_updates)
        if not math.isfinite(self.max_decay) or not 0 < self.max_decay < 1:
            raise ValueError("EMA_DECAY must lie strictly between zero and one.")
        if self.warmup_updates < 0 or self.warmup_updates != float(warmup_updates):
            raise ValueError("EMA_WARMUP_UPDATES must be a nonnegative integer.")
        self.model = copy.deepcopy(model).eval()
        self.model.requires_grad_(False)
        self.updates, self.last_decay = 0, 0.0

    @torch.no_grad()
    def update(self, model, optimizer_stepped=True):
        if not optimizer_stepped:
            return False
        self.updates += 1
        self.last_decay = (0.0 if self.updates == 1 else
                           min(self.max_decay, (1 + self.updates) / (self.warmup_updates + self.updates)))
        source_parameters = dict(model.named_parameters())
        target_parameters = dict(self.model.named_parameters())
        if source_parameters.keys() != target_parameters.keys():
            raise RuntimeError("EMA source and target parameter names differ.")
        for name, target in target_parameters.items():
            target.mul_(self.last_decay).add_(source_parameters[name].detach(), alpha=1 - self.last_decay)
        source_buffers = dict(model.named_buffers())
        target_buffers = dict(self.model.named_buffers())
        if source_buffers.keys() != target_buffers.keys():
            raise RuntimeError("EMA source and target buffer names differ.")
        for name, target in target_buffers.items():
            target.copy_(source_buffers[name].detach())
        return True


def fit_research_model(training, validation, cfg, device, stage, checkpoint_path,
                       train_crop=None, val_crop=None, epochs=None, seed=0):
    """Early-stop on inner data or refit for fixed epochs without validation.

    An integer epochs requests a final refit and the validation argument is
    intentionally never inspected. best_epoch is one-based, including refits.
    """
    device = torch.device(device)
    seed_everything(int(seed))
    fixed_epochs = epochs is not None
    horizon_epochs = int(cfg.COARSE_EPOCHS if stage == "coarse" else cfg.FINE_EPOCHS)
    total_epochs = int(epochs if fixed_epochs else horizon_epochs)
    if total_epochs < 1 or len(training) == 0:
        raise ValueError("Training needs nonempty samples and at least one epoch.")
    if total_epochs > horizon_epochs:
        raise ValueError("Fixed refit epochs cannot exceed the configured stage schedule horizon.")
    train_loader = make_research_loader(training, cfg, True, stage, train_crop, seed)
    val_loaders = None
    if not fixed_epochs:
        if validation is None or len(validation) == 0:
            raise ValueError("Early stopping needs a nonempty inner validation set.")
        if "group" in training and "group" in validation:
            if set(training.group.astype(str)) & set(validation.group.astype(str)):
                raise ValueError("Training/validation groups overlap.")
        val_loaders = research_validation_loaders(validation, cfg, stage, val_crop, seed)
    model = research_build_model(cfg, stage, device)
    optimizer = torch.optim.AdamW(research_optimizer_groups(model, cfg),
                                  lr=float(cfg.LR), weight_decay=float(cfg.WEIGHT_DECAY))
    group_metadata = [{"name": group["name"], "base_lr": group["lr"],
                       "parameter_tensors": len(group["params"]),
                       "parameters": sum(parameter.numel() for parameter in group["params"])}
                      for group in optimizer.param_groups]
    use_ema = bool(getattr(cfg, "USE_EMA", False))
    ema = ResearchEMA(model, cfg.EMA_DECAY, cfg.EMA_WARMUP_UPDATES) if use_ema else None
    total_successful_steps = 0
    def factor(step):
        return research_schedule_factor(step, len(train_loader), horizon_epochs, cfg.WARMUP_EPOCHS)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, factor)
    amp = bool(cfg.AMP) and device.type == "cuda"
    try:
        scaler = torch.amp.GradScaler("cuda", enabled=amp)
    except (AttributeError, TypeError):
        scaler = torch.cuda.amp.GradScaler(enabled=amp)
    checkpoint_path = Path(checkpoint_path)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    history, best_score, best_epoch, bad_epochs = [], math.inf, 0, 0
    for epoch in range(1, total_epochs + 1):
        started = time.perf_counter()
        model.train()
        total_loss, count = 0.0, 0
        crop_counts = {"crop_from_prediction": 0, "crop_was_enlarged": 0, "initial_crop_miss": 0}
        sampled_side_sum, final_side_sum, scale_counts = 0.0, 0.0, {}
        component_sums = {"heatmap": 0.0, "coordinate": 0.0, "visibility": 0.0}
        successful_steps = 0
        for cpu_batch in train_loader:
            batch = _to_device(cpu_batch, device)
            optimizer.zero_grad(set_to_none=True)
            with _autocast_context(device, cfg):
                output = model(batch["image"])
            loss, components = fovea_loss(output, batch, cfg, use_visibility=False)
            if not bool(torch.isfinite(loss)):
                raise FloatingPointError(f"Non-finite {stage} loss at epoch {epoch}")
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), float(cfg.GRAD_CLIP))
            previous_scale = scaler.get_scale()
            scaler.step(optimizer)
            scaler.update()
            optimizer_stepped = scaler.get_scale() >= previous_scale
            if optimizer_stepped:
                scheduler.step()
                successful_steps += 1
                total_successful_steps += 1
            if ema is not None:
                ema.update(model, optimizer_stepped=optimizer_stepped)
            n = int(batch["image"].shape[0])
            total_loss += float(loss.detach()) * n
            for component, value in components.items():
                component_sums[component] += value * n
            count += n
            for key in crop_counts:
                crop_counts[key] += int(batch[key].sum())
            sampled_side_sum += float(batch["crop_sampled_side"].sum())
            final_side_sum += float(batch["crop_final_side"].sum())
            for fraction in batch["crop_sampled_fraction"].detach().cpu().tolist():
                if fraction > 0:
                    key = f"{fraction:.4f}"
                    scale_counts[key] = scale_counts.get(key, 0) + 1
        evaluation_model = ema.model if ema is not None else model
        scale_scores = ({name: research_validation_rmse(evaluation_model, loader, cfg, device)
                         for name, loader in val_loaders} if not fixed_epochs else {})
        score = float(np.mean(list(scale_scores.values()))) if scale_scores else None
        if score is not None and not math.isfinite(score):
            raise FloatingPointError("Non-finite inner validation RMSE.")
        improved = fixed_epochs or score < best_score
        if improved:
            best_epoch, bad_epochs = epoch, 0
            if score is not None:
                best_score = float(score)
            if not fixed_epochs or epoch == total_epochs:
                torch.save({"model": {k: v.detach().cpu() for k, v in evaluation_model.state_dict().items()},
                            "epoch": epoch, "stage": stage, "use_visibility": False,
                            "heatmap_size": model.heatmap_size,
                            "schedule_horizon_epochs": horizon_epochs,
                            "validation_tta": bool(cfg.TTA),
                            "validation_scale_scores": scale_scores,
                            "weights_kind": "ema" if use_ema else "online",
                            "ema_updates": ema.updates if ema is not None else 0,
                            "optimizer_groups": group_metadata,
                            "selection": "fixed_epochs_no_validation" if fixed_epochs else "inner_validation_rmse",
                            "rmse": score, "seed": int(seed)}, checkpoint_path)
        else:
            bad_epochs += 1
        record = {"stage": stage, "epoch": epoch, "train_loss": total_loss / count,
                  "val_rmse": score, "seconds": time.perf_counter() - started,
                  "learning_rate": optimizer.param_groups[0]["lr"],
                  "learning_rates": {group["name"]: group["lr"] for group in optimizer.param_groups},
                  "train_heatmap_kl": component_sums["heatmap"] / count,
                  "train_coordinate_mse": component_sums["coordinate"] / count,
                  "train_visibility_bce": component_sums["visibility"] / count,
                  "successful_optimizer_steps": successful_steps,
                  "ema_updates": ema.updates if ema is not None else 0,
                  "ema_effective_decay": ema.last_decay if ema is not None else None,
                  "schedule_horizon_epochs": horizon_epochs, **crop_counts,
                  "crop_sampled_side_mean_px": sampled_side_sum / count,
                  "crop_final_side_mean_px": final_side_sum / count,
                  "crop_enlargement_rate": crop_counts["crop_was_enlarged"] / count,
                  "crop_sampled_fraction_counts": scale_counts,
                  "validation_scale_scores": scale_scores}
        history.append(record)
        validation_message = "fixed-epoch refit" if fixed_epochs else f"inner RMSE {score:.3f}px"
        print(f"{stage} | {checkpoint_path.stem} | epoch {epoch}/{total_epochs} | "
              f"loss {record['train_loss']:.4f} | {validation_message} | {record['seconds']:.1f}s", flush=True)
        if not fixed_epochs and int(cfg.PATIENCE) > 0 and bad_epochs >= int(cfg.PATIENCE):
            break
    load_model_checkpoint(model, checkpoint_path, device)
    info = {"best_epoch": best_epoch, "best_rmse": None if fixed_epochs else best_score,
            "history": history, "checkpoint_path": str(checkpoint_path),
            "fixed_epochs": fixed_epochs, "training_images": len(training),
            "schedule_horizon_epochs": horizon_epochs, "requested_training_epochs": total_epochs,
            "validation_tta": bool(cfg.TTA),
            "optimizer_groups": group_metadata,
            "weights_kind": "ema" if use_ema else "online",
            "successful_optimizer_steps": total_successful_steps,
            "ema_updates": ema.updates if ema is not None else 0,
            "ema_max_decay": ema.max_decay if ema is not None else None,
            "validation_metric": ("mean of prespecified inner-scale RMSEs" if stage == "fine" and bool(getattr(cfg, "FINE_MULTISCALE_EARLY_STOP", True)) else "inner RMSE"),
            "fine_multiscale_train": bool(getattr(cfg, "FINE_MULTISCALE_TRAIN", True)),
            "crop_training_policy": ("prediction centre + optional multiscale side + jitter; enlarge around that centre when training target misses; never change labels"
                                     if stage == "fine" and train_crop is not None else
                                     "GT centre + truncated jitter" if stage == "fine" else "full image")}
    checkpoint_path.with_suffix(".history.json").write_text(json.dumps(info, indent=2), encoding="utf-8")
    return model, info


def research_distribution_moments(logits, input_to_orig, input_size, undo_hflip=False):
    """Probability mean/covariance in original pixels, with affine propagation."""
    logits = logits.float()
    if undo_hflip:
        logits = torch.flip(logits, dims=[-1])
    b, _, h, w = logits.shape
    prob = logits.flatten(1).softmax(1)
    y, x = torch.meshgrid(torch.linspace(0, input_size - 1, h, device=logits.device),
                          torch.linspace(0, input_size - 1, w, device=logits.device), indexing="ij")
    grid = torch.stack([x.reshape(-1), y.reshape(-1)], dim=1)
    mean = prob @ grid
    delta = grid.unsqueeze(0) - mean.unsqueeze(1)
    covariance = torch.einsum("bn,bni,bnj->bij", prob, delta, delta)
    linear, shift = input_to_orig.float()[:, :, :2], input_to_orig.float()[:, :, 2]
    original_mean = torch.bmm(linear, mean.unsqueeze(-1)).squeeze(-1) + shift
    original_covariance = linear @ covariance @ linear.transpose(1, 2)
    return original_mean, original_covariance


@torch.inference_mode()
def research_predict_batch(model, batch, cfg, device):
    """Shared original-coordinate mean and total covariance for train selection/inference."""
    with _autocast_context(device, cfg):
        output = model(batch["image"])
    size = int(batch["image"].shape[-1])
    mean, covariance = research_distribution_moments(output["logits"], batch["input_to_orig"], size)
    means, covariances = [mean], [covariance]
    if cfg.TTA:
        with _autocast_context(device, cfg):
            flipped = model(torch.flip(batch["image"], dims=[-1]))
        mean, covariance = research_distribution_moments(flipped["logits"], batch["input_to_orig"], size, True)
        means.append(mean)
        covariances.append(covariance)
    stacked = torch.stack(means)
    mean = stacked.mean(0)
    offsets = stacked - mean.unsqueeze(0)
    covariance = (torch.stack(covariances) + offsets.unsqueeze(-1) @ offsets.unsqueeze(-2)).mean(0)
    covariance = (covariance + covariance.transpose(1, 2)) / 2
    mean = torch.minimum(mean.clamp_min(0), batch["orig_wh"].float() - 1)
    return mean, covariance


@torch.inference_mode()
def predict_research(model, frame, cfg, device, stage="coarse", crop_spec=None):
    """No label access. Mixture covariance includes disagreement between TTA views.

    uncertainty = sqrt(largest eigenvalue of original-pixel covariance), in px.
    inference_seconds includes data loading and inference, averaged per image.
    Crop-edge fraction is min distance to the crop's four pixel edges / side;
    it may be negative if final image-boundary clipping moves a point out of ROI.
    """
    device = torch.device(device)
    # An explicit whitelist prevents accidental use of x/y/visible at inference.
    inference_frame = frame[["image_id", "path", "width", "height"]].copy()
    loader = make_research_loader(inference_frame, cfg, False, stage, crop_spec, int(cfg.SEED))
    model.eval()
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    started, rows = time.perf_counter(), []
    for cpu_batch in loader:
        batch = _to_device(cpu_batch, device)
        mean, covariance = research_predict_batch(model, batch, cfg, device)
        uncertainty = torch.linalg.eigvalsh(covariance).clamp_min(0)[:, -1].sqrt()
        crop = batch["crop"]
        edge = 0.5 - ((mean - crop[:, :2]).abs() / crop[:, 2:3]).amax(1)
        for i, image_id in enumerate(batch["image_id"]):
            rows.append({"image_id": str(image_id), "x": float(mean[i, 0]), "y": float(mean[i, 1]),
                         "uncertainty": float(uncertainty[i]), "cov_xx": float(covariance[i, 0, 0]),
                         "cov_xy": float(covariance[i, 0, 1]), "cov_yy": float(covariance[i, 1, 1]),
                         "crop_edge_fraction": float(edge[i]) if stage == "fine" else np.nan})
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    columns = ["image_id", "x", "y", "uncertainty", "cov_xx", "cov_xy", "cov_yy", "crop_edge_fraction"]
    result = pd.DataFrame(rows, columns=columns)
    result["inference_seconds"] = (time.perf_counter() - started) / max(1, len(result))
    return result


def research_geometry_selfcheck():
    """Check half-pixel crop sampling, inverse affine, and flipped moments."""
    worst = 0.0
    for cx, cy, side, size in [(120.2, 90.8, 90, 128), (-10, 30, 121.5, 64),
                                (1450.3, 980.1, 600, 512)]:
        matrix = research_crop_transform(cx, cy, side, size)
        points = np.array([[cx, cy, 1], [cx + side * 0.21, cy - side * 0.3, 1]])
        mapped = (matrix @ points.T).T
        inverse = (np.linalg.inv(matrix) @ mapped.T).T
        worst = max(worst, float(np.abs(inverse - points).max()))
        np.testing.assert_allclose(mapped[0, :2], [(size - 1) / 2] * 2, atol=1e-8)
        np.testing.assert_allclose(inverse, points, atol=1e-8)
    logits = torch.full((1, 1, 9, 9), -50.0)
    logits[0, 0, 3, 6] = 50.0
    matrix = research_crop_transform(100, 150, 80, 64)
    inverse = torch.tensor(np.linalg.inv(matrix)[:2], dtype=torch.float32).unsqueeze(0)
    mean, covariance = research_distribution_moments(logits, inverse, 64)
    flipped_mean, flipped_cov = research_distribution_moments(torch.flip(logits, [-1]), inverse, 64, True)
    torch.testing.assert_close(mean, flipped_mean)
    torch.testing.assert_close(covariance, flipped_cov)
    assert torch.isfinite(covariance).all()
    # A coordinate ramp tests actual cv2 sampling, not just matrix inversion.
    ramp = np.tile(np.arange(200, dtype=np.float32), (180, 1))
    matrix = research_crop_transform(80, 90, 64, 64)
    sampled = cv2.warpAffine(ramp, matrix[:2], (64, 64), flags=cv2.INTER_LINEAR)
    expected_x = (np.linalg.inv(matrix) @ np.array([31, 31, 1]))[0]
    assert abs(float(sampled[31, 31]) - expected_x) <= 1 / 32 + 1e-5
    return {"max_inverse_error_px": worst, "hflip_distribution_roundtrip": True,
            "cv2_pixel_sampling": True}


"""Calibration, inference-only ROI decisions, and held-out research evaluation.

The calibration frame must be disjoint from every image used to train or select
the coarse/fine networks. Its in-sample fit scores are NOT validation scores.
All image alignment is by image_id; duplicate images are grouped by group.
"""
import math

import numpy as np
import pandas as pd


def _research_setting(cfg, name, default):
    return getattr(cfg, name, default) if cfg is not None else default


def _research_frame(frame, columns, name):
    if not isinstance(frame, pd.DataFrame):
        raise TypeError(f"{name} must be a pandas DataFrame")
    missing = set(["image_id", *columns]) - set(frame.columns)
    if missing:
        raise ValueError(f"{name} is missing columns: {sorted(missing)}")
    if not len(frame):
        raise ValueError(f"{name} is empty")
    out = frame.copy()
    if out.image_id.isna().any():
        raise ValueError(f"{name} has null image_id")
    out["image_id"] = out.image_id.astype(str)
    if out.image_id.duplicated().any():
        raise ValueError(f"{name} has duplicate image_id")
    for col in columns:
        if col == "group":
            if out[col].isna().any():
                raise ValueError(f"{name} has null groups")
            out[col] = out[col].astype(str)
        else:
            out[col] = pd.to_numeric(out[col], errors="raise")
            if not np.isfinite(out[col].to_numpy(dtype=float)).all():
                raise ValueError(f"{name}.{col} contains NaN or infinity")
    return out


def _research_align(pred, truth, pred_columns=("x", "y"), truth_columns=("x", "y")):
    pred = _research_frame(pred, pred_columns, "predictions")
    truth = _research_frame(truth, truth_columns, "reference frame")
    if set(pred.image_id) != set(truth.image_id):
        raise ValueError("Prediction/reference image_id sets differ")
    truth = truth.set_index("image_id").loc[pred.image_id].reset_index()
    return pred.reset_index(drop=True), truth.reset_index(drop=True)


def _research_uncertainty(pred, floor_px):
    floor_px = float(floor_px)
    if not math.isfinite(floor_px) or floor_px <= 0:
        raise ValueError("UNCERTAINTY_FLOOR_PX must be finite and positive")
    value = pred.uncertainty.to_numpy(dtype=float)
    if not np.isfinite(value).all() or np.any(value < 0):
        raise ValueError("uncertainty must be finite and nonnegative")
    return np.maximum(value, floor_px)


def _research_higher_quantile(values, probability):
    values = np.asarray(values, dtype=float)
    probability = float(probability)
    if not len(values) or not np.isfinite(values).all() or not 0 < probability <= 1:
        raise ValueError("Invalid empirical quantile input")
    # Explicit order statistic avoids NumPy-version-dependent interpolation.
    return float(np.sort(values)[int(np.ceil((len(values) - 1) * probability))])


def fit_radius_calibration(pred, truth, cfg):
    """Fit an empirical square-radius multiplier on held-out calibration groups.

    uncertainty is sqrt(maximum eigenvalue of the original-pixel heatmap
    covariance). It is a prediction-spread proxy, not calibrated confidence.
    One maximum standardized residual per duplicate/patient group is used.
    No conformal or finite-sample coverage guarantee is claimed.
    """
    pred, truth = _research_align(pred, truth, ("x", "y", "uncertainty"),
                                  ("x", "y", "group"))
    coverage = float(_research_setting(cfg, "ROI_COVERAGE", 0.9))
    floor_px = float(_research_setting(cfg, "UNCERTAINTY_FLOOR_PX", 2.0))
    uncertainty = _research_uncertainty(pred, floor_px)
    residual = np.abs(pred[["x", "y"]].to_numpy(float) - truth[["x", "y"]].to_numpy(float)).max(axis=1)
    standardized = residual / uncertainty
    if not np.isfinite(standardized).all():
        raise ValueError("Calibration standardized residuals must be finite")
    groups = pd.DataFrame({"group": truth.group, "score": standardized})
    group_scores = groups.groupby("group", sort=True).score.max()
    probabilities = sorted(set([0.5, 0.8, 0.9, 0.95, coverage]))
    alpha_by_coverage = {
        str(p): _research_higher_quantile(group_scores.to_numpy(), p) for p in probabilities
    }
    alpha = alpha_by_coverage[str(coverage)]
    return {
        "method": "group_max_empirical_higher_quantile",
        "target_coverage": coverage,
        "alpha": alpha,
        "alpha_by_coverage": alpha_by_coverage,
        "uncertainty_floor_px": floor_px,
        "n_images": int(len(pred)),
        "n_groups": int(len(group_scores)),
        "calibration_image_ids": pred.image_id.tolist(),
        "calibration_group_ids": group_scores.index.astype(str).tolist(),
        "fit_image_coverage_before_margin_and_clamp": float(np.mean(standardized <= alpha)),
        "fit_group_coverage_before_margin_and_clamp": float(np.mean(group_scores.to_numpy() <= alpha)),
        "score_definition": "max(abs(dx),abs(dy))/max(uncertainty,floor_px); group maximum",
        "coverage_note": "Empirical calibration only; ROI bounds, radius clamps and domain shift can reduce actual coverage. Fit coverage is not held-out performance.",
    }


def make_roi_spec(pred, frame, cfg, mode="fixed", calibration=None, fixed_fraction=None):
    """Create centered square ROIs using predictions and dimensions only.

    crop_x/crop_y are CENTER coordinates in original pixels. Crops extending
    beyond the image use padding; effective image intersections are recorded.
    All modes apply side=clip(base_side+2*margin,min_fraction*short,max_fraction*short).
    A fixed fraction describes base_side before this common margin and clamp.
    """
    if mode not in {"fixed", "raw", "calibrated"}:
        raise ValueError(f"Unsupported ROI mode: {mode}")
    required = ("x", "y") if mode == "fixed" else ("x", "y", "uncertainty")
    pred, frame = _research_align(pred, frame, required, ("width", "height"))
    dimensions = frame[["width", "height"]].to_numpy(float)
    if np.any(dimensions <= 1):
        raise ValueError("Image dimensions must exceed one pixel")
    short = dimensions.min(axis=1)
    min_fraction = float(_research_setting(cfg, "ROI_MIN_FRACTION", 0.15))
    max_fraction = float(_research_setting(cfg, "ROI_MAX_FRACTION", 0.60))
    margin = float(_research_setting(cfg, "ROI_MARGIN_PX", 16.0))
    if not 0 < min_fraction <= max_fraction <= 1 or not math.isfinite(margin) or margin < 0:
        raise ValueError("Invalid ROI fraction bounds or pixel margin")
    floor_px = float(_research_setting(cfg, "UNCERTAINTY_FLOOR_PX", 2.0))
    if mode == "fixed":
        fraction = float(fixed_fraction if fixed_fraction is not None else
                         _research_setting(cfg, "FIXED_ROI_FRACTION", 0.30))
        if not math.isfinite(fraction) or fraction <= 0:
            raise ValueError("Fixed ROI fraction must be finite and positive")
        base_side = fraction * short
    else:
        if mode == "calibrated":
            if calibration is None or "alpha" not in calibration:
                raise ValueError("Calibrated ROIs require fitted alpha")
            multiplier = float(calibration["alpha"])
            floor_px = float(calibration.get("uncertainty_floor_px", floor_px))
        else:
            multiplier = float(_research_setting(cfg, "RAW_RADIUS_MULTIPLIER", 2.0))
        if not math.isfinite(multiplier) or multiplier < 0:
            raise ValueError("Radius multiplier must be finite and nonnegative")
        base_side = 2 * multiplier * _research_uncertainty(pred, floor_px)
    proposed = base_side + 2 * margin
    if not np.isfinite(proposed).all():
        raise ValueError("Proposed ROI sides overflowed or contain non-finite values")
    side = np.clip(proposed, min_fraction * short, max_fraction * short)
    center = pred[["x", "y"]].to_numpy(float)
    if np.any(center < 0) or np.any(center > dimensions - 1 + 1e-6):
        raise ValueError("ROI centers must be within their original image")
    lower = center - side[:, None] / 2
    upper = center + side[:, None] / 2
    effective_lower = np.maximum(lower, 0)
    effective_upper = np.minimum(upper, dimensions - 1)
    # Areas are geometric fields in pixel-center coordinates; do not add +1.
    # Geometric crop area remains side**2 for matching the resampling field.
    effective_area = np.maximum(effective_upper[:, 0] - effective_lower[:, 0], 0) * np.maximum(effective_upper[:, 1] - effective_lower[:, 1], 0)
    return pd.DataFrame({
        "image_id": pred.image_id,
        "crop_x": center[:, 0], "crop_y": center[:, 1], "crop_side": side,
        "roi_x0": effective_lower[:, 0], "roi_y0": effective_lower[:, 1],
        "roi_x1": effective_upper[:, 0], "roi_y1": effective_upper[:, 1],
        "roi_area_px2": side ** 2, "roi_effective_area_px2": effective_area,
        "roi_area_fraction": side ** 2 / np.prod(dimensions, axis=1),
        "roi_at_min": proposed < min_fraction * short,
        "roi_at_max": proposed > max_fraction * short,
        "roi_has_padding": np.any(lower < 0, axis=1) | np.any(upper > dimensions - 1, axis=1),
        "roi_mode": mode,
    })


def roi_contains_truth(roi, truth):
    """Evaluation only. Never call this to select per-image inference behavior."""
    roi, truth = _research_align(roi, truth, ("roi_x0", "roi_y0", "roi_x1", "roi_y1"))
    point = truth[["x", "y"]].to_numpy(float)
    return pd.Series(
        (point[:, 0] >= roi.roi_x0.to_numpy(float)) & (point[:, 0] <= roi.roi_x1.to_numpy(float)) &
        (point[:, 1] >= roi.roi_y0.to_numpy(float)) & (point[:, 1] <= roi.roi_y1.to_numpy(float)),
        index=roi.image_id, name="contains_truth",
    )


def localization_metrics(pred, truth, roi=None):
    pred, truth = _research_align(pred, truth)
    delta = pred[["x", "y"]].to_numpy(float) - truth[["x", "y"]].to_numpy(float)
    distance = np.linalg.norm(delta, axis=1)
    result = {
        "n_images": int(len(pred)), "rmse": float(np.sqrt(np.mean(delta ** 2))),
        "mean_distance_px": float(distance.mean()),
        "median_distance_px": float(np.median(distance)),
        "p90_distance_px": float(np.quantile(distance, 0.90)),
        "max_distance_px": float(distance.max()),
    }
    if "group" in truth:
        result["n_groups"] = int(truth.group.nunique())
    if "fallback_used" in pred:
        result["fallback_rate"] = float(pred.fallback_used.astype(bool).mean())
    if roi is not None:
        roi, truth_roi = _research_align(roi, truth, ("crop_side", "roi_x0", "roi_y0", "roi_x1", "roi_y1"))
        result["roi_coverage"] = float(roi_contains_truth(roi, truth_roi).mean())
        result["mean_roi_area_px2"] = float(np.mean(roi.crop_side.to_numpy(float) ** 2))
        for column, key in [("roi_area_fraction", "mean_roi_area_fraction"),
                            ("roi_effective_area_px2", "mean_roi_effective_area_px2"),
                            ("roi_at_max", "roi_max_clamp_rate"),
                            ("roi_at_min", "roi_min_clamp_rate"),
                            ("roi_has_padding", "roi_padding_rate")]:
            if column in roi:
                value = pd.to_numeric(roi[column], errors="raise").to_numpy(float)
                if not np.isfinite(value).all():
                    raise ValueError(f"ROI metric {column} contains non-finite values")
                result[key] = float(value.mean())
    return result


def apply_fallback(coarse, fine, rule):
    """Use only prediction features at inference; no reference labels accepted."""
    coarse, fine = _research_align(coarse, fine, ("x", "y", "uncertainty"),
                                   ("x", "y", "crop_edge_fraction"))
    edge = fine.crop_edge_fraction.to_numpy(float)
    if np.any(edge < -1e-6) or np.any(edge > 0.5 + 1e-6):
        raise ValueError("crop_edge_fraction must be in [0,0.5]")
    edge = np.clip(edge, 0.0, 0.5)  # Numerical tolerance must not trigger a zero-edge rule.
    spread = _research_uncertainty(coarse, rule.get("uncertainty_floor_px", 2.0))
    disagreement = np.linalg.norm(fine[["x", "y"]].to_numpy(float) - coarse[["x", "y"]].to_numpy(float), axis=1) / spread
    if not np.isfinite(disagreement).all():
        raise ValueError("Standardized coarse/fine disagreement must be finite")
    used = np.zeros(len(fine), dtype=bool)
    if rule.get("enabled", False):
        edge_threshold = float(rule["edge_threshold"])
        disagreement_threshold = float(rule["disagreement_threshold"])
        if not math.isfinite(edge_threshold) or not 0 <= edge_threshold <= 0.5 or not math.isfinite(disagreement_threshold) or disagreement_threshold <= 0:
            raise ValueError("Fallback thresholds need edge in [0,0.5] and positive finite disagreement")
        used = (edge < edge_threshold) | (disagreement > disagreement_threshold)
    result = fine.copy()
    result.loc[used, ["x", "y"]] = coarse.loc[used, ["x", "y"]].to_numpy(float)
    if "uncertainty" in result:
        result.loc[used, "uncertainty"] = coarse.loc[used, "uncertainty"].to_numpy(float)
    covariance_columns = [column for column in ("cov_xx", "cov_xy", "cov_yy")
                          if column in coarse and column in result]
    if covariance_columns:
        result.loc[used, covariance_columns] = coarse.loc[used, covariance_columns].to_numpy(float)
    result["fallback_used"] = used
    result["coarse_fine_disagreement_standardized"] = disagreement
    return result


def fit_fallback_rule(coarse, fine, truth, cfg):
    """Small prespecified search on calibration labels, with no-fallback default.

    Select a rule only with enough calibration groups, a material absolute AND
    relative RMSE gain, and no unacceptable P90 increase. Three prespecified
    paired candidates replace the legacy Cartesian threshold grid. This is
    NOT a per-image oracle and does not promise improvement on held-out images.
    """
    coarse, truth = _research_align(coarse, truth, ("x", "y", "uncertainty"), ("x", "y", "group"))
    coarse, fine = _research_align(coarse, fine, ("x", "y", "uncertainty"), ("x", "y", "crop_edge_fraction"))
    floor_px = float(_research_setting(cfg, "UNCERTAINTY_FLOOR_PX", 2.0))
    baseline_rule = {"enabled": False, "edge_threshold": 0.0,
                     "disagreement_threshold": 1e9, "uncertainty_floor_px": floor_px}
    baseline_pred = apply_fallback(coarse, fine, baseline_rule)
    baseline = localization_metrics(baseline_pred, truth)
    min_gain = float(_research_setting(cfg, "FALLBACK_MIN_GAIN_PX", 1.0))
    relative_gain = float(_research_setting(cfg, "FALLBACK_MIN_RELATIVE_GAIN", 0.01))
    min_groups_value = float(_research_setting(cfg, "FALLBACK_MIN_GROUPS", 12))
    p90_tolerance = float(_research_setting(cfg, "FALLBACK_P90_TOLERANCE", 0.0))
    if (not math.isfinite(min_gain) or min_gain < 0 or not math.isfinite(relative_gain)
            or not 0 <= relative_gain <= 1 or not math.isfinite(p90_tolerance) or p90_tolerance < 0):
        raise ValueError("Fallback improvement/tolerance must be finite and nonnegative")
    if not math.isfinite(min_groups_value) or min_groups_value < 2 or not min_groups_value.is_integer():
        raise ValueError("FALLBACK_MIN_GROUPS must be an integer of at least two")
    min_groups = int(min_groups_value)
    n_groups = int(truth.group.nunique())
    required_gain = max(min_gain, relative_gain * baseline["rmse"])
    table = []
    winner, best = baseline_rule, baseline
    candidate_settings = _research_setting(cfg, "FALLBACK_RULE_CANDIDATES", ((0.05, 1e9), (0.0, 4.0), (0.05, 4.0)))
    candidates = []
    for pair in candidate_settings:
        if len(pair) != 2:
            raise ValueError("Each fallback candidate must contain (edge, disagreement)")
        edge, disagreement = map(float, pair)
        if not math.isfinite(edge) or not 0 <= edge <= 0.5:
            raise ValueError("Fallback candidate edge must be finite and in [0,0.5]")
        if not math.isfinite(disagreement) or disagreement <= 0:
            raise ValueError("Fallback candidate disagreement must be finite and positive")
        if (edge, disagreement) not in candidates:
            candidates.append((edge, disagreement))
    if n_groups >= min_groups:
        for edge, disagreement in candidates:
            rule = {"enabled": True, "edge_threshold": float(edge),
                    "disagreement_threshold": float(disagreement), "uncertainty_floor_px": floor_px}
            metrics = localization_metrics(apply_fallback(coarse, fine, rule), truth)
            gain = baseline["rmse"] - metrics["rmse"]
            eligible = (gain > 0 and gain >= required_gain and
                        metrics["p90_distance_px"] <= baseline["p90_distance_px"] * (1 + p90_tolerance) + 1e-9)
            table.append({**rule, "rmse": metrics["rmse"], "p90_distance_px": metrics["p90_distance_px"],
                          "fallback_rate": metrics["fallback_rate"], "gain_px": float(gain),
                          "required_gain_px": float(required_gain), "eligible": bool(eligible)})
            if eligible and metrics["rmse"] < best["rmse"] - 1e-12:
                winner, best = rule, metrics
    return {
        **winner, "n_images": int(len(truth)), "n_groups": n_groups,
        "calibration_image_ids": truth.image_id.tolist(),
        "calibration_group_ids": sorted(truth.group.unique().tolist()),
        "baseline_calibration_metrics": baseline, "selected_calibration_metrics": best,
        "search": table, "p90_relative_tolerance": p90_tolerance,
        "min_calibration_groups": min_groups, "min_gain_px": min_gain,
        "min_relative_gain": relative_gain, "required_gain_px": float(required_gain),
        "candidate_count": len(candidates), "evaluated_candidate_count": len(table),
        "selection_reason": ("insufficient_calibration_groups" if n_groups < min_groups else
                             "material_gain_and_p90_passed" if winner["enabled"] else
                             "no_candidate_met_material_gain_and_p90_requirements"),
        "note": "Rule fitted on calibration labels only. Selection metrics are optimistic calibration-fit scores; evaluate all images on untouched outer validation folds.",
    }


def research_calibration_diagnostics(coarse, truth, roi, radius_cfg, cfg):
    """Read-only diagnostics; never fit a parameter or change an ROI/prediction.

    May be called on calibration-fit or untouched outer-validation data. The
    role is labelled from IDs and groups saved in radius_cfg. A calibration-fit
    coverage or error association is not a held-out performance estimate.
    """
    coarse, truth = _research_align(coarse, truth, ("x", "y", "uncertainty"),
                                    ("x", "y", "group", "width", "height"))
    required_roi = ("crop_side", "roi_x0", "roi_y0", "roi_x1", "roi_y1")
    _, roi = _research_align(coarse, roi, ("x", "y", "uncertainty"), required_roi)
    sides = roi.crop_side.to_numpy(float)
    short = truth[["width", "height"]].to_numpy(float).min(axis=1)
    if np.any(sides <= 0) or np.any(short <= 1):
        raise ValueError("Diagnostic ROI sides must be positive and dimensions exceed one pixel")
    if np.any(roi.roi_x0 > roi.roi_x1) or np.any(roi.roi_y0 > roi.roi_y1):
        raise ValueError("Diagnostic ROI lower bounds must not exceed upper bounds")
    radius_cfg = {} if radius_cfg is None else dict(radius_cfg)
    target = float(radius_cfg.get("target_coverage", _research_setting(cfg, "ROI_COVERAGE", .90)))
    if not math.isfinite(target) or not 0 < target <= 1:
        raise ValueError("Diagnostic target coverage must lie in (0,1]")
    floor = float(radius_cfg.get("uncertainty_floor_px", _research_setting(cfg, "UNCERTAINTY_FLOOR_PX", 2.0)))
    spread = _research_uncertainty(coarse, floor)
    distance = np.hypot(coarse.x.to_numpy(float) - truth.x.to_numpy(float),
                        coarse.y.to_numpy(float) - truth.y.to_numpy(float))
    if not np.isfinite(distance).all():
        raise ValueError("Diagnostic localization errors must be finite")
    ranks_u, ranks_error = pd.Series(spread).rank(), pd.Series(distance).rank()
    rho = ranks_u.corr(ranks_error) if ranks_u.nunique() > 1 and ranks_error.nunique() > 1 else None
    rho = float(rho) if rho is not None and math.isfinite(float(rho)) else None
    covered = roi_contains_truth(roi, truth).to_numpy(bool)
    group_coverage = pd.DataFrame({"group": truth.group, "covered": covered}).groupby("group").covered.min()
    side_fraction = sides / short
    side_cv = float(np.std(sides) / np.mean(sides))
    fraction_cv = float(np.std(side_fraction) / np.mean(side_fraction))
    minimum = float(_research_setting(cfg, "ROI_MIN_FRACTION", .15))
    maximum = float(_research_setting(cfg, "ROI_MAX_FRACTION", .60))
    if not 0 < minimum <= maximum <= 1:
        raise ValueError("Invalid diagnostic ROI fraction bounds")
    clamp_rates = {}
    for column, fallback in (("roi_at_min", np.isclose(side_fraction, minimum)),
                              ("roi_at_max", np.isclose(side_fraction, maximum))):
        values = pd.to_numeric(roi[column], errors="raise").to_numpy(float) if column in roi else fallback.astype(float)
        if not np.isfinite(values).all() or np.any((values != 0) & (values != 1)):
            raise ValueError(f"Diagnostic {column} must contain boolean or binary values")
        clamp_rates[column] = float(values.mean())
    evaluation_ids, evaluation_groups = set(coarse.image_id), set(truth.group)
    calibration_ids = set(map(str, radius_cfg.get("calibration_image_ids", [])))
    calibration_groups = set(map(str, radius_cfg.get("calibration_group_ids", [])))
    overlap_ids = len(evaluation_ids & calibration_ids)
    overlap_groups = len(evaluation_groups & calibration_groups)
    if not calibration_ids and not calibration_groups:
        role = "unknown_calibration_membership"
    elif evaluation_ids <= calibration_ids:
        role = "calibration_fit"
    elif overlap_ids or overlap_groups:
        role = "mixed_or_overlapping_calibration_membership"
    else:
        role = "held_out_from_radius_calibration"
    n_calibration = int(radius_cfg.get("n_groups", len(calibration_groups)))
    if n_calibration < 0:
        raise ValueError("Diagnostic calibration group count must not be negative")
    warnings = []
    if n_calibration < 20:
        warnings.append("Small calibration sample: tail quantiles and coverage estimates are imprecise.")
    if clamp_rates["roi_at_min"] >= .80:
        warnings.append("At least 80% of ROIs are lower-clamped; adaptive sizes may behave like a fixed fractional ROI.")
    if clamp_rates["roi_at_max"] >= .80:
        warnings.append("At least 80% of ROIs are upper-clamped; intended coverage may be truncated.")
    if fraction_cv < .02:
        warnings.append("ROI side fractions have below 2% coefficient of variation; adaptation is nearly constant.")
    if rho is None:
        warnings.append("Dispersion/error association is undefined because a ranked variable is constant or there are too few cases.")
    elif rho <= 0:
        warnings.append("Observed dispersion/error rank association is nonpositive; dispersion has not demonstrated useful error ordering here.")
    actual = float(covered.mean())
    if actual < target:
        warnings.append("Observed bounded-ROI coverage is below the empirical target; this observation does not retune parameters.")
    if role != "held_out_from_radius_calibration":
        warnings.append("These diagnostics are not an untouched outer-validation performance estimate.")
    return {
        "n_images": int(len(coarse)), "n_groups": int(truth.group.nunique()),
        "sample_role": role, "calibration_n_groups": n_calibration,
        "calibration_id_overlap": overlap_ids, "calibration_group_overlap": overlap_groups,
        "empirical_quantile_group_resolution": 1.0 / n_calibration if n_calibration else None,
        "target_coverage": target, "bounded_roi_image_coverage": actual,
        "bounded_roi_group_coverage": float(group_coverage.mean()), "coverage_minus_target": actual - target,
        "roi_side_mean_px": float(sides.mean()), "roi_side_min_px": float(sides.min()),
        "roi_side_max_px": float(sides.max()), "roi_side_cv": side_cv,
        "roi_side_fraction_cv": fraction_cv, "roi_min_clamp_rate": clamp_rates["roi_at_min"],
        "roi_max_clamp_rate": clamp_rates["roi_at_max"],
        "adaptive_roi_nearly_constant": bool(fraction_cv < .02),
        "dispersion_error_spearman": rho, "warnings": warnings,
        "changes_parameters": False,
        "interpretation": "Read-only empirical diagnostics of heatmap dispersion and bounded ROI geometry; no probability calibration, finite-sample coverage, or accuracy improvement is asserted.",
    }


def uncertainty_diagnostics(pred, truth, calibration=None, cfg=None):
    """Risk-coverage sorts by PREDICTED uncertainty, never by true error.

    The separately named oracle curve is a descriptive lower bound only.
    Report full-coverage RMSE regardless of any selective-risk curve.
    """
    pred, truth = _research_align(pred, truth, ("x", "y", "uncertainty"))
    uncertainty = _research_uncertainty(pred, _research_setting(cfg, "UNCERTAINTY_FLOOR_PX", 2.0))
    delta = pred[["x", "y"]].to_numpy(float) - truth[["x", "y"]].to_numpy(float)
    squared = np.mean(delta ** 2, axis=1)
    distance = np.linalg.norm(delta, axis=1)
    ranks_u, ranks_e = pd.Series(uncertainty).rank(), pd.Series(distance).rank()
    rho = ranks_u.corr(ranks_e) if ranks_u.nunique() > 1 and ranks_e.nunique() > 1 else None
    predicted_order = np.argsort(uncertainty, kind="stable")
    oracle_order = np.argsort(distance, kind="stable")
    risk = []
    for fraction in (0.10, 0.25, 0.50, 0.75, 1.0):
        retained = max(1, int(np.ceil(fraction * len(pred))))
        risk.append({"requested_retention": fraction, "actual_retention": retained / len(pred),
                     "n_retained": retained,
                     "rmse_by_predicted_uncertainty": float(np.sqrt(squared[predicted_order[:retained]].mean())),
                     "oracle_rmse_by_true_error": float(np.sqrt(squared[oracle_order[:retained]].mean()))})
    result = {"n_images": int(len(pred)), "full_coverage_rmse": float(np.sqrt(squared.mean())),
              "uncertainty_error_spearman": None if rho is None or not np.isfinite(rho) else float(rho),
              "uncertainty_definition": "sqrt(max eigenvalue of original-pixel heatmap covariance); spread proxy, not ERE or epistemic uncertainty",
              "risk_coverage": risk}
    if calibration is not None:
        if cfg is None:
            raise ValueError("cfg is required for actual bounded-ROI calibration diagnostics")
        coverage_rows = []
        for level, alpha in sorted(calibration.get("alpha_by_coverage", {}).items(), key=lambda item: float(item[0])):
            roi = make_roi_spec(pred, truth, cfg, "calibrated", {**calibration, "alpha": alpha})
            actual = roi_contains_truth(roi, truth)
            row = {"nominal_coverage": float(level), "alpha": float(alpha),
                   "actual_bounded_roi_image_coverage": float(actual.mean())}
            if "group" in truth:
                grouped = pd.DataFrame({"group": truth.group.to_numpy(), "covered": actual.to_numpy()})
                row["actual_bounded_roi_group_coverage"] = float(grouped.groupby("group").covered.min().mean())
            coverage_rows.append(row)
        result["bounded_roi_calibration_curve"] = coverage_rows
        result["calibration_note"] = "Actual coverage includes margin and min/max size clamps; empirical target is not a guarantee."
    return result


def paired_group_bootstrap(predictions_by_method, truth, baseline=None, n_bootstrap=2000, seed=2026):
    """Paired percentile intervals conditional on these fitted OOF predictions.

    Resample whole groups; preserve all labeled rows within each sampled group.
    This does not account for retraining variation or unknown patient overlap.
    """
    if not predictions_by_method:
        raise ValueError("At least one prediction method is required")
    names = list(predictions_by_method)
    baseline = names[0] if baseline is None else baseline
    if baseline not in names or int(n_bootstrap) < 100:
        raise ValueError("Invalid bootstrap baseline or fewer than 100 resamples")
    truth = _research_frame(truth, ("x", "y", "group"), "bootstrap truth")
    # Establish one reference order for all methods, regardless of their row order.
    reference_ids = truth.image_id.tolist()
    group_codes, group_names = pd.factorize(truth.group, sort=True)
    n_groups = len(group_names)
    counts = np.bincount(group_codes, minlength=n_groups).astype(float)
    sums_by_method = []
    point_rmse = []
    for name in names:
        pred, aligned_truth = _research_align(predictions_by_method[name], truth)
        pred = pred.set_index("image_id").loc[reference_ids]
        error = pred[["x", "y"]].to_numpy(float) - truth[["x", "y"]].to_numpy(float)
        squared = np.mean(error ** 2, axis=1)
        sums_by_method.append(np.bincount(group_codes, weights=squared, minlength=n_groups))
        point_rmse.append(float(np.sqrt(squared.mean())))
    rng = np.random.default_rng(int(seed))
    multiplicity = rng.multinomial(n_groups, np.full(n_groups, 1 / n_groups), size=int(n_bootstrap))
    denominator = multiplicity @ counts
    scores = np.sqrt((multiplicity @ np.asarray(sums_by_method).T) / denominator[:, None])
    baseline_index = names.index(baseline)
    records = []
    for index, name in enumerate(names):
        delta = scores[:, index] - scores[:, baseline_index]
        records.append({"method": name, "baseline": baseline, "rmse": point_rmse[index],
                        "rmse_ci_low": float(np.quantile(scores[:, index], 0.025)),
                        "rmse_ci_high": float(np.quantile(scores[:, index], 0.975)),
                        "delta_rmse": point_rmse[index] - point_rmse[baseline_index],
                        "delta_ci_low": float(np.quantile(delta, 0.025)),
                        "delta_ci_high": float(np.quantile(delta, 0.975)),
                        "n_images": int(len(truth)), "n_groups": int(n_groups),
                        "n_bootstrap": int(n_bootstrap), "seed": int(seed),
                        "ci_note": "Paired group percentile bootstrap of fixed OOF predictions; excludes retraining variation and unobserved patient dependence."})
    return pd.DataFrame(records)


PIPELINE_VERSION = "2026task3-fovea-1.2-real-data"


def json_save(value, path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)


def audit_xml_boxes(train, root, output_dir, coordinate_origin=0):
    """Read optional Pascal VOC XML for audit only; CSV points remain the targets."""
    import xml.etree.ElementTree as ET
    xml_paths = sorted(Path(root).rglob("*.xml"))
    rows, errors = [], []
    train_index = build_image_index([Path(path) for path in train.path], root=root)
    path_to_id = dict(zip(train.path, train.image_id))
    for path in xml_paths:
        try:
            tree = ET.parse(path).getroot()
            if image_split_role(path, root) == "test":
                continue
            image_path = resolve_image(train_index, tree.findtext("filename") or path.stem, "train")
            key = path_to_id[str(image_path)]
            for obj in tree.findall(".//object"):
                box = obj.find("bndbox")
                if box is None:
                    continue
                values = [float(box.findtext(name)) - coordinate_origin
                          for name in ("xmin", "ymin", "xmax", "ymax")]
                if not np.isfinite(values).all() or values[2] < values[0] or values[3] < values[1]:
                    raise ValueError("Invalid bounding box")
                rows.append({"image_id": key, "class": obj.findtext("name") or "",
                             "xmin": values[0], "ymin": values[1],
                             "xmax": values[2], "ymax": values[3], "xml_path": str(path)})
        except Exception as exc:
            errors.append({"path": str(path), "error": str(exc)})
    columns = ["image_id", "class", "xmin", "ymin", "xmax", "ymax", "xml_path"]
    boxes = pd.DataFrame(rows, columns=columns)
    if len(boxes):
        boxes = boxes.merge(train[["image_id", "x", "y", "visible"]], on="image_id", validate="many_to_one")
        boxes["point_inside"] = boxes.visible & boxes.x.between(boxes.xmin, boxes.xmax) & boxes.y.between(boxes.ymin, boxes.ymax)
        boxes["box_center_distance_px"] = np.where(
            boxes.visible,
            np.hypot(boxes.x - (boxes.xmin + boxes.xmax) / 2,
                     boxes.y - (boxes.ymin + boxes.ymax) / 2), np.nan)
    boxes.to_csv(Path(output_dir) / "xml_box_audit.csv", index=False)
    json_save(errors, Path(output_dir) / "xml_parse_warnings.json")
    print(f"XML audit: {len(xml_paths)} files, {len(boxes)} boxes, {len(errors)} parse warnings.")
    return boxes


def plot_label_audit(train, boxes, output_dir, show=True):
    from matplotlib.patches import Rectangle
    for start in range(0, len(train), 16):
        block = train.iloc[start:start + 16]
        fig, axes = plt.subplots(math.ceil(len(block) / 4), 4, figsize=(16, 4 * math.ceil(len(block) / 4)),
                                 squeeze=False)
        for axis, (_, row) in zip(axes.flat, block.iterrows()):
            axis.imshow(read_rgb(row.path))
            if row.visible:
                axis.scatter(row.x, row.y, marker="+", s=120, color="lime", linewidths=2)
            if len(boxes):
                for _, box in boxes[boxes.image_id == row.image_id].iterrows():
                    axis.add_patch(Rectangle((box.xmin, box.ymin), box.xmax - box.xmin,
                                             box.ymax - box.ymin, fill=False, edgecolor="yellow"))
            axis.set_title(f"{row.image_id} | {'visible' if row.visible else '(0,0) invisible'}", fontsize=9)
            axis.axis("off")
        for axis in list(axes.flat)[len(block):]:
            axis.axis("off")
        fig.tight_layout()
        fig.savefig(Path(output_dir) / f"labels_{start:03d}.png", dpi=120, bbox_inches="tight")
        if show and start == 0:
            plt.show()
        plt.close(fig)


def apply_visibility(frame, threshold, coordinate_origin=0):
    prediction = frame[["x", "y"]].to_numpy(dtype=np.float64).copy()
    keep = frame.visibility_prob.to_numpy(float) >= threshold
    prediction[keep] += coordinate_origin
    prediction[~keep] = 0.0
    return prediction


def truth_in_submission_coordinates(frame, coordinate_origin=0):
    truth = frame[["x_true", "y_true"]].to_numpy(dtype=np.float64).copy()
    truth[frame.visible_true.to_numpy(bool)] += coordinate_origin
    return truth


def score_frame(frame, threshold, coordinate_origin=0):
    return float(np.sqrt(np.mean((apply_visibility(frame, threshold, coordinate_origin) -
                                  truth_in_submission_coordinates(frame, coordinate_origin))**2)))


def choose_threshold(oof, cfg, visibility_supported):
    baseline = score_frame(oof, cfg.VISIBILITY_THRESHOLD, cfg.COORDINATE_ORIGIN)
    threshold = cfg.VISIBILITY_THRESHOLD
    results = {"fixed_threshold": threshold, "oof_rmse_fixed_threshold": baseline,
               "selected_threshold": threshold, "threshold_tuned": False,
               "score_note": "Development CV estimate: checkpoints selected on held-out fold RMSE."}
    if not cfg.TUNE_VISIBILITY_THRESHOLD:
        return threshold, results
    counts = oof.visible_true.astype(bool).value_counts()
    if not visibility_supported or len(counts) != 2 or counts.min() < 5:
        warnings.warn("Threshold tuning skipped: require at least five examples of each visibility class.")
        return threshold, results
    candidates = sorted(set(np.linspace(0.0, 0.95, 20).tolist() + [cfg.VISIBILITY_THRESHOLD]))
    table = [{"threshold": float(t), "rmse": score_frame(oof, t, cfg.COORDINATE_ORIGIN)}
             for t in candidates]
    winner = min(table, key=lambda row: (row["rmse"], abs(row["threshold"] - cfg.VISIBILITY_THRESHOLD)))
    threshold = winner["threshold"]
    results.update({"selected_threshold": threshold, "threshold_tuned": True,
                    "oof_rmse_threshold_tuned_optimistic": winner["rmse"],
                    "threshold_search": table,
                    "threshold_note": "Threshold selected on these OOF labels; tuned score is optimistic, not independent validation."})
    return threshold, results


def write_submission(sample, layout, prediction_frame, metadata, cfg, threshold):
    if prediction_frame.image_id.duplicated().any():
        raise ValueError("Duplicate predictions for a test image.")
    if set(prediction_frame.image_id) != set(layout.image_id):
        raise ValueError("Predicted image IDs do not match the sample submission.")
    prediction = apply_visibility(prediction_frame, threshold, cfg.COORDINATE_ORIGIN)
    mapping = {key: point for key, point in zip(prediction_frame.image_id, prediction)}
    values = [mapping[row.image_id][0 if row.axis == "x" else 1] for row in layout.itertuples()]
    result = sample.copy()
    result[metadata["value_column"]] = np.asarray(values, dtype=np.float64)
    assert len(result) == len(sample)
    assert result[metadata["id_column"]].equals(sample[metadata["id_column"]])
    if not np.isfinite(result[metadata["value_column"]]).all():
        raise ValueError("Submission contains non-finite values.")
    for row, point in zip(prediction_frame.itertuples(), prediction):
        if np.all(point == 0):
            continue
        low = cfg.COORDINATE_ORIGIN
        upper = np.array([row.width - 1, row.height - 1]) + low
        if np.any(point < low - 1e-6) or np.any(point > upper + 1e-6):
            raise ValueError(f"Out-of-bounds prediction: {row.image_id} -> {point}")
    path = Path(cfg.SUBMISSION_PATH)
    path.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(path, index=False, float_format="%.6f")
    # Round-trip through disk: verify actual exported IDs and numeric values.
    reread = pd.read_csv(path, dtype={metadata["id_column"]: str}, keep_default_na=False)
    assert reread[metadata["id_column"]].tolist() == sample[metadata["id_column"]].tolist()
    assert np.isfinite(pd.to_numeric(reread[metadata["value_column"]], errors="raise")).all()
    print(f"Submission saved: {path} ({len(result)} rows)")
    print(result.head(8).to_string(index=False))
    return result


def plot_oof_errors(oof, train, threshold, cfg, output_dir):
    points = apply_visibility(oof, threshold, coordinate_origin=0)
    scored_points = apply_visibility(oof, threshold, cfg.COORDINATE_ORIGIN)
    scored_truth = truth_in_submission_coordinates(oof, cfg.COORDINATE_ORIGIN)
    errors = np.linalg.norm(scored_points - scored_truth, axis=1)
    report = oof.copy()
    report["pred_x"] = points[:, 0]
    report["pred_y"] = points[:, 1]
    report["euclidean_error_px"] = errors
    report = report.merge(train[["image_id", "path"]], on="image_id", validate="one_to_one")
    report.to_csv(Path(output_dir) / "oof_error_details.csv", index=False)
    worst = report.nlargest(min(12, len(report)), "euclidean_error_px")
    fig, axes = plt.subplots(math.ceil(len(worst) / 4), 4, figsize=(16, 4 * math.ceil(len(worst) / 4)), squeeze=False)
    for axis, row in zip(axes.flat, worst.itertuples()):
        axis.imshow(read_rgb(row.path))
        if row.visible_true:
            axis.scatter(row.x_true, row.y_true, marker="+", color="lime", s=120, label="truth")
        if row.visibility_prob >= threshold:
            axis.scatter(row.pred_x, row.pred_y, marker="x", color="red", s=90, label="prediction")
        axis.set_title(f"{row.image_id} | error {row.euclidean_error_px:.1f}px\np_visible={row.visibility_prob:.2f}", fontsize=9)
        axis.axis("off")
    for axis in list(axes.flat)[len(worst):]:
        axis.axis("off")
    fig.tight_layout()
    fig.savefig(Path(output_dir) / "oof_worst_cases.png", dpi=120, bbox_inches="tight")
    if cfg.SHOW_PLOTS:
        plt.show()
    plt.close(fig)
    return report




def _research_cleanup(model=None):
    if model is not None:
        model.cpu()
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def research_group_holdout(frame, fraction, seed):
    groups = frame.group.unique().copy()
    if len(groups) < 5:
        raise ValueError("Too few groups for a development/calibration split.")
    np.random.default_rng(seed).shuffle(groups)
    count = min(len(groups) - 3, max(2, int(round(len(groups) * fraction))))
    selected = set(groups[:count])
    fit = frame.loc[~frame.group.isin(selected)].reset_index(drop=True)
    held = frame.loc[frame.group.isin(selected)].reset_index(drop=True)
    assert not set(fit.group) & set(held.group)
    return fit, held


def _research_predictions(frame, path):
    frame.to_csv(path, index=False)


def research_file_digest(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def research_runtime_code_digest():
    """Detect edited notebook function bodies without relying on source files.

    Cell execution filenames/line offsets are normalized so Run All can reuse
    the same code. Nested code objects are normalized recursively as well.
    """
    import types
    import inspect
    def normalize(value):
        # marshal encodes object sharing/interning flags which can change after
        # imports or training. Hash semantic values instead of those flags.
        if isinstance(value, types.CodeType):
            return {"bytecode": value.co_code.hex(), "constants": normalize(value.co_consts),
                    "names": value.co_names, "variables": value.co_varnames,
                    "free": value.co_freevars, "cells": value.co_cellvars,
                    "arguments": [value.co_argcount, value.co_posonlyargcount, value.co_kwonlyargcount],
                    "flags": value.co_flags}
        if isinstance(value, (tuple, list)):
            return [normalize(item) for item in value]
        if isinstance(value, frozenset):
            return {"frozenset": sorted([normalize(item) for item in value], key=lambda item: json.dumps(item, sort_keys=True))}
        if isinstance(value, bytes):
            return {"bytes": value.hex()}
        if isinstance(value, (float, complex)):
            return {type(value).__name__: repr(value)}
        if value is Ellipsis:
            return {"ellipsis": True}
        return value
    digest = hashlib.sha256()
    for name, value in sorted(list(globals().items())):
        entries = []
        if isinstance(value, types.FunctionType):
            value = inspect.unwrap(value)
        if isinstance(value, types.FunctionType) and value.__globals__ is globals():
            entries.append((name, value))
        elif isinstance(value, type) and value.__module__ == __name__:
            for member, function in sorted(vars(value).items()):
                if isinstance(function, (staticmethod, classmethod)):
                    function = function.__func__
                if isinstance(function, types.FunctionType):
                    function = inspect.unwrap(function)
                    entries.append((name + "." + member, function))
        for label, function in entries:
            digest.update(label.encode())
            digest.update(json.dumps(normalize(function.__code__), sort_keys=True, ensure_ascii=True).encode())
            digest.update(json.dumps(normalize(function.__defaults__), sort_keys=True, default=str).encode())
            digest.update(json.dumps(function.__kwdefaults__, sort_keys=True, default=str).encode())
    return digest.hexdigest()


def research_training_environment():
    import importlib.metadata
    return {"python": platform.python_version(), "torch": str(torch.__version__),
            "torchvision": importlib.metadata.version("torchvision"),
            "numpy": np.__version__, "pandas": pd.__version__, "opencv": cv2.__version__,
            "device_type": "cuda" if torch.cuda.is_available() else "cpu"}


def research_cache_manifest(folder, paths):
    folder = Path(folder).resolve()
    result = {}
    for path in sorted(map(Path, paths)):
        resolved = path.resolve()
        relative = resolved.relative_to(folder).as_posix()
        if not resolved.is_file():
            raise FileNotFoundError(resolved)
        result[relative] = {"bytes": resolved.stat().st_size, "sha256": research_file_digest(resolved)}
    if not result:
        raise ValueError("Cannot mark an empty cache as complete.")
    return result


def research_verify_cache(folder, manifest, required_paths=()):
    folder = Path(folder).resolve()
    if not isinstance(manifest, dict) or not manifest:
        raise ValueError(f"{folder}: cache has no integrity manifest; use a new OUTPUT_DIR or REUSE_CHECKPOINTS=False.")
    missing = set(required_paths) - set(manifest)
    if missing:
        raise ValueError(f"Cache integrity manifest is missing required entries: {sorted(missing)}")
    for relative, expected in manifest.items():
        path = (folder / relative).resolve()
        try:
            path.relative_to(folder)
        except ValueError as exc:
            raise ValueError("Cache manifest escapes its output directory") from exc
        if (not path.is_file() or path.stat().st_size != expected.get("bytes") or
                research_file_digest(path) != expected.get("sha256")):
            raise ValueError(f"Cache integrity check failed: {path}. Restore the file or set REUSE_CHECKPOINTS=False.")


def research_stage_signature(training, validation, cfg, stage, seed, epochs, train_crop, val_crop):
    def frame_payload(frame, crop=False):
        if frame is None:
            return None
        names = (["image_id", "crop_x", "crop_y", "crop_side"] if crop else
                 ["image_id", "pixel_hash", "x", "y", "visible", "group", "width", "height"])
        cols = [name for name in names if name in frame]
        return frame[cols].to_dict("records")
    config = asdict(cfg)
    # Worker count affects augmentation RNG. Keep it in the identity.
    for name in ("OUTPUT_DIR", "SUBMISSION_PATH", "REUSE_CHECKPOINTS", "SHOW_PLOTS",
                 "PRETRAINED_WEIGHTS_PATH", "INPUT_ROOT", "DATA_DIR", "LABEL_CSV", "SAMPLE_CSV"):
        config.pop(name, None)
    payload = {"source": globals().get("RESEARCH_SOURCE_HASH", RESEARCH_VERSION),
               "runtime_code": research_runtime_code_digest(),
               "environment": research_training_environment(), "config": config, "stage": stage,
               "seed": int(seed), "epochs": epochs, "training": frame_payload(training),
               "validation": frame_payload(validation) if epochs is None else None,
               "train_crop": frame_payload(train_crop, True),
               "val_crop": frame_payload(val_crop, True) if epochs is None else None}
    return hashlib.sha256(json.dumps(payload, sort_keys=True, default=str).encode()).hexdigest()


def _research_stage(training, validation, cfg, device, stage, folder, seed,
                    train_crop=None, val_crop=None, epochs=None):
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    seed = int(seed) + int(cfg.MODEL_SEED_OFFSET)
    cols = [c for c in ("image_id", "group", "pixel_hash") if c in training]
    training[cols].to_csv(folder / "actual_training_inputs.csv", index=False)
    if validation is not None:
        validation[cols].to_csv(folder / "actual_early_stop_inputs.csv", index=False)
    if train_crop is not None:
        train_crop.to_csv(folder / "actual_training_crops.csv", index=False)
    marker = folder / "stage_complete.json"
    signature = research_stage_signature(training, validation, cfg, stage, seed, epochs, train_crop, val_crop)
    if cfg.REUSE_CHECKPOINTS and marker.exists():
        record = json.loads(marker.read_text(encoding="utf-8"))
        if record.get("signature") != signature:
            raise ValueError(f"{folder}: cached stage configuration/data/code changed; use a new OUTPUT_DIR or REUSE_CHECKPOINTS=False.")
        research_verify_cache(folder, record.get("files"), ("model.pt", "history.csv", "training_summary.json"))
        model = research_build_model(cfg, stage, device)
        load_model_checkpoint(model, folder / "model.pt", device)
        info = json.loads((folder / "training_summary.json").read_text(encoding="utf-8"))
        info["history"] = pd.read_csv(folder / "history.csv").to_dict("records")
        info["reused_completed_stage"] = True
        print(f"Reuse completed training stage: {folder.name}", flush=True)
        return model, info
    if marker.exists():
        marker.unlink()
    model, info = fit_research_model(
        training, validation, cfg, device, stage, folder / "model.pt",
        train_crop=train_crop, val_crop=val_crop, epochs=epochs, seed=seed)
    pd.DataFrame(info["history"]).to_csv(folder / "history.csv", index=False)
    safe_info = {k: v for k, v in info.items() if k != "history"}
    json_save(safe_info, folder / "training_summary.json")
    files = [folder / name for name in ("model.pt", "history.csv", "training_summary.json")]
    record = {"signature": signature, "files": research_cache_manifest(folder, files),
              "resume_policy": "Reuse completed stages only; interrupted epochs restart this stage."}
    temporary = marker.with_suffix(".partial.json")
    json_save(record, temporary)
    temporary.replace(marker)
    return model, info


def research_crossfit_crops(development, cfg, device, folder, epochs, seed):
    """OOF coarse centres for fine TRAINING only, not for reporting accuracy.

    Epoch count is selected within development. These training augmentation
    predictions are not claimed to be an unbiased inner validation estimate.
    The separate calibration and outer validation groups never enter this step.
    """
    split_cfg = replace(cfg, N_FOLDS=cfg.INNER_CROP_FOLDS, SEED=seed)
    inner = assign_folds(development, split_cfg)
    parts, audit = [], []
    for f in sorted(inner.fold.unique()):
        fitting = inner.loc[inner.fold != f].reset_index(drop=True)
        held = inner.loc[inner.fold == f].reset_index(drop=True)
        assert not set(fitting.group) & set(held.group)
        model, _ = _research_stage(fitting, None, cfg, device, "coarse",
                                   folder / f"inner_{f}", seed + int(f), epochs=epochs)
        pred = predict_research(model, held, cfg, device, stage="coarse")
        parts.append(pred)
        audit.append({"inner_fold": int(f), "fit_ids": fitting.image_id.tolist(),
                      "held_ids": held.image_id.tolist(),
                      "fit_groups": fitting.group.astype(str).tolist(),
                      "held_groups": held.group.astype(str).tolist()})
        _research_cleanup(model)
        del model
    pred = pd.concat(parts, ignore_index=True)
    assert len(pred) == len(development) and pred.image_id.is_unique
    pred = pred.set_index("image_id").loc[development.image_id].reset_index()
    pred.to_csv(folder / "training_oof_centres.csv", index=False)
    json_save(audit, folder / "crossfit_audit.json")
    return make_roi_spec(pred, development, cfg, mode="fixed")


def _research_attach_truth(pred, truth, fold, experiment):
    cols = ["image_id", "x", "y", "group", "width", "height"]
    labels = truth[cols].rename(columns={"x": "x_true", "y": "y_true"})
    base = pred.drop(columns=[c for c in ("group", "width", "height", "x_true", "y_true")
                              if c in pred], errors="ignore")
    out = base.merge(labels, on="image_id", validate="one_to_one")
    out["fold"], out["experiment"] = fold, experiment
    out["error_px"] = np.hypot(out.x - out.x_true, out.y - out.y_true)
    return out


def _research_match_fixed_roi(coarse, frame, adaptive_roi, cfg):
    target = float(np.mean(adaptive_roi.crop_side.to_numpy() ** 2))
    # Search the BASE fraction before the common margin and clamp. Starting
    # at ROI_MIN_FRACTION would make a minimum-area adaptive crop unmatchable.
    low, high = 1e-8, float(cfg.ROI_MAX_FRACTION)
    for _ in range(40):
        middle = (low + high) / 2
        area = float(np.mean(make_roi_spec(coarse, frame, cfg, mode="fixed",
                                          fixed_fraction=middle).crop_side.to_numpy() ** 2))
        if area < target:
            low = middle
        else:
            high = middle
    fraction = (low + high) / 2
    matched = float(np.mean(make_roi_spec(coarse, frame, cfg, mode="fixed",
                                         fixed_fraction=fraction).crop_side.to_numpy() ** 2))
    return fraction, {"adaptive_mean_area_px2": target,
                                  "matched_mean_area_px2": matched,
                                  "relative_area_difference": abs(matched-target)/max(target, 1e-12),
                                  "selection_source": "calibration image geometry and predictions only"}


def run_research_fold(train, test, fold, cfg, device, output, signature):
    folder = output / f"fold_{fold}"
    folder.mkdir(parents=True, exist_ok=True)
    complete = folder / "complete.json"
    if cfg.REUSE_CHECKPOINTS and complete.exists():
        record = json.loads(complete.read_text(encoding="utf-8"))
        if record.get("signature") != signature:
            raise ValueError(f"{folder}: completed run has a different configuration/data/code. Use a new OUTPUT_DIR.")
        required = [f"{name}_{kind}.csv" for name in cfg.EXPERIMENTS for kind in ("oof", "test")]
        required += ["radius_calibration.json", "fallback_rule.json", "fine_policy_selection.json",
                     "split_roles.csv", "split_audit.json", "coarse_outer_validation.csv", "metrics.csv",
                     "calibration_coarse.csv", "calibration_fine.csv", "calibration_roi.csv",
                     "calibration_health.json", "area_matched_control.json",
                     "coarse_final/model.pt", "fine_final/model.pt",
                     "coarse_final/stage_complete.json", "fine_final/stage_complete.json"]
        research_verify_cache(folder, record.get("files"), required)
        print(f"Reuse completed outer fold {fold}", flush=True)
        read = lambda name, kind: pd.read_csv(folder / f"{name}_{kind}.csv", dtype={"image_id": str, "group": str})
        return ({n: read(n, "oof") for n in cfg.EXPERIMENTS},
                {n: read(n, "test") for n in cfg.EXPERIMENTS}, record)

    # A failed forced rerun must never leave a stale completion marker that
    # legitimizes a mixture of old and newly written fold outputs.
    if complete.exists():
        complete.unlink()

    outer_train = train.loc[train.fold != fold].reset_index(drop=True)
    validation = train.loc[train.fold == fold].reset_index(drop=True)
    seed = cfg.SEED + int(fold) * 1000
    development, calibration = research_group_holdout(outer_train, cfg.CALIBRATION_FRACTION, seed)
    fitting, stopping = research_group_holdout(development, cfg.EARLY_STOP_FRACTION, seed + 1)
    for a, b in ((development, calibration), (outer_train, validation), (fitting, stopping)):
        assert not set(a.group) & set(b.group)
    assignments = []
    for role, frame in (("model_fit", fitting), ("early_stop", stopping),
                        ("calibration_only", calibration), ("outer_validation", validation)):
        assignments.append(frame[["image_id", "group"]].assign(role=role))
    pd.concat(assignments).to_csv(folder / "split_roles.csv", index=False)
    json_save({"outer_fold": int(fold), "development_ids": development.image_id.tolist(),
               "calibration_ids": calibration.image_id.tolist(),
               "validation_ids": validation.image_id.tolist(),
               "protocol": "Select epochs inside development; refit on development only; hold calibration aside; outer validation used once."},
              folder / "split_audit.json")
    print(f"Outer fold {fold}: development={len(development)} (fit={len(fitting)}, stop={len(stopping)}), "
          f"calibration={len(calibration)}, validation={len(validation)}", flush=True)

    # Select epochs without ever supplying outer validation/calibration labels.
    coarse_select, coarse_info = _research_stage(fitting, stopping, cfg, device, "coarse",
                                                folder / "coarse_select", seed + 10)
    stop_coarse = predict_research(coarse_select, stopping, cfg, device, stage="coarse")
    stop_roi = make_roi_spec(stop_coarse, stopping, cfg, mode="fixed")
    _research_cleanup(coarse_select)
    del coarse_select
    coarse_epochs = int(coarse_info["best_epoch"])

    if cfg.FINE_TRAIN_CROPS == "oof":
        fit_oof_crop = research_crossfit_crops(fitting, cfg, device, folder / "crop_crossfit",
                                              coarse_epochs, seed + 100)
        # Models producing fitting crops never train on early-stop groups.
        # Stopping crops come from the model fitted on fitting groups only.
        # All final fine-training centres are consequently image-held-out,
        # albeit produced using different training-set sizes.
        train_crop = pd.concat([fit_oof_crop, stop_roi], ignore_index=True)
    else:
        train_crop = None
    fit_crop = None if train_crop is None else train_crop.loc[train_crop.image_id.isin(fitting.image_id)]
    policy_candidates = ([False, True] if cfg.SELECT_FINE_CROP_POLICY and fit_crop is not None
                         else [bool(cfg.FINE_MULTISCALE_TRAIN)])
    policy_results = []
    for multiscale in policy_candidates:
        policy_name = "multiscale" if multiscale else "fixed"
        candidate_cfg = replace(cfg, FINE_MULTISCALE_TRAIN=multiscale)
        fine_select, candidate_info = _research_stage(
            fitting, stopping, candidate_cfg, device, "fine",
            folder / ("fine_select_" + policy_name), seed + 20,
            train_crop=fit_crop, val_crop=stop_roi)
        policy_results.append({"policy": policy_name, "multiscale": multiscale,
                               "best_rmse": candidate_info["best_rmse"],
                               "best_epoch": candidate_info["best_epoch"]})
        _research_cleanup(fine_select)
        del fine_select
    # Stable ties prefer the first, fixed-crop candidate. Never inspect outer
    # validation or calibration labels when choosing this training policy.
    fine_info = min(policy_results, key=lambda result: result["best_rmse"])
    fine_cfg = replace(cfg, FINE_MULTISCALE_TRAIN=fine_info["multiscale"])
    json_save({"selection_role": "inner_early_stop_only", "candidates": policy_results,
               "selected": fine_info, "stopping_image_ids": stopping.image_id.tolist(),
               "stopping_group_ids": sorted(stopping.group.astype(str).unique().tolist())},
              folder / "fine_policy_selection.json")
    fine_epochs = int(fine_info["best_epoch"])

    coarse, _ = _research_stage(development, None, cfg, device, "coarse",
                                folder / "coarse_final", seed + 30, epochs=coarse_epochs)
    coarse_cal = predict_research(coarse, calibration, cfg, device, stage="coarse")
    coarse_val = predict_research(coarse, validation, cfg, device, stage="coarse")
    coarse_test = predict_research(coarse, test, cfg, device, stage="coarse")
    coarse_val.to_csv(folder / "coarse_outer_validation.csv", index=False)
    _research_cleanup(coarse)
    del coarse
    radius_calibration = fit_radius_calibration(coarse_cal, calibration, cfg)
    roi_cal = make_roi_spec(coarse_cal, calibration, cfg, mode="calibrated", calibration=radius_calibration)
    matched_fraction, match_info = _research_match_fixed_roi(coarse_cal, calibration, roi_cal, cfg)

    fine, _ = _research_stage(development, None, fine_cfg, device, "fine",
                              folder / "fine_final", seed + 40, train_crop=train_crop, epochs=fine_epochs)
    fine_cal = predict_research(fine, calibration, cfg, device, stage="fine", crop_spec=roi_cal)
    fallback_rule = fit_fallback_rule(coarse_cal, fine_cal, calibration, cfg)
    # Both stages were fitted WITHOUT these calibration images. Parameters fit
    # here are developmental; only outer validation measures their generalization.
    coarse_cal.to_csv(folder / "calibration_coarse.csv", index=False)
    fine_cal.to_csv(folder / "calibration_fine.csv", index=False)
    roi_cal.to_csv(folder / "calibration_roi.csv", index=False)
    diagnostics = research_calibration_diagnostics(coarse_cal, calibration, roi_cal, radius_calibration, cfg)
    json_save(diagnostics, folder / "calibration_health.json")
    for message in diagnostics.get("warnings", []):
        warnings.warn(f"Fold {fold} calibration: {message}")
    json_save(radius_calibration, folder / "radius_calibration.json")
    json_save(fallback_rule, folder / "fallback_rule.json")
    json_save({"fixed_fraction": matched_fraction, **match_info}, folder / "area_matched_control.json")

    oof_parts, test_parts, metrics = {}, {}, []
    for kind, coarse_pred, truth in (("oof", coarse_val, validation), ("test", coarse_test, test)):
        preds = {"B0": coarse_pred}
        rois = {}
        modes = {"B1": "fixed", "B2": "raw", "B3": "calibrated", "C1": "fixed"}
        needed = set(cfg.EXPERIMENTS)
        if "B4" in needed:
            needed.add("B3")
        for name, mode in modes.items():
            if name not in needed:
                continue
            roi = make_roi_spec(coarse_pred, truth, cfg, mode=mode, calibration=radius_calibration,
                                fixed_fraction=matched_fraction if name == "C1" else None)
            pred = predict_research(fine, truth, cfg, device, stage="fine", crop_spec=roi)
            rois[name], preds[name] = roi, pred
            roi.to_csv(folder / f"{name}_{kind}_roi.csv", index=False)
        if "B4" in needed:
            preds["B4"] = apply_fallback(coarse_pred, preds["B3"], fallback_rule)
            rois["B4"] = rois["B3"]
        for name in cfg.EXPERIMENTS:
            pred = preds[name].copy()
            if kind == "oof":
                part = _research_attach_truth(pred, truth, fold, name)
                if name in rois:
                    roi = rois[name].set_index("image_id").loc[part.image_id]
                    part["crop_side"] = roi.crop_side.to_numpy()
                    part["roi_contains_truth"] = ((np.abs(part.x_true.to_numpy() - roi.crop_x.to_numpy()) <= roi.crop_side.to_numpy()/2) &
                                                   (np.abs(part.y_true.to_numpy() - roi.crop_y.to_numpy()) <= roi.crop_side.to_numpy()/2))
                oof_parts[name] = part
                m = localization_metrics(pred, truth, roi=rois.get(name))
                metrics.append({"fold": int(fold), "experiment": name, **m})
            else:
                part = pred.assign(fold=fold, experiment=name)
                test_parts[name] = part
            part.to_csv(folder / f"{name}_{kind}.csv", index=False)
    _research_cleanup(fine)
    del fine
    pd.DataFrame(metrics).to_csv(folder / "metrics.csv", index=False)
    record = {"signature": signature, "outer_fold": int(fold), "coarse_epochs": coarse_epochs,
              "selected_fine_training_policy": fine_info["policy"],
              "fine_epochs": fine_epochs, "n_development": len(development),
              "n_calibration": len(calibration), "n_validation": len(validation), "metrics": metrics}
    files = [p for p in folder.rglob("*") if p.is_file() and p.name not in
             {"complete.json", "complete.partial.json", "uncertainty_oof_diagnostics.json"}]
    record["files"] = research_cache_manifest(folder, files)
    marker = complete.with_suffix(".partial.json")
    json_save(record, marker)
    marker.replace(complete)  # Completion marker is written atomically, last.
    return oof_parts, test_parts, record


def research_summary(oof_by_experiment, train, cfg, output):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    summaries = []
    base = oof_by_experiment.get("B0")
    for name, pred in oof_by_experiment.items():
        metrics = localization_metrics(pred, train)
        metrics.update({"experiment": name, "description": EXPERIMENT_DESCRIPTIONS[name]})
        if "roi_contains_truth" in pred:
            if pred.roi_contains_truth.isna().any() or pred.crop_side.isna().any():
                raise ValueError(f"{name}: missing ROI evaluation values")
            metrics["roi_coverage"] = float(pred.roi_contains_truth.mean())
            metrics["mean_roi_area_px2"] = float(np.mean(pred.crop_side.to_numpy() ** 2))
        summaries.append(metrics)
    summary = pd.DataFrame(summaries)
    summary.to_csv(output / "ablation_summary.csv", index=False)
    reference = "B0" if base is not None else next(iter(oof_by_experiment))
    bootstrap = paired_group_bootstrap(oof_by_experiment, train, baseline=reference,
                                        n_bootstrap=int(cfg.BOOTSTRAP_REPEATS), seed=cfg.SEED)
    bootstrap.to_csv(output / "paired_group_bootstrap.csv", index=False)
    bootstrap.rename(columns={"method": "experiment", "baseline": "reference",
                              "delta_rmse": "rmse_difference",
                              "delta_ci_low": "paired_group_bootstrap_ci_low",
                              "delta_ci_high": "paired_group_bootstrap_ci_high",
                              "n_bootstrap": "bootstrap_repeats"}).to_csv(output / "paired_comparisons.csv", index=False)
    if base is not None:
        diagnostics = uncertainty_diagnostics(base, train, cfg=cfg)
        diagnostics["note"] = "Pooled OOF raw heatmap spread across separately trained folds; no cases excluded from main RMSE. Fold-specific bounded ROI coverage is evaluated separately."
        pd.DataFrame(diagnostics["risk_coverage"]).to_csv(output / "uncertainty_risk_coverage.csv", index=False)
        json_save(diagnostics, output / "uncertainty_diagnostics.json")
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    names = list(oof_by_experiment)
    axes[0].bar(names, [localization_metrics(p, train)["rmse"] for p in oof_by_experiment.values()])
    axes[0].set(ylabel="OOF coordinate RMSE (original pixels)", title="All validation images")
    distances = []
    for pred in oof_by_experiment.values():
        p, t = _research_align(pred, train)
        distances.append(np.linalg.norm(p[["x", "y"]].to_numpy(float) - t[["x", "y"]].to_numpy(float), axis=1))
    axes[1].boxplot(distances, showfliers=True)
    axes[1].set_xticks(range(1, len(names) + 1))
    axes[1].set_xticklabels(names)
    axes[1].set(ylabel="Euclidean localization error (pixels)", title="Error distribution")
    fig.tight_layout()
    fig.savefig(output / "ablation_comparison.png", dpi=160)
    plt.close(fig)
    print(summary.to_string(index=False))
    return summary


def research_uncertainty_evaluation(oof_by_experiment, train, cfg, output):
    """Evaluate each outer fold with ONLY that fold's saved calibration.

    No labels in this function are used to refit a multiplier or choose a rule.
    Counts, rather than unweighted fold averages, form the pooled coverage.
    The curve describes bounded ROI coverage, including min/max clamps and
    margins; it is not a guarantee or a calibrated full probability density.
    """
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    base = oof_by_experiment.get("B0")
    if base is None:
        files = sorted(output.glob("fold_*/coarse_outer_validation.csv"))
        if files:
            frames = []
            for path in files:
                frame = pd.read_csv(path, dtype={"image_id": str, "group": str})
                fold_from_path = int(path.parent.name.removeprefix("fold_"))
                if "fold" in frame and not (frame.fold == fold_from_path).all():
                    raise ValueError(f"Stored coarse prediction fold disagrees with folder: {path}")
                frame["fold"] = fold_from_path
                frames.append(frame)
            base = pd.concat(frames, ignore_index=True)
        else:
            result = {"status": "unavailable", "reason": "Coarse B0 OOF predictions were not retained; fine-model uncertainty must not substitute for coarse ROI uncertainty."}
            json_save(result, output / "uncertainty_calibration_diagnostics.json")
            return result
    base, truth = _research_align(base, train, ("x", "y", "uncertainty", "fold"),
                                  ("x", "y", "width", "height", "group"))
    group_folds = pd.DataFrame({"group": truth.group, "fold": base.fold}).groupby("group").fold.nunique()
    if (group_folds > 1).any():
        raise ValueError("A duplicate/patient group occurs in multiple outer validation folds")
    per_fold_rows, risk_rows, fold_reports = [], [], []
    nominal_levels = (0.50, 0.80, 0.90, 0.95)
    for fold in sorted(base.fold.unique()):
        if float(fold) != int(fold):
            raise ValueError("Outer fold must be an integer")
        folder = output / f"fold_{int(fold)}"
        calibration_path = folder / "radius_calibration.json"
        if not calibration_path.is_file():
            raise FileNotFoundError(f"Missing saved calibration: {calibration_path}")
        calibration = json.loads(calibration_path.read_text(encoding="utf-8"))
        indices = np.flatnonzero(base.fold.to_numpy() == fold)
        pred_fold, truth_fold = base.iloc[indices].reset_index(drop=True), truth.iloc[indices].reset_index(drop=True)
        if set(pred_fold.image_id) & set(map(str, calibration.get("calibration_image_ids", []))):
            raise ValueError("Calibration IDs overlap outer validation IDs")
        if set(truth_fold.group) & set(map(str, calibration.get("calibration_group_ids", []))):
            raise ValueError("Calibration groups overlap outer validation groups")
        report = uncertainty_diagnostics(pred_fold, truth_fold, calibration, cfg)
        report["outer_fold"] = int(fold)
        report["n_groups"] = int(truth_fold.group.nunique())
        report["calibration_n_images"] = int(calibration["n_images"])
        report["calibration_n_groups"] = int(calibration["n_groups"])
        json_save(report, folder / "uncertainty_oof_diagnostics.json")
        fold_reports.append(report)
        for entry in report["risk_coverage"]:
            risk_rows.append({"fold": int(fold), **entry})
        alpha_map = {float(level): float(alpha) for level, alpha in calibration["alpha_by_coverage"].items()}
        for level in nominal_levels:
            if level not in alpha_map:
                raise ValueError(f"Fold {fold}: saved calibration lacks nominal level {level}; do not refit on OOF labels")
            alpha = alpha_map[level]
            roi = make_roi_spec(pred_fold, truth_fold, cfg, mode="calibrated",
                                calibration={**calibration, "alpha": alpha})
            covered = roi_contains_truth(roi, truth_fold).to_numpy(bool)
            group_covered = pd.DataFrame({"group": truth_fold.group.to_numpy(), "covered": covered}).groupby("group").covered.min()
            per_fold_rows.append({
                "fold": int(fold), "nominal_coverage": float(level), "alpha": alpha,
                "n_images": int(len(pred_fold)), "covered_images": int(covered.sum()),
                "n_groups": int(len(group_covered)), "covered_groups": int(group_covered.sum()),
                "actual_image_coverage": float(covered.mean()),
                "actual_group_coverage": float(group_covered.mean()),
                "n_min_clamped": int(roi.roi_at_min.sum()), "n_max_clamped": int(roi.roi_at_max.sum()),
                "n_padded": int(roi.roi_has_padding.sum()),
                "sum_roi_area_px2": float(roi.roi_area_px2.sum()),
                "calibration_n_groups": int(calibration["n_groups"]),
            })
    per_fold = pd.DataFrame(per_fold_rows)
    per_fold.to_csv(output / "uncertainty_calibration_by_fold.csv", index=False)
    pd.DataFrame(risk_rows).to_csv(output / "uncertainty_risk_coverage_by_fold.csv", index=False)
    pooled_rows = []
    for level in nominal_levels:
        rows = per_fold.loc[per_fold.nominal_coverage == level]
        n_images, n_groups = int(rows.n_images.sum()), int(rows.n_groups.sum())
        covered_images, covered_groups = int(rows.covered_images.sum()), int(rows.covered_groups.sum())
        pooled_rows.append({
            "nominal_coverage": float(level), "n_images": n_images, "covered_images": covered_images,
            "n_groups": n_groups, "covered_groups": covered_groups,
            "actual_image_coverage": covered_images / n_images,
            "actual_group_coverage": covered_groups / n_groups,
            "min_clamp_rate": int(rows.n_min_clamped.sum()) / n_images,
            "max_clamp_rate": int(rows.n_max_clamped.sum()) / n_images,
            "padding_rate": int(rows.n_padded.sum()) / n_images,
            "mean_roi_area_px2": float(rows.sum_roi_area_px2.sum() / n_images),
        })
    pooled = pd.DataFrame(pooled_rows)
    pooled.to_csv(output / "uncertainty_calibration_curve.csv", index=False)
    result = {"status": "evaluated", "n_outer_folds": int(len(fold_reports)),
              "n_images": int(len(base)), "n_groups": int(truth.group.nunique()),
              "pooled_bounded_roi_coverage": pooled_rows, "outer_fold_diagnostics": fold_reports,
              "note": "Every OOF image uses its own outer fold's held-out calibration parameters. Counts are pooled across folds. No refitting on OOF labels; margins/clamps are included. Nominal coverage is an empirical target, not a finite-sample guarantee."}
    json_save(result, output / "uncertainty_calibration_diagnostics.json")
    fig, axis = plt.subplots(figsize=(6, 5))
    axis.plot([0, 1], [0, 1], "--", color="grey", label="Nominal target")
    axis.plot(pooled.nominal_coverage, pooled.actual_image_coverage, "o-", label="Held-out images")
    axis.plot(pooled.nominal_coverage, pooled.actual_group_coverage, "s-", label="Held-out groups (all members)")
    axis.set(xlabel="Nominal empirical coverage", ylabel="Actual bounded ROI coverage",
             title="Outer-fold calibration assessment", xlim=(0, 1), ylim=(0, 1.02))
    axis.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(output / "uncertainty_calibration_curve.png", dpi=160)
    plt.close(fig)
    return result


def validate_research_config(cfg):
    validate_config(cfg)
    if cfg.FINE_INPUT_SIZE < 64 or cfg.FINE_INPUT_SIZE % 32 or cfg.FINE_HEATMAP_SIZE < 8:
        raise ValueError("FINE_INPUT_SIZE must be >=64 and divisible by32; FINE_HEATMAP_SIZE >=8.")
    if cfg.COARSE_EPOCHS < 1 or cfg.FINE_EPOCHS < 1:
        raise ValueError("Both training stages require at least one epoch.")
    if not (0 < cfg.ROI_MIN_FRACTION <= cfg.ROI_MAX_FRACTION <= 1 and cfg.ROI_MARGIN_PX >= 0):
        raise ValueError("ROI fractions must satisfy 0 < min <= max <= 1; margin must be nonnegative.")
    if not 0 < cfg.ROI_COVERAGE < 1:
        raise ValueError("ROI_COVERAGE must lie strictly between zero and one.")
    if cfg.NUM_WORKERS < 0:
        raise ValueError("NUM_WORKERS must be nonnegative.")


def research_preflight(config):
    validate_research_config(config)
    print("Geometry:", research_geometry_selfcheck())
    if not config.SMOKE_TEST and config.REQUIRE_GPU and not torch.cuda.is_available():
        raise RuntimeError("请先在Kaggle Notebook设置中启用GPU。CPU流程检查可设SMOKE_TEST=True；正式训练不建议使用CPU。")
    print("Device:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU (smoke test)")
    print("Verified pretrained weights:", resolve_research_weights(config))
    train, test, sample, layout, metadata = prepare_tables(config)
    print(f"Data: {len(train)} training images, {train.pixel_hash.nunique()} unique images, {len(test)} test images, {len(sample)} submission rows")
    print("Default outer folds:", config.N_FOLDS, "| Submission method:", config.SUBMISSION_EXPERIMENT)
    print("All completed stages are checksummed; save the output directory to reuse them in a future session.")
    return {"train_images": len(train), "test_images": len(test), "submission_rows": len(sample)}


def run_research(config):
    cfg = replace(config)
    if cfg.SMOKE_TEST:
        cfg.N_FOLDS = cfg.SMOKE_FOLDS
        cfg.COARSE_EPOCHS = cfg.FINE_EPOCHS = cfg.SMOKE_EPOCHS
        cfg.INPUT_SIZE = cfg.FINE_INPUT_SIZE = cfg.SMOKE_INPUT_SIZE
        cfg.HEATMAP_SIZE = cfg.FINE_HEATMAP_SIZE = cfg.SMOKE_INPUT_SIZE // 4
        cfg.PATIENCE = cfg.SMOKE_EPOCHS
        cfg.NUM_WORKERS, cfg.BATCH_SIZE = 0, 4
        cfg.BOOTSTRAP_REPEATS = 100
        cfg.OUTPUT_DIR = str(Path(cfg.OUTPUT_DIR) / "smoke_test")
        cfg.SUBMISSION_PATH = str(Path(cfg.SUBMISSION_PATH).with_name("submission_SMOKE_TEST.csv"))
        warnings.warn("SMOKE TEST: reduced training, not a competitive result. Formal run requires SMOKE_TEST=False.")
    validate_research_config(cfg)
    if not set(cfg.EXPERIMENTS) <= set(EXPERIMENT_DESCRIPTIONS) or not cfg.EXPERIMENTS:
        raise ValueError("Unknown or empty EXPERIMENTS.")
    if cfg.SUBMISSION_EXPERIMENT not in cfg.EXPERIMENTS:
        raise ValueError("SUBMISSION_EXPERIMENT must be listed in EXPERIMENTS.")
    if cfg.FINE_TRAIN_CROPS not in ("oof", "jitter"):
        raise ValueError("FINE_TRAIN_CROPS must be 'oof' or 'jitter'.")
    if not 0.1 <= cfg.CALIBRATION_FRACTION <= .4 or not .1 <= cfg.EARLY_STOP_FRACTION <= .4:
        raise ValueError("Holdout fractions must be between .1 and .4.")
    if cfg.INNER_CROP_FOLDS < 2 or cfg.BOOTSTRAP_REPEATS < 100:
        raise ValueError("INNER_CROP_FOLDS >= 2 and BOOTSTRAP_REPEATS >= 100 required.")
    seed_everything(cfg.SEED)
    cv2.setNumThreads(0)
    output = Path(cfg.OUTPUT_DIR)
    output.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cpu" and cfg.REQUIRE_GPU and not cfg.SMOKE_TEST:
        raise RuntimeError("Enable a Kaggle GPU accelerator for the full experiment, or explicitly set REQUIRE_GPU=False.")
    cfg.PRETRAINED_WEIGHTS_PATH = resolve_research_weights(cfg)
    train, test, sample, layout, metadata = prepare_tables(cfg)
    if not train.visible.all():
        raise ValueError("Research protocol requires visible fovea point labels. Invisible-class modelling needs a separate validated protocol.")
    train = assign_folds(train, cfg)
    if train.fold.nunique() != cfg.N_FOLDS:
        raise ValueError("Insufficient groups for the requested number of outer folds.")
    train.to_csv(output / "folds.csv", index=False)
    test.to_csv(output / "test_manifest.csv", index=False)
    pd.DataFrame(metadata["duplicate_label_audit"]).to_csv(output / "duplicate_label_audit.csv", index=False)
    pd.concat([train[["image_id", "path"]].assign(split="train"),
               test[["image_id", "path"]].assign(split="test")]).to_csv(output / "filename_mapping.csv", index=False)
    boxes = audit_xml_boxes(train, metadata["root"], output, cfg.COORDINATE_ORIGIN)
    if cfg.SHOW_PLOTS:
        plot_label_audit(train, boxes, output, show=False)
    signature_cfg = asdict(cfg)
    for key in ("REUSE_CHECKPOINTS", "SHOW_PLOTS", "PRETRAINED_WEIGHTS_PATH"):
        signature_cfg.pop(key, None)
    fingerprint = {"version": RESEARCH_VERSION, "source_hash": globals().get("RESEARCH_SOURCE_HASH", RESEARCH_VERSION),
                   "runtime_code": research_runtime_code_digest(), "environment": research_training_environment(),
                   "config": signature_cfg,
                   "train": train[["image_id", "pixel_hash", "x", "y", "group", "fold"]].to_dict("records"),
                   "test": test[["image_id", "pixel_hash"]].to_dict("records")}
    signature = hashlib.sha256(json.dumps(fingerprint, sort_keys=True).encode()).hexdigest()
    json_save({"config": asdict(cfg), "signature": signature, "version": RESEARCH_VERSION,
               "note": 'Calibration groups remain held out from BOTH final networks. Outer labels never select epochs/ROI/fallback. .'},
              output / "run_config.json")
    json_save({"python": sys.version, "torch": str(torch.__version__), "device": str(device),
               "numpy": np.__version__, "pandas": pd.__version__, "opencv": cv2.__version__,
               "platform": platform.platform()}, output / "environment.json")
    oof_all = {n: [] for n in cfg.EXPERIMENTS}
    test_all = {n: [] for n in cfg.EXPERIMENTS}
    folds = []
    for fold in sorted(train.fold.unique()):
        oof, tests, record = run_research_fold(train, test, int(fold), cfg, device, output, signature)
        folds.append(record)
        for name in cfg.EXPERIMENTS:
            oof_all[name].append(oof[name])
            test_all[name].append(tests[name])
    oof_combined = {}
    submissions = {}
    for name in cfg.EXPERIMENTS:
        pred = pd.concat(oof_all[name], ignore_index=True)
        if len(pred) != len(train) or not pred.image_id.is_unique:
            raise AssertionError("OOF must contain each labelled image exactly once.")
        pred = pred.set_index("image_id").loc[train.image_id].reset_index()
        oof_combined[name] = pred
        pred.to_csv(output / f"oof_{name}.csv", index=False)
        all_test = pd.concat(test_all[name], ignore_index=True)
        if not (all_test.groupby("image_id").size() == cfg.N_FOLDS).all():
            raise AssertionError("Each test image requires one prediction from every outer fold.")
        all_test.to_csv(output / f"test_all_folds_{name}.csv", index=False)
        average = all_test.groupby("image_id", as_index=False)[["x", "y"]].mean()
        ensemble = test[["image_id", "width", "height"]].merge(average, on="image_id", validate="one_to_one")
        ensemble["visibility_prob"] = 1.0
        ensemble.to_csv(output / f"test_ensemble_{name}.csv", index=False)
        export_cfg = replace(cfg, SUBMISSION_PATH=str(output / f"submission_{name}.csv"))
        submissions[name] = write_submission(sample, layout, ensemble, metadata, export_cfg, .5)
    summary = research_summary(oof_combined, train, cfg, output)
    research_uncertainty_evaluation(oof_combined, train, cfg, output)
    chosen = submissions[cfg.SUBMISSION_EXPERIMENT]
    Path(cfg.SUBMISSION_PATH).parent.mkdir(parents=True, exist_ok=True)
    chosen.to_csv(cfg.SUBMISSION_PATH, index=False, float_format="%.6f")
    diagnostic = oof_combined[cfg.SUBMISSION_EXPERIMENT].copy()
    diagnostic["visibility_prob"], diagnostic["visible_true"] = 1.0, True
    plot_oof_errors(diagnostic, train, .5, cfg, output)
    json_save({"version": RESEARCH_VERSION, "smoke_test": cfg.SMOKE_TEST,
               "n_folds": cfg.N_FOLDS, "n_train": len(train), "n_test": len(test),
               "submission_rows": len(chosen), "submission_experiment": cfg.SUBMISSION_EXPERIMENT,
               "submission_path": cfg.SUBMISSION_PATH, "outer_folds": folds,
               "uncertainty": "Empirical error-radius calibration, not a theoretical conformal coverage guarantee.",
               "selection": "Submission method preselected in configuration, not auto-selected from outer OOF."},
              output / "run_summary.json")
    print(f"Complete: {cfg.SUBMISSION_PATH}")
    return {"config": cfg, "train": train, "test": test, "oof": oof_combined,
            "summary": summary, "submission": chosen, "submissions": submissions}


RESEARCH_SOURCE_HASH = "ff7cda2ef1fa2bb551e7f72eee3dc24d254db255601947b2ae497fc170cf4563"
