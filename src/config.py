"""Configuration loader and utilities for SonicScope AI.

Follows specifications from docs/PLAN.md (§05, §07, §08, §09, Addenda A3-A4).
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Mapping
import yaml
import torch


class ConfigDict(dict):
    """Dictionary supporting dot-notation attribute access alongside dict indexing."""

    def __getattr__(self, name: str) -> Any:
        try:
            return self[name]
        except KeyError:
            raise AttributeError(f"'ConfigDict' object has no attribute '{name}'")

    def __setattr__(self, name: str, value: Any) -> None:
        self[name] = _wrap_config(value)

    def __delattr__(self, name: str) -> None:
        try:
            del self[name]
        except KeyError:
            raise AttributeError(f"'ConfigDict' object has no attribute '{name}'")


def _wrap_config(item: Any) -> Any:
    """Recursively wrap mappings in ConfigDict."""
    if isinstance(item, Mapping):
        return ConfigDict({k: _wrap_config(v) for k, v in item.items()})
    elif isinstance(item, list):
        return [_wrap_config(x) for x in item]
    return item


def get_project_root() -> Path:
    """Return the absolute path to the project root directory."""
    return Path(__file__).resolve().parent.parent


def resolve_path(relative_path: str | Path, project_root: Path | None = None) -> Path:
    """Resolve a path relative to the project root directory."""
    root = project_root if project_root is not None else get_project_root()
    p = Path(relative_path)
    return p if p.is_absolute() else (root / p).resolve()


def compute_spectrogram_frames(duration: float, sample_rate: int, hop_length: int) -> int:
    """Compute deterministic spectrogram frame count for centered STFT windowing.

    Formula: 1 + int(duration * sample_rate) // hop_length
    For 10.0 s @ 22050 Hz and hop 512 -> 431 frames.
    """
    n_samples = int(duration * sample_rate)
    return 1 + (n_samples // hop_length)


def get_expected_spectrogram_frames(cfg: Mapping[str, Any] | None = None) -> int:
    """Compute expected spectrogram frames dynamically from configuration."""
    if cfg is None:
        cfg = load_config()
    audio_cfg = cfg.get("audio", cfg)
    duration = float(audio_cfg["window_duration"])
    sample_rate = int(audio_cfg["sample_rate"])
    hop_length = int(audio_cfg["hop_length"])
    return compute_spectrogram_frames(duration, sample_rate, hop_length)


def resolve_device(device_setting: Any = "auto") -> str:
    """Resolve device setting ('auto', 'cuda', 'cpu') to a concrete device name."""
    if isinstance(device_setting, Mapping):
        device_setting = device_setting.get("device", "auto")

    dev_str = str(device_setting).lower().strip()
    if dev_str == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    elif "cuda" in dev_str and not torch.cuda.is_available():
        return "cpu"
    return dev_str


def load_config(config_path: str | Path | None = None) -> ConfigDict:
    """Load configuration YAML file into a ConfigDict with attribute and dict access.

    If config_path is not specified, defaults to configs/config.yaml under project root.
    """
    root = get_project_root()
    if config_path is None:
        target_path = root / "configs" / "config.yaml"
    else:
        target_path = Path(config_path)
        if not target_path.is_absolute():
            target_path = root / target_path

    if not target_path.exists():
        raise FileNotFoundError(f"Configuration file not found at: {target_path}")

    with open(target_path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}

    wrapped: ConfigDict = _wrap_config(data)
    return wrapped
