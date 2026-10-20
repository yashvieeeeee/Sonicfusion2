"""Unit tests for configuration loading and validation.

Verifies requirements from docs/PLAN.md (§05, §07, §08, §09, Addenda A3-A4).
"""

from pathlib import Path
import pytest
from src.config import (
    load_config,
    get_project_root,
    resolve_path,
    compute_spectrogram_frames,
    get_expected_spectrogram_frames,
    resolve_device,
    ConfigDict,
)


@pytest.fixture
def config():
    """Fixture providing loaded project configuration."""
    return load_config()


def test_load_config_default(config):
    """Test loading configuration from default path."""
    assert config is not None
    assert isinstance(config, (dict, ConfigDict))


def test_seed_is_42(config):
    """Test that global random seed is 42."""
    assert config.seed == 42
    assert config["seed"] == 42


def test_device_setting(config):
    """Test device setting is 'auto' and resolves to a valid hardware device."""
    assert config.device == "auto"
    resolved = resolve_device(config)
    assert resolved in ("cpu", "cuda")


def test_paths_relative_to_root(config):
    """Test that all configured paths are relative and resolve properly."""
    root = get_project_root()
    assert (root / "configs" / "config.yaml").exists()

    paths = config.paths
    assert paths.data_root == "data"
    assert "raw_dir" in paths
    assert "processed_dir" in paths
    assert "cache_dir" in paths
    assert "manifest_path" in paths
    assert "features_path" in paths
    assert "label_map_path" in paths
    assert "cnn_checkpoint_path" in paths
    assert "baseline_model_path" in paths

    for key, rel_path in paths.items():
        assert isinstance(rel_path, str)
        # Paths must be relative, not hardcoded absolute drives or root slashes
        p = Path(rel_path)
        assert not p.is_absolute(), f"Path {key}='{rel_path}' must be relative to project root"
        resolved = resolve_path(rel_path)
        assert str(resolved).startswith(str(root)), f"Resolved path for {key} must reside under project root"


def test_audio_settings_section_07(config):
    """Verify deterministic audio processing contract from PLAN §07."""
    audio = config.audio
    assert audio.sample_rate == 22050
    assert audio.duration == 30
    assert audio.window_duration == 10
    assert audio.windows_per_track == 3
    assert audio.n_mels == 128
    assert audio.n_fft == 2048
    assert audio.hop_length == 512
    assert audio.mono is True


def test_expected_spectrogram_frames(config):
    """Verify spectrogram frame count is computed from config rather than hardcoded.

    Addendum A4: ~431 frames for 10 s @ 22,050 Hz, hop 512.
    """
    # 10 s * 22050 = 220500 samples; 220500 // 512 = 430; 430 + 1 = 431 frames
    computed = compute_spectrogram_frames(
        duration=config.audio.window_duration,
        sample_rate=config.audio.sample_rate,
        hop_length=config.audio.hop_length,
    )
    assert computed == 431
    assert get_expected_spectrogram_frames(config) == 431


def test_baseline_grid_section_08(config):
    """Verify classical baseline parameter grid from PLAN §08."""
    baseline = config.baseline
    assert baseline.random_state == 42
    assert baseline.param_grid.n_estimators == [200, 400]
    assert baseline.param_grid.max_depth == [12, None]


def test_cnn_settings_section_09(config):
    """Verify 2D CNN architecture and training settings from PLAN §09."""
    cnn = config.cnn
    arch = cnn.architecture
    assert arch.in_channels == 1
    assert arch.dropout == 0.3

    training = cnn.training
    assert training.optimizer == "AdamW"
    assert training.learning_rate == 0.001
    assert training.batch_size == 32
    assert training.epochs == 25
    assert training.early_stopping_patience == 5


def test_no_hardcoded_genre_list(config):
    """Verify no hardcoded genre list exists in configuration (PLAN Addendum A4)."""
    assert "genres" not in config
    assert "genre_list" not in config
    assert "classes" not in config
    assert "classes" not in config.cnn.architecture


def test_load_config_custom_path():
    """Verify load_config works with explicit custom path as str and Path."""
    root = get_project_root()
    cfg_path = root / "configs" / "config.yaml"
    cfg1 = load_config(cfg_path)
    cfg2 = load_config(str(cfg_path))
    assert cfg1.seed == cfg2.seed == 42


def test_config_dict_attribute_and_item_access(config):
    """Verify ConfigDict supports both item access and attribute access."""
    assert config["seed"] == config.seed == 42
    assert config["audio"]["sample_rate"] == config.audio.sample_rate == 22050
    assert config["cnn"]["training"]["batch_size"] == config.cnn.training.batch_size == 32
