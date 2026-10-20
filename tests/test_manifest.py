"""Unit tests for dataset manifest generation and audio verification.

Verifies requirements from docs/PLAN.md (§05, §06, Addenda A4):
- No track_id appears in two splits.
- Exactly 8 genres defined in alphabetical label map.
- All manifest audio file paths exist on disk.
- Label map is strictly alphabetical (0..7).
- Verification detects missing and corrupt files.
"""

import json
from pathlib import Path
import pandas as pd
import pytest

from src.audio.validation import verify_audio_file
from src.config import load_config, resolve_path, get_project_root


@pytest.fixture
def cfg():
    """Project configuration fixture."""
    return load_config()


@pytest.fixture
def manifest_df(cfg):
    """Loaded manifest DataFrame fixture."""
    manifest_path = resolve_path(cfg.paths.manifest_path)
    assert manifest_path.exists(), f"Manifest file does not exist at {manifest_path}. Run scripts/01_build_manifest.py first."
    return pd.read_csv(manifest_path)


@pytest.fixture
def label_map(cfg):
    """Loaded label map fixture."""
    label_map_path = resolve_path(cfg.paths.label_map_path)
    assert label_map_path.exists(), f"Label map does not exist at {label_map_path}. Run scripts/01_build_manifest.py first."
    with open(label_map_path, "r", encoding="utf-8") as f:
        return json.load(f)


def test_label_map_alphabetical_and_8_genres(label_map):
    """Verify label map has 8 genres, values 0..7, and is sorted strictly alphabetically."""
    assert len(label_map) == 8, f"Expected 8 genres in label_map, found {len(label_map)}"
    genres = list(label_map.keys())
    assert genres == sorted(genres), "Label map keys are not sorted alphabetically"
    indices = list(label_map.values())
    assert indices == list(range(8)), "Label map values must be contiguous integers 0..7"


def test_no_track_id_in_two_splits(manifest_df):
    """Verify that every track_id is unique and appears in exactly one split (PLAN §06)."""
    assert not manifest_df.empty, "Manifest is empty"
    assert manifest_df["track_id"].is_unique, "Duplicate track_id found! A track must never appear in multiple splits"
    grouped = manifest_df.groupby("track_id")["split"].nunique()
    assert (grouped == 1).all(), "Found track_ids assigned to more than one split"


def test_manifest_columns(manifest_df):
    """Verify required column schema in manifest.csv (PLAN §06)."""
    expected_cols = ["track_id", "path", "genre", "split", "artist_id"]
    assert list(manifest_df.columns) == expected_cols


def test_manifest_splits_valid(manifest_df):
    """Verify all split assignments match the official FMA values."""
    valid_splits = {"training", "validation", "test"}
    unique_splits = set(manifest_df["split"].unique())
    assert unique_splits.issubset(valid_splits), f"Unexpected splits found: {unique_splits - valid_splits}"


def test_manifest_paths_exist(manifest_df):
    """Verify that every audio file listed in the manifest exists on disk."""
    for path_str in manifest_df["path"]:
        full_path = resolve_path(path_str)
        assert full_path.exists(), f"Audio file does not exist: {full_path}"
        assert full_path.is_file(), f"Path is not a regular file: {full_path}"


def test_manifest_genres_match_label_map(manifest_df, label_map):
    """Verify every genre in the manifest exists in the official label map."""
    manifest_genres = set(manifest_df["genre"].unique())
    label_map_genres = set(label_map.keys())
    assert manifest_genres.issubset(label_map_genres), (
        f"Manifest contains genres not in label_map: {manifest_genres - label_map_genres}"
    )


def test_exclusions_file_format(cfg):
    """Verify data_exclusions.csv format and headers."""
    exclusions_path = resolve_path(cfg.paths.metrics_dir) / "data_exclusions.csv"
    assert exclusions_path.exists(), "data_exclusions.csv does not exist"
    df = pd.read_csv(exclusions_path)
    # Accepts new audit schema with reasons or legacy schema
    expected_new = {"track_id", "genre", "split", "reason"}
    assert list(df.columns) == ["track_id", "path", "reason"] or expected_new.issubset(df.columns)


def test_audit_file_format(cfg):
    """Verify manifest_audit.json exists and contains required audit fields."""
    audit_path = resolve_path(cfg.paths.metrics_dir) / "manifest_audit.json"
    assert audit_path.exists(), "manifest_audit.json does not exist"
    with open(audit_path, "r", encoding="utf-8") as f:
        audit = json.load(f)

    assert "splits" in audit
    assert "genres" in audit
    assert "processed_tracks" in audit
    assert "valid_tracks" in audit
    assert "excluded_tracks" in audit
    assert "artist_overlap_limitation" in audit


def test_audio_verification_detects_bad_files(tmp_path):
    """Verify verify_audio_file properly identifies missing, empty, and valid files."""
    # 1. Non-existent file
    missing_file = tmp_path / "missing.mp3"
    valid, reason = verify_audio_file(missing_file)
    assert not valid
    assert "not found" in reason.lower()

    # 2. 0-byte file
    empty_file = tmp_path / "empty.mp3"
    empty_file.touch()
    valid, reason = verify_audio_file(empty_file)
    assert not valid
    assert "0 bytes" in reason.lower()

    # 3. Known valid track in FMA small
    sample_mp3 = resolve_path("data/raw/fma_small/000/000002.mp3")
    if sample_mp3.exists():
        valid, reason = verify_audio_file(sample_mp3)
        assert valid
        assert reason is None


def test_strict_decode_check_rejects_truncated_and_corrupt():
    """Verify verify_audio_file_strict rejects stubs and truncated audio without lenient ffmpeg salvage."""
    from src.audio.validation import verify_audio_file_strict

    # 1. Corrupt stub file (track 99134: 1.2 KB, missing headers)
    stub_path = resolve_path("data/raw/fma_small/099/099134.mp3")
    if stub_path.exists():
        is_valid, dur, size, reason = verify_audio_file_strict(stub_path, min_duration=25.0)
        assert not is_valid
        assert dur == 0.0
        assert size < 2000
        assert "corrupt" in reason.lower() or "libsndfile" in reason.lower()

    # 2. Truncated audio (track 98565: 1.63s of decodable audio)
    trunc_path = resolve_path("data/raw/fma_small/098/098565.mp3")
    if trunc_path.exists():
        is_valid, dur, size, reason = verify_audio_file_strict(trunc_path, min_duration=25.0)
        assert not is_valid
        assert 0.0 < dur < 2.0
        assert "min_track_seconds" in reason

    # 3. Valid track (track 2: ~30s of valid audio)
    valid_path = resolve_path("data/raw/fma_small/000/000002.mp3")
    if valid_path.exists():
        is_valid, dur, size, reason = verify_audio_file_strict(valid_path, min_duration=25.0)
        assert is_valid
        assert dur >= 25.0
        assert reason is None


def test_manifest_audit_only_preserves_production_and_lists_exclusions(tmp_path):
    """Verify --audit-only writes data_exclusions.csv and manifest_audit.json without modifying manifest.csv."""
    import importlib
    build_module = importlib.import_module("scripts.01_build_manifest")
    build_manifest = build_module.build_manifest

    # Snapshot mtime and hash of real manifest.csv and label_map.json
    prod_manifest = resolve_path("data/processed/manifest.csv")
    prod_label_map = resolve_path("data/processed/label_map.json")
    manifest_bytes_before = prod_manifest.read_bytes()
    label_map_bytes_before = prod_label_map.read_bytes()

    temp_metrics = tmp_path / "metrics"
    audit_report = build_manifest(
        audit_only=True,
        metrics_dir_override=temp_metrics,
    )

    # 1. Production files completely untouched
    assert prod_manifest.read_bytes() == manifest_bytes_before
    assert prod_label_map.read_bytes() == label_map_bytes_before

    # 2. Audit report contents
    assert audit_report["total_small_tracks_in_manifest"] == 7994
    assert audit_report["total_exclusions"] == 6
    assert audit_report["splits_metadata"] == {"training": 6400, "validation": 800, "test": 800}
    assert audit_report["splits_manifest"] == {"training": 6394, "validation": 800, "test": 800}
    assert audit_report["splits_exclusions"] == {"training": 6, "validation": 0, "test": 0}
    assert "artist_overlap_limitation" in audit_report

    # 3. data_exclusions.csv generated in temp metrics with all 6 tracks
    exclusions_csv = temp_metrics / "data_exclusions.csv"
    assert exclusions_csv.exists()
    ex_df = pd.read_csv(exclusions_csv)
    assert len(ex_df) == 6
    expected_tids = {98565, 98567, 98569, 99134, 108925, 133297}
    assert set(ex_df["track_id"]) == expected_tids
    assert (ex_df["split"] == "training").all()
    # Ensure reason is non-empty for every row
    assert ex_df["reason"].notna().all()
    assert (ex_df["reason"].str.len() > 0).all()

    # 4. Strict path sanitization check: no absolute paths, drive letters, Users or backslashes
    for reason in ex_df["reason"]:
        assert "C:\\" not in reason, f"Found drive path in reason: {reason}"
        assert "Users" not in reason, f"Found Users directory in reason: {reason}"
        assert "\\" not in reason, f"Found backslash in reason: {reason}"

    # 5. Verify the 3 tiny stub MP3s are cleanly described
    stub_tids = [99134, 108925, 133297]
    for tid in stub_tids:
        row = ex_df[ex_df["track_id"] == tid].iloc[0]
        assert row["reason"].startswith("Unreadable/corrupt MP3 (file size"), (
            f"Expected stub MP3 reason for track {tid}, got: {row['reason']}"
        )
        assert "no decodable audio" in row["reason"]

    # 6. Verify artist overlap note matches measured overlap (0 overlap)
    artist_lim = audit_report["artist_overlap_limitation"]
    assert artist_lim["artists_in_multiple_splits_count"] == 0
    assert "No artist_id appears in more than one split" in artist_lim["note"]


def test_manifest_rebuild_in_temp_dir_reproduces_exact_manifest(tmp_path):
    """Verify full rebuild into a temp dir reproduces exactly the 7,994 track IDs, genres, and splits."""
    import importlib
    build_module = importlib.import_module("scripts.01_build_manifest")
    build_manifest = build_module.build_manifest

    temp_out = tmp_path / "processed"
    temp_metrics = tmp_path / "metrics"

    # Full rebuild pointing --out-dir to temp_out
    audit = build_manifest(
        out_dir=temp_out,
        audit_only=False,
        metrics_dir_override=temp_metrics,
        workers=8,
    )

    # Load rebuilt manifest
    rebuilt_manifest_path = temp_out / "manifest.csv"
    assert rebuilt_manifest_path.exists()
    rebuilt_df = pd.read_csv(rebuilt_manifest_path)
    assert len(rebuilt_df) == 7994

    # Load frozen production manifest
    prod_manifest_path = resolve_path("data/processed/manifest.csv")
    prod_df = pd.read_csv(prod_manifest_path)
    assert len(prod_df) == 7994

    # Compare track_id, genre, and split
    rebuilt_sorted = rebuilt_df.sort_values("track_id").reset_index(drop=True)
    prod_sorted = prod_df.sort_values("track_id").reset_index(drop=True)

    pd.testing.assert_series_equal(rebuilt_sorted["track_id"], prod_sorted["track_id"])
    pd.testing.assert_series_equal(rebuilt_sorted["genre"], prod_sorted["genre"])
    pd.testing.assert_series_equal(rebuilt_sorted["split"], prod_sorted["split"])

    # Exclusions in rebuilt output
    excl_path = temp_metrics / "data_exclusions.csv"
    assert excl_path.exists()
    excl_df = pd.read_csv(excl_path)
    assert set(excl_df["track_id"]) == {98565, 98567, 98569, 99134, 108925, 133297}

