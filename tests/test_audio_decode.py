"""Tests for audio decoding robustness, format signature checks, ffmpeg fallback, and cache consistency.

Follows user specifications:
- (a) MP3 with ID3v2 tag + APIC cover art
- (b) AAC/M4A file renamed .mp3 (signature rejection with REJECT_NON_MP3=True)
- (c) Corrupt file upload (HTTP 400 with sanitized error message, no path leaks)
- (d) Cache consistency regression test across 3 FMA tracks (000002, 000005, 000010) against float16 logmels
- (e) load_audio does NOT pad short audio; padding happens in segment_audio
- (f) Missing ffmpeg on fallback path raises 500-class error, never 400
- (g) Guaranteed temporary file cleanup
"""

from __future__ import annotations

import io
from pathlib import Path
import shutil
import subprocess
import tempfile
from unittest.mock import patch

from fastapi.testclient import TestClient
import numpy as np
import pytest
import soundfile as sf
import torch

from backend.app.main import app
from src.audio.loading import AudioDecodeError, get_audio_duration, load_audio, segment_audio
from src.audio.spectrogram import compute_logmel_spectrogram, normalize_spectrogram
from src.audio.validation import (
    REJECT_NON_MP3,
    check_known_non_mp3_signatures,
    get_container_format,
    verify_audio_file,
)
from src.config import resolve_path

client = TestClient(app)


def _create_synthetic_mp3(duration_sec: float = 3.0, sr: int = 22050) -> bytes:
    """Create a minimal valid MP3 in memory using ffmpeg."""
    ffmpeg_bin = shutil.which("ffmpeg")
    if not ffmpeg_bin:
        pytest.skip("ffmpeg not available on test system")

    cmd = [
        ffmpeg_bin,
        "-y",
        "-f", "lavfi",
        "-i", f"sine=frequency=440:duration={duration_sec}:sample_rate={sr}",
        "-c:a", "libmp3lame",
        "-b:a", "128k",
        "-f", "mp3",
        "-",
    ]
    res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=False, timeout=15)
    return res.stdout


def _create_mp3_with_cover_art(duration_sec: float = 2.0, sr: int = 22050) -> Path:
    """Create a temporary MP3 with ID3v2 metadata and embedded APIC cover art."""
    ffmpeg_bin = shutil.which("ffmpeg")
    if not ffmpeg_bin:
        pytest.skip("ffmpeg not available on test system")

    temp_dir = Path(tempfile.gettempdir())
    out_path = temp_dir / f"test_cover_{tempfile.mktemp().split('tmp')[-1]}.mp3"

    cmd = [
        ffmpeg_bin,
        "-y",
        "-f", "lavfi",
        "-i", f"sine=frequency=523:duration={duration_sec}:sample_rate={sr}",
        "-f", "lavfi",
        "-i", f"color=c=navy:s=120x120:d={duration_sec}",
        "-c:a", "libmp3lame",
        "-c:v", "mjpeg",
        "-id3v2_version", "3",
        "-metadata:s:v", "title=CoverArt",
        "-t", str(duration_sec),
        str(out_path),
    ]
    res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=False, timeout=20)
    assert res.returncode == 0, f"ffmpeg failed to create cover art MP3: {res.stderr.decode()}"
    return out_path


# ==============================================================================
# (a) MP3 with ID3v2 tag + APIC cover art
# ==============================================================================
def test_mp3_with_cover_art_decodes_cleanly():
    """Verify an MP3 with embedded cover art (APIC frame) decodes correctly without failing."""
    mp3_path = _create_mp3_with_cover_art(duration_sec=2.0)
    try:
        assert mp3_path.exists()
        y = load_audio(mp3_path, sr=22050, duration=2.0, mono=True)
        assert isinstance(y, np.ndarray)
        assert y.dtype == np.float32
        assert len(y) == 22050 * 2
        assert not np.all(y == 0.0)
    finally:
        if mp3_path.exists():
            mp3_path.unlink()


# ==============================================================================
# (b) AAC / M4A file renamed to .mp3 (signature rejection & format policy)
# ==============================================================================
def test_aac_m4a_signature_rejection():
    """Verify files with 'ftyp' at offset 4 (M4A/AAC container) are rejected when REJECT_NON_MP3 is True."""
    # Synthetic M4A header with 'ftyp' at offset 4
    fake_m4a_header = b"\x00\x00\x00\x18ftypM4A \x00\x00\x00\x00isommp42" + b"\x00" * 100

    temp_dir = Path(tempfile.gettempdir())
    fake_path = temp_dir / "fake_m4a_renamed.mp3"
    fake_path.write_bytes(fake_m4a_header)

    try:
        # 1. Direct signature check with default policy (REJECT_NON_MP3 = True)
        valid, err = check_known_non_mp3_signatures(fake_path, reject_non_mp3=True)
        assert not valid
        assert "Non-MP3 format detected" in err
        assert "ftyp" in err

        # 2. Signature check with policy disabled
        valid_allow, _ = check_known_non_mp3_signatures(fake_path, reject_non_mp3=False)
        assert valid_allow

        # 3. verify_audio_file rejects it
        valid_v, reason_v = verify_audio_file(fake_path, reject_non_mp3=True)
        assert not valid_v
        assert "Non-MP3 format detected" in reason_v

        # 4. POST /api/analyze rejects it with HTTP 400
        res = client.post(
            "/api/analyze",
            files={"file": ("fake_m4a.mp3", fake_m4a_header, "audio/mpeg")},
        )
        assert res.status_code == 400
        assert "audio could not be decoded" in res.json()["detail"].lower()
    finally:
        if fake_path.exists():
            fake_path.unlink()


# ==============================================================================
# (c) Corrupt file upload (sanitized HTTP 400, no path or username leak)
# ==============================================================================
def test_corrupt_file_sanitized_error():
    """Verify corrupt file upload fails validation with HTTP 400 and does NOT leak file paths or usernames."""
    corrupt_content = b"\xff\xfb\x90\x64" + b"\xde\xad\xbe\xef\xca\xfe\xba\xbe" * 40

    res = client.post(
        "/api/analyze",
        files={"file": ("corrupt_track.mp3", corrupt_content, "audio/mpeg")},
    )
    assert res.status_code == 400
    detail = res.json()["detail"]
    assert "audio could not be decoded" in detail.lower()

    # Crucial security check: must NOT leak absolute paths, tmp directory, or user home paths
    assert "c:" not in detail.lower()
    assert "\\" not in detail
    assert "temp" not in detail.lower()
    assert "users" not in detail.lower()
    assert "sonicscope_" not in detail.lower()


# ==============================================================================
# (d) Cache consistency regression test (FMA tracks 000002, 000005, 000010)
# ==============================================================================
def test_cache_consistency_regression():
    """Verify recomputing log-mels via load_audio matches existing cached .npy arrays within float16 tolerance.

    Does NOT rewrite or invalidate any cache files.
    """
    ckpt_path = resolve_path("models/checkpoints/genre_cnn.pt")
    if not ckpt_path.exists():
        pytest.skip("Checkpoint not found; cannot verify normalization stats")

    checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    stats = checkpoint.get("normalization_stats", {"mean": -43.0, "std": 15.0})
    mean = float(stats["mean"])
    std = float(stats["std"])

    tracks_to_check = ["000002", "000005", "000010"]
    for tid in tracks_to_check:
        mp3_path = resolve_path(f"data/raw/fma_small/000/{tid}.mp3")
        if not mp3_path.exists():
            pytest.skip(f"FMA track {mp3_path} not found")

        # Recompute via unified load_audio + segmentation pipeline
        y = load_audio(mp3_path, sr=22050, duration=30.0, mono=True)
        segments = segment_audio(y, sr=22050, window_sec=10.0, num_windows=3)

        for w_idx, seg in enumerate(segments):
            cache_file = resolve_path(f"data/cache/logmels/{tid}_{w_idx}.npy")
            if not cache_file.exists():
                pytest.skip(f"Cache file {cache_file} does not exist")

            cached_arr = np.load(cache_file)
            raw_logmel = compute_logmel_spectrogram(
                segment=seg,
                sr=22050,
                n_fft=2048,
                hop_length=512,
                n_mels=128,
            )
            recomputed_arr = normalize_spectrogram(raw_logmel, mean=mean, std=std).astype(np.float16)

            # Compare using float16-appropriate tolerance
            assert cached_arr.shape == recomputed_arr.shape
            assert np.allclose(cached_arr, recomputed_arr, rtol=1e-2, atol=1e-2), (
                f"Cache mismatch for track {tid} window {w_idx}: max diff = {np.max(np.abs(cached_arr - recomputed_arr))}"
            )


# ==============================================================================
# (e) load_audio does NOT pad short audio (padding happens in segment_audio)
# ==============================================================================
def test_load_audio_does_not_pad_short_audio():
    """Verify load_audio returns at most duration seconds and does not pad short audio with zeros."""
    sr = 22050
    short_seconds = 5.0
    synthetic_mp3_bytes = _create_synthetic_mp3(duration_sec=short_seconds, sr=sr)

    temp_dir = Path(tempfile.gettempdir())
    temp_file = temp_dir / "short_test_track.mp3"
    temp_file.write_bytes(synthetic_mp3_bytes)

    try:
        # Load requesting 30.0s max duration
        loaded = load_audio(temp_file, sr=sr, duration=30.0, mono=True)

        # Must NOT pad to 30.0s (661,500 samples)
        assert len(loaded) < sr * 30
        assert abs(len(loaded) - int(sr * short_seconds)) < sr * 0.5  # within encoder margin

        # Padding occurs when segment_audio is invoked per PLAN §07
        segments = segment_audio(loaded, sr=sr, window_sec=10.0, num_windows=3)
        assert len(segments) == 3
        for seg in segments:
            assert len(seg) == int(sr * 10.0)
    finally:
        if temp_file.exists():
            temp_file.unlink()


# ==============================================================================
# (f) Missing ffmpeg on fallback path raises 500-class error, never 400
# ==============================================================================
def test_missing_ffmpeg_raises_500():
    """Verify that if ffmpeg is missing when fallback decode is required, a 500-class error is raised."""
    temp_dir = Path(tempfile.gettempdir())
    dummy_file = temp_dir / "unsupported_codec.mp3"
    dummy_file.write_bytes(b"\x00\x00\x00\x00" * 50)

    try:
        # Mock shutil.which to simulate ffmpeg missing on the host system
        with patch("shutil.which", return_value=None):
            # load_audio directly raises RuntimeError
            with pytest.raises(RuntimeError) as exc_info:
                load_audio(dummy_file, sr=22050, duration=5.0)
            assert "ffmpeg" in str(exc_info.value).lower()

            # POST /api/analyze converts this to HTTP 500, NOT HTTP 400
            res = client.post(
                "/api/analyze",
                files={"file": ("unsupported.mp3", b"\x00\x00\x00\x00" * 50, "audio/mpeg")},
            )
            assert res.status_code == 500
            assert "server audio processing error" in res.json()["detail"].lower()
    finally:
        if dummy_file.exists():
            dummy_file.unlink()


# ==============================================================================
# (g) Guaranteed temporary upload file cleanup
# ==============================================================================
def test_temporary_upload_file_cleanup():
    """Verify temporary files are cleanly unlinked after request completion or failure."""
    temp_dir = Path(tempfile.gettempdir())
    existing_temps = set(temp_dir.glob("sonicscope_*.tmp"))

    # Test failure case (bad bytes)
    res_fail = client.post(
        "/api/analyze",
        files={"file": ("bad.mp3", b"not-real-audio-data", "audio/mpeg")},
    )
    assert res_fail.status_code == 400
    after_fail_temps = set(temp_dir.glob("sonicscope_*.tmp"))
    assert after_fail_temps == existing_temps, "Temporary file was leaked on failure!"

    # Test success case
    mp3_bytes = _create_synthetic_mp3(duration_sec=3.0)
    res_ok = client.post(
        "/api/analyze",
        files={"file": ("ok.mp3", mp3_bytes, "audio/mpeg")},
    )
    assert res_ok.status_code == 200
    after_ok_temps = set(temp_dir.glob("sonicscope_*.tmp"))
    assert after_ok_temps == existing_temps, "Temporary file was leaked on success!"
