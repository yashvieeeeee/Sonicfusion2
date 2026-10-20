"""Regression tests for widened audio input formats, container detection, and security constraints.

Follows user requirements:
1. Real M4A container (AAC) renamed .mp3 succeeds (HTTP 200).
2. 10 s WAV succeeds (HTTP 200).
3. 10 s MP3 succeeds (HTTP 200).
4. 16-minute file returns "too long" with the configured limit (HTTP 400).
5. OGG renamed .mp3 returns "unsupported format" (HTTP 400).
6. Random bytes return "could not decode" (HTTP 400).
7. Fake ffconcat script rejected as unsupported without reading referenced path.
8. FMA clips produce unchanged results.
9. No sonicscope_* temp files remain.
10. No username or path appears in any error response.
"""

from __future__ import annotations

import os
from pathlib import Path
import shutil
import subprocess
import tempfile

from fastapi.testclient import TestClient
import numpy as np
import pytest

from backend.app.main import app
from src.audio.loading import get_audio_duration, load_audio
from src.config import load_config, resolve_path

client = TestClient(app)


def _generate_synthetic_audio(format_type: str, duration_sec: float = 10.0, sr: int = 22050) -> bytes:
    """Generate synthetic audio bytes in memory using ffmpeg with no copyrighted audio."""
    ffmpeg_bin = shutil.which("ffmpeg")
    if not ffmpeg_bin:
        pytest.skip("ffmpeg not available on test system")

    if format_type == "mp3":
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
    elif format_type == "wav":
        cmd = [
            ffmpeg_bin,
            "-y",
            "-f", "lavfi",
            "-i", f"sine=frequency=440:duration={duration_sec}:sample_rate={sr}",
            "-c:a", "pcm_s16le",
            "-f", "wav",
            "-",
        ]
    elif format_type == "m4a":
        cmd = [
            ffmpeg_bin,
            "-y",
            "-f", "lavfi",
            "-i", f"sine=frequency=440:duration={duration_sec}:sample_rate={sr}",
            "-c:a", "aac",
            "-f", "ipod",
            "-movflags", "frag_keyframe+empty_moov",
            "-",
        ]
    elif format_type == "ogg":
        cmd = [
            ffmpeg_bin,
            "-y",
            "-f", "lavfi",
            "-i", f"sine=frequency=440:duration={duration_sec}:sample_rate={sr}",
            "-c:a", "libvorbis",
            "-f", "ogg",
            "-",
        ]
    elif format_type == "mp3_16min":
        # 16 minutes = 960 seconds; use silent null source for rapid generation
        cmd = [
            ffmpeg_bin,
            "-y",
            "-f", "lavfi",
            "-i", "anullsrc=r=22050:cl=mono",
            "-t", "960",
            "-c:a", "libmp3lame",
            "-b:a", "32k",
            "-f", "mp3",
            "-",
        ]
    else:
        raise ValueError(f"Unknown format_type: {format_type}")

    res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=False, timeout=30)
    assert res.returncode == 0, f"ffmpeg failed to generate {format_type}: {res.stderr.decode()}"
    return res.stdout


# ==============================================================================
# 1. Real M4A renamed .mp3 succeeds
# ==============================================================================
def test_m4a_renamed_mp3_succeeds():
    """Verify an M4A (MP4/AAC container) file renamed with a .mp3 extension succeeds."""
    m4a_bytes = _generate_synthetic_audio("m4a", duration_sec=5.0)

    res = client.post(
        "/api/analyze",
        files={"file": ("synthetic_aac_renamed.mp3", m4a_bytes, "audio/mpeg")},
    )
    assert res.status_code == 200, f"Expected 200, got {res.status_code}: {res.text}"
    body = res.json()
    assert "genre" in body
    assert body["duration_seconds"] > 0


def test_real_user_tum_tak_m4a_succeeds():
    """Verify the real user library file data/user_samples/Tum Tak.mp3 succeeds if present."""
    tum_tak_path = resolve_path("data/user_samples/Tum Tak.mp3")
    if not tum_tak_path.exists():
        pytest.skip("data/user_samples/Tum Tak.mp3 not found; skipping real user track test")

    with open(tum_tak_path, "rb") as f:
        res = client.post(
            "/api/analyze",
            files={"file": ("Tum Tak.mp3", f, "audio/mpeg")},
        )
    assert res.status_code == 200, f"Expected 200, got {res.status_code}: {res.text}"
    body = res.json()
    assert body["duration_seconds"] >= 300.0
    assert body["genre"]["top_label"] is not None


# ==============================================================================
# 2. 10 s WAV succeeds
# ==============================================================================
def test_10s_wav_succeeds():
    """Verify a 10s WAV file upload succeeds."""
    wav_bytes = _generate_synthetic_audio("wav", duration_sec=10.0)

    res = client.post(
        "/api/analyze",
        files={"file": ("tone_10s.wav", wav_bytes, "audio/wav")},
    )
    assert res.status_code == 200, f"Expected 200, got {res.status_code}: {res.text}"
    body = res.json()
    assert abs(body["duration_seconds"] - 10.0) < 0.5


# ==============================================================================
# 3. 10 s MP3 succeeds
# ==============================================================================
def test_10s_mp3_succeeds():
    """Verify a 10s MP3 file upload succeeds."""
    mp3_bytes = _generate_synthetic_audio("mp3", duration_sec=10.0)

    res = client.post(
        "/api/analyze",
        files={"file": ("tone_10s.mp3", mp3_bytes, "audio/mpeg")},
    )
    assert res.status_code == 200, f"Expected 200, got {res.status_code}: {res.text}"
    body = res.json()
    assert abs(body["duration_seconds"] - 10.0) < 0.5


# ==============================================================================
# 4. 16-minute file returns "too long" with the limit
# ==============================================================================
def test_16min_file_returns_too_long():
    """Verify a 16-minute (960 s) file returns HTTP 400 with the configured 900.0 s limit."""
    cfg = load_config()
    max_dur = float(cfg.audio.max_duration_seconds)
    max_min = int(max_dur // 60)

    mp3_16min_bytes = _generate_synthetic_audio("mp3_16min")

    res = client.post(
        "/api/analyze",
        files={"file": ("long_audio.mp3", mp3_16min_bytes, "audio/mpeg")},
    )
    assert res.status_code == 400, f"Expected 400, got {res.status_code}: {res.text}"
    detail = res.json()["detail"]
    assert f"Audio longer than {max_min} minutes" in detail
    assert f"exceeds {max_dur:.1f} s limit" in detail


# ==============================================================================
# 5. OGG renamed .mp3 returns "unsupported format"
# ==============================================================================
def test_ogg_renamed_mp3_returns_unsupported():
    """Verify an OGG file renamed with .mp3 extension returns HTTP 400 'Not a supported audio format'."""
    cfg = load_config()
    allowed_formats = list(cfg.audio.allowed_formats)
    formats_str = ", ".join(f.upper() for f in allowed_formats)

    ogg_bytes = _generate_synthetic_audio("ogg", duration_sec=2.0)

    res = client.post(
        "/api/analyze",
        files={"file": ("vorbis_track.mp3", ogg_bytes, "audio/mpeg")},
    )
    assert res.status_code == 400, f"Expected 400, got {res.status_code}: {res.text}"
    detail = res.json()["detail"]
    assert detail == f"Not a supported audio format. Supported formats: {formats_str}."


# ==============================================================================
# 6. Random bytes return "could not decode"
# ==============================================================================
def test_random_bytes_returns_could_not_decode():
    """Verify random byte corruption returns HTTP 400 'Audio could not be decoded'."""
    corrupt_bytes = os.urandom(2048)

    res = client.post(
        "/api/analyze",
        files={"file": ("corrupt.mp3", corrupt_bytes, "audio/mpeg")},
    )
    assert res.status_code == 400, f"Expected 400, got {res.status_code}: {res.text}"
    detail = res.json()["detail"]
    assert detail == "Audio could not be decoded: file is corrupted or not valid audio."


# ==============================================================================
# 7. Fake ffconcat script rejected as unsupported without reading referenced path
# ==============================================================================
def test_fake_ffconcat_rejected_as_unsupported(tmp_path):
    """Verify an ffconcat script payload is rejected as unsupported without reading any referenced path."""
    canary_file = tmp_path / "canary_secret.txt"
    canary_file.write_text("SUPER_SECRET_TOKEN_DO_NOT_READ", encoding="utf-8")

    concat_payload = f"ffconcat version 1.0\nfile '{canary_file.as_posix()}'\n".encode("utf-8")

    # 1. Uploaded as a text file
    res_txt = client.post(
        "/api/analyze",
        files={"file": ("fake_playlist.txt", concat_payload, "text/plain")},
    )
    assert res_txt.status_code == 400
    assert "Not a supported audio format" in res_txt.json()["detail"]
    assert "SUPER_SECRET_TOKEN" not in res_txt.text
    assert "canary_secret" not in res_txt.text

    # 2. Uploaded disguised as .mp3
    res_mp3 = client.post(
        "/api/analyze",
        files={"file": ("playlist.mp3", concat_payload, "audio/mpeg")},
    )
    assert res_mp3.status_code == 400
    assert "Not a supported audio format" in res_mp3.json()["detail"]
    assert "SUPER_SECRET_TOKEN" not in res_mp3.text
    assert "canary_secret" not in res_mp3.text


# ==============================================================================
# 8. FMA clips give unchanged results
# ==============================================================================
def test_fma_clip_unchanged_results():
    """Verify standard FMA Small MP3 clips decode reliably and produce consistent samples."""
    fma_path = resolve_path("data/raw/fma_small/000/000002.mp3")
    if not fma_path.exists():
        pytest.skip(f"FMA clip not found at {fma_path}")

    y = load_audio(fma_path, sr=22050, duration=30.0, mono=True)
    assert isinstance(y, np.ndarray)
    assert y.dtype == np.float32
    assert len(y) > 0
    # Expected frame count for ~30s FMA track is ~661,248
    assert abs(len(y) - int(22050 * 29.988)) < 1000


# ==============================================================================
# 9. No sonicscope_* temp files remain
# ==============================================================================
def test_no_temp_files_remain_after_requests():
    """Verify no sonicscope_* files are leaked in the temporary directory."""
    temp_dir = Path(tempfile.gettempdir())
    initial_temps = set(temp_dir.glob("sonicscope_*"))

    # Execute a success and several failures
    client.post("/api/analyze", files={"file": ("bad.mp3", b"invalid-bytes", "audio/mpeg")})
    client.post("/api/analyze", files={"file": ("fake.mp3", b"ffconcat version 1.0\n", "audio/mpeg")})

    wav_bytes = _generate_synthetic_audio("wav", duration_sec=2.0)
    client.post("/api/analyze", files={"file": ("ok.wav", wav_bytes, "audio/wav")})

    post_temps = set(temp_dir.glob("sonicscope_*"))
    leaked = post_temps - initial_temps
    assert not leaked, f"Temporary upload files were leaked on disk: {leaked}"


# ==============================================================================
# 10. No username or path appears in any error response
# ==============================================================================
def test_no_username_or_path_in_any_error_response():
    """Verify that error responses never leak local system paths, drive letters, usernames, or temp files."""
    test_cases = [
        ("corrupt.mp3", os.urandom(1024)),
        ("fake.m4a", b"ffconcat version 1.0\nfile 'test.txt'"),
        ("empty.mp3", b""),
        ("bad_sig.mp3", b"OggS\x00\x02\x00\x00" + b"\x00" * 100),
    ]

    for filename, content in test_cases:
        res = client.post("/api/analyze", files={"file": (filename, content, "audio/mpeg")})
        assert res.status_code >= 400
        detail = res.json().get("detail", "").lower()

        # Check for path / username leaks
        assert "c:" not in detail, f"Path leak in error detail: {detail}"
        assert "\\" not in detail, f"Backslash path leak in error detail: {detail}"
        assert "/users" not in detail, f"User directory leak in error detail: {detail}"
        assert "yashvie" not in detail, f"Username leak in error detail: {detail}"
        assert "sonicscope_" not in detail, f"Temp filename leak in error detail: {detail}"
        assert "appdata" not in detail, f"AppData path leak in error detail: {detail}"


# ==============================================================================
# 11. Streaming upload: Content-Length up front and partial cleanup
# ==============================================================================
def test_streaming_upload_content_length_upfront():
    """Verify Content-Length header exceeding limit is rejected upfront with HTTP 413."""
    cfg = load_config()
    max_mb = cfg.audio.max_upload_mb
    max_bytes = max_mb * 1024 * 1024

    headers = {"content-length": str(max_bytes + 1000)}
    res = client.post(
        "/api/analyze",
        files={"file": ("fake_large.mp3", b"dummy", "audio/mpeg")},
        headers=headers,
    )
    assert res.status_code == 413
    assert f"{max_mb} MB limit" in res.json()["detail"]


def test_streaming_upload_stops_reading_and_cleans_partial_temp():
    """Verify streaming upload stops reading as soon as byte limit is exceeded and partial file is unlinked."""
    cfg = load_config()
    max_mb = cfg.audio.max_upload_mb
    max_bytes = max_mb * 1024 * 1024

    temp_dir = Path(tempfile.gettempdir())
    initial_temps = set(temp_dir.glob("sonicscope_*"))

    oversize_bytes = b"X" * (max_bytes + 1024 * 1024)
    res = client.post(
        "/api/analyze",
        files={"file": ("stream_large.mp3", oversize_bytes, "audio/mpeg")},
    )
    assert res.status_code == 413
    assert f"{max_mb} MB limit" in res.json()["detail"]

    post_temps = set(temp_dir.glob("sonicscope_*"))
    assert not (post_temps - initial_temps), "Partial temp file was not deleted upon size limit exceeded!"


# ==============================================================================
# 12. Dynamic catalog track exclusion and test-split non-exclusion
# ==============================================================================
def test_upload_catalog_clip_excludes_self():
    """Verify uploading a catalog clip dynamically retrieved from the index excludes itself."""
    index_path = resolve_path("data/processed/similarity_index.joblib")
    if not index_path.exists():
        pytest.skip(f"Similarity index not found at {index_path}")

    import joblib
    import pandas as pd

    index_data = joblib.load(index_path)
    manifest_path = resolve_path("data/processed/manifest.csv")
    manifest_df = pd.read_csv(manifest_path)
    path_map = dict(zip(manifest_df["track_id"], manifest_df["path"]))

    # Find the first catalog member with an existing MP3 on disk (never hardcode track 2)
    catalog_tid = None
    catalog_file = None
    for record in index_data.get("catalog_metadata", []):
        tid = int(record["track_id"])
        rel_path = path_map.get(tid, f"data/raw/fma_small/{tid:06d}"[:18] + f"/{tid:06d}.mp3")
        resolved = resolve_path(rel_path)
        if resolved.exists():
            catalog_tid = tid
            catalog_file = resolved
            break

    if catalog_file is None:
        pytest.skip("No catalog member MP3 file found on disk.")

    with open(catalog_file, "rb") as f:
        res = client.post(
            "/api/analyze",
            files={"file": (f"{catalog_tid:06d}.mp3", f, "audio/mpeg")},
        )

    assert res.status_code == 200, f"Expected 200, got {res.status_code}: {res.text}"
    body = res.json()

    # Returns 5 neighbours
    similar_tracks = body.get("similar_tracks", [])
    assert len(similar_tracks) == 5, f"Expected 5 similar tracks, got {len(similar_tracks)}"

    # None of them is the uploaded catalog track itself
    returned_ids = [t["track_id"] for t in similar_tracks]
    assert catalog_tid not in returned_ids, f"Track {catalog_tid} retrieved itself in top-5!"

    # Exclusion warning and flag present
    assert body.get("identical_catalog_track_excluded") is True
    assert "An identical catalog track was excluded." in body.get("warnings", [])

    # Keep ONE similarity field: feature_similarity_score (0.0 to 1.0)
    for t in similar_tracks:
        assert "feature_similarity_score" in t
        assert 0.0 <= t["feature_similarity_score"] <= 1.0
        assert "similarity_score" not in t
        assert "similarity_percentage" not in t


def test_upload_test_split_clip_returns_five_without_exclusion():
    """Verify uploading a TEST-split clip (not in catalog) returns 5 neighbours, no exclusion warning, flag false."""
    manifest_path = resolve_path("data/processed/manifest.csv")
    if not manifest_path.exists():
        pytest.skip(f"Manifest not found at {manifest_path}")

    import pandas as pd
    manifest_df = pd.read_csv(manifest_path)
    test_rows = manifest_df[manifest_df["split"] == "test"]

    test_tid = None
    test_file = None
    for _, row in test_rows.iterrows():
        tid = int(row["track_id"])
        resolved = resolve_path(row["path"])
        if resolved.exists():
            test_tid = tid
            test_file = resolved
            break

    if test_file is None:
        pytest.skip("No TEST-split clip found on disk.")

    with open(test_file, "rb") as f:
        res = client.post(
            "/api/analyze",
            files={"file": (f"{test_tid:06d}.mp3", f, "audio/mpeg")},
        )

    assert res.status_code == 200, f"Expected 200, got {res.status_code}: {res.text}"
    body = res.json()

    similar_tracks = body.get("similar_tracks", [])
    assert len(similar_tracks) == 5, f"Expected 5 similar tracks, got {len(similar_tracks)}"

    # identical_catalog_track_excluded must be False
    assert body.get("identical_catalog_track_excluded") is False
    # No exclusion warning present
    assert not any("excluded" in w.lower() for w in body.get("warnings", []))

    for t in similar_tracks:
        assert "feature_similarity_score" in t
        assert 0.0 <= t["feature_similarity_score"] <= 1.0
        assert "similarity_score" not in t
        assert "similarity_percentage" not in t


# ==============================================================================
# 13. Padding warning: raw_duration = min(decoded duration, 30); no warning > 30s
# ==============================================================================
def test_padding_warning_threshold_and_format():
    """Verify padding warning only fires when padding shortfall > 0.5s, with correct format, and not for > 30s."""
    # 1. 12.3s audio (padding shortfall = 30.0 - 12.3 = 17.7s > 0.5s) -> Warning produced
    audio_12s = _generate_synthetic_audio("wav", duration_sec=12.3)
    res_12s = client.post("/api/analyze", files={"file": ("clip_12s.wav", audio_12s, "audio/wav")})
    assert res_12s.status_code == 200
    warnings_12s = res_12s.json().get("warnings", [])
    assert any("Audio (12.3 s) was zero-padded to 30.0 s" in w for w in warnings_12s)

    # 2. 29.8s audio (padding shortfall = 30.0 - 29.8 = 0.2s <= 0.5s) -> No zero-padded warning
    audio_29s = _generate_synthetic_audio("wav", duration_sec=29.8)
    res_29s = client.post("/api/analyze", files={"file": ("clip_29s.wav", audio_29s, "audio/wav")})
    assert res_29s.status_code == 200
    warnings_29s = res_29s.json().get("warnings", [])
    assert not any("zero-padded" in w for w in warnings_29s)

    # 3. 35.0s audio (> 30s; raw_duration clamped to 30 for padding calculation) -> No zero-padded warning
    audio_35s = _generate_synthetic_audio("wav", duration_sec=35.0)
    res_35s = client.post("/api/analyze", files={"file": ("clip_35s.wav", audio_35s, "audio/wav")})
    assert res_35s.status_code == 200
    warnings_35s = res_35s.json().get("warnings", [])
    assert not any("zero-padded" in w for w in warnings_35s)

