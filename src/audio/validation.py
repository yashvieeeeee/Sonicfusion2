"""Audio file validation and container probe utilities.

Follows audio validation requirements from docs/PLAN.md (§05, §06, §13, §17, Addendum A).
Enforces:
- Candidate demuxer selection by magic bytes ("RIFF....WAVE" -> wav, "ftyp" -> mov, else mp3 candidate).
- Explicit -f <demuxer> and -protocol_whitelist file,pipe to ffprobe and ffmpeg.
- Post-probe format name and audio stream verification.
- Detection of known unsupported formats (Ogg, FLAC, ffconcat).
"""

from __future__ import annotations

import os
from pathlib import Path
import shutil
import subprocess
from typing import Any, List, Optional, Tuple, Union

# Single boolean policy controlling whether non-MP3 formats are rejected
REJECT_NON_MP3: bool = False


def get_candidate_demuxer(file_path: Union[str, Path]) -> str:
    """Choose candidate demuxer by magic bytes without whitelisting MP3 leading bytes.

    - "RIFF....WAVE" (offset 0 RIFF, offset 8 WAVE) -> 'wav'
    - 'ftyp' at offset 4 -> 'mov'
    - Anything else -> 'mp3' candidate (do NOT whitelist leading bytes for MP3)

    Args:
        file_path: Path to the audio file.

    Returns:
        Demuxer name string ('wav', 'mov', or 'mp3').
    """
    path = Path(file_path)
    try:
        with open(path, "rb") as f:
            header = f.read(16)
    except OSError:
        return "mp3"

    if len(header) >= 12 and header[:4] == b"RIFF" and header[8:12] == b"WAVE":
        return "wav"
    if len(header) >= 8 and header[4:8] == b"ftyp":
        return "mov"
    return "mp3"


def check_known_unsupported_signatures(file_path: Union[str, Path]) -> bool:
    """Check if file starts with known unsupported containers or script headers.

    Specifically catches:
    - OggS at offset 0 (Ogg Vorbis / Opus)
    - fLaC at offset 0 (FLAC)
    - ffconcat at offset 0 (fake concat scripts)
    - RIFF at offset 0 but not WAVE (e.g. AVI)

    Args:
        file_path: Path to the file.

    Returns:
        True if the file matches a known unsupported signature, False otherwise.
    """
    path = Path(file_path)
    try:
        with open(path, "rb") as f:
            header = f.read(16)
    except OSError:
        return False

    if header.startswith(b"OggS") or header.startswith(b"fLaC") or header.startswith(b"ffconcat"):
        return True
    if len(header) >= 12 and header[:4] == b"RIFF" and header[8:12] != b"WAVE":
        return True
    return False


def get_container_format(file_path: Union[str, Path]) -> Optional[str]:
    """Inspect audio container format using candidate demuxer with ffprobe.

    Args:
        file_path: Path to the audio file.

    Returns:
        Container format string (e.g. 'mp3', 'mov,mp4,m4a,3gp,3g2,mj2', 'wav') or None.
    """
    ffprobe_bin = shutil.which("ffprobe")
    if not ffprobe_bin:
        return None

    path = Path(file_path)
    if not path.exists():
        return None

    demuxer = get_candidate_demuxer(path)
    try:
        res = subprocess.run(
            [
                ffprobe_bin,
                "-v", "error",
                "-protocol_whitelist", "file,pipe",
                "-f", demuxer,
                "-show_entries", "format=format_name",
                "-of", "default=noprint_wrappers=1:nokey=1",
                str(path),
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            shell=False,
            timeout=10,
        )
        if res.returncode == 0 and res.stdout.strip():
            return res.stdout.strip()
    except Exception:
        pass
    return None


def probe_audio_container(
    file_path: Union[str, Path],
    allowed_formats: Optional[List[str]] = None,
    max_duration_seconds: float = 900.0,
) -> Tuple[str, Optional[str], Optional[float]]:
    """Probe audio file container, streams, and duration using explicit demuxer.

    Enforces:
    1. Known unsupported signature check (Ogg, FLAC, ffconcat).
    2. Explicit demuxer passed with -f (no auto-detection).
    3. At least one audio stream required (-select_streams a:0).
    4. Second-layer verification of ffprobe format_name against allowed list.
    5. Bounded duration inspection (never full decode if N/A).

    Args:
        file_path: Path to the audio file.
        allowed_formats: List of allowed formats (default ['mp3', 'wav', 'm4a']).
        max_duration_seconds: Maximum allowed duration for bounded fallback check.

    Returns:
        Tuple of (status, canonical_format, duration_seconds)
        where status is 'ok', 'unsupported_format', or 'decode_error'.
    """
    if allowed_formats is None:
        allowed_formats = ["mp3", "wav", "m4a"]

    path = Path(file_path)
    if not path.exists():
        return "decode_error", None, None

    try:
        size = path.stat().st_size
        if size == 0:
            return "decode_error", None, None
    except OSError:
        return "decode_error", None, None

    # 1. Upfront check for known non-supported signatures (Ogg, FLAC, ffconcat)
    if check_known_unsupported_signatures(path):
        return "unsupported_format", None, None

    # 2. Determine candidate demuxer by magic bytes
    demuxer = get_candidate_demuxer(path)

    ffprobe_bin = shutil.which("ffprobe")
    if not ffprobe_bin:
        raise RuntimeError("ffprobe executable not found on system PATH; cannot validate container")

    # 3. Run ffprobe with explicit candidate demuxer and whitelist
    cmd = [
        ffprobe_bin,
        "-v", "error",
        "-protocol_whitelist", "file,pipe",
        "-f", demuxer,
        "-show_entries", "format=format_name,duration",
        "-select_streams", "a:0",
        "-show_entries", "stream=codec_type",
        "-of", "default=noprint_wrappers=1",
        str(path),
    ]

    try:
        res = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            shell=False,
            timeout=10,
        )
    except subprocess.TimeoutExpired:
        return "decode_error", None, None
    except Exception:
        return "decode_error", None, None

    if res.returncode != 0:
        return "decode_error", None, None

    output_lines = [line.strip() for line in res.stdout.strip().splitlines() if line.strip()]
    metadata: dict[str, str] = {}
    for line in output_lines:
        if "=" in line:
            k, v = line.split("=", 1)
            metadata[k.strip()] = v.strip()

    # 4. Require at least one audio stream
    codec_type = metadata.get("codec_type")
    if codec_type != "audio":
        return "unsupported_format", None, None

    # 5. Second layer: post-probe format-name verification
    format_name = metadata.get("format_name", "")
    if not format_name:
        return "decode_error", None, None

    # Reject playlist/network/concat formats
    bad_tokens = {"hls", "applehttp", "concat", "m3u8", "subrip"}
    tokens = set(t.strip().lower() for t in format_name.split(",") if t.strip())
    if tokens.intersection(bad_tokens):
        return "unsupported_format", None, None

    canonical_format: Optional[str] = None
    if demuxer == "wav":
        if "wav" in tokens:
            canonical_format = "wav"
    elif demuxer == "mov":
        mov_tokens = {"mov", "mp4", "m4a", "3gp", "3g2", "mj2"}
        if tokens.intersection(mov_tokens):
            canonical_format = "m4a"
    elif demuxer == "mp3":
        if "mp3" in tokens:
            canonical_format = "mp3"

    if canonical_format is None or canonical_format not in allowed_formats:
        return "unsupported_format", None, None

    # 6. Parse duration
    dur_str = metadata.get("duration", "")
    duration: Optional[float] = None
    if dur_str and dur_str != "N/A":
        try:
            val = float(dur_str)
            if val > 0:
                duration = val
        except ValueError:
            pass

    # If duration was missing / N/A, measure with bounded decode
    if duration is None:
        try:
            from src.audio.loading import bounded_decode_duration
            duration = bounded_decode_duration(
                path,
                demuxer=demuxer,
                max_bounded_seconds=max_duration_seconds + 1.0,
            )
        except Exception:
            return "decode_error", None, None

    return "ok", canonical_format, duration


def check_known_non_mp3_signatures(
    file_path: Union[str, Path],
    reject_non_mp3: bool = True,
) -> Tuple[bool, Optional[str]]:
    """Legacy signature check maintained for backwards compatibility."""
    path = Path(file_path)
    try:
        with open(path, "rb") as f:
            header = f.read(32)
    except OSError as e:
        return False, f"Could not read file header: {e}"

    if len(header) < 8:
        return False, "File too small to be valid audio"

    if header[4:8] == b"ftyp":
        if reject_non_mp3:
            return False, "Non-MP3 format detected: ISO Media container (M4A/AAC/MP4 with ftyp signature)"
    elif header[:4] == b"OggS":
        if reject_non_mp3:
            return False, "Non-MP3 format detected: Ogg container (OggS signature)"
    elif header[:4] == b"fLaC":
        if reject_non_mp3:
            return False, "Non-MP3 format detected: FLAC audio (fLaC signature)"
    elif header[:4] == b"RIFF":
        if reject_non_mp3:
            return False, "Non-MP3 format detected: RIFF/WAV container (RIFF signature)"

    return True, None


def verify_audio_file(
    file_path: Union[str, Path],
    reject_non_mp3: bool = False,
    test_decode_sec: float = 5.0,
    allowed_formats: Optional[List[str]] = None,
) -> Tuple[bool, Optional[str]]:
    """Verify that an audio file exists, is non-empty, and decodes successfully.

    Maintained for existing callers in build_manifest and manifest tests.
    """
    path = Path(file_path)
    if not path.exists():
        return False, "File not found"

    try:
        size = path.stat().st_size
        if size == 0:
            return False, "Empty file (0 bytes)"
    except OSError as e:
        return False, f"Could not stat file: {e}"

    # If caller specifically asked for reject_non_mp3 (e.g. legacy tests)
    if reject_non_mp3:
        valid_sig, sig_err = check_known_non_mp3_signatures(path, reject_non_mp3=True)
        if not valid_sig:
            return False, sig_err

    status, canonical_format, duration = probe_audio_container(
        path,
        allowed_formats=allowed_formats or (["mp3"] if reject_non_mp3 else ["mp3", "wav", "m4a"]),
    )
    if status == "unsupported_format":
        return False, "Not a supported audio format"
    if status != "ok":
        return False, "Audio could not be decoded: file is corrupted or not valid audio"

    try:
        from src.audio.loading import load_audio
        data = load_audio(path, sr=22050, duration=test_decode_sec, mono=True)
        if len(data) == 0:
            return False, "Decoded 0 audio frames"
        return True, None
    except RuntimeError:
        raise
    except Exception as exc:
        return False, f"Decode error: {exc}"


def sanitize_path_str(text: str, project_root: Optional[Union[str, Path]] = None) -> str:
    """Sanitize path strings by converting backslashes and stripping absolute system paths.

    Ensures that usernames, Windows drive letters, and full machine paths are never
    leaked into logs, error messages, or reports.
    """
    import re
    s = str(text).replace("\\", "/")
    if project_root is not None:
        root_s = str(project_root).replace("\\", "/")
        s = s.replace(root_s + "/", "").replace(root_s, "")
    # Remove Windows drive paths (e.g. C:/Users/.../Sonicfusion2/)
    s = re.sub(r"[A-Za-z]:/[^:\"']*/Sonicfusion2/", "", s, flags=re.IGNORECASE)
    # Remove any absolute user paths (e.g. /Users/<username>/ or C:/Users/<username>/)
    s = re.sub(r"[A-Za-z]:/(?:Users|home)/[^/]+/[^:\"']*/", "", s, flags=re.IGNORECASE)
    s = re.sub(r"/(?:Users|home)/[^/]+/[^:\"']*/", "", s, flags=re.IGNORECASE)
    # Generic drive letter strip
    s = re.sub(r"[A-Za-z]:/", "", s)
    return s


def verify_audio_file_strict(
    file_path: Union[str, Path],
    min_duration: float = 25.0,
) -> Tuple[bool, float, int, Optional[str]]:
    """Strictly verify dataset audio file integrity and duration without lenient ffmpeg salvage.

    Enforces:
    1. File existence and non-zero byte size.
    2. Strict audio header / container parse with soundfile (does not use lenient ffmpeg salvage).
    3. Rejection of any file whose strictly decodable duration is below min_duration.

    Args:
        file_path: Path to the audio file.
        min_duration: Minimum required duration in seconds (default: 25.0).

    Returns:
        Tuple of (is_valid, decodable_seconds, file_size_bytes, reason).
    """
    import soundfile as sf

    path = Path(file_path)
    if not path.exists():
        return False, 0.0, 0, "File not found"

    try:
        size = path.stat().st_size
        if size == 0:
            return False, 0.0, 0, "Empty file (0 bytes)"
    except OSError as e:
        return False, 0.0, 0, f"Could not stat file: {e}"

    try:
        info = sf.info(str(path))
        dur = float(info.duration) if info.duration is not None else 0.0
    except Exception as exc:
        exc_str = sanitize_path_str(str(exc))
        # Handle small stub/corrupt MP3s with unreadable headers cleanly
        if ("pipe" in exc_str or "not a regular file" in exc_str or "Error opening" in exc_str) and size < 10000:
            clean_reason = f"Unreadable/corrupt MP3 (file size {size} bytes, no decodable audio)"
        else:
            clean_reason = f"Corrupt audio header: {type(exc).__name__}: {exc_str}"
        return False, 0.0, size, clean_reason

    if dur < min_duration:
        return (
            False,
            dur,
            size,
            f"Decodable duration ({dur:.2f}s) is below min_track_seconds ({min_duration:.1f}s)",
        )

    return True, dur, size, None

