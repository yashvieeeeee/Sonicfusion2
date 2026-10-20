"""Audio loading, normalization, and window segmentation.

Follows the deterministic audio contract from docs/PLAN.md (§07).
"""

from __future__ import annotations

from pathlib import Path
import os
import shutil
import subprocess
from typing import List, Optional, Union
import librosa
import numpy as np
import soundfile as sf


class AudioDecodeError(Exception):
    """Raised when an audio file cannot be decoded by any supported engine."""
    pass


def bounded_decode_duration(
    file_path: Union[str, Path],
    demuxer: Optional[str] = None,
    max_bounded_seconds: float = 901.0,
    sr: int = 22050,
) -> float:
    """Measure duration by bounded decode without decoding whole file.

    Decodes at most max_bounded_seconds with ffmpeg.
    """
    path = Path(file_path)
    ffmpeg_bin = shutil.which("ffmpeg")
    if not ffmpeg_bin:
        raise RuntimeError("ffmpeg executable not found on system PATH; cannot perform bounded decode")

    if demuxer is None:
        from src.audio.validation import get_candidate_demuxer
        demuxer = get_candidate_demuxer(path)

    cmd = [
        ffmpeg_bin,
        "-nostdin",
        "-v", "error",
        "-protocol_whitelist", "file,pipe",
        "-f", demuxer,
        "-i", str(path),
        "-vn",
        "-map", "0:a:0",
        "-t", str(max_bounded_seconds),
        "-ac", "1",
        "-ar", str(sr),
        "-f", "f32le",
        "-",
    ]
    try:
        res = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            shell=False,
            timeout=35,
        )
    except Exception as exc:
        raise AudioDecodeError(f"Bounded decode failed for {path.name}: {exc}") from exc

    if res.returncode != 0 or len(res.stdout) == 0:
        raise AudioDecodeError(f"Could not decode audio frames from {path.name}")

    num_samples = len(res.stdout) // 4
    return float(num_samples) / float(sr)


def get_audio_duration(
    file_path: Union[str, Path],
    max_bounded_seconds: float = 901.0,
) -> float:
    """Retrieve audio duration in seconds without fully decoding audio bitstream.

    1. Tries soundfile header info first.
    2. Tries ffprobe container metadata with candidate demuxer and -protocol_whitelist file,pipe.
    3. If duration is missing, empty, or 'N/A', measures duration via bounded decode (at most max_bounded_seconds).
       Never falls back to full unbounded decode.

    Args:
        file_path: Path to the audio file.
        max_bounded_seconds: Maximum seconds to inspect in bounded decode fallback.

    Returns:
        Duration in seconds as a float.
    """
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"Audio file not found: {path}")

    # 1. Try soundfile info (fast header parse for standard containers)
    try:
        info = sf.info(str(path))
        if info.duration and float(info.duration) > 0:
            return float(info.duration)
    except Exception:
        pass

    # 2. Try ffprobe container metadata with candidate demuxer and whitelist
    ffprobe_bin = shutil.which("ffprobe")
    if ffprobe_bin:
        from src.audio.validation import get_candidate_demuxer
        demuxer = get_candidate_demuxer(path)
        try:
            cmd = [
                ffprobe_bin,
                "-v", "error",
                "-protocol_whitelist", "file,pipe",
                "-f", demuxer,
                "-show_entries", "format=duration",
                "-of", "default=noprint_wrappers=1:nokey=1",
                str(path),
            ]
            res = subprocess.run(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                shell=False,
                timeout=10,
            )
            if res.returncode == 0 and res.stdout.strip():
                dur_str = res.stdout.strip()
                if dur_str != "N/A":
                    dur = float(dur_str)
                    if dur > 0:
                        return dur
        except Exception:
            pass

    # 3. Fallback to bounded decode measurement (NEVER full decode)
    try:
        return bounded_decode_duration(path, max_bounded_seconds=max_bounded_seconds)
    except Exception as exc:
        raise AudioDecodeError(f"Could not determine audio duration for file: {path.name}") from exc


def _load_audio_ffmpeg(
    path: Path,
    sr: int = 22050,
    duration: Optional[float] = None,
    mono: bool = True,
) -> np.ndarray:
    """Decode audio file to float32 PCM using ffmpeg subprocess with candidate demuxer.

    Uses specific fallback flags:
    -nostdin -v error -protocol_whitelist file,pipe -f <demuxer> -vn -map 0:a:0 -t <duration> -ac 1 -ar <sr> -f f32le
    """
    ffmpeg_bin = shutil.which("ffmpeg")
    if not ffmpeg_bin:
        # Missing ffmpeg is a 500-class server environment error, not a 400 user error
        raise RuntimeError("ffmpeg executable not found on system PATH; cannot perform fallback decode")

    from src.audio.validation import get_candidate_demuxer
    demuxer = get_candidate_demuxer(path)

    cmd = [
        ffmpeg_bin,
        "-nostdin",
        "-v", "error",
        "-protocol_whitelist", "file,pipe",
        "-f", demuxer,
        "-i", str(path),
        "-vn",
        "-map", "0:a:0",
    ]
    if duration is not None:
        cmd.extend(["-t", str(duration)])

    cmd.extend([
        "-ac", "1" if mono else "2",
        "-ar", str(sr),
        "-f", "f32le",
        "-",
    ])

    try:
        res = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            shell=False,
            timeout=30,
        )
    except subprocess.TimeoutExpired as exc:
        raise AudioDecodeError(f"ffmpeg decode timed out after 30s: {path.name}") from exc
    except Exception as exc:
        raise AudioDecodeError(f"ffmpeg subprocess execution failed: {exc}") from exc

    if res.returncode != 0:
        err_msg = res.stderr.decode("utf-8", errors="replace").strip()
        raise AudioDecodeError(f"ffmpeg failed to decode audio (code {res.returncode}): {err_msg}")

    raw_bytes = res.stdout
    if len(raw_bytes) == 0:
        raise AudioDecodeError(f"ffmpeg decoded 0 audio frames from: {path.name}")

    y = np.frombuffer(raw_bytes, dtype=np.float32)
    return y


def load_audio(
    file_path: Union[str, Path],
    sr: int = 22050,
    duration: Optional[float] = 30.0,
    mono: bool = True,
) -> np.ndarray:
    """Load an audio file, convert to mono, resample to sr, and truncate to at most duration.

    Tries librosa.load first. If it raises, falls back to ffmpeg subprocess decode.
    Per PLAN §07 and project audio contracts, load_audio must NOT pad short audio.
    Zero-padding is strictly handled in segment_audio() or downstream inference preppers.

    Args:
        file_path: Path to the audio file.
        sr: Target sample rate (default 22050 Hz).
        duration: Target maximum duration in seconds (default 30.0 s).
        mono: Whether to convert multi-channel audio to mono (default True).

    Returns:
        1D float32 normalized waveform of length at most int(sr * duration).
    """
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"Audio file not found: {path}")

    from src.audio.validation import get_candidate_demuxer
    demuxer = get_candidate_demuxer(path)

    # For MOV/M4A containers, libsoundfile does not support AAC/MP4; use direct ffmpeg with -f mov
    if demuxer == "mov":
        y = _load_audio_ffmpeg(path, sr=sr, duration=duration, mono=mono)
    else:
        # Standard MP3/WAV: try librosa.load, fallback to ffmpeg with explicit candidate demuxer if it raises
        try:
            y, _ = librosa.load(str(path), sr=sr, mono=mono, duration=duration)
            y = y.astype(np.float32)
        except Exception:
            try:
                y = _load_audio_ffmpeg(path, sr=sr, duration=duration, mono=mono)
            except RuntimeError:
                raise
            except Exception as ffmpeg_exc:
                raise AudioDecodeError(
                    f"Failed to decode audio: {ffmpeg_exc}"
                ) from ffmpeg_exc

    # Enforce maximum duration / target samples by truncation only (NO PADDING)
    if duration is not None:
        target_samples = int(sr * duration)
        if len(y) > target_samples:
            y = y[:target_samples]

    return y


def segment_audio(
    y: np.ndarray,
    sr: int = 22050,
    window_sec: float = 10.0,
    num_windows: int = 3,
) -> List[np.ndarray]:
    """Segment a 1D audio array into non-overlapping temporal windows.

    Args:
        y: 1D audio waveform.
        sr: Sample rate in Hz.
        window_sec: Duration of each window in seconds (default 10.0 s).
        num_windows: Number of consecutive windows to extract (default 3).

    Returns:
        List of 1D float32 arrays, each of shape (int(sr * window_sec),).
    """
    window_samples = int(sr * window_sec)
    required_samples = window_samples * num_windows

    if len(y) < required_samples:
        pad_width = required_samples - len(y)
        y = np.pad(y, (0, pad_width), mode="constant", constant_values=0.0)

    segments = []
    for i in range(num_windows):
        start = i * window_samples
        end = start + window_samples
        seg = y[start:end].astype(np.float32)
        segments.append(seg)

    return segments
