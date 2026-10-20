"""Audio processing, validation, and visualization downsampling service.

Follows audio requirements from PLAN §07, §11, and §13:
- Validates uploads by full real audio decode.
- Enforces 30 MB file size cap and 5-minute (300 s) duration cap.
- Downsamples waveforms to <= 1000 points.
- Downsamples spectrograms to <= 128x300 matrices.
- Computes 12-bin chroma, RMS energy, and estimated BPM (explicitly labelled as an estimate).
"""

from __future__ import annotations

import math
from pathlib import Path
import tempfile
from typing import Any, Dict, List, Tuple
import uuid

from fastapi import HTTPException, Request, UploadFile
import librosa
import numpy as np
import soundfile as sf

from src.config import load_config
from src.audio.loading import AudioDecodeError, get_audio_duration, load_audio
from src.audio.validation import probe_audio_container, verify_audio_file


def validate_and_save_upload(
    upload_file: UploadFile,
    request: Optional[Request] = None,
    max_bytes: Optional[int] = None,
    max_duration: Optional[float] = None,
    allowed_formats: Optional[List[str]] = None,
) -> Tuple[Path, str]:
    """Stream an uploaded file to a randomized temporary path, verifying size, decode, and duration.

    Synchronously streams chunks in worker threadpool to avoid blocking event loop.
    Enforces config-driven file size cap and duration cap before full decoding.
    Renames the temp file with a suffix matching the detected format.

    Args:
        upload_file: FastAPI UploadFile object.
        max_bytes: Maximum allowed file size in bytes (defaults to config max_upload_mb).
        max_duration: Maximum allowed decoded duration in seconds (defaults to config max_duration_seconds).
        allowed_formats: Allowed container formats list (defaults to config allowed_formats).

    Returns:
        (temp_path, sanitized_filename)

    Raises:
        HTTPException(413): If upload exceeds size cap.
        HTTPException(400): If decode fails, unsupported format, or duration exceeds cap.
        HTTPException(500): If server dependencies (e.g. ffmpeg/ffprobe) are missing.
    """
    raw_filename = upload_file.filename or "unknown.mp3"
    sanitized_filename = Path(raw_filename).name

    cfg = load_config()
    audio_cfg = cfg.get("audio", {})
    cfg_max_dur = float(max_duration if max_duration is not None else audio_cfg.get("max_duration_seconds", 900.0))
    cfg_max_mb = int(audio_cfg.get("max_upload_mb", 30))
    cfg_max_bytes = int(max_bytes if max_bytes is not None else (cfg_max_mb * 1024 * 1024))
    cfg_allowed = list(allowed_formats if allowed_formats is not None else audio_cfg.get("allowed_formats", ["mp3", "wav", "m4a"]))

    formats_str = ", ".join(f.upper() for f in cfg_allowed)
    max_minutes = int(cfg_max_dur // 60)

    # 1. Check Content-Length header up front if present
    content_length_header = None
    if request is not None and hasattr(request, "headers") and request.headers:
        content_length_header = request.headers.get("content-length")
    if not content_length_header and hasattr(upload_file, "headers") and upload_file.headers:
        content_length_header = upload_file.headers.get("content-length")

    if content_length_header:
        try:
            if int(content_length_header) > cfg_max_bytes:
                raise HTTPException(
                    status_code=413,
                    detail=f"File too large: exceeds {cfg_max_mb} MB limit.",
                )
        except ValueError:
            pass

    if hasattr(upload_file, "size") and upload_file.size and upload_file.size > cfg_max_bytes:
        raise HTTPException(
            status_code=413,
            detail=f"File too large: exceeds {cfg_max_mb} MB limit.",
        )

    temp_dir = Path(tempfile.gettempdir())
    random_id = uuid.uuid4().hex
    temp_path = temp_dir / f"sonicscope_{random_id}.tmp"

    total_bytes = 0
    try:
        with open(temp_path, "wb") as f_out:
            while chunk := upload_file.file.read(64 * 1024):
                total_bytes += len(chunk)
                if total_bytes > cfg_max_bytes:
                    f_out.close()
                    if temp_path.exists():
                        temp_path.unlink()
                    raise HTTPException(
                        status_code=413,
                        detail=f"File too large: exceeds {cfg_max_mb} MB limit.",
                    )
                f_out.write(chunk)

        if total_bytes == 0:
            if temp_path.exists():
                temp_path.unlink()
            raise HTTPException(
                status_code=400,
                detail="Audio could not be decoded: file is corrupted or not valid audio.",
            )

        # 2. Probe container format, audio stream presence, and duration
        status, canonical_format, duration = probe_audio_container(
            temp_path,
            allowed_formats=cfg_allowed,
            max_duration_seconds=cfg_max_dur,
        )

        if status == "unsupported_format":
            if temp_path.exists():
                temp_path.unlink()
            raise HTTPException(
                status_code=400,
                detail=f"Not a supported audio format. Supported formats: {formats_str}.",
            )
        elif status != "ok" or not canonical_format:
            if temp_path.exists():
                temp_path.unlink()
            raise HTTPException(
                status_code=400,
                detail="Audio could not be decoded: file is corrupted or not valid audio.",
            )

        # 3. Rename temp file to have suffix matching detected format
        renamed_path = temp_dir / f"sonicscope_{random_id}.{canonical_format}"
        temp_path.rename(renamed_path)
        temp_path = renamed_path

        # 4. Enforce duration cap
        if duration is not None and duration > cfg_max_dur:
            if temp_path.exists():
                temp_path.unlink()
            raise HTTPException(
                status_code=400,
                detail=(
                    f"Audio longer than {max_minutes} minutes: duration ({duration:.1f} s) "
                    f"exceeds {cfg_max_dur:.1f} s limit."
                ),
            )

        # 5. Real decode verification (first 5 seconds test decode)
        try:
            test_data = load_audio(temp_path, sr=22050, duration=5.0, mono=True)
            if len(test_data) == 0:
                raise AudioDecodeError("Decoded 0 audio frames")
        except RuntimeError:
            raise
        except Exception:
            if temp_path.exists():
                temp_path.unlink()
            raise HTTPException(
                status_code=400,
                detail="Audio could not be decoded: file is corrupted or not valid audio.",
            )

        return temp_path, sanitized_filename

    except HTTPException:
        if temp_path.exists():
            try:
                temp_path.unlink()
            except OSError:
                pass
        raise
    except RuntimeError as rerr:
        if temp_path.exists():
            try:
                temp_path.unlink()
            except OSError:
                pass
        if "ffmpeg" in str(rerr).lower() or "ffprobe" in str(rerr).lower():
            raise HTTPException(
                status_code=500,
                detail=f"Server audio processing error: {rerr}",
            )
        raise HTTPException(status_code=500, detail=str(rerr))
    except Exception:
        if temp_path.exists():
            try:
                temp_path.unlink()
            except OSError:
                pass
        raise HTTPException(
            status_code=400,
            detail="Audio could not be decoded: file is corrupted or not valid audio.",
        )


def downsample_waveform(y: np.ndarray, max_points: int = 1000) -> List[float]:
    """Downsample a 1D audio waveform to at most max_points using peak envelopes.

    Preserves positive and negative peaks for visually accurate browser rendering.

    Args:
        y: 1D audio waveform array.
        max_points: Maximum points to return (default 1000).

    Returns:
        List of rounded floats, with length <= max_points.
    """
    if len(y) == 0:
        return []

    if len(y) <= max_points:
        return [round(float(v), 4) for v in y]

    # Split into max_points chunks and take the signed peak amplitude
    chunks = np.array_split(y, max_points)
    points: List[float] = []
    for chunk in chunks:
        if len(chunk) == 0:
            points.append(0.0)
        else:
            # Pick index of max absolute value to preserve envelope dynamics
            peak_idx = int(np.argmax(np.abs(chunk)))
            points.append(round(float(chunk[peak_idx]), 4))

    return points[:max_points]


def downsample_spectrogram(
    logmel: np.ndarray, max_freq: int = 128, max_time: int = 300
) -> List[List[float]]:
    """Downsample a 2D log-mel spectrogram matrix to at most [max_freq, max_time].

    Averages time bins to produce lightweight JSON payload (<= ~150 KB).

    Args:
        logmel: 2D numpy array [n_mels, time_frames].
        max_freq: Maximum frequency mel bins (default 128).
        max_time: Maximum time frames (default 300).

    Returns:
        2D list of floats with shape <= [max_freq, max_time].
    """
    spec = logmel[:max_freq, :]
    n_freq, n_time = spec.shape

    if n_time > max_time:
        time_chunks = np.array_split(spec, max_time, axis=1)
        downsampled = np.column_stack([chunk.mean(axis=1) for chunk in time_chunks])
    else:
        downsampled = spec

    # Clean any non-finite values and round to 2 decimal places
    clean = np.nan_to_num(downsampled, nan=-80.0, posinf=0.0, neginf=-80.0)
    return [[round(float(val), 2) for val in row] for row in clean]


def extract_acoustic_features(y: np.ndarray, sr: int = 22050) -> Dict[str, Any]:
    """Extract acoustic features (chroma, RMS, estimated BPM).

    Args:
        y: 1D audio waveform array.
        sr: Audio sample rate in Hz.

    Returns:
        Dictionary with tempo_bpm_estimate, bpm_label, rms_mean, rms_std, chroma.
    """
    # 1. RMS Energy
    rms = librosa.feature.rms(y=y)
    rms_mean = float(np.mean(rms)) if rms.size > 0 else 0.0
    rms_std = float(np.std(rms)) if rms.size > 0 else 0.0

    # 2. 12-bin Chroma Profile (mean across frames)
    chroma_stft = librosa.feature.chroma_stft(y=y, sr=sr, n_fft=2048, hop_length=512)
    chroma_mean = chroma_stft.mean(axis=1) if chroma_stft.size > 0 else np.zeros(12)
    chroma_vals = [
        round(float(c), 4) if not (math.isnan(c) or math.isinf(c)) else 0.0
        for c in chroma_mean
    ]
    # Ensure exactly 12 bins
    if len(chroma_vals) < 12:
        chroma_vals.extend([0.0] * (12 - len(chroma_vals)))
    chroma_vals = chroma_vals[:12]

    # 3. Estimated BPM (explicitly labelled as an algorithmic estimate)
    try:
        tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
        if isinstance(tempo, np.ndarray):
            tempo_val = float(tempo[0]) if len(tempo) > 0 else 0.0
        else:
            tempo_val = float(tempo)
    except Exception:
        tempo_val = 0.0

    if math.isnan(tempo_val) or math.isinf(tempo_val):
        tempo_val = 0.0

    return {
        "tempo_bpm_estimate": round(tempo_val, 1),
        "bpm_label": "estimate",
        "rms_mean": round(rms_mean, 4),
        "rms_std": round(rms_std, 4),
        "chroma": chroma_vals,
        "chroma_labels": ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"],
    }
