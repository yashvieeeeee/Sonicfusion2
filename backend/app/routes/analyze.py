"""POST /api/analyze endpoint for audio upload, validation, acoustic analysis, and genre inference.

Follows PLAN §11 and §13:
- Enforces 30 MB file size cap and 5-minute duration cap.
- Real decode validation (fails on renamed .txt or corrupt MP3).
- Uses a randomized temporary file, guaranteed to be unlinked after request.
- Downsamples waveform to <= 1000 points.
- Downsamples spectrogram to <= 128x300 matrix.
- Extracts 12-bin chroma, RMS energy, and estimated BPM (explicitly labelled estimate).
- Performs GenreCNN inference and returns structured analysis payload.
"""

from __future__ import annotations

import json
from pathlib import Path
import uuid
from typing import Any, Dict, List

from fastapi import APIRouter, File, HTTPException, Request, UploadFile
import numpy as np
import soundfile as sf

from backend.app.schemas import (
    AcousticFeatures,
    AnalysisResponse,
    GenreResult,
    SimilarTrack,
    Visualizations,
)
from backend.app.services.audio import (
    downsample_spectrogram,
    downsample_waveform,
    extract_acoustic_features,
    validate_and_save_upload,
)
from backend.app.services.genre import get_genre_service
from backend.app.services.similarity import get_similarity_service
from src.audio.loading import get_audio_duration, load_audio
from src.audio.spectrogram import compute_logmel_spectrogram
from src.features.extraction import extract_track_features

router = APIRouter(prefix="/api", tags=["Analysis"])


@router.post("/analyze", response_model=AnalysisResponse)
def analyze_audio(request: Request, file: UploadFile = File(...)) -> AnalysisResponse:
    """Analyze an uploaded MP3 audio file.

    Synchronously executed in FastAPI's worker threadpool to avoid blocking event loop.
    Performs real decode validation, extracts downsampled waveforms and spectrograms,
    computes acoustic metrics, and classifies genre using GenreCNN.
    """
    # 1. Stream upload to random temp file, enforcing byte and duration caps + real decode
    temp_path, filename = validate_and_save_upload(file, request=request)

    try:
        # 2. Check model service availability
        genre_svc = get_genre_service()
        if not genre_svc.is_loaded():
            raise HTTPException(
                status_code=503,
                detail=f"Model unavailable: {genre_svc.get_load_error() or 'Checkpoint not loaded'}. Please train the model first.",
            )

        # 3. Read duration and waveform
        try:
            raw_duration = float(get_audio_duration(temp_path))
        except Exception:
            raw_duration = 30.0

        # Load audio (sr=22050, mono, duration <= 30.0 s, unpadded)
        y_raw = load_audio(temp_path, sr=22050, duration=30.0, mono=True)

        # Standardize for CNN: pad to exactly 30s if audio is shorter
        target_cnn_samples = 22050 * 30
        if len(y_raw) < target_cnn_samples:
            pad_len = target_cnn_samples - len(y_raw)
            y_cnn = np.pad(y_raw, (0, pad_len), mode="constant", constant_values=0.0)
        else:
            y_cnn = y_raw[:target_cnn_samples]

        # Use unpadded waveform for visualization
        y_viz = y_raw

        # 4. Waveform downsampling (<= 1000 points)
        waveform_points = downsample_waveform(y_viz, max_points=1000)

        # 5. Spectrogram generation & downsampling (<= 128x300)
        raw_logmel = compute_logmel_spectrogram(
            y_viz,
            sr=22050,
            n_fft=2048,
            hop_length=512,
            n_mels=128,
        )
        spectrogram_matrix = downsample_spectrogram(raw_logmel, max_freq=128, max_time=300)

        # 6. Acoustic features (chroma, RMS, estimated BPM)
        acoustic_feats = extract_acoustic_features(y_viz, sr=22050)

        # 7. Song similarity search (PLAN §12)
        similarity_svc = get_similarity_service()
        similar_tracks: List[SimilarTrack] = []
        similarity_status = "unavailable"
        identical_catalog_track_excluded = False

        if similarity_svc.is_available():
            try:
                # Extract canonical 51 features for nearest neighbor retrieval
                track_51_feats = extract_track_features(y_cnn, sr=22050)
                sim_res = similarity_svc.query_similar(track_51_feats, top_k=5, filter_exact_matches=True)
                similarity_status = sim_res["status"]
                similar_tracks = [SimilarTrack(**t) for t in sim_res.get("similar_tracks", [])]
                identical_catalog_track_excluded = bool(sim_res.get("identical_catalog_track_excluded", False))
            except Exception as e:
                similarity_status = "unavailable"
        else:
            similarity_status = "unavailable"

        # 8. Genre classification
        genre_result = genre_svc.predict(y_cnn, sr=22050)

        # 9. Compile academic warnings
        warnings: List[str] = [
            "BPM is an algorithmic estimate; may represent half-time or double-time.",
            "Genre values are model scores, not ground truth.",
        ]

        # Padding warning only emitted when padding shortfall exceeds 0.5 s (raw_duration = min(decoded duration, 30))
        effective_dur = min(raw_duration, 30.0)
        padding_duration = 30.0 - effective_dur
        if padding_duration > 0.5:
            warnings.append(
                f"Audio ({raw_duration:.1f} s) was zero-padded to 30.0 s for standardized CNN classification."
            )
        elif raw_duration > 30.0:
            warnings.append(
                f"Audio duration is {raw_duration:.1f} s; the first 30.0 s were analyzed for standardized CNN classification."
            )

        if identical_catalog_track_excluded:
            warnings.append("An identical catalog track was excluded.")

        if genre_result["is_uncertain"]:
            warnings.append(
                "Prediction uncertainty detected: model score is below threshold or margin between top genres is narrow."
            )
        if similarity_status == "unavailable":
            warnings.append(
                "Similarity engine unavailable: catalog index has not been generated on this instance."
            )

        return AnalysisResponse(
            analysis_id=str(uuid.uuid4()),
            filename=filename,
            duration_seconds=round(raw_duration, 2),
            sample_rate=22050,
            genre=GenreResult(**genre_result),
            acoustic_features=AcousticFeatures(**acoustic_feats),
            visualizations=Visualizations(
                waveform=waveform_points,
                spectrogram=spectrogram_matrix,
                waveform_points=len(waveform_points),
                spectrogram_shape=[
                    len(spectrogram_matrix),
                    len(spectrogram_matrix[0]) if spectrogram_matrix else 0,
                ],
            ),
            similar_tracks=similar_tracks,
            similarity_status=similarity_status,
            identical_catalog_track_excluded=identical_catalog_track_excluded,
            warnings=warnings,
        )

    finally:
        # Guaranteed cleanup of temporary upload file
        if temp_path.exists():
            try:
                temp_path.unlink()
            except OSError:
                pass
