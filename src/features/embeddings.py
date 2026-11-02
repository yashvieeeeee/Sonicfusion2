"""Pretrained audio embedding extraction using MERT-v1-95M.

Implements 5-second non-overlapping windowing across 30-second clips:
- Model: m-a-p/MERT-v1-95M via Hugging Face transformers.
- Audio contract: 24,000 Hz mono waveform.
- 6 non-overlapping windows per 30s track (window_sec=5.0, hop_sec=5.0).
- Per-window embedding shape: (6, 768).
- Per-track mean embedding shape: (768,).
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import torch
from transformers import AutoModel, Wav2Vec2FeatureExtractor

from src.audio.loading import load_audio

MERT_MODEL_ID = "m-a-p/MERT-v1-95M"
MODEL_ALIAS = "MERT-v1-95M-5s"
DEFAULT_SAMPLE_RATE = 24000
DEFAULT_DURATION_SEC = 30.0
DEFAULT_WINDOW_SEC = 5.0
DEFAULT_HOP_SEC = 5.0


def segment_audio_windows(
    y: np.ndarray,
    sr: int = DEFAULT_SAMPLE_RATE,
    window_sec: float = DEFAULT_WINDOW_SEC,
    hop_sec: float = DEFAULT_HOP_SEC,
    target_duration: float = DEFAULT_DURATION_SEC,
) -> np.ndarray:
    """Segment 1D audio into fixed-duration windows with zero-padding/truncation.

    Args:
        y: 1D audio array.
        sr: Audio sample rate in Hz (default 24000).
        window_sec: Window length in seconds (default 5.0).
        hop_sec: Hop length in seconds between window starts (default 5.0, no overlap).
        target_duration: Target total audio duration in seconds (default 30.0).

    Returns:
        2D numpy array of shape (num_windows, window_samples).
    """
    target_samples = int(target_duration * sr)
    window_samples = int(window_sec * sr)
    hop_samples = int(hop_sec * sr)

    # Pad with zeros or truncate to exactly target_samples
    if len(y) < target_samples:
        pad_width = target_samples - len(y)
        y = np.pad(y, (0, pad_width), mode="constant")
    elif len(y) > target_samples:
        y = y[:target_samples]

    max_start = target_samples - window_samples
    starts = np.arange(0, max_start + 1, hop_samples)

    windows = [y[s : s + window_samples] for s in starts]
    return np.stack(windows, axis=0).astype(np.float32)


class AudioEmbedder:
    """Extracts pretrained audio representations using MERT-v1-95M."""

    def __init__(
        self,
        model_name_or_path: str = MERT_MODEL_ID,
        device: Optional[str] = None,
        sample_rate: int = DEFAULT_SAMPLE_RATE,
    ):
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        self.model_name = model_name_or_path
        self.sr = sample_rate

        self.processor = Wav2Vec2FeatureExtractor.from_pretrained(
            self.model_name,
            trust_remote_code=True,
        )
        self.model = AutoModel.from_pretrained(
            self.model_name,
            trust_remote_code=True,
        )
        self.model.to(self.device)
        self.model.eval()

    def embed_windows(
        self,
        windows: np.ndarray,
        batch_size: int = 6,
    ) -> np.ndarray:
        """Extract embeddings for an array of windows of shape (N, window_samples).

        Args:
            windows: 2D array of shape (num_windows, window_samples).
            batch_size: Batch size for model inference.

        Returns:
            2D array of shape (num_windows, 768) representing mean-pooled window embeddings.
        """
        all_embeddings: List[np.ndarray] = []
        num_windows = len(windows)

        for i in range(0, num_windows, batch_size):
            batch = [w for w in windows[i : i + batch_size]]
            inputs = self.processor(
                batch,
                sampling_rate=self.sr,
                return_tensors="pt",
                padding=True,
            )
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

            with torch.no_grad():
                outputs = self.model(**inputs)
                # Mean pool across time dimension (dim=1)
                emb = outputs.last_hidden_state.mean(dim=1)
                all_embeddings.append(emb.cpu().numpy().astype(np.float32))

        return np.concatenate(all_embeddings, axis=0)

    def extract_track_embeddings(
        self,
        audio_path_or_array: Union[str, Path, np.ndarray],
        window_sec: float = DEFAULT_WINDOW_SEC,
        hop_sec: float = DEFAULT_HOP_SEC,
        target_duration: float = DEFAULT_DURATION_SEC,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Extract both per-window and track-mean embeddings.

        Args:
            audio_path_or_array: Path to audio file or 1D audio waveform array.
            window_sec: Window length in seconds (default 5.0).
            hop_sec: Hop length in seconds (default 5.0, no overlap).
            target_duration: Total target clip length in seconds (default 30.0).

        Returns:
            (window_embeddings, track_mean_embedding)
            - window_embeddings: shape (num_windows, 768) -> (6, 768)
            - track_mean_embedding: shape (768,)
        """
        if isinstance(audio_path_or_array, (str, Path)):
            y = load_audio(
                audio_path_or_array,
                sr=self.sr,
                duration=target_duration,
                mono=True,
            )
        else:
            y = np.asarray(audio_path_or_array, dtype=np.float32)

        windows = segment_audio_windows(
            y,
            sr=self.sr,
            window_sec=window_sec,
            hop_sec=hop_sec,
            target_duration=target_duration,
        )

        window_embeddings = self.embed_windows(windows, batch_size=len(windows))
        track_mean_embedding = window_embeddings.mean(axis=0).astype(np.float32)

        return window_embeddings, track_mean_embedding
