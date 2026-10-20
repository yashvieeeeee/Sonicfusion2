"""Audio loading, validation, segmentation, and spectrogram processing."""

from src.audio.validation import (
    REJECT_NON_MP3,
    check_known_non_mp3_signatures,
    check_known_unsupported_signatures,
    get_candidate_demuxer,
    get_container_format,
    probe_audio_container,
    sanitize_path_str,
    verify_audio_file,
)
from src.audio.loading import (
    AudioDecodeError,
    bounded_decode_duration,
    get_audio_duration,
    load_audio,
    segment_audio,
)
from src.audio.spectrogram import (
    RunningStats,
    compute_logmel_spectrogram,
    normalize_spectrogram,
    to_cnn_tensor,
    save_norm_stats,
    load_norm_stats,
)

__all__ = [
    "REJECT_NON_MP3",
    "AudioDecodeError",
    "RunningStats",
    "bounded_decode_duration",
    "check_known_non_mp3_signatures",
    "check_known_unsupported_signatures",
    "get_audio_duration",
    "get_candidate_demuxer",
    "get_container_format",
    "probe_audio_container",
    "sanitize_path_str",
    "verify_audio_file",
    "load_audio",
    "segment_audio",
    "compute_logmel_spectrogram",
    "normalize_spectrogram",
    "to_cnn_tensor",
    "save_norm_stats",
    "load_norm_stats",
]

