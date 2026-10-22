export interface HealthResponse {
  status: "ok" | "unavailable" | string;
  model_loaded: boolean;
  version: string;
  device: string;
  detail?: string | null;
}

export interface GenreScore {
  label: string;
  score: number;
}

export interface GenreResult {
  model: string;
  top_label: string;
  confidence: number;
  is_uncertain: boolean;
  status: string;
  scores: GenreScore[];
}

export interface AcousticFeatures {
  tempo_bpm_estimate: number;
  bpm_label: string;
  rms_mean: number;
  rms_std: number;
  chroma: number[];
  chroma_labels: string[];
}

export interface Visualizations {
  waveform: number[];
  spectrogram: number[][];
  waveform_points: number;
  spectrogram_shape: number[];
}

export interface SimilarTrack {
  track_id: number;
  title: string;
  artist: string;
  genre: string;
  feature_similarity_score: number;
  explanation: string;
}

export interface AnalysisResponse {
  analysis_id: string;
  filename: string;
  duration_seconds: number;
  sample_rate: number;
  genre: GenreResult;
  acoustic_features: AcousticFeatures;
  visualizations: Visualizations;
  similar_tracks?: SimilarTrack[];
  similarity_status?: string;
  identical_catalog_track_excluded?: boolean;
  warnings: string[];
}

export interface ModelsResponse {
  model_name: string;
  status: string;
  architecture: string;
  num_classes: number;
  class_names: string[];
  checkpoint_version: string;
  sample_rate: number;
  preprocessing_params: Record<string, unknown>;
  normalization_stats: Record<string, number>;
  max_duration_seconds: number;
  max_upload_mb: number;
  allowed_formats: string[];
}

export interface MetricPerClass {
  precision: number;
  recall: number;
  "f1-score": number;
  support: number;
}

export interface ClassificationReport {
  [className: string]: MetricPerClass | number | Record<string, number>;
  accuracy: number;
  "macro avg": MetricPerClass;
  "weighted avg": MetricPerClass;
}

export interface CNNValidationMetrics {
  timestamp: string;
  status?: string;
  model_architecture: string;
  num_classes: number;
  class_names: string[];
  best_epoch: number;
  epochs_run?: number;
  epochs_configured?: number;
  best_validation_macro_f1: number;
  classification_report: ClassificationReport;
  training_history: {
    train_loss: number[];
    val_loss: number[];
    val_macro_f1: number[];
    val_accuracy: number[];
  };
  num_train_tracks: number;
  num_val_tracks: number;
  note: string;
}

export interface BaselineValidationMetrics {
  timestamp: string;
  model_type: string;
  random_state: number;
  best_params: {
    n_estimators: number;
    max_depth: number | null;
  };
  validation_macro_f1: number;
  validation_accuracy: number;
  validation_macro_precision: number;
  validation_macro_recall: number;
  classification_report: ClassificationReport;
  top_feature_importances: Record<string, number>;
}

export interface ConfusedPair {
  true_class: string;
  predicted_class: string;
  count: number;
  description: string;
}

export interface MisclassifiedExample {
  track_id: number;
  true_label: string;
  predicted_label: string;
  top_3_scores: Array<{ label: string; score: number }>;
}

export interface ModelTestStats {
  accuracy: number;
  macro_f1: number;
  bootstrap_ci_95?: {
    accuracy: [number, number];
    macro_f1: [number, number];
  };
  classification_report?: ClassificationReport;
  most_confused_pairs?: ConfusedPair[];
  misclassified_examples_sample?: MisclassifiedExample[];
}

export interface TestMetrics {
  timestamp: string;
  note: string;
  git?: {
    commit_hash: string;
    is_dirty: boolean;
    dirty_files?: string[];
  };
  hardware?: {
    platform?: string;
    processor?: string;
    device?: string;
    device_name?: string;
    cuda_device_name?: string;
  };
  package_versions?: Record<string, string>;
  sha256_checksums?: Record<string, string>;
  split_counts?: {
    training: number;
    validation: number;
    test: number;
  };
  evaluated_test_tracks_count: number;
  exclusions_count: number;
  cnn_training_metadata?: {
    status: string;
    best_epoch: number;
  };
  random_forest: ModelTestStats;
  cnn: ModelTestStats;
  statistical_comparison?: {
    difference_cnn_minus_rf?: {
      accuracy_difference_ci_95: [number, number];
      mean_accuracy_difference: number;
      paired_bootstrap_p_value: number;
    };
    mcnemar_test?: {
      contingency_table: {
        both_correct: number;
        cnn_correct_rf_incorrect: number;
        rf_correct_cnn_incorrect: number;
        both_incorrect: number;
      };
      statistic: number;
      p_value: number;
      exact_binomial_p_value: number;
    };
  };
  latency_benchmark?: {
    device?: string;
    hardware_name?: string;
    n_samples_measured?: number;
    median_decode_ms?: number;
    median_feature_extraction_ms?: number;
    median_rf_inference_ms?: number;
    median_cnn_forward_ms?: number;
  };
}

export interface MetricsResponse {
  status: string;
  cnn_status?: string;
  cnn_validation?: CNNValidationMetrics | null;
  baseline_validation?: BaselineValidationMetrics | null;
  manifest_summary?: {
    manifest_available: boolean;
    manifest_path?: string;
  } | null;
  run_status?: string | null;
  epochs_run?: number | null;
  epochs_configured?: number | null;
  test_status?: string;
  test_metrics?: TestMetrics | null;
  figures_available?: string[];
  eda_summary?: Record<string, any> | null;
}

