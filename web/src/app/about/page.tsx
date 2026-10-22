"use client";

import React, { useEffect, useState } from "react";
import Link from "next/link";
import {
  Activity,
  AlertTriangle,
  ArrowRight,
  BarChart2,
  BookOpen,
  CheckCircle2,
  Cpu,
  Database,
  ExternalLink,
  FileText,
  Info,
  Layers,
  Scale,
  ShieldCheck,
  Sliders,
  TrendingUp,
} from "lucide-react";
import { getMetrics, getModels } from "@/lib/api";
import { MetricsResponse, ModelsResponse } from "@/types/api";
import GenreTopologyGraph from "@/components/GenreTopologyGraph";

export default function AboutPage() {
  const [modelInfo, setModelInfo] = useState<ModelsResponse | null>(null);
  const [metrics, setMetrics] = useState<MetricsResponse | null>(null);
  const [loading, setLoading] = useState<boolean>(true);

  useEffect(() => {
    Promise.all([getModels(), getMetrics()])
      .then(([modelsData, metricsData]) => {
        setModelInfo(modelsData);
        setMetrics(metricsData);
      })
      .catch((err) => {
        console.warn("Could not fetch model card data:", err);
      })
      .finally(() => {
        setLoading(false);
      });
  }, []);

  const prep = modelInfo?.preprocessing_params || {};
  const testData = metrics?.test_metrics;

  const cnnAcc = testData?.cnn?.accuracy ?? 0.5075;
  const cnnPrec = testData?.cnn?.classification_report?.["macro avg"]?.precision ?? 0.5131;
  const cnnRec = testData?.cnn?.classification_report?.["macro avg"]?.recall ?? 0.5075;
  const cnnF1 = testData?.cnn?.macro_f1 ?? 0.4985;

  const rfAcc = testData?.random_forest?.accuracy ?? 0.4375;
  const rfPrec = testData?.random_forest?.classification_report?.["macro avg"]?.precision ?? 0.4369;
  const rfRec = testData?.random_forest?.classification_report?.["macro avg"]?.recall ?? 0.4375;
  const rfF1 = testData?.random_forest?.macro_f1 ?? 0.4357;

  const evaluatedCount = testData?.evaluated_test_tracks_count ?? 800;

  return (
    <div className="max-w-5xl mx-auto px-4 sm:px-6 lg:px-8 py-8 space-y-8">
      {/* Page Header */}
      <div className="pb-6 border-b border-[#E3DDD4]">
        <div className="flex items-center gap-2 mb-1.5">
          <span className="text-[10px] font-mono uppercase tracking-wider font-semibold text-[#76556F] bg-[#F0E8EE] px-2 py-0.5 rounded">
            Scientific Documentation
          </span>
          <span className="text-[10px] font-mono text-[#6B626E]">
            Sonic Fusion v1.2
          </span>
        </div>
        <h1 className="text-3xl sm:text-4xl font-serif font-bold text-[#27232B] tracking-tight">
          Model Card &amp; System Provenance
        </h1>
        <p className="text-sm text-[#6B626E] mt-1 max-w-3xl leading-relaxed">
          Comprehensive technical specification documenting dataset provenance, quantitative model performance, acoustic signal transformations, 2D CNN topology, baseline configurations, and system boundary policies.
        </p>
      </div>

      {/* Row 1: Dataset Provenance (Left) beside Model Performance (Right) */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-6 items-stretch">
        {/* Left: Dataset Provenance Box (Preserved exactly as requested) */}
        <div className="editorial-panel p-6 flex flex-col justify-between">
          <div>
            <div className="flex items-center gap-2.5 pb-3 border-b border-[#E3DDD4] mb-4">
              <Database className="h-4 w-4 text-[#76556F]" />
              <div>
                <h3 className="text-sm font-semibold text-[#27232B]">Dataset Provenance</h3>
                <p className="text-[11px] text-[#6B626E]">Free Music Archive (FMA Small)</p>
              </div>
            </div>
            <p className="text-xs text-[#6B626E] leading-relaxed mb-4">
              Audited subset of the official Free Music Archive benchmark published by Defferrard et al. (ISMIR 2017). Strictly partitioned to eliminate artist leakage across evaluation splits.
            </p>
            <ul className="space-y-2 text-xs">
              <li className="flex justify-between py-1 border-b border-[#E3DDD4]/60">
                <span className="text-[#6B626E]">Total Track Corpus:</span>
                <span className="font-mono text-[#27232B]">8,000 raw clips (30s MP3s)</span>
              </li>
              <li className="flex justify-between py-1 border-b border-[#E3DDD4]/60">
                <span className="text-[#6B626E]">Decodable Tracks:</span>
                <span className="font-mono text-[#27232B]">7,994 tracks (99.93%)</span>
              </li>
              <li className="flex justify-between py-1 border-b border-[#E3DDD4]/60">
                <span className="text-[#6B626E]">Official Partition Splits:</span>
                <span className="font-mono text-[#76556F] font-semibold">6,394 train / 800 val / 800 test</span>
              </li>
              <li className="flex justify-between py-1 border-b border-[#E3DDD4]/60">
                <span className="text-[#6B626E]">Audited Exclusions:</span>
                <span className="font-mono text-[#D77C63]">6 excluded (corrupt or &lt;25s)</span>
              </li>
              <li className="flex justify-between py-1">
                <span className="text-[#6B626E]">Artist Overlap:</span>
                <span className="font-mono text-[#76556F] font-semibold">0 tracks (zero leakage)</span>
              </li>
            </ul>
          </div>
          <div className="pt-4 border-t border-[#E3DDD4] mt-4 text-[11px] font-mono text-[#8E8692]">
            Licensing: Creative Commons By-NC-SA (FMA)
          </div>
        </div>

        {/* Right: New Dedicated Model Performance Panel */}
        <div className="editorial-panel p-6 flex flex-col justify-between">
          <div>
            <div className="flex items-center justify-between pb-3 border-b border-[#E3DDD4] mb-4">
              <div className="flex items-center gap-2.5">
                <ShieldCheck className="h-4 w-4 text-[#76556F]" />
                <div>
                  <h3 className="text-sm font-semibold text-[#27232B]">Model Performance</h3>
                  <p className="text-[11px] text-[#6B626E]">
                    Held-Out Test Split ({evaluatedCount} Tracks)
                  </p>
                </div>
              </div>
              <span className="text-[10px] font-mono px-2 py-0.5 rounded-full bg-[#F0E8EE] text-[#76556F] font-bold border border-[#E3DDD4]">
                Verified
              </span>
            </div>

            <p className="text-xs text-[#6B626E] leading-relaxed mb-3">
              Unbiased held-out test evaluation with zero artist overlap across splits. Metrics compare the deep 2D CNN against the classical Random Forest baseline.
            </p>

            {/* Compact Comparative Metrics Table */}
            <div className="space-y-2.5">
              <div className="grid grid-cols-3 gap-2 text-center text-xs pb-1 border-b border-[#E3DDD4]/60 text-[#8E8692] font-mono font-medium text-[10px] uppercase">
                <span className="text-left font-sans">Metric</span>
                <span>GenreCNN (2D)</span>
                <span>Random Forest</span>
              </div>

              {/* Accuracy Row */}
              <div className="grid grid-cols-3 gap-2 text-center text-xs items-center py-1 border-b border-[#E3DDD4]/40">
                <span className="text-left font-medium text-[#27232B]">Accuracy</span>
                <span className="font-mono font-bold text-[#76556F] bg-[#F0E8EE] py-0.5 rounded">
                  {(cnnAcc * 100).toFixed(2)}%
                </span>
                <span className="font-mono text-[#6B626E]">
                  {(rfAcc * 100).toFixed(2)}%
                </span>
              </div>

              {/* Macro F1 Row */}
              <div className="grid grid-cols-3 gap-2 text-center text-xs items-center py-1 border-b border-[#E3DDD4]/40">
                <span className="text-left font-medium text-[#27232B]">Macro F1</span>
                <span className="font-mono font-bold text-[#76556F] bg-[#F0E8EE] py-0.5 rounded">
                  {(cnnF1 * 100).toFixed(2)}%
                </span>
                <span className="font-mono text-[#6B626E]">
                  {(rfF1 * 100).toFixed(2)}%
                </span>
              </div>

              {/* Macro Precision */}
              <div className="grid grid-cols-3 gap-2 text-center text-xs items-center py-1 border-b border-[#E3DDD4]/40">
                <span className="text-left font-medium text-[#27232B]">Macro Precision</span>
                <span className="font-mono text-[#27232B] font-semibold">
                  {(cnnPrec * 100).toFixed(2)}%
                </span>
                <span className="font-mono text-[#6B626E]">
                  {(rfPrec * 100).toFixed(2)}%
                </span>
              </div>

              {/* Macro Recall */}
              <div className="grid grid-cols-3 gap-2 text-center text-xs items-center py-1 border-b border-[#E3DDD4]/40">
                <span className="text-left font-medium text-[#27232B]">Macro Recall</span>
                <span className="font-mono text-[#27232B] font-semibold">
                  {(cnnRec * 100).toFixed(2)}%
                </span>
                <span className="font-mono text-[#6B626E]">
                  {(rfRec * 100).toFixed(2)}%
                </span>
              </div>

              {/* Statistical Significance Note */}
              <div className="flex items-center justify-between text-[11px] pt-1 text-[#6B626E]">
                <span>McNemar Test:</span>
                <span className="font-mono text-[#76556F] font-semibold">
                  p = 0.00013 (p &lt; 0.001 ***)
                </span>
              </div>
            </div>
          </div>

          {/* Action Link: View Detailed Evaluation */}
          <div className="pt-4 border-t border-[#E3DDD4] mt-4 flex items-center justify-between">
            <span className="text-[10px] font-mono text-[#8E8692]">
              Evaluated on {evaluatedCount} test tracks
            </span>
            <Link
              href="/evaluation"
              className="inline-flex items-center gap-1.5 px-3 py-1.5 rounded-lg bg-[#76556F] hover:bg-[#583C52] text-white text-xs font-semibold transition-all shadow-2xs group"
            >
              <span>View Detailed Evaluation</span>
              <ArrowRight className="h-3.5 w-3.5 text-[#D77C63] group-hover:translate-x-0.5 transition-transform" />
            </Link>
          </div>
        </div>
      </div>

      {/* Row 2: 2D CNN Architecture & Classical Baseline Pipeline */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
        {/* 2D CNN Architecture */}
        <div className="editorial-panel p-6">
          <div className="flex items-center gap-2.5 pb-3 border-b border-[#E3DDD4] mb-4">
            <Cpu className="h-4 w-4 text-[#76556F]" />
            <div>
              <h3 className="text-sm font-semibold text-[#27232B]">GenreCNN Topology</h3>
              <p className="text-[11px] text-[#6B626E]">3-Block 2D Convolutional Neural Network</p>
            </div>
          </div>
          <p className="text-xs text-[#6B626E] leading-relaxed mb-4">
            Custom lightweight 2D CNN optimized for real-time CPU forward passes, processing normalized 128-band log-mel power spectrogram tensors.
          </p>
          <div className="p-3 rounded-lg bg-[#FAF7F2] border border-[#E3DDD4] font-mono text-[11px] text-[#27232B] space-y-1.5">
            <div>Input: 1 &times; 128 &times; ~431 normalized log-mel tensor</div>
            <div className="text-[#76556F]">&bull; Block 1: Conv2d(1, 16, 3, p=1) + BatchNorm + ReLU + MaxPool(2)</div>
            <div className="text-[#76556F]">&bull; Block 2: Conv2d(16, 32, 3, p=1) + BatchNorm + ReLU + MaxPool(2)</div>
            <div className="text-[#76556F]">&bull; Block 3: Conv2d(32, 64, 3, p=1) + BatchNorm + ReLU + MaxPool(2)</div>
            <div className="text-[#6B626E]">&bull; Classifier: AdaptiveAvgPool2d(1) + Dropout(0.3) + Linear(64, 8)</div>
            <div className="text-[#D77C63] font-semibold pt-1 border-t border-[#E3DDD4]">
              Output: 8 raw logits &rarr; Softmax class distribution
            </div>
          </div>
        </div>

        {/* Audio Preprocessing & Classical Baseline */}
        <div className="editorial-panel p-6">
          <div className="flex items-center gap-2.5 pb-3 border-b border-[#E3DDD4] mb-4">
            <Sliders className="h-4 w-4 text-[#76556F]" />
            <div>
              <h3 className="text-sm font-semibold text-[#27232B]">Audio Preprocessing</h3>
              <p className="text-[11px] text-[#6B626E]">Deterministic STFT Transformation</p>
            </div>
          </div>
          <ul className="space-y-2 text-xs">
            <li className="flex justify-between py-1 border-b border-[#E3DDD4]/60">
              <span className="text-[#6B626E]">Standard Sample Rate:</span>
              <span className="font-mono text-[#27232B]">{modelInfo?.sample_rate || 22050} Hz Mono</span>
            </li>
            <li className="flex justify-between py-1 border-b border-[#E3DDD4]/60">
              <span className="text-[#6B626E]">STFT FFT Window (n_fft):</span>
              <span className="font-mono text-[#27232B]">{(prep as any).n_fft || 2048} samples</span>
            </li>
            <li className="flex justify-between py-1 border-b border-[#E3DDD4]/60">
              <span className="text-[#6B626E]">STFT Hop Length:</span>
              <span className="font-mono text-[#27232B]">{(prep as any).hop_length || 512} samples</span>
            </li>
            <li className="flex justify-between py-1 border-b border-[#E3DDD4]/60">
              <span className="text-[#6B626E]">Mel Filterbank Channels:</span>
              <span className="font-mono text-[#27232B]">{(prep as any).n_mels || 128} bins</span>
            </li>
            <li className="flex justify-between py-1">
              <span className="text-[#6B626E]">Normalization Scheme:</span>
              <span className="font-mono text-[#27232B]">Global train split Z-score</span>
            </li>
          </ul>
        </div>
      </div>

      {/* Uncertainty Policy & Limitations */}
      <div className="editorial-panel p-6 space-y-4">
        <div className="flex items-center gap-2.5 pb-3 border-b border-[#E3DDD4]">
          <AlertTriangle className="h-4 w-4 text-[#D6A75D]" />
          <h3 className="text-sm font-semibold text-[#27232B]">
            Documented System Limitations &amp; Uncertainty Policy
          </h3>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-2 gap-6 text-xs text-[#6B626E] leading-relaxed">
          <div className="space-y-2">
            <h4 className="font-semibold text-[#27232B]">Uncertainty Detection Thresholds</h4>
            <p>
              When the leading class softmax score is &lt; 40% or the margin over the runner-up is &lt; 5%, Sonic Fusion explicitly flags the prediction as <em>Uncertain / Hybrid Influences</em>. Softmax outputs must not be treated as absolute frequentist probabilities.
            </p>
          </div>
          <div className="space-y-2">
            <h4 className="font-semibold text-[#27232B]">Domain and Data Boundaries</h4>
            <p>
              GenreCNN was trained on 30-second studio/live excerpts from the Free Music Archive. Performance on non-western musical traditions, low-fidelity phone recordings, or speech-dominant audio will exhibit higher variance.
            </p>
          </div>
        </div>
      </div>

      {/* ========================================================================= */}
      {/*                            GENRE TOPOLOGY                                 */}
      {/* ========================================================================= */}
      <GenreTopologyGraph />
    </div>
  );
}
