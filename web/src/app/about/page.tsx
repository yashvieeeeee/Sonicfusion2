"use client";

import React, { useEffect, useState } from "react";
import {
  AlertTriangle,
  BookOpen,
  CheckCircle,
  Cpu,
  Database,
  ExternalLink,
  Layers,
  Radio,
  Sliders,
  Terminal,
} from "lucide-react";
import { getModels } from "@/lib/api";
import { ModelsResponse } from "@/types/api";

export default function AboutPage() {
  const [modelInfo, setModelInfo] = useState<ModelsResponse | null>(null);
  const [loading, setLoading] = useState<boolean>(true);

  useEffect(() => {
    async function loadInfo() {
      try {
        const res = await getModels();
        setModelInfo(res);
      } catch (err) {
        console.warn("Could not fetch models info:", err);
      } finally {
        setLoading(false);
      }
    }
    loadInfo();
  }, []);

  return (
    <div className="flex-1 py-8 sm:py-12 px-4 sm:px-6 lg:px-8 max-w-5xl mx-auto w-full space-y-8">
      {/* Page Header */}
      <div className="border-b border-slate-800 pb-6">
        <div className="flex items-center gap-2 mb-2">
          <span className="text-xs uppercase tracking-wider font-semibold text-indigo-400">
            PLAN §14 Product Specification
          </span>
          <span className="text-[10px] font-mono px-2 py-0.5 rounded bg-slate-800 text-slate-300 border border-slate-700">
            Model Card v1.0
          </span>
        </div>
        <h1 className="text-3xl sm:text-4xl font-extrabold text-white tracking-tight">
          SonicScope AI Model Card & Architecture
        </h1>
        <p className="text-sm text-slate-400 mt-2 leading-relaxed">
          Detailed technical documentation on dataset provenance, signal processing specifications,
          neural network architecture, uncertainty policies, and system limitations.
        </p>
      </div>

      {/* Grid of Sections */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
        {/* Dataset Provenance */}
        <div className="rounded-2xl border border-slate-800 bg-slate-900/80 p-6 backdrop-blur-md">
          <div className="flex items-center gap-3 mb-4">
            <div className="flex h-9 w-9 items-center justify-center rounded-xl bg-indigo-500/10 text-indigo-400 border border-indigo-500/20">
              <Database className="h-5 w-5" />
            </div>
            <h3 className="text-base font-bold text-white">Dataset Provenance</h3>
          </div>
          <p className="text-xs text-slate-400 leading-relaxed mb-4">
            Trained and evaluated on the official <strong>Free Music Archive (FMA) Small</strong> dataset
            published by Defferrard et al. (ISMIR 2017).
          </p>
          <ul className="space-y-2 text-xs text-slate-300">
            <li className="flex justify-between py-1 border-b border-slate-800/80">
              <span className="text-slate-400">Total Track Corpus:</span>
              <span className="font-mono text-white">8,000 tracks (30s clips)</span>
            </li>
            <li className="flex justify-between py-1 border-b border-slate-800/80">
              <span className="text-slate-400">Genre Classes:</span>
              <span className="font-mono text-indigo-400">8 balanced genres</span>
            </li>
            <li className="flex justify-between py-1 border-b border-slate-800/80">
              <span className="text-slate-400">Split Assignment:</span>
              <span className="font-mono text-white">Official song-level splits</span>
            </li>
            <li className="flex justify-between py-1">
              <span className="text-slate-400">Licensing:</span>
              <span className="font-mono text-slate-400">Creative Commons (FMA)</span>
            </li>
          </ul>
        </div>

        {/* 2D CNN Architecture */}
        <div className="rounded-2xl border border-slate-800 bg-slate-900/80 p-6 backdrop-blur-md">
          <div className="flex items-center gap-3 mb-4">
            <div className="flex h-9 w-9 items-center justify-center rounded-xl bg-cyan-500/10 text-cyan-400 border border-cyan-500/20">
              <Cpu className="h-5 w-5" />
            </div>
            <h3 className="text-base font-bold text-white">GenreCNN Architecture</h3>
          </div>
          <p className="text-xs text-slate-400 leading-relaxed mb-4">
            A lightweight 3-block 2D convolutional neural network designed for rapid CPU inference.
          </p>
          <div className="space-y-1.5 font-mono text-[11px] text-slate-300 p-3 rounded-xl bg-slate-950/70 border border-slate-800">
            <div>Input: 1 x 128 x ~431 log-mel spectrogram</div>
            <div className="text-cyan-400">&bull; Block 1: Conv2d(1, 16, 3) + BN + ReLU + MaxPool(2)</div>
            <div className="text-indigo-400">&bull; Block 2: Conv2d(16, 32, 3) + BN + ReLU + MaxPool(2)</div>
            <div className="text-purple-400">&bull; Block 3: Conv2d(32, 64, 3) + BN + ReLU + MaxPool(2)</div>
            <div className="text-slate-400">&bull; Head: AdaptiveAvgPool2d(1) + Dropout(0.3) + Linear(64, 8)</div>
          </div>
        </div>

        {/* Signal Preprocessing Contract */}
        <div className="rounded-2xl border border-slate-800 bg-slate-900/80 p-6 backdrop-blur-md">
          <div className="flex items-center gap-3 mb-4">
            <div className="flex h-9 w-9 items-center justify-center rounded-xl bg-emerald-500/10 text-emerald-400 border border-emerald-500/20">
              <Sliders className="h-5 w-5" />
            </div>
            <h3 className="text-base font-bold text-white">Signal Preprocessing</h3>
          </div>
          <ul className="space-y-2 text-xs text-slate-300">
            <li className="flex justify-between py-1 border-b border-slate-800/80">
              <span className="text-slate-400">Sampling Rate:</span>
              <span className="font-mono text-emerald-400">22,050 Hz Mono</span>
            </li>
            <li className="flex justify-between py-1 border-b border-slate-800/80">
              <span className="text-slate-400">STFT Window (n_fft):</span>
              <span className="font-mono text-white">2,048 samples</span>
            </li>
            <li className="flex justify-between py-1 border-b border-slate-800/80">
              <span className="text-slate-400">Hop Length:</span>
              <span className="font-mono text-white">512 samples</span>
            </li>
            <li className="flex justify-between py-1 border-b border-slate-800/80">
              <span className="text-slate-400">Mel Frequency Bands:</span>
              <span className="font-mono text-white">128 bins (0 - 11,025 Hz)</span>
            </li>
            <li className="flex justify-between py-1">
              <span className="text-slate-400">Window Segmentation:</span>
              <span className="font-mono text-white">10.0s (~431 frames / window)</span>
            </li>
          </ul>
        </div>

        {/* Uncertainty Thresholds */}
        <div className="rounded-2xl border border-slate-800 bg-slate-900/80 p-6 backdrop-blur-md">
          <div className="flex items-center gap-3 mb-4">
            <div className="flex h-9 w-9 items-center justify-center rounded-xl bg-amber-500/10 text-amber-400 border border-amber-500/20">
              <AlertTriangle className="h-5 w-5" />
            </div>
            <h3 className="text-base font-bold text-white">Uncertainty Logic</h3>
          </div>
          <p className="text-xs text-slate-400 leading-relaxed mb-4">
            Softmax outputs represent model distribution scores, not true calibrated probabilities.
            To prevent false confidence, predictions are marked <strong>uncertain</strong> when:
          </p>
          <div className="space-y-2 text-xs">
            <div className="p-2.5 rounded-lg bg-amber-950/30 border border-amber-800/40 text-amber-200">
              <strong>1. Low Top Score:</strong> Top confidence is below <strong>40% (0.40)</strong>.
            </div>
            <div className="p-2.5 rounded-lg bg-amber-950/30 border border-amber-800/40 text-amber-200">
              <strong>2. Narrow Top Margin:</strong> The gap between rank #1 and rank #2 is less than <strong>15% (0.15)</strong>.
            </div>
          </div>
        </div>
      </div>

      {/* Target Genre Classes List */}
      <div className="rounded-2xl border border-slate-800 bg-slate-900/80 p-6">
        <h3 className="text-sm font-semibold uppercase tracking-wider text-slate-300 mb-4">
          Trained 8-Genre Target Taxonomy (FMA Small)
        </h3>
        <div className="grid grid-cols-2 sm:grid-cols-4 gap-3">
          {(
            modelInfo?.class_names || [
              "Electronic",
              "Experimental",
              "Folk",
              "Hip-Hop",
              "Instrumental",
              "International",
              "Pop",
              "Rock",
            ]
          ).map((name) => (
            <div
              key={name}
              className="p-3 rounded-xl border border-slate-800 bg-slate-950/60 text-center font-semibold text-slate-200 text-xs hover:border-indigo-500/40 transition-colors"
            >
              {name}
            </div>
          ))}
        </div>
      </div>

      {/* Quantitative Performance Summary */}
      <div className="rounded-2xl border border-slate-800 bg-slate-900/80 p-6">
        <div className="flex items-center gap-3 mb-4">
          <div className="flex h-9 w-9 items-center justify-center rounded-xl bg-teal-500/10 text-teal-400 border border-teal-500/20">
            <Radio className="h-5 w-5" />
          </div>
          <h3 className="text-base font-bold text-white">Quantitative Performance Summary</h3>
        </div>
        <div className="grid grid-cols-1 sm:grid-cols-3 gap-4 text-xs">
          <div className="p-4 rounded-xl bg-slate-950/60 border border-slate-800/80">
            <span className="text-slate-400 block mb-1">Random Forest Baseline</span>
            <div className="text-lg font-mono font-bold text-emerald-400">0.5093 (50.9%)</div>
            <span className="text-[11px] text-slate-500">Validation Macro-F1 (52.38% Acc)</span>
          </div>
          <div className="p-4 rounded-xl bg-slate-950/60 border border-slate-800/80">
            <span className="text-slate-400 block mb-1">GenreCNN (2D PyTorch)</span>
            <div className="text-lg font-mono font-bold text-cyan-400">23,928 Params</div>
            <span className="text-[11px] text-slate-500">3-Block CNN (AdamW, Batch 16)</span>
          </div>
          <div className="p-4 rounded-xl bg-slate-950/60 border border-slate-800/80">
            <span className="text-slate-400 block mb-1">Similarity Engine</span>
            <div className="text-lg font-mono font-bold text-purple-400">0.3893 (38.9%)</div>
            <span className="text-[11px] text-slate-500">Mean Precision@5 (2k Catalog)</span>
          </div>
        </div>
      </div>

      {/* Ethical Considerations & Disclosures */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
        <div className="rounded-2xl border border-slate-800 bg-slate-900/60 p-6">
          <h3 className="text-sm font-semibold uppercase tracking-wider text-slate-300 mb-2">
            Dataset Errata &amp; Disclosures
          </h3>
          <p className="text-xs text-slate-400 leading-relaxed">
            FMA Small split assignments are defined at the individual track level rather than artist level. Consequently, tracks by the same artist appear across both training and validation splits. This is a documented characteristic of the benchmark corpus (Defferrard et al., 2017).
          </p>
        </div>
        <div className="rounded-2xl border border-slate-800 bg-slate-900/60 p-6">
          <h3 className="text-sm font-semibold uppercase tracking-wider text-slate-300 mb-2">
            Privacy &amp; Data Governance
          </h3>
          <p className="text-xs text-slate-400 leading-relaxed">
            All user audio uploads are processed strictly in ephemeral memory and temporary disk space. Temporary files are immediately unlinked upon completion of the request. SonicScope does not store uploaded audio, retain user telemetry, or redistribute copyrighted media.
          </p>
        </div>
      </div>

      {/* Limitations and Non-Goals (§20) */}
      <div className="rounded-2xl border border-slate-800/80 bg-slate-900/40 p-6">
        <h3 className="text-sm font-semibold uppercase tracking-wider text-slate-400 mb-2">
          Documented Academic Non-Goals (PLAN §20)
        </h3>
        <p className="text-xs text-slate-500 leading-relaxed">
          SonicScope AI is built strictly for scientific research comparing feature-engineered machine learning with convolutional deep learning on music audio. It explicitly excludes: commercial audio fingerprinting (Shazam), copyrighted song recognition, exact artist identification, music generation, stem separation, and public audio streaming.
        </p>
      </div>
    </div>
  );
}
