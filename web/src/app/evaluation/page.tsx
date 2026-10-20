"use client";

import React, { useEffect, useState } from "react";
import {
  Activity,
  AlertCircle,
  BarChart2,
  CheckCircle2,
  Cpu,
  Database,
  FileText,
  LineChart,
  RefreshCw,
  Scale,
  ShieldCheck,
  TrendingUp,
  Zap,
} from "lucide-react";
import { getMetrics } from "@/lib/api";
import { MetricsResponse } from "@/types/api";

export default function EvaluationPage() {
  const [metrics, setMetrics] = useState<MetricsResponse | null>(null);
  const [loading, setLoading] = useState<boolean>(true);
  const [error, setError] = useState<string | null>(null);

  const fetchValidationMetrics = async () => {
    setLoading(true);
    setError(null);
    try {
      const data = await getMetrics();
      setMetrics(data);
    } catch (err: unknown) {
      const msg = err instanceof Error ? err.message : "Failed to load validation metrics";
      setError(msg);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchValidationMetrics();
  }, []);

  const cnn = metrics?.cnn_validation;
  const baseline = metrics?.baseline_validation;

  return (
    <div className="flex-1 py-8 sm:py-12 px-4 sm:px-6 lg:px-8 max-w-7xl mx-auto w-full space-y-8">
      {/* Page Header */}
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4 pb-6 border-b border-slate-800">
        <div>
          <div className="flex items-center gap-2 mb-1">
            <span className="text-xs uppercase tracking-wider font-semibold text-indigo-400">
              PLAN §16 Evaluation Protocol
            </span>
            <span className="text-[10px] font-mono px-2 py-0.5 rounded bg-emerald-950/80 text-emerald-400 border border-emerald-800">
              Live Verified Reports
            </span>
          </div>
          <h1 className="text-2xl sm:text-4xl font-extrabold text-white tracking-tight">
            Model Performance & Validation
          </h1>
          <p className="text-sm text-slate-400 mt-1 max-w-2xl">
            Empirical comparison between classical handcrafted feature engineering (Random Forest)
            and 2D log-mel spectrogram deep learning (GenreCNN) on held-out FMA Small tracks.
          </p>
        </div>

        <button
          onClick={fetchValidationMetrics}
          disabled={loading}
          className="flex items-center gap-2 px-4 py-2 rounded-xl border border-slate-700 bg-slate-800/80 hover:bg-slate-700 text-slate-200 text-xs font-semibold transition-colors shrink-0"
        >
          <RefreshCw className={`h-4 w-4 ${loading ? "animate-spin" : ""}`} />
          <span>Refresh Metrics</span>
        </button>
      </div>

      {loading ? (
        <div className="py-20 flex flex-col items-center justify-center text-center">
          <RefreshCw className="h-8 w-8 text-indigo-400 animate-spin mb-3" />
          <p className="text-sm text-slate-300 font-medium">
            Fetching validation metrics from /api/metrics...
          </p>
        </div>
      ) : error ? (
        <div className="p-5 rounded-2xl bg-rose-950/60 border border-rose-600/60 text-rose-200 text-xs flex items-start gap-3">
          <AlertCircle className="h-5 w-5 text-rose-400 shrink-0 mt-0.5" />
          <div>
            <strong className="font-semibold text-rose-300">
              Failed to load metrics:
            </strong>{" "}
            {error}
          </div>
        </div>
      ) : (
        <div className="space-y-8">
          {/* Key Metric Comparison Cards */}
          <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
            {/* Random Forest Baseline Card */}
            <div className="rounded-2xl border border-slate-800 bg-slate-900/80 p-6 backdrop-blur-md">
              <div className="flex items-center justify-between pb-4 border-b border-slate-800">
                <div className="flex items-center gap-3">
                  <div className="flex h-10 w-10 items-center justify-center rounded-xl bg-cyan-500/10 text-cyan-400 border border-cyan-500/20">
                    <BarChart2 className="h-5 w-5" />
                  </div>
                  <div>
                    <h3 className="text-base font-bold text-white">
                      Random Forest Baseline
                    </h3>
                    <p className="text-xs text-slate-400">
                      Handcrafted Summary Vectors (518 Features)
                    </p>
                  </div>
                </div>
                <span className="text-[11px] font-mono px-2.5 py-1 rounded-lg bg-cyan-950/80 text-cyan-300 border border-cyan-800">
                  n=400, depth=12
                </span>
              </div>

              <div className="grid grid-cols-2 gap-4 mt-6">
                <div className="p-4 rounded-xl bg-slate-950/60 border border-slate-800">
                  <span className="text-xs text-slate-400 font-medium">
                    Validation Macro F1
                  </span>
                  <div className="text-3xl font-extrabold font-mono text-cyan-400 mt-1">
                    {baseline?.validation_macro_f1
                      ? (baseline.validation_macro_f1 * 100).toFixed(1) + "%"
                      : "N/A"}
                  </div>
                  <span className="text-[10px] text-slate-500">
                    Balanced across 8 classes
                  </span>
                </div>

                <div className="p-4 rounded-xl bg-slate-950/60 border border-slate-800">
                  <span className="text-xs text-slate-400 font-medium">
                    Validation Accuracy
                  </span>
                  <div className="text-3xl font-extrabold font-mono text-white mt-1">
                    {baseline?.validation_accuracy
                      ? (baseline.validation_accuracy * 100).toFixed(1) + "%"
                      : "N/A"}
                  </div>
                  <span className="text-[10px] text-slate-500">
                    Overall correct fraction
                  </span>
                </div>
              </div>

              {/* Top Feature Importances */}
              {baseline?.top_feature_importances && (
                <div className="mt-6">
                  <h4 className="text-xs font-semibold uppercase tracking-wider text-slate-400 mb-3">
                    Top Predictive Acoustic Features
                  </h4>
                  <div className="space-y-2">
                    {Object.entries(baseline.top_feature_importances)
                      .slice(0, 5)
                      .map(([feat, imp]) => (
                        <div key={feat} className="text-xs">
                          <div className="flex justify-between text-slate-300 mb-1">
                            <span className="font-mono text-[11px]">{feat}</span>
                            <span className="font-mono text-cyan-400 font-semibold">
                              {(imp * 100).toFixed(2)}%
                            </span>
                          </div>
                          <div className="h-1.5 rounded-full bg-slate-950 overflow-hidden">
                            <div
                              className="h-full bg-gradient-to-r from-cyan-500 to-indigo-500 rounded-full"
                              style={{ width: `${Math.min(imp * 1200, 100)}%` }}
                            />
                          </div>
                        </div>
                      ))}
                  </div>
                </div>
              )}
            </div>

            {/* 2D CNN Deep Learning Card */}
            <div className="rounded-2xl border border-indigo-500/30 bg-slate-900/80 p-6 backdrop-blur-md shadow-xl glow-primary">
              <div className="flex flex-col sm:flex-row sm:items-center justify-between pb-4 border-b border-slate-800 gap-2">
                <div className="flex items-center gap-3">
                  <div className="flex h-10 w-10 items-center justify-center rounded-xl bg-indigo-500/20 text-indigo-400 border border-indigo-500/30 shrink-0">
                    <Cpu className="h-5 w-5" />
                  </div>
                  <div>
                    <h3 className="text-base font-bold text-white">
                      GenreCNN (2D Deep Learning)
                    </h3>
                    <p className="text-xs text-slate-400">
                      Log-Mel Spectrograms (1 x 128 x 431)
                    </p>
                  </div>
                </div>
                {cnn ? (
                  <span className="text-[11px] font-mono px-2.5 py-1 rounded-lg bg-indigo-950 text-indigo-300 border border-indigo-800 self-start sm:self-auto">
                    Status: {cnn.status || metrics?.run_status || "completed"} | Epoch {cnn.best_epoch} / {cnn.epochs_configured ?? 25}
                  </span>
                ) : (
                  <span className="text-[11px] font-mono px-2.5 py-1 rounded-lg bg-amber-950/80 text-amber-300 border border-amber-800 self-start sm:self-auto">
                    {metrics?.cnn_status || "validation metrics not available yet"}
                  </span>
                )}
              </div>

              {!cnn ? (
                <div className="py-8 text-center space-y-3">
                  <div className="flex h-12 w-12 items-center justify-center rounded-2xl bg-amber-500/10 text-amber-400 border border-amber-500/20 mx-auto">
                    <AlertCircle className="h-6 w-6" />
                  </div>
                  <h4 className="text-sm font-semibold text-slate-200">
                    Validation Metrics Not Available Yet
                  </h4>
                  <p className="text-xs text-slate-400 max-w-md mx-auto leading-relaxed">
                    Official CNN validation metrics (<code className="text-indigo-300 font-mono">cnn_val.json</code>) have not been produced for this run.
                    {metrics?.run_status && (
                      <span className="block mt-1 text-slate-300">
                        Run status: <strong className="text-amber-300 font-mono">{metrics.run_status}</strong>
                        {metrics.epochs_run !== undefined && metrics.epochs_run !== null ? ` (${metrics.epochs_run} epochs completed)` : ""}.
                      </span>
                    )}
                  </p>
                  <p className="text-[11px] text-slate-500 font-mono">
                    Run <code className="text-indigo-400">python scripts/05_train_cnn.py</code> to train or evaluate on the validation split.
                  </p>
                </div>
              ) : (
                <>
                  <div className="grid grid-cols-2 gap-4 mt-6">
                    <div className="p-4 rounded-xl bg-slate-950/60 border border-slate-800">
                      <span className="text-xs text-slate-400 font-medium">
                        Validation Macro F1
                      </span>
                      <div className="text-3xl font-extrabold font-mono text-indigo-400 mt-1">
                        {cnn.best_validation_macro_f1
                          ? (cnn.best_validation_macro_f1 * 100).toFixed(1) + "%"
                          : "N/A"}
                      </div>
                      <span className="text-[10px] text-slate-500">
                        Unseen validation split
                      </span>
                    </div>

                    <div className="p-4 rounded-xl bg-slate-950/60 border border-slate-800">
                      <span className="text-xs text-slate-400 font-medium">
                        Validation Accuracy
                      </span>
                      <div className="text-3xl font-extrabold font-mono text-white mt-1">
                        {cnn.classification_report?.accuracy !== undefined
                          ? (
                              Number(cnn.classification_report.accuracy) * 100
                            ).toFixed(1) + "%"
                          : "N/A"}
                      </div>
                      <span className="text-[10px] text-slate-500">
                        Segment-averaged validation predictions
                      </span>
                    </div>
                  </div>

                  {/* CNN Training Curves */}
                  {cnn.training_history && cnn.training_history.train_loss?.length > 0 && (
                    <div className="mt-6">
                      <h4 className="text-xs font-semibold uppercase tracking-wider text-slate-400 mb-3">
                        Validation & Training History (Cross-Entropy Loss)
                      </h4>
                      <div className="p-3 rounded-xl bg-slate-950/60 border border-slate-800 space-y-2">
                        <div className="flex justify-between text-xs text-slate-400">
                          <span>Initial Train Loss (Epoch 1):</span>
                          <span className="font-mono text-slate-200">
                            {cnn.training_history.train_loss[0]?.toFixed(4)}
                          </span>
                        </div>
                        {cnn.training_history.train_loss.length > 1 && (
                          <div className="flex justify-between text-xs text-slate-400">
                            <span>Latest Train Loss (Epoch {cnn.training_history.train_loss.length}):</span>
                            <span className="font-mono text-indigo-400 font-semibold">
                              {cnn.training_history.train_loss[cnn.training_history.train_loss.length - 1]?.toFixed(4)}
                            </span>
                          </div>
                        )}
                        {cnn.training_history.val_loss?.length > 0 && (
                          <div className="flex justify-between text-xs text-slate-400 pt-1 border-t border-slate-900">
                            <span>Best Validation Loss:</span>
                            <span className="font-mono text-emerald-400">
                              {Math.min(...cnn.training_history.val_loss).toFixed(4)}
                            </span>
                          </div>
                        )}
                      </div>
                    </div>
                  )}
                </>
              )}
            </div>
          </div>

          {/* Academic Safeguards Banner (§16) */}
          <div className="p-5 rounded-2xl bg-slate-900/60 border border-slate-800 flex items-start gap-4">
            <ShieldCheck className="h-6 w-6 text-emerald-400 shrink-0 mt-0.5" />
            <div className="space-y-1">
              <h4 className="text-sm font-semibold text-white">
                Academic Integrity & Split Safeguards (PLAN §16)
              </h4>
              <p className="text-xs text-slate-400 leading-relaxed">
                All metrics above are evaluated strictly on the official FMA Small validation split.
                Per Addendum A5, the test split is frozen and touched only once during final evaluation.
                No fabricated target numbers or artificial smoothing are permitted.
              </p>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
