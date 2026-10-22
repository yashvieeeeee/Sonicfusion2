"use client";

import React, { useEffect, useState } from "react";
import Link from "next/link";
import {
  Activity,
  AlertCircle,
  BarChart2,
  CheckCircle2,
  Clock,
  Compass,
  Cpu,
  Database,
  FileText,
  HelpCircle,
  Info,
  Layers,
  LineChart,
  RefreshCw,
  Scale,
  ShieldAlert,
  ShieldCheck,
  TrendingUp,
} from "lucide-react";
import { getFigureUrl, getMetrics } from "@/lib/api";
import { MetricsResponse, TestMetrics } from "@/types/api";

export default function EvaluationPage() {
  const [metrics, setMetrics] = useState<MetricsResponse | null>(null);
  const [loading, setLoading] = useState<boolean>(true);
  const [error, setError] = useState<string | null>(null);
  const [activeTab, setActiveTab] = useState<"test" | "validation" | "eda">("test");

  const fetchMetrics = async () => {
    setLoading(true);
    setError(null);
    try {
      const data = await getMetrics();
      setMetrics(data);
    } catch (err: unknown) {
      const msg = err instanceof Error ? err.message : "Failed to load metrics";
      setError(msg);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchMetrics();
  }, []);

  const testData = metrics?.test_metrics;
  const isTestAvailable = metrics?.test_status === "available" && !!testData;
  const cnnVal = metrics?.cnn_validation;
  const baselineVal = metrics?.baseline_validation;
  const edaData = metrics?.eda_summary;
  const figures = metrics?.figures_available || [];

  return (
    <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8 space-y-8">
      {/* Page Header */}
      <div className="flex flex-col sm:flex-row sm:items-end justify-between gap-4 pb-6 border-b border-[#E3DDD4]">
        <div>
          <div className="flex items-center gap-2 mb-1.5">
            <span className="text-[10px] font-mono uppercase tracking-wider font-semibold text-[#76556F] bg-[#F0E8EE] px-2 py-0.5 rounded">
              Empirical Benchmark
            </span>
            <span className="text-[10px] font-mono text-[#6B626E]">
              FMA Small Partitions
            </span>
          </div>
          <h1 className="text-3xl sm:text-4xl font-serif font-bold text-[#27232B] tracking-tight">
            Model Evaluation &amp; Benchmarks
          </h1>
          <p className="text-sm text-[#6B626E] mt-1 max-w-3xl leading-relaxed">
            Scientific comparison evaluating the frozen Classical Baseline (Random Forest 51-d) against the Spectrogram Deep Learning model (GenreCNN) on held-out test audio.
          </p>
        </div>

        <button
          onClick={fetchMetrics}
          disabled={loading}
          className="flex items-center gap-2 px-3.5 py-1.5 rounded-md border border-[#E3DDD4] bg-white hover:bg-[#FAF7F2] text-[#27232B] text-xs font-medium transition-colors shrink-0 shadow-2xs"
        >
          <RefreshCw className={`h-3.5 w-3.5 text-[#76556F] ${loading ? "animate-spin" : ""}`} />
          <span>Refresh Metrics</span>
        </button>
      </div>

      {/* Tabs navigation */}
      <div className="flex flex-wrap gap-2 border-b border-[#E3DDD4] pb-3">
        <button
          onClick={() => setActiveTab("test")}
          className={`flex items-center gap-2 px-3.5 py-1.5 rounded-md text-xs font-medium transition-colors ${
            activeTab === "test"
              ? "bg-[#76556F] text-white font-semibold shadow-2xs"
              : "bg-[#FAF7F2] text-[#6B626E] hover:text-[#27232B] border border-[#E3DDD4]"
          }`}
        >
          <ShieldCheck className="h-3.5 w-3.5" />
          <span>Official Test Split (Held-Out)</span>
          {isTestAvailable ? (
            <span className="text-[10px] font-mono px-1.5 py-0.2 rounded bg-white/20 text-white">
              Available
            </span>
          ) : (
            <span className="text-[10px] font-mono px-1.5 py-0.2 rounded bg-[#FBF6EC] text-[#D6A75D]">
              Pending
            </span>
          )}
        </button>

        <button
          onClick={() => setActiveTab("validation")}
          className={`flex items-center gap-2 px-3.5 py-1.5 rounded-md text-xs font-medium transition-colors ${
            activeTab === "validation"
              ? "bg-[#76556F] text-white font-semibold shadow-2xs"
              : "bg-[#FAF7F2] text-[#6B626E] hover:text-[#27232B] border border-[#E3DDD4]"
          }`}
        >
          <Layers className="h-3.5 w-3.5" />
          <span>Validation Split (Tuning Only)</span>
          {baselineVal || cnnVal ? (
            <span className="text-[10px] font-mono px-1.5 py-0.2 rounded bg-[#F0E8EE] text-[#76556F]">
              Available
            </span>
          ) : null}
        </button>

        <button
          onClick={() => setActiveTab("eda")}
          className={`flex items-center gap-2 px-3.5 py-1.5 rounded-md text-xs font-medium transition-colors ${
            activeTab === "eda"
              ? "bg-[#76556F] text-white font-semibold shadow-2xs"
              : "bg-[#FAF7F2] text-[#6B626E] hover:text-[#27232B] border border-[#E3DDD4]"
          }`}
        >
          <BarChart2 className="h-3.5 w-3.5" />
          <span>Corpus Exploratory Data Analysis</span>
          {edaData ? (
            <span className="text-[10px] font-mono px-1.5 py-0.2 rounded bg-[#F0E8EE] text-[#76556F]">
              Available
            </span>
          ) : null}
        </button>
      </div>

      {loading ? (
        <div className="py-24 text-center">
          <RefreshCw className="h-6 w-6 text-[#76556F] animate-spin mx-auto mb-2" />
          <p className="text-xs text-[#6B626E]">Loading reproducible metrics from /api/metrics...</p>
        </div>
      ) : error ? (
        <div className="p-4 rounded-md bg-[#FDF1EE] border border-[#F1D0CD] text-xs text-[#D77C63] flex items-center gap-2">
          <AlertCircle className="h-4 w-4 shrink-0" />
          <span>{error}</span>
        </div>
      ) : activeTab === "test" ? (
        /* HELD-OUT TEST SPLIT CONTENT */
        <div className="space-y-8">
          {/* Integrity Note */}
          <div className="editorial-panel p-5 flex flex-col sm:flex-row sm:items-center justify-between gap-4">
            <div className="flex items-start gap-3">
              <ShieldCheck className="h-5 w-5 text-[#76556F] shrink-0 mt-0.5" />
              <div>
                <h3 className="text-xs font-semibold uppercase tracking-wider text-[#27232B] flex items-center gap-2">
                  <span>Official Test Split Protocol</span>
                  <span className="text-[10px] font-mono px-2 py-0.2 rounded bg-[#F0E8EE] text-[#76556F]">
                    800 Held-Out Tracks
                  </span>
                </h3>
                <p className="text-xs text-[#6B626E] mt-1 leading-relaxed">
                  Hyperparameter selection was strictly restricted to training and validation splits. The test split is evaluated exactly once after checkpoints are frozen to guarantee unbiased out-of-sample metrics with 0 artist ID leakage.
                </p>
              </div>
            </div>
            {isTestAvailable && testData?.timestamp && (
              <div className="text-right shrink-0 text-xs text-[#6B626E] font-mono">
                <span className="block text-[10px]">Evaluated UTC</span>
                <span>{new Date(testData.timestamp).toLocaleDateString()}</span>
              </div>
            )}
          </div>

          {!isTestAvailable ? (
            <div className="editorial-panel p-10 text-center space-y-3">
              <div className="flex h-10 w-10 items-center justify-center rounded-lg bg-[#FBF6EC] text-[#D6A75D] mx-auto">
                <Info className="h-5 w-5" />
              </div>
              <h3 className="text-sm font-semibold text-[#27232B]">Test Results Not Available Yet</h3>
              <p className="text-xs text-[#6B626E] max-w-md mx-auto leading-relaxed">
                Official test split evaluation has not been run on this server yet. Metrics are never fabricated or simulated.
              </p>
              <div className="inline-block p-2 rounded bg-[#FAF7F2] border border-[#E3DDD4] text-[11px] font-mono text-[#27232B]">
                python scripts/06_evaluate.py
              </div>
            </div>
          ) : (
            <>
              {/* Primary Comparison Cards */}
              <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                {/* Random Forest Test Card */}
                <div className="editorial-panel p-6">
                  <div className="flex items-center justify-between pb-3 border-b border-[#E3DDD4] mb-4">
                    <div>
                      <h3 className="text-sm font-semibold text-[#27232B]">
                        Random Forest Baseline
                      </h3>
                      <p className="text-[11px] text-[#6B626E]">
                        51-d Handcrafted Summary Vector
                      </p>
                    </div>
                    <span className="text-[10px] font-mono px-2 py-0.5 rounded bg-[#FAF7F2] border border-[#E3DDD4] text-[#6B626E]">
                      Baseline
                    </span>
                  </div>

                  <div className="grid grid-cols-2 gap-4">
                    <div className="p-3.5 rounded-lg bg-[#FAF7F2] border border-[#E3DDD4]">
                      <span className="text-[10px] uppercase tracking-wider text-[#6B626E] block">
                        Test Accuracy
                      </span>
                      <div className="text-2xl font-mono font-bold text-[#27232B] mt-1 tabular-nums">
                        {(testData.random_forest.accuracy * 100).toFixed(2)}%
                      </div>
                      {testData.random_forest.bootstrap_ci_95?.accuracy && (
                        <span className="text-[10px] font-mono text-[#8E8692] mt-1 block">
                          95% CI: [{(testData.random_forest.bootstrap_ci_95.accuracy[0] * 100).toFixed(1)}%,{" "}
                          {(testData.random_forest.bootstrap_ci_95.accuracy[1] * 100).toFixed(1)}%]
                        </span>
                      )}
                    </div>

                    <div className="p-3.5 rounded-lg bg-[#FAF7F2] border border-[#E3DDD4]">
                      <span className="text-[10px] uppercase tracking-wider text-[#6B626E] block">
                        Test Macro-F1
                      </span>
                      <div className="text-2xl font-mono font-bold text-[#27232B] mt-1 tabular-nums">
                        {(testData.random_forest.macro_f1 * 100).toFixed(2)}%
                      </div>
                      {testData.random_forest.bootstrap_ci_95?.macro_f1 && (
                        <span className="text-[10px] font-mono text-[#8E8692] mt-1 block">
                          95% CI: [{(testData.random_forest.bootstrap_ci_95.macro_f1[0] * 100).toFixed(1)}%,{" "}
                          {(testData.random_forest.bootstrap_ci_95.macro_f1[1] * 100).toFixed(1)}%]
                        </span>
                      )}
                    </div>
                  </div>

                  <div className="mt-4 pt-3 border-t border-[#E3DDD4] flex justify-between text-xs text-[#6B626E]">
                    <span>Median Feature Latency:</span>
                    <span className="font-mono text-[#27232B]">
                      {testData.latency_benchmark?.median_rf_inference_ms !== undefined
                        ? `${testData.latency_benchmark.median_rf_inference_ms.toFixed(2)} ms / track`
                        : "—"}
                    </span>
                  </div>
                </div>

                {/* 2D CNN Test Card */}
                <div className="editorial-panel p-6">
                  <div className="flex items-center justify-between pb-3 border-b border-[#E3DDD4] mb-4">
                    <div>
                      <h3 className="text-sm font-semibold text-[#76556F]">
                        GenreCNN (2D Spectrogram)
                      </h3>
                      <p className="text-[11px] text-[#6B626E]">
                        3-Window Track Probability Average
                      </p>
                    </div>
                    <span className="text-[10px] font-mono px-2 py-0.5 rounded bg-[#F0E8EE] text-[#76556F] font-medium">
                      Deep Learning
                    </span>
                  </div>

                  <div className="grid grid-cols-2 gap-4">
                    <div className="p-3.5 rounded-lg bg-[#FAF7F2] border border-[#E3DDD4]">
                      <span className="text-[10px] uppercase tracking-wider text-[#6B626E] block">
                        Test Accuracy
                      </span>
                      <div className="text-2xl font-mono font-bold text-[#76556F] mt-1 tabular-nums">
                        {(testData.cnn.accuracy * 100).toFixed(2)}%
                      </div>
                      {testData.cnn.bootstrap_ci_95?.accuracy && (
                        <span className="text-[10px] font-mono text-[#8E8692] mt-1 block">
                          95% CI: [{(testData.cnn.bootstrap_ci_95.accuracy[0] * 100).toFixed(1)}%,{" "}
                          {(testData.cnn.bootstrap_ci_95.accuracy[1] * 100).toFixed(1)}%]
                        </span>
                      )}
                    </div>

                    <div className="p-3.5 rounded-lg bg-[#FAF7F2] border border-[#E3DDD4]">
                      <span className="text-[10px] uppercase tracking-wider text-[#6B626E] block">
                        Test Macro-F1
                      </span>
                      <div className="text-2xl font-mono font-bold text-[#76556F] mt-1 tabular-nums">
                        {(testData.cnn.macro_f1 * 100).toFixed(2)}%
                      </div>
                      {testData.cnn.bootstrap_ci_95?.macro_f1 && (
                        <span className="text-[10px] font-mono text-[#8E8692] mt-1 block">
                          95% CI: [{(testData.cnn.bootstrap_ci_95.macro_f1[0] * 100).toFixed(1)}%,{" "}
                          {(testData.cnn.bootstrap_ci_95.macro_f1[1] * 100).toFixed(1)}%]
                        </span>
                      )}
                    </div>
                  </div>

                  <div className="mt-4 pt-3 border-t border-[#E3DDD4] flex justify-between text-xs text-[#6B626E]">
                    <span>Median Forward Latency:</span>
                    <span className="font-mono text-[#D77C63]">
                      {testData.latency_benchmark?.median_cnn_forward_ms !== undefined
                        ? `${testData.latency_benchmark.median_cnn_forward_ms.toFixed(2)} ms / track`
                        : "—"}
                    </span>
                  </div>
                </div>
              </div>

              {/* Statistical Significance & Paired Tests */}
              {testData.statistical_comparison && (
                <div className="editorial-panel p-6">
                  <div className="flex items-center gap-2 mb-4 pb-3 border-b border-[#E3DDD4]">
                    <Scale className="h-4 w-4 text-[#76556F]" />
                    <h3 className="text-xs font-semibold uppercase tracking-wider text-[#27232B]">
                      Statistical Hypotheses &amp; Significance Testing
                    </h3>
                  </div>

                  <div className="grid grid-cols-1 sm:grid-cols-3 gap-4 text-xs">
                    <div className="p-3.5 rounded-lg bg-[#FAF7F2] border border-[#E3DDD4]">
                      <span className="text-[#6B626E] block text-[10px] uppercase tracking-wider">
                        Accuracy Difference (CNN &minus; RF)
                      </span>
                      <div className="text-lg font-mono font-bold text-[#27232B] mt-1 tabular-nums">
                        {testData.statistical_comparison.difference_cnn_minus_rf?.mean_accuracy_difference !== undefined
                          ? `${(testData.statistical_comparison.difference_cnn_minus_rf.mean_accuracy_difference * 100).toFixed(2)}%`
                          : "—"}
                      </div>
                      <span className="text-[10px] font-mono text-[#8E8692] mt-1 block">
                        95% CI: [
                        {testData.statistical_comparison.difference_cnn_minus_rf?.accuracy_difference_ci_95
                          ? `${(testData.statistical_comparison.difference_cnn_minus_rf.accuracy_difference_ci_95[0] * 100).toFixed(2)}%, ${(testData.statistical_comparison.difference_cnn_minus_rf.accuracy_difference_ci_95[1] * 100).toFixed(2)}%`
                          : "—"}
                        ]
                      </span>
                    </div>

                    <div className="p-3.5 rounded-lg bg-[#FAF7F2] border border-[#E3DDD4]">
                      <span className="text-[#6B626E] block text-[10px] uppercase tracking-wider">
                        Paired Bootstrap Test
                      </span>
                      <div className="text-lg font-mono font-bold text-[#27232B] mt-1 tabular-nums">
                        p = {testData.statistical_comparison.difference_cnn_minus_rf?.paired_bootstrap_p_value !== undefined
                          ? testData.statistical_comparison.difference_cnn_minus_rf.paired_bootstrap_p_value.toFixed(4)
                          : "—"}
                      </div>
                      <span className="text-[10px] text-[#8E8692] mt-1 block">
                        1,000 paired resamples on test tracks
                      </span>
                    </div>

                    <div className="p-3.5 rounded-lg bg-[#FAF7F2] border border-[#E3DDD4]">
                      <span className="text-[#6B626E] block text-[10px] uppercase tracking-wider">
                        McNemar Chi-Square Test
                      </span>
                      <div className="text-lg font-mono font-bold text-[#76556F] mt-1 tabular-nums">
                        &chi;&sup2; = {testData.statistical_comparison.mcnemar_test?.statistic !== undefined
                          ? testData.statistical_comparison.mcnemar_test.statistic.toFixed(2)
                          : "—"}
                      </div>
                      <span className="text-[10px] text-[#8E8692] mt-1 block font-mono">
                        p = {testData.statistical_comparison.mcnemar_test?.p_value !== undefined
                          ? testData.statistical_comparison.mcnemar_test.p_value.toFixed(4)
                          : "—"}
                      </span>
                    </div>
                  </div>
                </div>
              )}

              {/* Model Comparison Table */}
              <div className="editorial-panel p-6">
                <h3 className="text-xs font-semibold uppercase tracking-wider text-[#27232B] mb-3">
                  Summary Benchmark Comparison
                </h3>
                <div className="overflow-x-auto">
                  <table className="w-full text-left editorial-table">
                    <thead>
                      <tr>
                        <th>Model</th>
                        <th>Feature Representation</th>
                        <th>Test Accuracy (95% CI)</th>
                        <th>Macro-F1 (95% CI)</th>
                        <th>Inference Latency</th>
                      </tr>
                    </thead>
                    <tbody>
                      <tr>
                        <td className="font-semibold text-[#27232B]">Random Forest Baseline</td>
                        <td className="text-[#6B626E]">51-d Handcrafted Summary</td>
                        <td className="font-mono tabular-nums">
                          {(testData.random_forest.accuracy * 100).toFixed(2)}%{" "}
                          {testData.random_forest.bootstrap_ci_95?.accuracy && (
                            <span className="text-[#8E8692] text-xs">
                              [{(testData.random_forest.bootstrap_ci_95.accuracy[0] * 100).toFixed(1)}%,{" "}
                              {(testData.random_forest.bootstrap_ci_95.accuracy[1] * 100).toFixed(1)}%]
                            </span>
                          )}
                        </td>
                        <td className="font-mono tabular-nums text-[#27232B]">
                          {(testData.random_forest.macro_f1 * 100).toFixed(2)}%
                        </td>
                        <td className="font-mono text-[#8E8692]">
                          {testData.latency_benchmark?.median_rf_inference_ms !== undefined
                            ? `${testData.latency_benchmark.median_rf_inference_ms.toFixed(2)} ms`
                            : "—"}
                        </td>
                      </tr>
                      <tr>
                        <td className="font-semibold text-[#76556F]">GenreCNN (2D Spectrogram)</td>
                        <td className="text-[#6B626E]">128-Band Log-Mel Spectrogram</td>
                        <td className="font-mono tabular-nums text-[#76556F] font-semibold">
                          {(testData.cnn.accuracy * 100).toFixed(2)}%{" "}
                          {testData.cnn.bootstrap_ci_95?.accuracy && (
                            <span className="text-[#8E8692] text-xs">
                              [{(testData.cnn.bootstrap_ci_95.accuracy[0] * 100).toFixed(1)}%,{" "}
                              {(testData.cnn.bootstrap_ci_95.accuracy[1] * 100).toFixed(1)}%]
                            </span>
                          )}
                        </td>
                        <td className="font-mono tabular-nums text-[#76556F] font-semibold">
                          {(testData.cnn.macro_f1 * 100).toFixed(2)}%
                        </td>
                        <td className="font-mono text-[#D77C63]">
                          {testData.latency_benchmark?.median_cnn_forward_ms !== undefined
                            ? `${testData.latency_benchmark.median_cnn_forward_ms.toFixed(2)} ms`
                            : "—"}
                        </td>
                      </tr>
                    </tbody>
                  </table>
                </div>
              </div>

              {/* Confusion Matrices Figures */}
              <div className="editorial-panel p-6 space-y-4">
                <div>
                  <h3 className="text-xs font-semibold uppercase tracking-wider text-[#27232B]">
                    Official Confusion Matrices
                  </h3>
                  <p className="text-xs text-[#6B626E] mt-0.5">
                    Row-normalized class recall and raw classification error counts across the 800 test tracks.
                  </p>
                </div>

                <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                  <div className="p-4 rounded-lg border border-[#E3DDD4] bg-[#FAF7F2] space-y-2">
                    <span className="text-xs font-semibold text-[#27232B] block">
                      GenreCNN (Row-Normalized Recall)
                    </span>
                    {figures.includes("cnn_confusion_matrix_norm.png") ? (
                      <div className="aspect-square bg-white rounded border border-[#E3DDD4] overflow-hidden">
                        <img
                          src={getFigureUrl("cnn_confusion_matrix_norm.png")}
                          alt="GenreCNN Normalized Confusion Matrix"
                          className="w-full h-full object-contain"
                        />
                      </div>
                    ) : (
                      <div className="py-16 text-center text-xs text-[#8E8692]">Figure pending</div>
                    )}
                  </div>

                  <div className="p-4 rounded-lg border border-[#E3DDD4] bg-[#FAF7F2] space-y-2">
                    <span className="text-xs font-semibold text-[#27232B] block">
                      Random Forest (Row-Normalized Recall)
                    </span>
                    {figures.includes("rf_confusion_matrix_norm.png") ? (
                      <div className="aspect-square bg-white rounded border border-[#E3DDD4] overflow-hidden">
                        <img
                          src={getFigureUrl("rf_confusion_matrix_norm.png")}
                          alt="Random Forest Normalized Confusion Matrix"
                          className="w-full h-full object-contain"
                        />
                      </div>
                    ) : (
                      <div className="py-16 text-center text-xs text-[#8E8692]">Figure pending</div>
                    )}
                  </div>
                </div>
              </div>

              {/* Most Confused Genre Pairs */}
              {testData.cnn.most_confused_pairs && (
                <div className="editorial-panel p-6">
                  <h3 className="text-xs font-semibold uppercase tracking-wider text-[#27232B] mb-3">
                    Frequent Confusion Boundaries (Error Analysis)
                  </h3>
                  <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
                    <div className="p-4 rounded-lg border border-[#E3DDD4] bg-[#FAF7F2]">
                      <span className="text-xs font-semibold text-[#76556F] block mb-2">
                        Top Confused Pairs (GenreCNN)
                      </span>
                      <ul className="space-y-1.5 text-xs text-[#27232B]">
                        {testData.cnn.most_confused_pairs.slice(0, 5).map((p) => (
                          <li key={p.description} className="flex justify-between py-1 border-b border-[#E3DDD4]">
                            <span className="font-mono text-[#6B626E]">{p.description}</span>
                            <span className="font-mono font-medium">{p.count} tracks</span>
                          </li>
                        ))}
                      </ul>
                    </div>

                    <div className="p-4 rounded-lg border border-[#E3DDD4] bg-[#FAF7F2]">
                      <span className="text-xs font-semibold text-[#27232B] block mb-2">
                        Top Confused Pairs (Random Forest)
                      </span>
                      <ul className="space-y-1.5 text-xs text-[#27232B]">
                        {testData.random_forest.most_confused_pairs?.slice(0, 5).map((p) => (
                          <li key={p.description} className="flex justify-between py-1 border-b border-[#E3DDD4]">
                            <span className="font-mono text-[#6B626E]">{p.description}</span>
                            <span className="font-mono font-medium">{p.count} tracks</span>
                          </li>
                        )) || <li className="text-[#8E8692]">Not available</li>}
                      </ul>
                    </div>
                  </div>
                </div>
              )}
            </>
          )}
        </div>
      ) : activeTab === "validation" ? (
        /* VALIDATION TUNING SPLIT CONTENT */
        <div className="space-y-6">
          <div className="editorial-panel p-5 flex items-start gap-3">
            <Info className="h-5 w-5 text-[#76556F] shrink-0 mt-0.5" />
            <div>
              <h3 className="text-xs font-semibold uppercase tracking-wider text-[#27232B]">
                Validation Split Diagnostics (Used for Model Selection Only)
              </h3>
              <p className="text-xs text-[#6B626E] mt-1 leading-relaxed">
                Validation metrics computed on the 800-track validation partition during model tuning. Used strictly for hyperparameter grid search (Random Forest) and early stopping (CNN).
              </p>
            </div>
          </div>

          <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
            <div className="editorial-panel p-6">
              <div className="flex items-center justify-between pb-3 border-b border-[#E3DDD4] mb-4">
                <h3 className="text-sm font-semibold text-[#27232B]">Random Forest (Validation)</h3>
                <span className="font-mono text-xs text-[#6B626E]">
                  {baselineVal?.best_params ? `trees=${baselineVal.best_params.n_estimators}` : "n=400"}
                </span>
              </div>
              <div className="grid grid-cols-2 gap-4">
                <div className="p-3.5 rounded-lg bg-[#FAF7F2] border border-[#E3DDD4]">
                  <span className="text-[10px] uppercase text-[#6B626E]">Accuracy</span>
                  <div className="text-2xl font-mono font-bold text-[#27232B] mt-1 tabular-nums">
                    {baselineVal?.validation_accuracy !== undefined
                      ? `${(baselineVal.validation_accuracy * 100).toFixed(2)}%`
                      : "—"}
                  </div>
                </div>
                <div className="p-3.5 rounded-lg bg-[#FAF7F2] border border-[#E3DDD4]">
                  <span className="text-[10px] uppercase text-[#6B626E]">Macro-F1</span>
                  <div className="text-2xl font-mono font-bold text-[#76556F] mt-1 tabular-nums">
                    {baselineVal?.validation_macro_f1 !== undefined
                      ? `${(baselineVal.validation_macro_f1 * 100).toFixed(2)}%`
                      : "—"}
                  </div>
                </div>
              </div>
            </div>

            <div className="editorial-panel p-6">
              <div className="flex items-center justify-between pb-3 border-b border-[#E3DDD4] mb-4">
                <h3 className="text-sm font-semibold text-[#76556F]">GenreCNN (Validation)</h3>
                <span className="font-mono text-xs text-[#76556F]">
                  {cnnVal?.best_epoch ? `Best Epoch: ${cnnVal.best_epoch}` : "—"}
                </span>
              </div>
              <div className="grid grid-cols-2 gap-4">
                <div className="p-3.5 rounded-lg bg-[#FAF7F2] border border-[#E3DDD4]">
                  <span className="text-[10px] uppercase text-[#6B626E]">Accuracy</span>
                  <div className="text-2xl font-mono font-bold text-[#27232B] mt-1 tabular-nums">
                    {cnnVal?.classification_report?.accuracy !== undefined
                      ? `${(Number(cnnVal.classification_report.accuracy) * 100).toFixed(2)}%`
                      : "—"}
                  </div>
                </div>
                <div className="p-3.5 rounded-lg bg-[#FAF7F2] border border-[#E3DDD4]">
                  <span className="text-[10px] uppercase text-[#6B626E]">Macro-F1</span>
                  <div className="text-2xl font-mono font-bold text-[#76556F] mt-1 tabular-nums">
                    {cnnVal?.best_validation_macro_f1 !== undefined
                      ? `${(cnnVal.best_validation_macro_f1 * 100).toFixed(2)}%`
                      : "—"}
                  </div>
                </div>
              </div>
            </div>
          </div>
        </div>
      ) : (
        /* EDA TAB CONTENT */
        <div className="space-y-6">
          <div className="editorial-panel p-5 flex items-start gap-3">
            <BarChart2 className="h-5 w-5 text-[#76556F] shrink-0 mt-0.5" />
            <div>
              <h3 className="text-xs font-semibold uppercase tracking-wider text-[#27232B]">
                Dataset Exploratory Profiles &amp; Figures
              </h3>
              <p className="text-xs text-[#6B626E] mt-1 leading-relaxed">
                Distribution of genre classes across splits, duration distributions, and representative spectrogram examples.
              </p>
            </div>
          </div>

          <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
            {figures.includes("eda_class_counts_by_split.png") && (
              <div className="editorial-panel p-5 space-y-2">
                <span className="text-xs font-semibold text-[#27232B] block">
                  Genre Class Counts Across Train/Val/Test
                </span>
                <div className="aspect-16/10 bg-white rounded border border-[#E3DDD4] overflow-hidden">
                  <img
                    src={getFigureUrl("eda_class_counts_by_split.png")}
                    alt="Class counts by split"
                    className="w-full h-full object-contain"
                  />
                </div>
              </div>
            )}

            {figures.includes("eda_feature_distributions.png") && (
              <div className="editorial-panel p-5 space-y-2">
                <span className="text-xs font-semibold text-[#27232B] block">
                  Acoustic Feature Distributions
                </span>
                <div className="aspect-16/10 bg-white rounded border border-[#E3DDD4] overflow-hidden">
                  <img
                    src={getFigureUrl("eda_feature_distributions.png")}
                    alt="Feature distributions"
                    className="w-full h-full object-contain"
                  />
                </div>
              </div>
            )}
          </div>
        </div>
      )}
    </div>
  );
}
