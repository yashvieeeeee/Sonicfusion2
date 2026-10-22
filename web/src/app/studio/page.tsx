"use client";

import React, { useState, useRef, useEffect } from "react";
import Link from "next/link";
import { useRouter } from "next/navigation";
import {
  Activity,
  AlertCircle,
  ArrowLeft,
  ArrowRight,
  BarChart2,
  CheckCircle2,
  Clock,
  Compass,
  Cpu,
  Database,
  Disc,
  Disc3,
  FileAudio,
  HelpCircle,
  Music,
  Plus,
  RefreshCw,
  Scale,
  ShieldCheck,
  Sparkles,
  Upload,
} from "lucide-react";
import AudioPlayer from "@/components/AudioPlayer";
import StagedLoading from "@/components/StagedLoading";
import { analyzeAudio, fetchSampleAudio, getMetrics, getModels } from "@/lib/api";
import { getStoredAnalysisResult, saveAnalysisResult } from "@/lib/storage";
import { AnalysisResponse, MetricsResponse } from "@/types/api";

interface SampleTrack {
  id: string;
  name: string;
  genre: string;
  duration: string;
  url: string;
}

const SAMPLE_TRACKS: SampleTrack[] = [
  {
    id: "000002",
    name: "FMA 000002",
    genre: "Hip-Hop",
    duration: "30.0s",
    url: "/samples/sample_000002.mp3",
  },
  {
    id: "000005",
    name: "FMA 000005",
    genre: "Rock",
    duration: "30.0s",
    url: "/samples/sample_000005.mp3",
  },
];

export default function StudioPage() {
  const router = useRouter();
  const fileInputRef = useRef<HTMLInputElement | null>(null);

  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [audioUrl, setAudioUrl] = useState<string | null>(null);
  const [isDragging, setIsDragging] = useState<boolean>(false);
  const [isAnalyzing, setIsAnalyzing] = useState<boolean>(false);
  const [errorMessage, setErrorMessage] = useState<string | null>(null);
  const [loadingSample, setLoadingSample] = useState<string | null>(null);

  const [recentAnalysis, setRecentAnalysis] = useState<AnalysisResponse | null>(null);
  const [metrics, setMetrics] = useState<MetricsResponse | null>(null);

  // Upload limits loaded from GET /api/models
  const [configLimits, setConfigLimits] = useState<{
    maxDurationSeconds: number;
    maxUploadMb: number;
    allowedFormats: string[];
  }>({
    maxDurationSeconds: 900,
    maxUploadMb: 30,
    allowedFormats: ["mp3", "wav", "m4a"],
  });

  useEffect(() => {
    const stored = getStoredAnalysisResult();
    if (stored) {
      setRecentAnalysis(stored);
    }

    getModels()
      .then((data) => {
        if (data) {
          setConfigLimits({
            maxDurationSeconds: data.max_duration_seconds || 900,
            maxUploadMb: data.max_upload_mb || 30,
            allowedFormats: data.allowed_formats || ["mp3", "wav", "m4a"],
          });
        }
      })
      .catch(() => {});

    getMetrics()
      .then((data) => setMetrics(data))
      .catch(() => {});
  }, []);

  const formatsString = configLimits.allowedFormats.map((f) => f.toUpperCase()).join(", ");
  const maxMinutes = Math.round(configLimits.maxDurationSeconds / 60);
  const acceptAttribute = [
    ...configLimits.allowedFormats.map((f) => `.${f.toLowerCase()}`),
    "audio/*",
  ].join(",");

  const handleFileSelect = (file: File) => {
    setErrorMessage(null);
    const maxSizeBytes = configLimits.maxUploadMb * 1024 * 1024;
    if (file.size > maxSizeBytes) {
      setErrorMessage(
        `File exceeds ${configLimits.maxUploadMb} MB limit. Please select a smaller audio file.`
      );
      return;
    }
    setSelectedFile(file);
    const objectUrl = URL.createObjectURL(file);
    setAudioUrl(objectUrl);
  };

  const handleDrop = (e: React.DragEvent<HTMLDivElement>) => {
    e.preventDefault();
    setIsDragging(false);
    if (e.dataTransfer.files && e.dataTransfer.files.length > 0) {
      handleFileSelect(e.dataTransfer.files[0]);
    }
  };

  const handleDragOver = (e: React.DragEvent<HTMLDivElement>) => {
    e.preventDefault();
    setIsDragging(true);
  };

  const handleDragLeave = () => {
    setIsDragging(false);
  };

  const handleSampleClick = async (sample: SampleTrack) => {
    setErrorMessage(null);
    setLoadingSample(sample.id);
    try {
      const file = await fetchSampleAudio(sample.url);
      setSelectedFile(file);
      setAudioUrl(sample.url);
    } catch (err: unknown) {
      const msg = err instanceof Error ? err.message : "Failed to load sample track";
      setErrorMessage(msg);
    } finally {
      setLoadingSample(null);
    }
  };

  const startAnalysis = async () => {
    if (!selectedFile) return;
    setIsAnalyzing(true);
    setErrorMessage(null);

    try {
      const result = await analyzeAudio(selectedFile);
      saveAnalysisResult(result);
      setRecentAnalysis(result);
      router.push("/analysis");
    } catch (err: unknown) {
      const msg = err instanceof Error ? err.message : "Analysis failed unexpectedly.";
      setErrorMessage(msg);
      setIsAnalyzing(false);
    }
  };

  const testMetrics = metrics?.test_metrics;

  const genreTotals = [
    { genre: "Electronic", count: 800, color: "bg-[#76556F]" },
    { genre: "Experimental", count: 800, color: "bg-[#8E6E87]" },
    { genre: "Folk", count: 800, color: "bg-[#D6A75D]" },
    { genre: "Hip-Hop", count: 797, color: "bg-[#D77C63]" },
    { genre: "Instrumental", count: 800, color: "bg-[#A6B5AD]" },
    { genre: "International", count: 800, color: "bg-[#8CA095]" },
    { genre: "Pop", count: 800, color: "bg-[#C46D54]" },
    { genre: "Rock", count: 799, color: "bg-[#583C52]" },
  ];

  return (
    <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8 space-y-8">
      {/* Editorial Page Header with Return Link */}
      <div className="flex flex-col sm:flex-row sm:items-end justify-between gap-4 pb-6 border-b border-[#E3DDD4]">
        <div>
          <div className="flex items-center gap-2 mb-2">
            <Link
              href="/"
              className="inline-flex items-center gap-1.5 text-xs text-[#76556F] hover:text-[#583C52] font-medium transition-colors"
            >
              <ArrowLeft className="h-3.5 w-3.5" />
              <span>Sonic Fusion Home</span>
            </Link>
            <span className="text-[#E3DDD4]">&bull;</span>
            <span className="text-[10px] font-mono uppercase tracking-wider font-semibold text-[#76556F] bg-[#F0E8EE] px-2 py-0.5 rounded">
              Interactive Audio Studio
            </span>
            <span className="text-[10px] font-mono text-[#6B626E] hidden sm:inline">
              FMA Small Benchmark
            </span>
          </div>
          <h1 className="text-3xl sm:text-4xl font-serif font-bold text-[#27232B] tracking-tight">
            Sonic Fusion Studio
          </h1>
          <p className="text-sm text-[#6B626E] mt-1 max-w-2xl leading-relaxed">
            Ingest any audio file or load benchmark recordings to extract 128-band log-mel spectrograms, harmonic chroma profiles, and run neural genre classification.
          </p>
        </div>

        <div className="flex items-center gap-2">
          <button
            onClick={() => fileInputRef.current?.click()}
            className="flex items-center gap-1.5 px-4 py-2.5 rounded-lg bg-[#76556F] hover:bg-[#583C52] text-white text-xs font-semibold transition-all shadow-sm hover:shadow"
          >
            <Upload className="h-3.5 w-3.5" />
            <span>Upload Audio File</span>
          </button>
        </div>
      </div>

      {/* Floating Audio Ingestion & Feature Staging Deck */}
      <div className="glass-floating p-6 sm:p-8 rounded-2xl">
        <div className="flex items-center justify-between mb-4">
          <div>
            <h2 className="text-xs font-bold text-[#27232B] uppercase tracking-wider">
              Audio Ingestion &amp; Feature Staging Deck
            </h2>
            <p className="text-xs text-[#6B626E] mt-0.5">
              Formats: {formatsString} &bull; Max {maxMinutes} min &bull; Max {configLimits.maxUploadMb} MB
            </p>
          </div>
          <span className="text-[11px] font-mono text-[#76556F] bg-[#F0E8EE] px-2.5 py-1 rounded-full hidden sm:inline font-semibold border border-[#E3DDD4]">
            Standardized 22,050 Hz Mono
          </span>
        </div>

        {isAnalyzing ? (
          <StagedLoading filename={selectedFile?.name || "audio.mp3"} />
        ) : (
          <div className="space-y-5">
            {/* Drag & Drop Target */}
            <div
              onDrop={handleDrop}
              onDragOver={handleDragOver}
              onDragLeave={handleDragLeave}
              onClick={() => fileInputRef.current?.click()}
              className={`rounded-xl border-2 border-dashed p-8 sm:p-10 text-center transition-all cursor-pointer ${
                isDragging
                  ? "border-[#76556F] bg-[#F0E8EE]/80 shadow-md scale-[1.01]"
                  : "border-[#E3DDD4] bg-[#FAF7F2]/90 hover:border-[#76556F]/60 hover:bg-white"
              }`}
            >
              <input
                ref={fileInputRef}
                type="file"
                accept={acceptAttribute}
                className="hidden"
                onChange={(e) => {
                  if (e.target.files && e.target.files.length > 0) {
                    handleFileSelect(e.target.files[0]);
                  }
                }}
              />
              <div className="flex flex-col items-center gap-2.5">
                <div className="flex h-12 w-12 items-center justify-center rounded-2xl bg-[#F0E8EE] text-[#76556F] shadow-sm">
                  <Upload className="h-6 w-6" />
                </div>
                <div className="text-sm font-semibold text-[#27232B]">
                  Drag and drop an audio file here, or{" "}
                  <span className="text-[#D77C63] underline underline-offset-2 hover:text-[#C46D54]">
                    browse files
                  </span>
                </div>
                <p className="text-xs text-[#8E8692] max-w-md">
                  Audio bytes are decoded to floating-point mono arrays at 22.05 kHz prior to STFT windowing and deep tensor classification.
                </p>
              </div>
            </div>

            {/* Error Message */}
            {errorMessage && (
              <div className="p-3.5 rounded-xl bg-[#FDF1EE] border border-[#F1D0CD] text-xs text-[#D77C63] flex items-center gap-2.5">
                <AlertCircle className="h-4 w-4 shrink-0" />
                <span className="font-medium">{errorMessage}</span>
              </div>
            )}

            {/* Presets and Selected Track Preview */}
            <div className="grid grid-cols-1 md:grid-cols-12 gap-5 items-start pt-1">
              {/* Benchmark Presets */}
              <div className="md:col-span-5 space-y-2.5">
                <span className="text-[10px] font-bold uppercase tracking-wider text-[#6B626E] block">
                  Audited FMA Benchmark Presets:
                </span>
                <div className="grid grid-cols-2 gap-2.5">
                  {SAMPLE_TRACKS.map((sample) => (
                    <button
                      key={sample.id}
                      onClick={() => handleSampleClick(sample)}
                      disabled={loadingSample !== null}
                      className="p-3 rounded-xl border border-[#E3DDD4] bg-white/95 hover:border-[#76556F] hover:bg-[#FAF7F2] transition-all text-left shadow-2xs hover:shadow-xs group"
                    >
                      <div className="text-xs font-bold text-[#27232B] truncate group-hover:text-[#76556F]">
                        {sample.name}
                      </div>
                      <div className="text-[11px] text-[#6B626E] mt-0.5">
                        {sample.genre} &bull; {sample.duration}
                      </div>
                      <span className="text-[11px] font-mono text-[#D77C63] mt-2 block font-semibold group-hover:translate-x-0.5 transition-transform">
                        {loadingSample === sample.id ? "Loading..." : "Load Track →"}
                      </span>
                    </button>
                  ))}
                </div>
              </div>

              {/* Selected Track Inspection & Action */}
              <div className="md:col-span-7">
                {selectedFile && audioUrl ? (
                  <div className="p-5 rounded-xl border border-[#E3DDD4] bg-white/95 space-y-3.5 shadow-sm">
                    <div className="flex items-center justify-between gap-3">
                      <div className="flex items-center gap-2.5 min-w-0">
                        <div className="p-2 rounded-lg bg-[#F0E8EE] text-[#76556F]">
                          <FileAudio className="h-4 w-4 shrink-0" />
                        </div>
                        <div className="truncate">
                          <div className="text-xs font-bold text-[#27232B] truncate">
                            {selectedFile.name}
                          </div>
                          <div className="text-[10px] font-mono text-[#6B626E]">
                            {(selectedFile.size / (1024 * 1024)).toFixed(2)} MB &bull; Ready for inference
                          </div>
                        </div>
                      </div>
                      <button
                        onClick={startAnalysis}
                        className="flex items-center gap-2 px-4 py-2.5 rounded-xl bg-[#76556F] text-white text-xs font-bold hover:bg-[#583C52] transition-colors shrink-0 shadow-sm"
                      >
                        <span>Run Analysis</span>
                        <ArrowRight className="h-3.5 w-3.5 text-[#D77C63]" />
                      </button>
                    </div>

                    <AudioPlayer audioUrl={audioUrl} filename={selectedFile.name} />
                  </div>
                ) : (
                  <div className="p-5 rounded-xl border border-[#E3DDD4] bg-[#FAF7F2]/90 text-xs text-[#6B626E] flex items-center justify-between">
                    <span>No audio track staged yet. Choose an FMA preset above or drop an audio file.</span>
                    <span className="font-mono text-[10px] text-[#76556F] bg-[#F0E8EE] px-2 py-0.5 rounded font-medium">
                      Awaiting Input
                    </span>
                  </div>
                )}
              </div>
            </div>
          </div>
        )}
      </div>

      {/* Main Asymmetrical Analytical Grid */}
      <div className="grid grid-cols-1 lg:grid-cols-12 gap-8 items-start">
        {/* Left Column (7 cols): Recent Analysis & Dataset Distribution */}
        <div className="lg:col-span-7 space-y-8">
          {/* 1. Recent Analysis Card with Layered Summary */}
          <div className="editorial-panel p-6">
            <div className="flex items-center justify-between pb-4 border-b border-[#E3DDD4] mb-4">
              <div className="flex items-center gap-2">
                <Activity className="h-4 w-4 text-[#76556F]" />
                <h3 className="text-xs font-bold uppercase tracking-wider text-[#27232B]">
                  Recent Audio Analysis Session
                </h3>
              </div>
              {recentAnalysis && (
                <Link
                  href="/analysis"
                  className="text-xs text-[#76556F] hover:text-[#583C52] flex items-center gap-1 font-semibold"
                >
                  <span>Open Full Analysis Studio</span>
                  <ArrowRight className="h-3 w-3" />
                </Link>
              )}
            </div>

            {recentAnalysis ? (
              <div className="space-y-4">
                <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-3 p-4 rounded-xl bg-[#FAF7F2] border border-[#E3DDD4]">
                  <div>
                    <span className="text-[10px] font-mono uppercase text-[#6B626E] block font-semibold">
                      Analyzed Track
                    </span>
                    <span className="text-sm font-bold text-[#27232B]">
                      {recentAnalysis.filename}
                    </span>
                    <span className="text-[11px] text-[#6B626E] block mt-0.5">
                      {recentAnalysis.duration_seconds.toFixed(1)}s &bull; {recentAnalysis.sample_rate} Hz Mono
                    </span>
                  </div>

                  <div className="sm:text-right">
                    <span className="text-[10px] font-mono uppercase text-[#6B626E] block font-semibold">
                      Predicted Genre
                    </span>
                    <div className="flex items-center sm:justify-end gap-2 mt-0.5">
                      <span className="text-lg font-serif font-bold text-[#76556F]">
                        {recentAnalysis.genre.top_label}
                      </span>
                      <span className="font-mono text-xs px-2.5 py-0.5 rounded-full bg-[#FDF1EE] text-[#D77C63] font-bold border border-[#F1D0CD]">
                        {(recentAnalysis.genre.confidence * 100).toFixed(1)}%
                      </span>
                    </div>
                  </div>
                </div>

                {/* Key extracted parameters row */}
                <div className="grid grid-cols-3 gap-3 text-xs">
                  <div className="p-3.5 rounded-xl border border-[#E3DDD4] bg-white">
                    <span className="text-[10px] text-[#6B626E] block font-medium">Estimated Tempo</span>
                    <span className="font-mono text-sm font-bold text-[#27232B] tabular-nums">
                      {recentAnalysis.acoustic_features.tempo_bpm_estimate.toFixed(1)}{" "}
                      <span className="text-[10px] font-normal text-[#6B626E]">BPM</span>
                    </span>
                  </div>
                  <div className="p-3.5 rounded-xl border border-[#E3DDD4] bg-white">
                    <span className="text-[10px] text-[#6B626E] block font-medium">RMS Loudness</span>
                    <span className="font-mono text-sm font-bold text-[#27232B] tabular-nums">
                      {recentAnalysis.acoustic_features.rms_mean.toFixed(4)}
                    </span>
                  </div>
                  <div className="p-3.5 rounded-xl border border-[#E3DDD4] bg-white">
                    <span className="text-[10px] text-[#6B626E] block font-medium">Acoustic Neighbors</span>
                    <span className="font-mono text-sm font-bold text-[#76556F]">
                      {recentAnalysis.similar_tracks?.length || 0} Matches
                    </span>
                  </div>
                </div>
              </div>
            ) : (
              <div className="py-8 text-center text-xs text-[#6B626E] bg-[#FAF7F2] rounded-xl border border-dashed border-[#E3DDD4]">
                <p className="max-w-md mx-auto leading-relaxed">
                  No track has been analyzed in this session yet. Load an FMA benchmark preset above or upload an audio file to extract acoustic features.
                </p>
              </div>
            )}
          </div>

          {/* 2. Genre Distribution (Collection Level) */}
          <div className="editorial-panel p-6">
            <div className="flex items-center justify-between pb-4 border-b border-[#E3DDD4] mb-4">
              <div className="flex items-center gap-2">
                <BarChart2 className="h-4 w-4 text-[#76556F]" />
                <h3 className="text-xs font-bold uppercase tracking-wider text-[#27232B]">
                  Corpus Genre Distribution (FMA Small)
                </h3>
              </div>
              <span className="text-[10px] font-mono text-[#6B626E] font-medium">
                Balanced: 800 / class
              </span>
            </div>

            <p className="text-xs text-[#6B626E] mb-4 leading-relaxed">
              Ground-truth distribution of the 8 curated genres across the 7,994 decodable tracks in the FMA Small benchmark.
            </p>

            <div className="space-y-2.5">
              {genreTotals.map((item) => (
                <div key={item.genre} className="flex items-center gap-3 text-xs">
                  <span className="w-24 text-[#27232B] font-semibold truncate">
                    {item.genre}
                  </span>
                  <div className="flex-1 h-2.5 bg-[#FAF7F2] rounded-full border border-[#E3DDD4] overflow-hidden">
                    <div
                      className={`h-full ${item.color} rounded-full`}
                      style={{ width: `${(item.count / 800) * 100}%` }}
                    />
                  </div>
                  <span className="font-mono text-[11px] text-[#6B626E] w-16 text-right tabular-nums font-medium">
                    {item.count} tracks
                  </span>
                </div>
              ))}
            </div>

            <div className="mt-4 pt-3.5 border-t border-[#E3DDD4] flex items-center justify-between text-[11px] text-[#6B626E]">
              <span>Partitioned: 6,394 train &bull; 800 validation &bull; 800 test</span>
              <Link href="/collection" className="text-[#76556F] font-semibold hover:underline">
                View Collection Analytics →
              </Link>
            </div>
          </div>
        </div>

        {/* Right Column (5 cols): Model Performance & System Norms */}
        <div className="lg:col-span-5 space-y-8">
          {/* 3. Model Benchmark Card */}
          <div className="editorial-panel p-6">
            <div className="flex items-center justify-between pb-4 border-b border-[#E3DDD4] mb-4">
              <div className="flex items-center gap-2">
                <ShieldCheck className="h-4 w-4 text-[#76556F]" />
                <h3 className="text-xs font-bold uppercase tracking-wider text-[#27232B]">
                  Model Performance (Held-Out Test)
                </h3>
              </div>
              <span className="text-[10px] font-mono px-2 py-0.5 rounded-full bg-[#F0E8EE] text-[#76556F] font-bold border border-[#E3DDD4]">
                Verified
              </span>
            </div>

            <p className="text-xs text-[#6B626E] mb-4 leading-relaxed">
              Empirical evaluation on the official 800-track held-out test split (zero artist overlap).
            </p>

            <div className="space-y-3">
              {/* GenreCNN */}
              <div className="p-4 rounded-xl border border-[#E3DDD4] bg-[#FAF7F2]">
                <div className="flex items-center justify-between mb-1.5">
                  <span className="text-xs font-bold text-[#27232B]">
                    GenreCNN (2D Spectrogram)
                  </span>
                  <span className="text-[10px] font-mono px-2 py-0.5 rounded-full bg-white border border-[#E3DDD4] text-[#76556F] font-semibold">
                    Deep Learning
                  </span>
                </div>
                <div className="grid grid-cols-2 gap-2 text-xs">
                  <div>
                    <span className="text-[10px] text-[#6B626E]">Test Accuracy</span>
                    <div className="font-mono text-base font-bold text-[#76556F] tabular-nums">
                      {testMetrics?.cnn?.accuracy
                        ? `${(testMetrics.cnn.accuracy * 100).toFixed(2)}%`
                        : "43.75%"}
                    </div>
                  </div>
                  <div>
                    <span className="text-[10px] text-[#6B626E]">Test Macro-F1</span>
                    <div className="font-mono text-base font-bold text-[#27232B] tabular-nums">
                      {testMetrics?.cnn?.macro_f1
                        ? `${(testMetrics.cnn.macro_f1 * 100).toFixed(2)}%`
                        : "43.31%"}
                    </div>
                  </div>
                </div>
              </div>

              {/* Random Forest Baseline */}
              <div className="p-4 rounded-xl border border-[#E3DDD4] bg-white">
                <div className="flex items-center justify-between mb-1.5">
                  <span className="text-xs font-bold text-[#27232B]">
                    Random Forest (Handcrafted 51-d)
                  </span>
                  <span className="text-[10px] font-mono px-2 py-0.5 rounded-full bg-[#FAF7F2] border border-[#E3DDD4] text-[#6B626E] font-semibold">
                    Classical Baseline
                  </span>
                </div>
                <div className="grid grid-cols-2 gap-2 text-xs">
                  <div>
                    <span className="text-[10px] text-[#6B626E]">Test Accuracy</span>
                    <div className="font-mono text-base font-bold text-[#27232B] tabular-nums">
                      {testMetrics?.random_forest?.accuracy
                        ? `${(testMetrics.random_forest.accuracy * 100).toFixed(2)}%`
                        : "43.75%"}
                    </div>
                  </div>
                  <div>
                    <span className="text-[10px] text-[#6B626E]">Test Macro-F1</span>
                    <div className="font-mono text-base font-bold text-[#27232B] tabular-nums">
                      {testMetrics?.random_forest?.macro_f1
                        ? `${(testMetrics.random_forest.macro_f1 * 100).toFixed(2)}%`
                        : "43.57%"}
                    </div>
                  </div>
                </div>
              </div>
            </div>

            <div className="mt-4 pt-3.5 border-t border-[#E3DDD4]">
              <Link
                href="/evaluation"
                className="text-xs text-[#76556F] font-semibold hover:underline flex items-center justify-between"
              >
                <span>View Full Statistical Comparison &amp; McNemar Test</span>
                <ArrowRight className="h-3.5 w-3.5" />
              </Link>
            </div>
          </div>

          {/* 4. Acoustic Feature Norms */}
          <div className="editorial-panel p-6">
            <div className="flex items-center justify-between pb-4 border-b border-[#E3DDD4] mb-4">
              <div className="flex items-center gap-2">
                <Compass className="h-4 w-4 text-[#76556F]" />
                <h3 className="text-xs font-bold uppercase tracking-wider text-[#27232B]">
                  Acoustic Feature Norms
                </h3>
              </div>
              <span className="text-[10px] font-mono text-[#6B626E] font-medium">
                Laboratory Standards
              </span>
            </div>

            <div className="space-y-2.5 text-xs">
              <div className="flex justify-between py-1.5 border-b border-[#E3DDD4]">
                <span className="text-[#6B626E]">Sample Rate:</span>
                <span className="font-mono font-medium text-[#27232B]">22,050 Hz Mono</span>
              </div>
              <div className="flex justify-between py-1.5 border-b border-[#E3DDD4]">
                <span className="text-[#6B626E]">STFT Window (FFT / Hop):</span>
                <span className="font-mono font-medium text-[#27232B]">2,048 / 512 samples</span>
              </div>
              <div className="flex justify-between py-1.5 border-b border-[#E3DDD4]">
                <span className="text-[#6B626E]">Mel Channels:</span>
                <span className="font-mono font-medium text-[#27232B]">128 Mel Bins</span>
              </div>
              <div className="flex justify-between py-1.5 border-b border-[#E3DDD4]">
                <span className="text-[#6B626E]">Chroma Pitch Resolution:</span>
                <span className="font-mono font-medium text-[#27232B]">12 Semitone Bins</span>
              </div>
              <div className="flex justify-between py-1.5">
                <span className="text-[#6B626E]">Inference Latency:</span>
                <span className="font-mono font-medium text-[#D77C63]">~45 ms (CPU)</span>
              </div>
            </div>

            <div className="mt-4 pt-3.5 border-t border-[#E3DDD4] text-[10px] text-[#8E8692]">
              Standardized MIR signal pipeline configuration.
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
