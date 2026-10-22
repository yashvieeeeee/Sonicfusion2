"use client";

import React, { useEffect, useState } from "react";
import Link from "next/link";
import {
  Activity,
  AlertCircle,
  ArrowLeft,
  Clock,
  Compass,
  Cpu,
  Database,
  Disc3,
  FileAudio,
  Gauge,
  GitCompare,
  HelpCircle,
  Info,
  Layers,
  Music,
  Music2,
  RefreshCw,
  Scale,
  ShieldCheck,
  Waves,
} from "lucide-react";
import AudioPlayer from "@/components/AudioPlayer";
import ChromaChart from "@/components/ChromaChart";
import GenreRanking from "@/components/GenreRanking";
import SpectrogramHeatmap from "@/components/SpectrogramHeatmap";
import WaveformChart from "@/components/WaveformChart";
import { analyzeAudio, fetchSampleAudio } from "@/lib/api";
import { getStoredAnalysisResult, saveAnalysisResult } from "@/lib/storage";
import { AnalysisResponse } from "@/types/api";

export default function AnalysisPage() {
  const [analysis, setAnalysis] = useState<AnalysisResponse | null>(null);
  const [currentTime, setCurrentTime] = useState<number>(0);
  const [seekTime, setSeekTime] = useState<number | null>(null);
  const [isLoadingSample, setIsLoadingSample] = useState<boolean>(false);
  const [sampleAudioUrl, setSampleAudioUrl] = useState<string | null>(null);

  useEffect(() => {
    const stored = getStoredAnalysisResult();
    if (stored) {
      setAnalysis(stored);
      if (stored.filename.includes("000002")) {
        setSampleAudioUrl("/samples/sample_000002.mp3");
      } else if (stored.filename.includes("000005")) {
        setSampleAudioUrl("/samples/sample_000005.mp3");
      }
    }
  }, []);

  const loadBenchmarkSample = async (samplePath: string) => {
    setIsLoadingSample(true);
    try {
      const file = await fetchSampleAudio(samplePath);
      setSampleAudioUrl(samplePath);
      const res = await analyzeAudio(file);
      saveAnalysisResult(res);
      setAnalysis(res);
    } catch (e) {
      console.error("Failed to load sample:", e);
    } finally {
      setIsLoadingSample(false);
    }
  };

  const handleSeek = (time: number) => {
    setSeekTime(time);
    setCurrentTime(time);
  };

  if (!analysis) {
    return (
      <div className="max-w-xl mx-auto py-24 px-4 text-center">
        <div className="editorial-panel p-8 space-y-4">
          <div className="flex h-12 w-12 items-center justify-center rounded-lg bg-[#F0E8EE] text-[#76556F] mx-auto">
            <Activity className="h-6 w-6" />
          </div>
          <h2 className="text-xl font-serif font-bold text-[#27232B]">
            No Analysis Available Yet
          </h2>
          <p className="text-xs text-[#6B626E] leading-relaxed max-w-md mx-auto">
            Upload an audio track from the overview dashboard or load an FMA Small benchmark track to inspect spectrograms, harmonic chroma, and 2D CNN genre predictions.
          </p>
          <div className="flex flex-col sm:flex-row items-center justify-center gap-3 pt-2">
            <Link
              href="/studio"
              className="w-full sm:w-auto px-4 py-2 rounded-lg bg-[#76556F] text-white text-xs font-semibold hover:bg-[#583C52] transition-colors shadow-2xs"
            >
              Go to Studio Workspace
            </Link>
            <button
              onClick={() => loadBenchmarkSample("/samples/sample_000002.mp3")}
              disabled={isLoadingSample}
              className="w-full sm:w-auto px-4 py-2 rounded-lg border border-[#E3DDD4] bg-white text-xs font-semibold text-[#27232B] hover:bg-[#FAF7F2] transition-colors"
            >
              {isLoadingSample ? "Analyzing..." : "Load FMA 000002 (Hip-Hop)"}
            </button>
          </div>
        </div>
      </div>
    );
  }

  const {
    filename,
    duration_seconds,
    sample_rate,
    genre,
    acoustic_features,
    visualizations,
    warnings,
  } = analysis;

  return (
    <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8 space-y-8">
      {/* Top Context Bar */}
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4 pb-5 border-b border-[#E3DDD4]">
        <div className="flex items-center gap-3">
          <Link
            href="/studio"
            className="flex items-center gap-1.5 text-xs text-[#6B626E] hover:text-[#76556F] font-semibold px-3 py-1.5 rounded-lg border border-[#E3DDD4] bg-white shadow-2xs transition-colors"
          >
            <ArrowLeft className="h-3.5 w-3.5" />
            <span>Studio Workspace</span>
          </Link>
          <div className="min-w-0">
            <div className="flex items-center gap-2">
              <h1 className="text-xl sm:text-2xl font-serif font-bold text-[#27232B] truncate max-w-md">
                {filename}
              </h1>
              <span className="font-mono text-[10px] px-2 py-0.5 rounded bg-[#FAF7F2] border border-[#E3DDD4] text-[#6B626E]">
                {sample_rate} Hz Mono
              </span>
            </div>
          </div>
        </div>

        <div className="flex items-center gap-2 shrink-0">
          <button
            onClick={() =>
              loadBenchmarkSample(
                filename.includes("000002")
                  ? "/samples/sample_000005.mp3"
                  : "/samples/sample_000002.mp3"
              )
            }
            disabled={isLoadingSample}
            className="flex items-center gap-1.5 text-xs font-medium px-3.5 py-1.5 rounded-md border border-[#E3DDD4] bg-white/90 text-[#27232B] hover:bg-[#FAF7F2] transition-colors shadow-2xs"
          >
            <RefreshCw
              className={`h-3.5 w-3.5 text-[#76556F] ${isLoadingSample ? "animate-spin" : ""}`}
            />
            <span>
              {isLoadingSample
                ? "Analyzing..."
                : filename.includes("000002")
                ? "Compare FMA 000005 (Rock)"
                : "Compare FMA 000002 (Hip-Hop)"}
            </span>
          </button>
        </div>
      </div>

      {/* Floating / Lightly Elevated Audio Player */}
      {sampleAudioUrl && (
        <div className="sticky top-20 z-20">
          <AudioPlayer
            audioUrl={sampleAudioUrl}
            filename={filename}
            onTimeUpdate={(t) => setCurrentTime(t)}
            seekTime={seekTime}
          />
        </div>
      )}

      {/* SECTION A: Track Overview & Quantitative Metric Chips */}
      <div className="grid grid-cols-2 sm:grid-cols-4 gap-4">
        <div className="editorial-panel p-4">
          <div className="flex items-center gap-1.5 text-[#6B626E] text-[10px] font-semibold uppercase tracking-wider mb-1">
            <Compass className="h-3.5 w-3.5 text-[#76556F]" />
            <span>Primary Prediction</span>
          </div>
          <div className="text-lg font-serif font-bold text-[#27232B] truncate">
            {genre.top_label}
          </div>
          <p className="text-[10px] font-mono text-[#D77C63] mt-0.5 font-medium">
            {(genre.confidence * 100).toFixed(1)}% Confidence
          </p>
        </div>

        <div className="editorial-panel p-4">
          <div className="flex items-center gap-1.5 text-[#6B626E] text-[10px] font-semibold uppercase tracking-wider mb-1">
            <Gauge className="h-3.5 w-3.5 text-[#D77C63]" />
            <span>Estimated Tempo</span>
          </div>
          <div className="text-lg font-mono font-bold text-[#27232B] tabular-nums">
            {acoustic_features.tempo_bpm_estimate.toFixed(1)}
          </div>
          <p className="text-[10px] text-[#6B626E] mt-0.5">
            Librosa beat tracking
          </p>
        </div>

        <div className="editorial-panel p-4">
          <div className="flex items-center gap-1.5 text-[#6B626E] text-[10px] font-semibold uppercase tracking-wider mb-1">
            <Clock className="h-3.5 w-3.5 text-[#76556F]" />
            <span>Decoded Window</span>
          </div>
          <div className="text-lg font-mono font-bold text-[#27232B] tabular-nums">
            {duration_seconds.toFixed(1)}s
          </div>
          <p className="text-[10px] text-[#6B626E] mt-0.5">
            Evaluated audio crop
          </p>
        </div>

        <div className="editorial-panel p-4">
          <div className="flex items-center gap-1.5 text-[#6B626E] text-[10px] font-semibold uppercase tracking-wider mb-1">
            <Scale className="h-3.5 w-3.5 text-[#A6B5AD]" />
            <span>Decision State</span>
          </div>
          <div
            className={`text-sm font-semibold truncate ${
              genre.is_uncertain ? "text-[#D6A75D]" : "text-[#76556F]"
            }`}
          >
            {genre.is_uncertain ? "Uncertain / Hybrid" : "Confident Class"}
          </div>
          <p className="text-[10px] text-[#6B626E] mt-0.5">
            {genre.is_uncertain ? "Narrow score margin" : "Clear winner margin"}
          </p>
        </div>
      </div>

      {/* SECTION B & C: Main Asymmetrical Visualization & Prediction Column */}
      <div id="genre" className="grid grid-cols-1 lg:grid-cols-12 gap-6 items-start">
        {/* Broad Main Visualization Area (7 cols) */}
        <div className="lg:col-span-7 space-y-6">
          <WaveformChart
            waveform={visualizations.waveform}
            durationSeconds={duration_seconds}
            currentTime={currentTime}
            onSeek={handleSeek}
          />

          <SpectrogramHeatmap
            spectrogram={visualizations.spectrogram}
            durationSeconds={duration_seconds}
          />
        </div>

        {/* Narrower Genre Prediction Column (5 cols) */}
        <div className="lg:col-span-5 space-y-6">
          <GenreRanking genre={genre} />

          {/* Model Specification Card */}
          <div className="editorial-panel p-4 text-xs text-[#6B626E] space-y-2">
            <div className="flex items-center gap-2 text-[#27232B] font-semibold">
              <Cpu className="h-4 w-4 text-[#76556F]" />
              <span>Inference Engine: GenreCNN</span>
            </div>
            <p className="leading-relaxed">
              Trained on 6,394 FMA Small tracks across 8 balanced classes. 10-second multi-window average over 128-band log-mel STFT features.
            </p>
            <div className="pt-2 border-t border-[#E3DDD4] flex justify-between font-mono text-[10px]">
              <span>Sample Rate: {sample_rate} Hz</span>
              <span className="text-[#76556F]">PyTorch CPU Engine</span>
            </div>
          </div>
        </div>
      </div>

      {/* SECTION C (Cont.): Harmonic Chroma & Signal Energy Profile */}
      <ChromaChart features={acoustic_features} />

      {/* SECTION D: Similar Tracks & Auxiliary Discovery */}
      <div id="similarity" className="editorial-panel p-6">
        <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-2 pb-4 border-b border-[#E3DDD4] mb-5">
          <div className="flex items-center gap-2">
            <GitCompare className="h-4 w-4 text-[#76556F]" />
            <h3 className="text-xs font-semibold uppercase tracking-wider text-[#27232B]">
              Acoustic Similarity Discovery (Top-5 Nearest Neighbors)
            </h3>
          </div>
          <span
            className={`text-[10px] font-mono px-2 py-0.5 rounded ${
              analysis.similarity_status === "available" &&
              analysis.similar_tracks &&
              analysis.similar_tracks.length > 0
                ? "bg-[#F0E8EE] text-[#76556F]"
                : "bg-[#FBF6EC] text-[#D6A75D]"
            }`}
          >
            {analysis.similarity_status === "available" &&
            analysis.similar_tracks &&
            analysis.similar_tracks.length > 0
              ? "Cosine Top-5 Catalog Index"
              : "Catalog Index Unbuilt"}
          </span>
        </div>

        {analysis.similarity_status === "available" &&
        analysis.similar_tracks &&
        analysis.similar_tracks.length > 0 ? (
          <div className="space-y-3">
            <p className="text-xs text-[#6B626E] leading-relaxed mb-3">
              Retrieved from the FMA catalog using cosine distance over standardized 513-dimensional audio descriptors (spectral contrast, chromagrams, MFCCs, and tempo rhythmics).
            </p>

            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
              {analysis.similar_tracks.map((track) => (
                <div
                  key={track.track_id}
                  className="p-3.5 rounded-lg border border-[#E3DDD4] bg-[#FAF7F2] flex flex-col justify-between"
                >
                  <div>
                    <div className="flex items-start justify-between gap-2 mb-1.5">
                      <div className="min-w-0">
                        <h4 className="text-xs font-semibold text-[#27232B] truncate">
                          {track.title}
                        </h4>
                        <p className="text-[11px] text-[#6B626E] truncate">
                          {track.artist}
                        </p>
                      </div>
                      <span className="font-mono text-[9px] px-1.5 py-0.2 rounded bg-white border border-[#E3DDD4] text-[#8E8692] shrink-0">
                        #{track.track_id}
                      </span>
                    </div>

                    <div className="flex items-center gap-2 mb-2">
                      <span className="text-[10px] font-mono px-2 py-0.5 rounded bg-[#F0E8EE] text-[#76556F] font-medium">
                        {track.genre}
                      </span>
                    </div>
                  </div>

                  <div className="space-y-1.5 pt-2 border-t border-[#E3DDD4]">
                    <div className="flex items-center justify-between text-[11px]">
                      <span className="text-[#6B626E]">Cosine Similarity</span>
                      <span className="font-mono font-semibold text-[#D77C63] tabular-nums">
                        {track.feature_similarity_score.toFixed(4)}
                      </span>
                    </div>
                    <div className="h-1 w-full bg-[#EAE2D8] rounded-xs overflow-hidden">
                      <div
                        className="h-full bg-[#76556F] rounded-xs"
                        style={{
                          width: `${Math.min(100, Math.max(0, track.feature_similarity_score * 100))}%`,
                        }}
                      />
                    </div>
                    <p className="text-[10px] text-[#8E8692] italic pt-0.5 leading-snug">
                      {track.explanation}
                    </p>
                  </div>
                </div>
              ))}
            </div>
          </div>
        ) : (
          <div className="p-4 rounded-lg bg-[#FAF7F2] border border-[#E3DDD4] text-xs text-[#6B626E] space-y-1.5">
            <div className="font-medium text-[#27232B]">Catalog Index Offline / Unbuilt</div>
            <p className="leading-relaxed">
              Acoustic nearest neighbor indexing has not been precomputed on this server instance. Sonic Fusion maintains scientific integrity and never fakes or hallucinates recommendations.
            </p>
            <p className="font-mono text-[10px] text-[#76556F] pt-1">
              Build command: python scripts/07_build_similarity_index.py
            </p>
          </div>
        )}
      </div>

      {/* SECTION E: Technical Disclaimers & Pipeline Diagnostics */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
        <div className="editorial-panel p-5">
          <div className="flex items-center gap-2 mb-3">
            <Info className="h-4 w-4 text-[#76556F]" />
            <h4 className="text-xs font-semibold uppercase tracking-wider text-[#27232B]">
              Acoustic Disclaimers &amp; Scope
            </h4>
          </div>
          <ul className="space-y-2 text-xs text-[#6B626E]">
            {warnings && warnings.length > 0 ? (
              warnings.map((w, idx) => (
                <li key={idx} className="flex items-start gap-2">
                  <span className="text-[#D77C63]">&bull;</span>
                  <span>{w}</span>
                </li>
              ))
            ) : (
              <li>Analysis completed with 0 signal anomalies or decode warnings reported.</li>
            )}
          </ul>
        </div>

        <div className="editorial-panel p-5">
          <div className="flex items-center gap-2 mb-3">
            <Layers className="h-4 w-4 text-[#76556F]" />
            <h4 className="text-xs font-semibold uppercase tracking-wider text-[#27232B]">
              Pipeline Specifications
            </h4>
          </div>
          <p className="text-xs text-[#6B626E] leading-relaxed mb-2">
            The input audio is standardized to a 30.0-second crop at 22,050 Hz mono. Log-mel power spectrograms are extracted using a 2,048-point STFT window with a 512-point hop.
          </p>
          <div className="flex items-center justify-between text-[10px] font-mono text-[#8E8692] pt-2 border-t border-[#E3DDD4]">
            <span>Zero-loss linear decimation</span>
            <span>Deterministic CPU inference</span>
          </div>
        </div>
      </div>
    </div>
  );
}
