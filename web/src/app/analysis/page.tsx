"use client";

import React, { useEffect, useState } from "react";
import Link from "next/link";
import {
  Activity,
  AlertTriangle,
  ArrowLeft,
  Clock,
  Cpu,
  Disc3,
  FileAudio,
  Gauge,
  HelpCircle,
  Info,
  Layers,
  Music2,
  RefreshCw,
  Sparkles,
  Waves,
  Zap,
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
      // If it matches a sample track, set its URL for audio playback
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
      <div className="flex-1 flex flex-col items-center justify-center p-6 text-center max-w-xl mx-auto">
        <div className="flex h-16 w-16 items-center justify-center rounded-2xl bg-indigo-500/10 text-indigo-400 border border-indigo-500/20 mb-4">
          <Activity className="h-8 w-8" />
        </div>
        <h2 className="text-2xl font-bold text-white mb-2">
          No Audio Analysis Available
        </h2>
        <p className="text-sm text-slate-400 mb-6">
          Upload an audio file to inspect deep learning genre recognition,
          spectral heatmaps, and signal metrics, or load a benchmark track now.
        </p>

        <div className="flex flex-col sm:flex-row items-center gap-3 w-full">
          <Link
            href="/"
            className="w-full sm:w-auto flex-1 flex items-center justify-center gap-2 px-5 py-2.5 rounded-xl bg-indigo-600 text-white font-semibold text-sm hover:bg-indigo-500 transition-colors shadow-lg shadow-indigo-500/20"
          >
            <FileAudio className="h-4 w-4" />
            <span>Go to Upload Page</span>
          </Link>

          <button
            onClick={() => loadBenchmarkSample("/samples/sample_000002.mp3")}
            disabled={isLoadingSample}
            className="w-full sm:w-auto flex-1 flex items-center justify-center gap-2 px-5 py-2.5 rounded-xl border border-slate-700 bg-slate-800 text-slate-200 font-semibold text-sm hover:bg-slate-700 transition-colors"
          >
            <Disc3 className="h-4 w-4 text-cyan-400" />
            <span>{isLoadingSample ? "Analyzing..." : "Load Sample 000002"}</span>
          </button>
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
    <div className="flex-1 py-6 sm:py-8 px-4 sm:px-6 lg:px-8 max-w-7xl mx-auto w-full space-y-6">
      {/* Top Breadcrumb & Actions Bar */}
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-3 pb-4 border-b border-slate-800">
        <div className="flex items-center gap-3">
          <Link
            href="/"
            className="flex items-center gap-1.5 text-xs font-semibold text-slate-400 hover:text-white transition-colors px-2.5 py-1 rounded-lg bg-slate-800/80 border border-slate-700/60"
          >
            <ArrowLeft className="h-3.5 w-3.5" />
            <span>Upload Another</span>
          </Link>
          <div className="flex items-center gap-2">
            <h1 className="text-lg font-bold text-white truncate max-w-md">
              {filename}
            </h1>
            <span className="text-[10px] font-mono px-2 py-0.5 rounded bg-slate-800 text-slate-400 border border-slate-700">
              {sample_rate} Hz Mono
            </span>
          </div>
        </div>

        <div className="flex items-center gap-2">
          <button
            onClick={() =>
              loadBenchmarkSample(
                filename.includes("000002")
                  ? "/samples/sample_000005.mp3"
                  : "/samples/sample_000002.mp3"
              )
            }
            disabled={isLoadingSample}
            className="flex items-center gap-1.5 text-xs text-indigo-400 hover:text-indigo-300 font-medium px-3 py-1.5 rounded-lg border border-indigo-500/30 bg-indigo-950/40 transition-colors"
          >
            <RefreshCw
              className={`h-3.5 w-3.5 ${isLoadingSample ? "animate-spin" : ""}`}
            />
            <span>
              {isLoadingSample ? "Analyzing..." : "Compare with Alternative Sample"}
            </span>
          </button>
        </div>
      </div>

      {/* AUDIO PLAYER PREVIEW IF AVAILABLE */}
      {sampleAudioUrl && (
        <AudioPlayer
          audioUrl={sampleAudioUrl}
          filename={filename}
          onTimeUpdate={(t) => setCurrentTime(t)}
          seekTime={seekTime}
        />
      )}

      {/* LEAD SECTION: Lead with Genre, BPM, Duration, and Waveform (PLAN §14) */}
      <div className="grid grid-cols-1 lg:grid-cols-12 gap-6">
        {/* Left Column: Top Genre & Probability Ranking */}
        <div className="lg:col-span-5 flex flex-col gap-6">
          <GenreRanking genre={genre} />

          {/* Quick Metrics Grid */}
          <div className="grid grid-cols-2 gap-3">
            <div className="p-4 rounded-xl border border-slate-800 bg-slate-900/70">
              <div className="flex items-center gap-1.5 text-slate-400 text-xs mb-1">
                <Gauge className="h-4 w-4 text-amber-400" />
                <span className="font-semibold uppercase tracking-wider text-[10px]">
                  Estimated BPM
                </span>
              </div>
              <div className="text-2xl font-extrabold font-mono text-white">
                {acoustic_features.tempo_bpm_estimate.toFixed(1)}
              </div>
              <p className="text-[10px] text-slate-500 mt-0.5">
                Librosa beat onset estimate
              </p>
            </div>

            <div className="p-4 rounded-xl border border-slate-800 bg-slate-900/70">
              <div className="flex items-center gap-1.5 text-slate-400 text-xs mb-1">
                <Clock className="h-4 w-4 text-cyan-400" />
                <span className="font-semibold uppercase tracking-wider text-[10px]">
                  Decoded Duration
                </span>
              </div>
              <div className="text-2xl font-extrabold font-mono text-white">
                {duration_seconds.toFixed(1)}s
              </div>
              <p className="text-[10px] text-slate-500 mt-0.5">
                Standard 30s window crop
              </p>
            </div>
          </div>
        </div>

        {/* Right Column: Waveform Peak Envelope and Spectrogram Heatmap */}
        <div className="lg:col-span-7 flex flex-col gap-6">
          {/* Waveform */}
          <WaveformChart
            waveform={visualizations.waveform}
            durationSeconds={duration_seconds}
            currentTime={currentTime}
            onSeek={handleSeek}
          />

          {/* Spectrogram Heatmap */}
          <SpectrogramHeatmap
            spectrogram={visualizations.spectrogram}
            durationSeconds={duration_seconds}
          />
        </div>
      </div>

      {/* SECOND SECTION: Chroma Pitch Classes & RMS Dynamics */}
      <ChromaChart features={acoustic_features} />

      {/* THIRD SECTION: Academic Warnings & Optional Tagging / Similarity Panels */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
        {/* Warnings / Caveats */}
        <div className="rounded-xl border border-slate-800 bg-slate-900/60 p-4">
          <div className="flex items-center gap-2 mb-3">
            <Info className="h-4 w-4 text-indigo-400" />
            <h4 className="text-xs font-semibold uppercase tracking-wider text-slate-300">
              Academic Caveats & Disclaimers
            </h4>
          </div>
          <ul className="space-y-2 text-xs text-slate-400">
            {warnings && warnings.length > 0 ? (
              warnings.map((w, idx) => (
                <li key={idx} className="flex items-start gap-2">
                  <span className="text-indigo-400 text-base leading-none">&bull;</span>
                  <span>{w}</span>
                </li>
              ))
            ) : (
              <li className="text-slate-500">No warnings reported.</li>
            )}
          </ul>
        </div>

        {/* Pretrained Tags Status Panel (§14) */}
        <div className="rounded-xl border border-slate-800 bg-slate-900/60 p-4">
          <div className="flex items-center gap-2 mb-3">
            <Layers className="h-4 w-4 text-cyan-400" />
            <h4 className="text-xs font-semibold uppercase tracking-wider text-slate-300">
              Mood & Instrument Tags (§10)
            </h4>
          </div>
          <div className="p-3 rounded-lg bg-slate-950/60 border border-slate-800/80 mb-2">
            <div className="flex items-center justify-between text-xs mb-1">
              <span className="text-slate-300 font-medium">Essentia EffNet</span>
              <span className="text-[10px] font-mono px-2 py-0.5 rounded bg-slate-800 text-slate-400">
                Disabled (CPU Safe)
              </span>
            </div>
            <p className="text-[11px] text-slate-500">
              Per PLAN §10 and §20, optional pretrained Discogs-EffNet mood/instrument heads are kept offline in local CPU mode to prevent TensorFlow/Essentia binary conflicts.
            </p>
          </div>
          <p className="text-[11px] text-slate-500">
            GenreCNN serves verified 8-genre recognition directly.
          </p>
        </div>

        {/* Similar Tracks Panel (§12 / §14) */}
        <div className="rounded-xl border border-slate-800 bg-slate-900/60 p-4">
          <div className="flex items-center justify-between mb-3">
            <div className="flex items-center gap-2">
              <Music2 className="h-4 w-4 text-emerald-400" />
              <h4 className="text-xs font-semibold uppercase tracking-wider text-slate-300">
                Song Similarity (§12)
              </h4>
            </div>
            <span
              className={`text-[10px] font-mono px-2 py-0.5 rounded border ${
                analysis?.similarity_status === "available" &&
                analysis?.similar_tracks &&
                analysis.similar_tracks.length > 0
                  ? "bg-emerald-950 text-emerald-400 border-emerald-800"
                  : "bg-amber-950/80 text-amber-400 border-amber-800"
              }`}
            >
              {analysis?.similarity_status === "available" &&
              analysis?.similar_tracks &&
              analysis.similar_tracks.length > 0
                ? "Cosine Top-5"
                : "Unavailable"}
            </span>
          </div>

          {analysis?.similarity_status === "available" &&
          analysis?.similar_tracks &&
          analysis.similar_tracks.length > 0 ? (
            <div className="space-y-2.5">
              {/* PLAN §12 Content-based similarity retrieval */}
              <p className="text-[11px] text-slate-400 mb-2 leading-relaxed">
                Top-5 acoustic neighbors retrieved from the standardized FMA catalog (training &amp; validation splits only, excluding self-matches). Feature similarity score is a rough content-based proxy (tempo, timbre, chroma and energy features), not a measure of musical similarity.
              </p>
              {analysis.similar_tracks.map((track) => (
                <div
                  key={track.track_id}
                  className="p-3 rounded-lg bg-slate-950/80 border border-slate-800/90 hover:border-emerald-500/40 transition-colors"
                >
                  <div className="flex items-start justify-between gap-2 mb-1.5">
                    <div className="min-w-0">
                      <h5 className="text-xs font-semibold text-slate-200 truncate">
                        {track.title}
                      </h5>
                      <p className="text-[11px] text-slate-400 truncate">
                        {track.artist}
                      </p>
                    </div>
                    <div className="flex items-center gap-1.5 shrink-0">
                      <span className="text-[10px] font-mono px-1.5 py-0.5 rounded bg-slate-800 text-slate-400 border border-slate-700">
                        #{track.track_id}
                      </span>
                      <span className="text-[10px] font-semibold px-2 py-0.5 rounded bg-cyan-950 text-cyan-400 border border-cyan-800">
                        {track.genre}
                      </span>
                    </div>
                  </div>

                  <div className="space-y-1">
                    <div className="flex items-center justify-between text-[11px]">
                      <span className="text-slate-400">Feature Similarity Score</span>
                      <span className="font-mono font-medium text-emerald-400">
                        {track.feature_similarity_score.toFixed(4)}
                      </span>
                    </div>
                    <div className="h-1.5 w-full bg-slate-800 rounded-full overflow-hidden">
                      <div
                        className="h-full bg-gradient-to-r from-emerald-500 to-teal-400 rounded-full"
                        style={{ width: `${Math.min(100, Math.max(0, track.feature_similarity_score * 100))}%` }}
                      />
                    </div>
                    <p className="text-[10px] text-slate-400 italic pt-0.5">
                      {track.explanation}
                    </p>
                  </div>
                </div>
              ))}
            </div>
          ) : (
            <div className="p-3.5 rounded-lg bg-slate-950/60 border border-slate-800 text-left space-y-1.5">
              <div className="flex items-center justify-between text-xs">
                <span className="text-slate-300 font-medium">Catalog Index Status</span>
                <span className="text-[10px] font-mono text-amber-400">unavailable</span>
              </div>
              <p className="text-[11px] text-slate-400 leading-relaxed">
                Similarity catalog index is currently unavailable on this instance. SonicScope reports honest unavailability and never synthesizes or fakes recommendation results. Run <code className="text-emerald-400 font-mono text-[10px]">python scripts/07_build_similarity_index.py</code> to build the index.
              </p>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
