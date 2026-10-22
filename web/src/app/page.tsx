"use client";

import React, { useState, useEffect } from "react";
import Link from "next/link";
import {
  Activity,
  ArrowRight,
  BarChart2,
  CheckCircle2,
  ChevronRight,
  Clock,
  Compass,
  Cpu,
  Database,
  Disc,
  Disc3,
  FileAudio,
  GitCompare,
  HelpCircle,
  Layers,
  Music,
  Radio,
  Scale,
  ShieldCheck,
  Sparkles,
  Upload,
  Waves,
  Zap,
} from "lucide-react";
import { checkHealth, getMetrics } from "@/lib/api";
import { HealthResponse, MetricsResponse } from "@/types/api";

export default function LandingPage() {
  const [health, setHealth] = useState<HealthResponse | null>(null);
  const [metrics, setMetrics] = useState<MetricsResponse | null>(null);
  const [activeFeatureTab, setActiveFeatureTab] = useState<number>(0);
  const [selectedDemoPreset, setSelectedDemoPreset] = useState<"hiphop" | "rock">("hiphop");

  useEffect(() => {
    checkHealth()
      .then((data) => setHealth(data))
      .catch(() => {});

    getMetrics()
      .then((data) => setMetrics(data))
      .catch(() => {});
  }, []);

  const testMetrics = metrics?.test_metrics;

  const demoTracks = {
    hiphop: {
      name: "FMA 000002",
      genre: "Hip-Hop",
      confidence: "82.4%",
      tempo: "113.7 BPM",
      rms: "0.2272",
      centroid: "2,276 Hz",
      keyRes: "C / C# minor",
      sampleUrl: "/samples/sample_000002.mp3",
      neighbors: ["FMA 000694", "FMA 001086", "FMA 001482", "FMA 002134", "FMA 003891"],
    },
    rock: {
      name: "FMA 000005",
      genre: "Rock",
      confidence: "89.1%",
      tempo: "121.8 BPM",
      rms: "0.2125",
      centroid: "2,137 Hz",
      keyRes: "E / G major",
      sampleUrl: "/samples/sample_000005.mp3",
      neighbors: ["FMA 000182", "FMA 000459", "FMA 001928", "FMA 002441", "FMA 003102"],
    },
  };

  const currentDemo = demoTracks[selectedDemoPreset];

  const featurePillars = [
    {
      id: "genre",
      title: "Genre Intelligence",
      subtitle: "2D Convolutional Neural Network",
      tag: "Deep Learning",
      tagColor: "bg-[#F0E8EE] text-[#76556F] border-[#E3DDD4]",
      icon: Radio,
      description:
        "Classify uploaded audio into 8 balanced genres from the Free Music Archive benchmark using an optimized 3-block 2D CNN with batch normalization and adaptive global pooling.",
      stats: [
        { label: "Target Classes", val: "8 Closed Genres" },
        { label: "Calibration", val: "Uncertainty Policy" },
        { label: "Inference Latency", val: "~45 ms (CPU)" },
      ],
      detail:
        "Outputs complete softmax probability distributions across Electronic, Experimental, Folk, Hip-Hop, Instrumental, International, Pop, and Rock. Prediction margins under 15% automatically trigger advisory boundary disclaimers.",
    },
    {
      id: "visualization",
      title: "Audio Visualization",
      subtitle: "Multi-Modal Spectral Exploration",
      tag: "Spectral Signal",
      tagColor: "bg-[#FDF1EE] text-[#D77C63] border-[#F1D0CD]",
      icon: Waves,
      description:
        "Explore real-time interactive audio representations: 128-band normalized log-mel spectrogram heatmaps, peak waveform envelopes, and 12-bin harmonic pitch-class profiles.",
      stats: [
        { label: "Mel Channels", val: "128 Frequency Bins" },
        { label: "Waveform Res", val: "22,050 Hz Mono" },
        { label: "Chroma Profile", val: "12 Semitone Bins" },
      ],
      detail:
        "Log-mel representations mirror human psychoacoustic pitch perception, providing deep intuitive visibility into track frequency energy, rhythm, and harmonic structure.",
    },
    {
      id: "acoustic",
      title: "Acoustic Analysis",
      subtitle: "Physical Signal & Dynamics Extraction",
      tag: "Signal Features",
      tagColor: "bg-[#FBF6EC] text-[#D6A75D] border-[#EEDFCA]",
      icon: Compass,
      description:
        "Quantify core physical audio characteristics: algorithmic tempo (BPM) from onset strength envelopes, dynamic RMS loudness contours, and spectral centroid brightness.",
      stats: [
        { label: "Tempo Tracking", val: "Onset Peak Window" },
        { label: "Dynamic Energy", val: "Frame RMS Mean & Std" },
        { label: "Spectral Centroid", val: "Timbral Brightness" },
      ],
      detail:
        "Deterministic feature extraction via standardized Librosa 0.10 algorithms, ensuring reproducible and comparable acoustic metrics across any ingested recording.",
    },
    {
      id: "comparison",
      title: "Model Comparison",
      subtitle: "Deep Learning vs. Classical Random Forest",
      tag: "Scientific Benchmark",
      tagColor: "bg-[#EEF3F0] text-[#6C7F75] border-[#D5E1DB]",
      icon: Scale,
      description:
        "Compare the 2D CNN against the classical Random Forest baseline trained on 51 handcrafted acoustic features, evaluated with 95% bootstrap confidence intervals on held-out test splits.",
      stats: [
        { label: "Baseline Features", val: "51-D Handcrafted" },
        { label: "Official Test Split", val: "800 Tracks (Zero Overlap)" },
        { label: "Significance", val: "McNemar Chi-Square" },
      ],
      detail:
        "Rigorous partition isolation ensures zero artist leakage across training, validation, and held-out test splits, upholding scientific machine learning integrity.",
    },
    {
      id: "similarity",
      title: "Similar Track Discovery",
      subtitle: "High-Dimensional Acoustic Neighbors",
      tag: "Vector Search",
      tagColor: "bg-[#F0E8EE] text-[#76556F] border-[#E3DDD4]",
      icon: GitCompare,
      description:
        "Find acoustically similar tracks across the 8,000-track FMA catalog using cosine distance over 513-dimensional standardized feature vectors.",
      stats: [
        { label: "Retrieved Neighbors", val: "Top-5 Cosine Matches" },
        { label: "Catalog Corpus", val: "7,994 Audited Tracks" },
        { label: "Feature Vector", val: "513-D Multimodal" },
      ],
      detail:
        "Identifies genuine acoustic relatives based on timbre, tempo, and chroma dynamics—revealing surprising aesthetic connections that transcend conventional genre labels.",
    },
    {
      id: "exploration",
      title: "Music Exploration",
      subtitle: "Collection-Level Distribution & Analytics",
      tag: "Corpus Analytics",
      tagColor: "bg-[#FDF1EE] text-[#D77C63] border-[#F1D0CD]",
      icon: Database,
      description:
        "Investigate macro-level patterns across the audited Free Music Archive benchmark: track duration distributions, split partitions, and acoustic differences across genres.",
      stats: [
        { label: "Total Tracks", val: "8,000 Raw MP3s" },
        { label: "Decodable Tracks", val: "7,994 (99.93%)" },
        { label: "Official Partitions", val: "6,394 / 800 / 800" },
      ],
      detail:
        "Transparent exploratory data analysis documenting excluded corrupt tracks, license distributions, and feature correlation matrices across the whole music corpus.",
    },
  ];

  return (
    <div className="w-full bg-[#F5F0E8] text-[#27232B] overflow-hidden">
      {/* ========================================================================= */}
      {/*                            1. HERO SECTION                                */}
      {/* ========================================================================= */}
      <section className="relative pt-12 pb-20 md:pt-20 md:pb-32 px-4 sm:px-6 lg:px-8 max-w-7xl mx-auto">
        {/* Ambient Sound Wave Gradient Mesh in Background */}
        <div className="absolute inset-0 pointer-events-none -z-10 bg-sound-mesh opacity-80" />

        <div className="grid grid-cols-1 lg:grid-cols-12 gap-12 lg:gap-8 items-center">
          {/* Left Column: Editorial Headline & Primary CTA */}
          <div className="lg:col-span-7 space-y-6 md:space-y-8">
            {/* Top Atmospheric Badge */}
            <div className="inline-flex items-center gap-2.5 px-3.5 py-1.5 rounded-full glass-pill text-[#76556F] text-xs font-semibold shadow-2xs">
              <span className="flex h-2 w-2 rounded-full bg-[#D77C63] animate-pulse" />
              <span className="tracking-wide">Sonic Fusion 2.0 &bull; Music Intelligence &amp; MIR Engine</span>
            </div>

            {/* Exact Required Headline */}
            <h1 className="text-4xl sm:text-5xl lg:text-6xl font-serif font-extrabold text-[#27232B] tracking-tight leading-[1.12]">
              Welcome to Sonic Fusion, your music genre classifier.
            </h1>

            {/* Exact Required Concise Description */}
            <p className="text-base sm:text-lg text-[#6B626E] leading-relaxed max-w-2xl font-normal">
              Sonic Fusion transforms audio into insights. Upload a track, explore its genre, visualize its acoustic features, and discover the patterns that make every piece of music unique.
            </p>

            {/* Two Prominent Actions */}
            <div className="flex flex-col sm:flex-row items-stretch sm:items-center gap-3.5 pt-2">
              <Link
                href="/studio"
                className="inline-flex items-center justify-center gap-2.5 px-6 py-3.5 rounded-xl bg-[#76556F] hover:bg-[#583C52] text-white font-semibold text-sm shadow-md hover:shadow-lg transition-all duration-200 group text-center cursor-pointer"
              >
                <span>Enter the Studio</span>
                <ArrowRight className="h-4 w-4 text-[#D77C63] group-hover:translate-x-1 transition-transform" />
              </Link>

              <a
                href="#features"
                className="inline-flex items-center justify-center gap-2 px-6 py-3.5 rounded-xl glass-pill hover:bg-white text-[#27232B] font-semibold text-sm transition-all duration-200 text-center cursor-pointer"
              >
                <span>Explore Features</span>
                <ChevronRight className="h-4 w-4 text-[#76556F]" />
              </a>
            </div>

            {/* Live Model & Provenance Status Pill */}
            <div className="pt-2 flex flex-wrap items-center gap-4 text-xs text-[#8E8692]">
              <div className="flex items-center gap-1.5 font-medium">
                <span className="h-2 w-2 rounded-full bg-[#6C7F75]" />
                <span>
                  Backend: {health?.model_loaded ? `FastAPI Ready (${health.device.toUpperCase()})` : "Active"}
                </span>
              </div>
              <span>&bull;</span>
              <span>Free Music Archive (FMA Small)</span>
              <span>&bull;</span>
              <span>8 Closed-World Genres</span>
            </div>
          </div>

          {/* Right Column: Hero Visual (Layered Waves, Rings & Translucent Floating Panels) */}
          <div className="lg:col-span-5 relative flex items-center justify-center min-h-[440px] sm:min-h-[500px]">
            {/* Concentric Sound Propagation Rings */}
            <div className="absolute w-[360px] h-[360px] sm:w-[440px] sm:h-[440px] concentric-ring animate-concentric opacity-40 -z-10" />
            <div className="absolute w-[280px] h-[280px] sm:w-[340px] sm:h-[340px] concentric-ring opacity-60 -z-10" />
            <div className="absolute w-[200px] h-[200px] sm:w-[240px] sm:h-[240px] concentric-ring opacity-80 -z-10" />

            {/* Flowing Sound Waveform Graphic Container */}
            <div className="relative w-full max-w-[420px] p-6 sm:p-7 rounded-3xl glass-floating space-y-5">
              {/* Top Bar of Floating Preview */}
              <div className="flex items-center justify-between pb-3.5 border-b border-[#E3DDD4]">
                <div className="flex items-center gap-2.5">
                  <div className="p-2 rounded-xl bg-[#F0E8EE] text-[#76556F] shadow-2xs">
                    <Radio className="h-4 w-4" />
                  </div>
                  <div>
                    <span className="text-[10px] font-mono uppercase text-[#76556F] font-bold block">
                      Live Signal Ingestion
                    </span>
                    <span className="text-xs font-bold text-[#27232B]">
                      {currentDemo.name} &bull; {currentDemo.genre}
                    </span>
                  </div>
                </div>
                <span className="font-mono text-xs px-2.5 py-0.5 rounded-full bg-[#FDF1EE] text-[#D77C63] font-bold border border-[#F1D0CD]">
                  {currentDemo.confidence} Match
                </span>
              </div>

              {/* Dynamic Animated Sound Waveform Visualization */}
              <div className="space-y-2">
                <div className="flex items-center justify-between text-[11px] text-[#6B626E]">
                  <span>Log-Mel Spectrogram Envelope</span>
                  <span className="font-mono text-[10px] text-[#8E8692]">128 Channels</span>
                </div>

                <div className="h-16 flex items-end justify-between gap-1 p-2.5 rounded-xl bg-[#FAF7F2] border border-[#E3DDD4] overflow-hidden">
                  {[
                    40, 65, 30, 85, 55, 95, 45, 80, 25, 90, 60, 40, 75, 50, 88, 35,
                    70, 45, 95, 60, 30, 85, 50, 75, 40, 90, 65, 35, 80, 55, 70, 45,
                  ].map((val, idx) => (
                    <div
                      key={idx}
                      className={`flex-1 rounded-full transition-all duration-300 ${
                        idx % 3 === 0
                          ? "bg-[#76556F] animate-eq-slow-1"
                          : idx % 3 === 1
                          ? "bg-[#D77C63] animate-eq-slow-2"
                          : "bg-[#D6A75D] animate-eq-slow-3"
                      }`}
                      style={{
                        height: `${Math.max(15, val * 0.6)}%`,
                        animationDelay: `${idx * 0.05}s`,
                      }}
                    />
                  ))}
                </div>
              </div>

              {/* Floating Real Acoustic Data Chips */}
              <div className="grid grid-cols-3 gap-2.5 text-center">
                <div className="p-2.5 rounded-xl bg-white border border-[#E3DDD4] shadow-2xs">
                  <span className="text-[10px] text-[#8E8692] block font-medium">Tempo</span>
                  <span className="font-mono text-xs font-bold text-[#27232B]">{currentDemo.tempo}</span>
                </div>
                <div className="p-2.5 rounded-xl bg-white border border-[#E3DDD4] shadow-2xs">
                  <span className="text-[10px] text-[#8E8692] block font-medium">Brightness</span>
                  <span className="font-mono text-xs font-bold text-[#76556F]">{currentDemo.centroid}</span>
                </div>
                <div className="p-2.5 rounded-xl bg-white border border-[#E3DDD4] shadow-2xs">
                  <span className="text-[10px] text-[#8E8692] block font-medium">Audio Sample</span>
                  <span className="font-mono text-xs font-bold text-[#D77C63]">22.05 kHz</span>
                </div>
              </div>

              {/* Toggle preset selector */}
              <div className="pt-2 flex items-center justify-between text-xs border-t border-[#E3DDD4]">
                <span className="text-[#6B626E] text-[11px]">Toggle Audited Benchmark:</span>
                <div className="flex items-center gap-1.5">
                  <button
                    onClick={() => setSelectedDemoPreset("hiphop")}
                    className={`px-2.5 py-1 rounded-md text-[11px] font-semibold transition-all ${
                      selectedDemoPreset === "hiphop"
                        ? "bg-[#76556F] text-white shadow-2xs"
                        : "bg-white text-[#6B626E] border border-[#E3DDD4] hover:bg-[#FAF7F2]"
                    }`}
                  >
                    Hip-Hop (000002)
                  </button>
                  <button
                    onClick={() => setSelectedDemoPreset("rock")}
                    className={`px-2.5 py-1 rounded-md text-[11px] font-semibold transition-all ${
                      selectedDemoPreset === "rock"
                        ? "bg-[#76556F] text-white shadow-2xs"
                        : "bg-white text-[#6B626E] border border-[#E3DDD4] hover:bg-[#FAF7F2]"
                    }`}
                  >
                    Rock (000005)
                  </button>
                </div>
              </div>
            </div>

            {/* Overlapping Floating Sub-Card (Nearest Neighbors Preview) */}
            <div className="hidden sm:block absolute -bottom-6 -left-6 p-4 rounded-2xl glass-surface w-64 shadow-lg border border-[#E3DDD4] animate-float-slow">
              <div className="flex items-center justify-between mb-1.5">
                <span className="text-[10px] font-bold uppercase tracking-wider text-[#76556F] flex items-center gap-1">
                  <GitCompare className="h-3 w-3" />
                  <span>Acoustic Neighbors</span>
                </span>
                <span className="text-[10px] font-mono text-[#6C7F75] font-semibold">513-D Cosine</span>
              </div>
              <p className="text-[11px] text-[#6B626E]">
                {currentDemo.neighbors.slice(0, 3).join(", ")} + 2 more
              </p>
            </div>
          </div>
        </div>
      </section>

      {/* ========================================================================= */}
      {/*                          2. FEATURES SECTION                              */}
      {/* ========================================================================= */}
      <section id="features" className="py-20 md:py-28 px-4 sm:px-6 lg:px-8 max-w-7xl mx-auto border-t border-[#E3DDD4]">
        <div className="space-y-4 mb-14 text-center max-w-3xl mx-auto">
          <div className="inline-flex items-center gap-2 px-3 py-1 rounded-full bg-[#F0E8EE] text-[#76556F] text-xs font-bold uppercase tracking-wider">
            <span>Capabilities &amp; MIR Architecture</span>
          </div>
          {/* Exact Required Title */}
          <h2 className="text-3xl sm:text-5xl font-serif font-bold text-[#27232B] tracking-tight">
            More than just a genre.
          </h2>
          <p className="text-sm sm:text-base text-[#6B626E] leading-relaxed">
            Genre classification is just the entry point. Sonic Fusion extracts multi-dimensional acoustic representations, providing unprecedented visibility into the physical, spectral, and emotional anatomy of sound.
          </p>
        </div>

        {/* Feature Pillars Asymmetric Grid */}
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6 sm:gap-8">
          {featurePillars.map((pillar, idx) => {
            const Icon = pillar.icon;
            return (
              <div
                key={pillar.id}
                className="group p-7 rounded-2xl glass-card-interactive flex flex-col justify-between"
              >
                <div className="space-y-4">
                  <div className="flex items-center justify-between">
                    <div className="p-3 rounded-xl bg-[#F0E8EE] text-[#76556F] group-hover:bg-[#76556F] group-hover:text-white transition-colors duration-300 shadow-2xs">
                      <Icon className="h-5 w-5" />
                    </div>
                    <span className={`text-[10px] font-mono font-bold px-2.5 py-0.5 rounded-full border ${pillar.tagColor}`}>
                      {pillar.tag}
                    </span>
                  </div>

                  <div>
                    <h3 className="text-xl font-serif font-bold text-[#27232B] group-hover:text-[#76556F] transition-colors">
                      {pillar.title}
                    </h3>
                    <p className="text-xs font-mono text-[#8E8692] mt-0.5">{pillar.subtitle}</p>
                  </div>

                  <p className="text-xs text-[#6B626E] leading-relaxed">
                    {pillar.description}
                  </p>
                </div>

                {/* Key Spec Stats */}
                <div className="pt-6 mt-6 border-t border-[#E3DDD4] space-y-2">
                  <div className="grid grid-cols-2 gap-2 text-[11px]">
                    {pillar.stats.slice(0, 2).map((s, sIdx) => (
                      <div key={sIdx} className="p-2 rounded-lg bg-[#FAF7F2] border border-[#E3DDD4]/60">
                        <span className="text-[10px] text-[#8E8692] block">{s.label}</span>
                        <span className="font-semibold text-[#27232B] truncate block">{s.val}</span>
                      </div>
                    ))}
                  </div>
                  <p className="text-[11px] text-[#8E8692] leading-tight pt-1">
                    {pillar.detail}
                  </p>
                </div>
              </div>
            );
          })}
        </div>

        {/* Interactive Feature Deep Dive Drawer / Live Studio Preview */}
        <div className="mt-14 p-8 sm:p-10 rounded-3xl glass-floating border border-[#E3DDD4]">
          <div className="flex flex-col lg:flex-row lg:items-center justify-between gap-6 pb-6 border-b border-[#E3DDD4]">
            <div className="space-y-1.5">
              <span className="text-[10px] font-mono font-bold uppercase tracking-wider text-[#76556F] bg-[#F0E8EE] px-2.5 py-0.5 rounded">
                Live Studio Demonstration
              </span>
              <h3 className="text-2xl font-serif font-bold text-[#27232B]">
                Interactive Signal Inspection in Action
              </h3>
              <p className="text-xs sm:text-sm text-[#6B626E] max-w-xl">
                Switch between audio feature views extracted live from the audited FMA Small benchmark clips.
              </p>
            </div>

            <Link
              href="/studio"
              className="inline-flex items-center gap-2 px-5 py-2.5 rounded-xl bg-[#76556F] hover:bg-[#583C52] text-white text-xs font-semibold transition-all shadow-sm shrink-0 self-start lg:self-center"
            >
              <span>Launch Studio With This Track</span>
              <ArrowRight className="h-4 w-4 text-[#D77C63]" />
            </Link>
          </div>

          <div className="pt-6 grid grid-cols-1 md:grid-cols-3 gap-6">
            <div className="p-5 rounded-2xl bg-white border border-[#E3DDD4] space-y-3 shadow-2xs">
              <div className="flex items-center gap-2 text-[#76556F]">
                <Radio className="h-4 w-4" />
                <span className="text-xs font-bold uppercase tracking-wider">Classification</span>
              </div>
              <div className="text-2xl font-serif font-bold text-[#27232B]">{currentDemo.genre}</div>
              <p className="text-xs text-[#6B626E]">
                Neural 2D CNN inference indicates high confidence ({currentDemo.confidence}) over secondary runner-up genres.
              </p>
              <div className="pt-2 border-t border-[#E3DDD4] text-[11px] font-mono text-[#8E8692]">
                Evaluated over 3x 10s windows
              </div>
            </div>

            <div className="p-5 rounded-2xl bg-white border border-[#E3DDD4] space-y-3 shadow-2xs">
              <div className="flex items-center gap-2 text-[#D77C63]">
                <Clock className="h-4 w-4" />
                <span className="text-xs font-bold uppercase tracking-wider">Tempo &amp; Dynamics</span>
              </div>
              <div className="text-2xl font-serif font-bold text-[#27232B]">{currentDemo.tempo}</div>
              <p className="text-xs text-[#6B626E]">
                Extracted via onset autocorrelation with RMS loudness mean of {currentDemo.rms} (normalized).
              </p>
              <div className="pt-2 border-t border-[#E3DDD4] text-[11px] font-mono text-[#8E8692]">
                Key Profile: {currentDemo.keyRes}
              </div>
            </div>

            <div className="p-5 rounded-2xl bg-white border border-[#E3DDD4] space-y-3 shadow-2xs">
              <div className="flex items-center gap-2 text-[#D6A75D]">
                <GitCompare className="h-4 w-4" />
                <span className="text-xs font-bold uppercase tracking-wider">Cosine Similarity</span>
              </div>
              <div className="text-2xl font-serif font-bold text-[#27232B]">5 Acoustic Neighbors</div>
              <p className="text-xs text-[#6B626E]">
                Nearest catalog tracks: {currentDemo.neighbors.slice(0, 3).join(", ")} (excluding self-matches).
              </p>
              <div className="pt-2 border-t border-[#E3DDD4] text-[11px] font-mono text-[#8E8692]">
                Cosine distance over 513-D vector
              </div>
            </div>
          </div>
        </div>
      </section>

      {/* ========================================================================= */}
      {/*                         3. HOW IT WORKS SECTION                           */}
      {/* ========================================================================= */}
      <section id="how-it-works" className="py-20 md:py-28 px-4 sm:px-6 lg:px-8 max-w-7xl mx-auto border-t border-[#E3DDD4]">
        <div className="text-center space-y-3 mb-16 max-w-2xl mx-auto">
          <span className="text-[10px] font-mono font-bold uppercase tracking-wider text-[#76556F] bg-[#F0E8EE] px-3 py-1 rounded-full">
            The Three-Step Journey
          </span>
          <h2 className="text-3xl sm:text-4xl font-serif font-bold text-[#27232B] tracking-tight">
            How Sonic Fusion Works
          </h2>
          <p className="text-sm text-[#6B626E] leading-relaxed">
            From raw audio bytes to multi-layered musical intelligence in seconds.
          </p>
        </div>

        {/* 3 Step Flowing Cards with Connecting Visual Motif */}
        <div className="relative grid grid-cols-1 md:grid-cols-3 gap-8">
          {/* Connecting Line on Desktop */}
          <div className="hidden md:block absolute top-1/2 left-16 right-16 h-0.5 bg-gradient-to-r from-[#76556F]/20 via-[#D77C63]/30 to-[#6C7F75]/20 -translate-y-6 -z-0" />

          {/* Step 1 */}
          <div className="relative z-10 p-8 rounded-2xl glass-surface space-y-4 hover:shadow-md transition-shadow">
            <div className="flex items-center justify-between">
              <div className="flex h-12 w-12 items-center justify-center rounded-2xl bg-[#76556F] text-white shadow-sm font-serif font-bold text-lg">
                1
              </div>
              <span className="text-[10px] font-mono text-[#8E8692]">Input Stage</span>
            </div>
            <div>
              <h3 className="text-lg font-serif font-bold text-[#27232B]">
                Bring Your Sound
              </h3>
              <p className="text-xs text-[#6B626E] mt-2 leading-relaxed">
                Upload an MP3, WAV, or M4A audio file (up to 30 MB / 15 minutes) or choose one of our audited Free Music Archive benchmark recordings.
              </p>
            </div>
            <div className="pt-4 border-t border-[#E3DDD4] text-[11px] font-mono text-[#76556F]">
              &bull; Decoded to 22.05 kHz Mono
            </div>
          </div>

          {/* Step 2 */}
          <div className="relative z-10 p-8 rounded-2xl glass-surface space-y-4 hover:shadow-md transition-shadow">
            <div className="flex items-center justify-between">
              <div className="flex h-12 w-12 items-center justify-center rounded-2xl bg-[#D77C63] text-white shadow-sm font-serif font-bold text-lg">
                2
              </div>
              <span className="text-[10px] font-mono text-[#8E8692]">Processing Stage</span>
            </div>
            <div>
              <h3 className="text-lg font-serif font-bold text-[#27232B]">
                Analyze the Audio
              </h3>
              <p className="text-xs text-[#6B626E] mt-2 leading-relaxed">
                Extract 128-band log-mel spectrogram frames, compute beat onset envelopes, and stream tensors through our trained PyTorch 2D CNN model.
              </p>
            </div>
            <div className="pt-4 border-t border-[#E3DDD4] text-[11px] font-mono text-[#D77C63]">
              &bull; 3x Non-Overlapping 10s Windows
            </div>
          </div>

          {/* Step 3 */}
          <div className="relative z-10 p-8 rounded-2xl glass-surface space-y-4 hover:shadow-md transition-shadow">
            <div className="flex items-center justify-between">
              <div className="flex h-12 w-12 items-center justify-center rounded-2xl bg-[#6C7F75] text-white shadow-sm font-serif font-bold text-lg">
                3
              </div>
              <span className="text-[10px] font-mono text-[#8E8692]">Discovery Stage</span>
            </div>
            <div>
              <h3 className="text-lg font-serif font-bold text-[#27232B]">
                Discover Its Identity
              </h3>
              <p className="text-xs text-[#6B626E] mt-2 leading-relaxed">
                View the predicted genre with calibrated confidence scores, interact with the waveform playhead, inspect chroma pitch profiles, and explore acoustic nearest neighbors.
              </p>
            </div>
            <div className="pt-4 border-t border-[#E3DDD4] text-[11px] font-mono text-[#6C7F75]">
              &bull; Full Model Card &amp; Provenance Included
            </div>
          </div>
        </div>
      </section>

      {/* ========================================================================= */}
      {/*                        4. FINAL CALL TO ACTION                            */}
      {/* ========================================================================= */}
      <section className="py-20 md:py-28 px-4 sm:px-6 lg:px-8 max-w-5xl mx-auto text-center">
        <div className="relative p-10 sm:p-16 rounded-3xl glass-floating border border-[#E3DDD4] space-y-6 overflow-hidden">
          {/* Subtle Concentric Rings Behind CTA */}
          <div className="absolute -top-24 left-1/2 -translate-x-1/2 w-96 h-96 concentric-ring opacity-30 pointer-events-none" />

          <div className="inline-flex items-center gap-2 px-3.5 py-1 rounded-full bg-[#F0E8EE] text-[#76556F] text-xs font-semibold">
            <Sparkles className="h-3.5 w-3.5 text-[#D77C63]" />
            <span>Ready for Acoustic Exploration</span>
          </div>

          {/* Exact Required Headline */}
          <h2 className="text-3xl sm:text-5xl font-serif font-extrabold text-[#27232B] tracking-tight leading-tight">
            Every track has a story. Discover yours.
          </h2>

          <p className="text-sm sm:text-base text-[#6B626E] max-w-xl mx-auto leading-relaxed">
            Step directly into the Sonic Fusion Studio to upload your audio, view spectrogram heatmaps, and unlock deep acoustic genre predictions.
          </p>

          <div className="pt-4 flex flex-col sm:flex-row items-center justify-center gap-4">
            {/* Required Action */}
            <Link
              href="/studio"
              className="w-full sm:w-auto inline-flex items-center justify-center gap-2.5 px-8 py-4 rounded-xl bg-[#76556F] hover:bg-[#583C52] text-white font-bold text-sm shadow-md hover:shadow-lg transition-all duration-200 group cursor-pointer"
            >
              <span>Enter the Studio</span>
              <ArrowRight className="h-4 w-4 text-[#D77C63] group-hover:translate-x-1 transition-transform" />
            </Link>

            <Link
              href="/evaluation"
              className="w-full sm:w-auto inline-flex items-center justify-center gap-2 px-6 py-4 rounded-xl glass-pill hover:bg-white text-[#27232B] font-semibold text-sm transition-all text-center cursor-pointer"
            >
              <ShieldCheck className="h-4 w-4 text-[#76556F]" />
              <span>Review Model Evaluation</span>
            </Link>
          </div>
        </div>
      </section>

      {/* ========================================================================= */}
      {/*                            5. EDITORIAL FOOTER                            */}
      {/* ========================================================================= */}
      <footer className="border-t border-[#E3DDD4] bg-[#FAF7F2]/80 py-12 px-4 sm:px-6 lg:px-8 text-xs text-[#6B626E]">
        <div className="max-w-7xl mx-auto grid grid-cols-1 md:grid-cols-4 gap-8 mb-8">
          <div className="space-y-3 md:col-span-1">
            <div className="flex items-center gap-2 text-base font-serif font-bold text-[#27232B]">
              <div className="h-7 w-7 rounded-lg bg-[#76556F] text-white flex items-center justify-center">
                <Radio className="h-4 w-4" />
              </div>
              <span>Sonic Fusion</span>
            </div>
            <p className="text-xs text-[#8E8692] leading-relaxed">
              Interactive music genre classification &amp; acoustic feature exploration on the Free Music Archive.
            </p>
          </div>

          <div className="space-y-2">
            <span className="font-mono text-[10px] font-bold uppercase tracking-wider text-[#27232B] block">
              Workspace
            </span>
            <ul className="space-y-1.5 text-xs">
              <li>
                <Link href="/studio" className="hover:text-[#76556F] transition-colors">
                  Studio Workspace
                </Link>
              </li>
              <li>
                <Link href="/analysis" className="hover:text-[#76556F] transition-colors">
                  Audio Analysis
                </Link>
              </li>
              <li>
                <Link href="/collection" className="hover:text-[#76556F] transition-colors">
                  Collection Analytics
                </Link>
              </li>
            </ul>
          </div>

          <div className="space-y-2">
            <span className="font-mono text-[10px] font-bold uppercase tracking-wider text-[#27232B] block">
              Scientific Benchmarks
            </span>
            <ul className="space-y-1.5 text-xs">
              <li>
                <Link href="/evaluation" className="hover:text-[#76556F] transition-colors">
                  Model Evaluation &amp; McNemar
                </Link>
              </li>
              <li>
                <Link href="/about" className="hover:text-[#76556F] transition-colors">
                  Model Card &amp; Provenance
                </Link>
              </li>
            </ul>
          </div>

          <div className="space-y-2">
            <span className="font-mono text-[10px] font-bold uppercase tracking-wider text-[#27232B] block">
              System Specifications
            </span>
            <ul className="space-y-1 text-[11px] font-mono text-[#8E8692]">
              <li>PyTorch CPU Checkpoint</li>
              <li>Librosa 0.10 Signal Pipeline</li>
              <li>FMA Small (7,994 Tracks)</li>
              <li>Zero Artist ID Partition Leakage</li>
            </ul>
          </div>
        </div>

        <div className="max-w-7xl mx-auto pt-6 border-t border-[#E3DDD4] flex flex-col sm:flex-row items-center justify-between gap-3 text-[11px]">
          <span>&copy; 2026 Sonic Fusion &bull; Academic MIR &amp; Deep Learning System</span>
          <span className="font-mono text-[#8E8692]">FastAPI + Next.js App Router Architecture</span>
        </div>
      </footer>
    </div>
  );
}
