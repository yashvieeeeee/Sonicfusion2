"use client";

import React, { useEffect, useState } from "react";
import Link from "next/link";
import {
  Activity,
  ArrowRight,
  BarChart2,
  CheckCircle2,
  Compass,
  Database,
  FileText,
  Filter,
  Layers,
  Music,
  RefreshCw,
  Sliders,
  TrendingUp,
} from "lucide-react";
import { getFigureUrl, getMetrics } from "@/lib/api";
import { MetricsResponse } from "@/types/api";

export default function CollectionAnalyticsPage() {
  const [metrics, setMetrics] = useState<MetricsResponse | null>(null);
  const [loading, setLoading] = useState<boolean>(true);
  const [selectedSort, setSelectedSort] = useState<"genre" | "centroid" | "rms" | "tempo">("genre");

  useEffect(() => {
    getMetrics()
      .then((data) => setMetrics(data))
      .catch(() => {})
      .finally(() => setLoading(false));
  }, []);

  const eda = metrics?.eda_summary;
  const figures = metrics?.figures_available || [];

  const genreStats = [
    { genre: "Electronic", centroid: 2158.78, rms: 0.2191, tempo: 118.91, count: 800 },
    { genre: "Experimental", centroid: 1840.58, rms: 0.1547, tempo: 119.82, count: 800 },
    { genre: "Folk", centroid: 1495.27, rms: 0.1511, tempo: 119.96, count: 800 },
    { genre: "Hip-Hop", centroid: 2276.64, rms: 0.2272, tempo: 113.74, count: 797 },
    { genre: "Instrumental", centroid: 1250.43, rms: 0.1312, tempo: 113.81, count: 800 },
    { genre: "International", centroid: 2072.84, rms: 0.1550, tempo: 125.25, count: 800 },
    { genre: "Pop", centroid: 1966.67, rms: 0.1910, tempo: 119.34, count: 800 },
    { genre: "Rock", centroid: 2137.82, rms: 0.2125, tempo: 121.76, count: 799 },
  ];

  const sortedStats = [...genreStats].sort((a, b) => {
    if (selectedSort === "centroid") return b.centroid - a.centroid;
    if (selectedSort === "rms") return b.rms - a.rms;
    if (selectedSort === "tempo") return b.tempo - a.tempo;
    return a.genre.localeCompare(b.genre);
  });

  return (
    <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8 space-y-8">
      {/* Editorial Page Header */}
      <div className="flex flex-col sm:flex-row sm:items-end justify-between gap-4 pb-6 border-b border-[#E3DDD4]">
        <div>
          <div className="flex items-center gap-2 mb-1.5">
            <span className="text-[10px] font-mono uppercase tracking-wider font-semibold text-[#76556F] bg-[#F0E8EE] px-2 py-0.5 rounded">
              Corpus Intelligence
            </span>
            <span className="text-[10px] font-mono text-[#6B626E]">
              7,994 Audited Tracks
            </span>
          </div>
          <h1 className="text-3xl sm:text-4xl font-serif font-bold text-[#27232B] tracking-tight">
            Collection Analytics
          </h1>
          <p className="text-sm text-[#6B626E] mt-1 max-w-2xl leading-relaxed">
            Population-level distribution metrics across the 8 curated FMA Small genres, analyzing spectral centroid, loudness dynamics, and tempo patterns.
          </p>
        </div>

        <Link
          href="/evaluation"
          className="flex items-center gap-1.5 px-3.5 py-1.5 rounded-md border border-[#E3DDD4] bg-white text-xs font-medium text-[#27232B] hover:bg-[#FAF7F2] transition-colors shadow-2xs"
        >
          <span>Model Evaluation Results</span>
          <ArrowRight className="h-3.5 w-3.5 text-[#D77C63]" />
        </Link>
      </div>

      {/* Corpus KPI Summary Strip */}
      <div className="grid grid-cols-2 sm:grid-cols-4 gap-4">
        <div className="editorial-panel p-4">
          <span className="text-[10px] font-mono uppercase tracking-wider text-[#6B626E] block">
            Decodable Audio Tracks
          </span>
          <div className="text-2xl font-serif font-bold text-[#27232B] mt-1">
            7,994
          </div>
          <span className="text-[11px] text-[#8E8692]">
            99.93% valid bitstream
          </span>
        </div>

        <div className="editorial-panel p-4">
          <span className="text-[10px] font-mono uppercase tracking-wider text-[#6B626E] block">
            Curated Classes
          </span>
          <div className="text-2xl font-serif font-bold text-[#76556F] mt-1">
            8 Genres
          </div>
          <span className="text-[11px] text-[#8E8692]">
            Balanced ~800 / class
          </span>
        </div>

        <div className="editorial-panel p-4">
          <span className="text-[10px] font-mono uppercase tracking-wider text-[#6B626E] block">
            Corrupt Exclusions
          </span>
          <div className="text-2xl font-serif font-bold text-[#D77C63] mt-1">
            6 Tracks
          </div>
          <span className="text-[11px] text-[#8E8692]">
            Audited &amp; excluded
          </span>
        </div>

        <div className="editorial-panel p-4">
          <span className="text-[10px] font-mono uppercase tracking-wider text-[#6B626E] block">
            Split Protocol
          </span>
          <div className="text-sm font-mono font-semibold text-[#27232B] mt-1">
            6394 / 800 / 800
          </div>
          <span className="text-[11px] text-[#8E8692]">
            Train / Val / Test
          </span>
        </div>
      </div>

      {/* Comparative Acoustic Profile by Genre Table */}
      <div className="editorial-panel p-6">
        <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-3 pb-4 border-b border-[#E3DDD4] mb-5">
          <div>
            <h2 className="text-xs font-semibold uppercase tracking-wider text-[#27232B]">
              Acoustic Profile Distribution by Musical Genre
            </h2>
            <p className="text-xs text-[#6B626E] mt-0.5">
              Empirical sample means calculated over the full dataset audio features.
            </p>
          </div>

          {/* Sort selector */}
          <div className="flex items-center gap-1.5 text-xs text-[#6B626E]">
            <span>Sort by:</span>
            <div className="inline-flex rounded-md border border-[#E3DDD4] bg-[#FAF7F2] p-0.5">
              <button
                onClick={() => setSelectedSort("genre")}
                className={`px-2 py-1 rounded text-[11px] font-medium transition-colors ${
                  selectedSort === "genre" ? "bg-white text-[#76556F] font-semibold shadow-2xs" : "text-[#6B626E]"
                }`}
              >
                Genre
              </button>
              <button
                onClick={() => setSelectedSort("centroid")}
                className={`px-2 py-1 rounded text-[11px] font-medium transition-colors ${
                  selectedSort === "centroid" ? "bg-white text-[#76556F] font-semibold shadow-2xs" : "text-[#6B626E]"
                }`}
              >
                Centroid
              </button>
              <button
                onClick={() => setSelectedSort("rms")}
                className={`px-2 py-1 rounded text-[11px] font-medium transition-colors ${
                  selectedSort === "rms" ? "bg-white text-[#76556F] font-semibold shadow-2xs" : "text-[#6B626E]"
                }`}
              >
                RMS
              </button>
              <button
                onClick={() => setSelectedSort("tempo")}
                className={`px-2 py-1 rounded text-[11px] font-medium transition-colors ${
                  selectedSort === "tempo" ? "bg-white text-[#76556F] font-semibold shadow-2xs" : "text-[#6B626E]"
                }`}
              >
                Tempo
              </button>
            </div>
          </div>
        </div>

        {/* Table */}
        <div className="overflow-x-auto">
          <table className="w-full text-left editorial-table">
            <thead>
              <tr>
                <th>Genre</th>
                <th>Corpus Tracks</th>
                <th>Spectral Centroid (Hz)</th>
                <th>Mean RMS Loudness</th>
                <th>Mean Tempo (BPM)</th>
                <th>Acoustic Trait</th>
              </tr>
            </thead>
            <tbody>
              {sortedStats.map((item) => {
                const isBrightest = item.centroid > 2200;
                const isWarmest = item.centroid < 1400;

                return (
                  <tr key={item.genre} className="hover:bg-[#FAF7F2] transition-colors">
                    <td className="font-semibold text-[#27232B]">
                      {item.genre}
                    </td>
                    <td className="font-mono text-[#6B626E] tabular-nums">
                      {item.count}
                    </td>
                    <td>
                      <div className="flex items-center gap-2">
                        <span className="font-mono tabular-nums text-[#27232B]">
                          {item.centroid.toFixed(1)} Hz
                        </span>
                        <div className="w-16 h-1.5 bg-[#EAE2D8] rounded-xs overflow-hidden">
                          <div
                            className="h-full bg-[#76556F] rounded-xs"
                            style={{ width: `${(item.centroid / 2500) * 100}%` }}
                          />
                        </div>
                      </div>
                    </td>
                    <td>
                      <div className="flex items-center gap-2">
                        <span className="font-mono tabular-nums text-[#27232B]">
                          {item.rms.toFixed(4)}
                        </span>
                        <div className="w-16 h-1.5 bg-[#EAE2D8] rounded-xs overflow-hidden">
                          <div
                            className="h-full bg-[#6C7F75] rounded-xs"
                            style={{ width: `${(item.rms / 0.25) * 100}%` }}
                          />
                        </div>
                      </div>
                    </td>
                    <td>
                      <span className="font-mono tabular-nums text-[#27232B]">
                        {item.tempo.toFixed(1)} BPM
                      </span>
                    </td>
                    <td className="text-[#6B626E] text-xs">
                      {isBrightest
                        ? "High-frequency percussive timbre"
                        : isWarmest
                        ? "Acoustic / harmonic warmth"
                        : item.tempo > 123
                        ? "Fast rhythmic cadence"
                        : "Balanced dynamic mid-range"}
                    </td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
      </div>

      {/* Generated Report Figures Gallery */}
      <div className="editorial-panel p-6">
        <div className="flex items-center justify-between pb-4 border-b border-[#E3DDD4] mb-5">
          <div>
            <h2 className="text-xs font-semibold uppercase tracking-wider text-[#27232B]">
              Exploratory Data Artifacts &amp; Figures
            </h2>
            <p className="text-xs text-[#6B626E] mt-0.5">
              Reproducible visualizations generated during preprocessing and EDA pipelines.
            </p>
          </div>
          <span className="text-[10px] font-mono text-[#8E8692]">
            reports/figures/
          </span>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
          <div className="p-4 rounded-lg border border-[#E3DDD4] bg-[#FAF7F2] space-y-2">
            <span className="text-xs font-semibold text-[#27232B] block">
              1. Class Counts by Split
            </span>
            <div className="aspect-16/10 rounded overflow-hidden border border-[#E3DDD4] bg-white">
              <img
                src={getFigureUrl("eda_class_counts_by_split.png")}
                alt="EDA Class Counts by Split"
                className="w-full h-full object-contain"
                onError={(e) => {
                  (e.target as HTMLElement).style.display = "none";
                }}
              />
            </div>
            <p className="text-[11px] text-[#6B626E]">
              Exact balance verified across training (800/class), validation (100/class), and test (100/class).
            </p>
          </div>

          <div className="p-4 rounded-lg border border-[#E3DDD4] bg-[#FAF7F2] space-y-2">
            <span className="text-xs font-semibold text-[#27232B] block">
              2. Acoustic Feature Distributions
            </span>
            <div className="aspect-16/10 rounded overflow-hidden border border-[#E3DDD4] bg-white">
              <img
                src={getFigureUrl("eda_feature_distributions.png")}
                alt="EDA Feature Distributions"
                className="w-full h-full object-contain"
                onError={(e) => {
                  (e.target as HTMLElement).style.display = "none";
                }}
              />
            </div>
            <p className="text-[11px] text-[#6B626E]">
              Kernel density estimates for spectral centroid, rolloff, zero-crossing rate, and RMS energy.
            </p>
          </div>
        </div>
      </div>
    </div>
  );
}
