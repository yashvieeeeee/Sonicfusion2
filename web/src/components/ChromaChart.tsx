"use client";

import React from "react";
import { Disc, Zap } from "lucide-react";
import { AcousticFeatures } from "@/types/api";

interface ChromaChartProps {
  features: AcousticFeatures;
}

export default function ChromaChart({ features }: ChromaChartProps) {
  const { chroma, chroma_labels, rms_mean, rms_std, tempo_bpm_estimate } =
    features;

  const maxChroma = Math.max(...chroma, 0.001);

  // Identify top 2 dominant pitch classes
  const sortedIndices = chroma
    .map((val, idx) => ({ val, idx }))
    .sort((a, b) => b.val - a.val);
  const dominant1 = chroma_labels[sortedIndices[0]?.idx || 0];
  const dominant2 = chroma_labels[sortedIndices[1]?.idx || 1];

  return (
    <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
      {/* 12-Bin Chroma Pitch Class Profile */}
      <div className="rounded-xl border border-slate-800 bg-slate-900/80 p-4">
        <div className="flex items-center justify-between mb-3">
          <div className="flex items-center gap-2">
            <Disc className="h-4 w-4 text-emerald-400" />
            <h4 className="text-xs font-semibold uppercase tracking-wider text-slate-300">
              12-Bin Chroma (Pitch Class Profile)
            </h4>
          </div>
          <span className="text-xs font-mono text-emerald-400">
            Tonal Centers: {dominant1}, {dominant2}
          </span>
        </div>

        <div className="h-40 flex items-end justify-between gap-1.5 pt-6 pb-2 px-1 bg-slate-950/60 rounded-lg border border-slate-800/80">
          {chroma.map((value, idx) => {
            const label = chroma_labels[idx] || `${idx}`;
            const heightPercent = Math.round((value / maxChroma) * 100);
            const isTop = idx === sortedIndices[0]?.idx;

            return (
              <div
                key={label}
                className="flex-1 flex flex-col items-center h-full justify-end group relative"
              >
                {/* Tooltip on hover */}
                <div className="opacity-0 group-hover:opacity-100 transition-opacity pointer-events-none absolute -top-7 px-1.5 py-0.5 rounded bg-slate-800 border border-slate-700 text-[10px] font-mono text-white z-10 whitespace-nowrap">
                  {label}: {value.toFixed(3)}
                </div>

                <div
                  className={`w-full rounded-t transition-all duration-300 ${
                    isTop
                      ? "bg-gradient-to-t from-emerald-600 to-emerald-400 shadow-sm shadow-emerald-500/40"
                      : "bg-gradient-to-t from-slate-700 to-slate-500 hover:from-indigo-600 hover:to-indigo-400"
                  }`}
                  style={{ height: `${Math.max(heightPercent, 4)}%` }}
                />
                <span
                  className={`mt-2 text-[10px] font-mono font-medium ${
                    isTop ? "text-emerald-400 font-bold" : "text-slate-400"
                  }`}
                >
                  {label}
                </span>
              </div>
            );
          })}
        </div>
        <p className="text-[11px] text-slate-500 mt-2">
          Mean 12-bin chromagram measuring harmonic pitch distribution.
        </p>
      </div>

      {/* Signal Energy & Dynamic Profile */}
      <div className="rounded-xl border border-slate-800 bg-slate-900/80 p-4 flex flex-col justify-between">
        <div>
          <div className="flex items-center justify-between mb-3">
            <div className="flex items-center gap-2">
              <Zap className="h-4 w-4 text-cyan-400" />
              <h4 className="text-xs font-semibold uppercase tracking-wider text-slate-300">
                Signal Energy & Rhythm Dynamics
              </h4>
            </div>
            <span className="text-xs font-mono text-cyan-400">
              RMS: {(rms_mean * 100).toFixed(1)}%
            </span>
          </div>

          <div className="space-y-4 my-2">
            <div>
              <div className="flex justify-between text-xs mb-1">
                <span className="text-slate-400">Mean RMS Amplitude</span>
                <span className="font-mono text-white font-medium">
                  {rms_mean.toFixed(4)}
                </span>
              </div>
              <div className="h-2 rounded-full bg-slate-950 overflow-hidden border border-slate-800">
                <div
                  className="h-full bg-gradient-to-r from-cyan-500 to-indigo-500 rounded-full transition-all duration-500"
                  style={{
                    width: `${Math.min(Math.max(rms_mean * 300, 5), 100)}%`,
                  }}
                />
              </div>
            </div>

            <div>
              <div className="flex justify-between text-xs mb-1">
                <span className="text-slate-400">Energy Variance (RMS Std)</span>
                <span className="font-mono text-white font-medium">
                  {rms_std.toFixed(4)}
                </span>
              </div>
              <div className="h-2 rounded-full bg-slate-950 overflow-hidden border border-slate-800">
                <div
                  className="h-full bg-gradient-to-r from-indigo-500 to-purple-500 rounded-full transition-all duration-500"
                  style={{
                    width: `${Math.min(Math.max(rms_std * 350, 5), 100)}%`,
                  }}
                />
              </div>
            </div>

            <div className="p-3 rounded-lg bg-slate-950/70 border border-slate-800/80">
              <div className="flex items-center justify-between">
                <span className="text-xs text-slate-400">
                  Estimated Tempo (BPM)
                </span>
                <span className="text-sm font-bold font-mono text-amber-400">
                  {tempo_bpm_estimate.toFixed(1)} BPM
                </span>
              </div>
              <p className="text-[10px] text-slate-500 mt-1">
                Extracted via librosa beat onset detection. Algorithmic estimate per PLAN §11.
              </p>
            </div>
          </div>
        </div>

        <p className="text-[11px] text-slate-500">
          RMS (Root Mean Square) energy indicates perceived track loudness and dynamic variability.
        </p>
      </div>
    </div>
  );
}
