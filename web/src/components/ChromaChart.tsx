"use client";

import React from "react";
import { Disc, Gauge } from "lucide-react";
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
    <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
      {/* 12-Bin Chroma Pitch Class Profile */}
      <div className="editorial-panel p-5 flex flex-col justify-between">
        <div>
          <div className="flex items-center justify-between mb-3">
            <div className="flex items-center gap-2">
              <Disc className="h-4 w-4 text-[#76556F]" />
              <h3 className="text-xs font-semibold uppercase tracking-wider text-[#27232B]">
                12-Bin Chroma (Pitch Class Profile)
              </h3>
            </div>
            <div className="flex items-center gap-1.5">
              <span className="text-[11px] font-mono text-[#D77C63] bg-[#FDF1EE] px-2 py-0.5 rounded font-medium">
                Root: {dominant1}
              </span>
              <span className="text-[11px] font-mono text-[#D6A75D] bg-[#FBF6EC] px-2 py-0.5 rounded font-medium">
                2nd: {dominant2}
              </span>
            </div>
          </div>

          <p className="text-xs text-[#6B626E] mb-4">
            Octave-folded harmonic energy distribution across the 12 chromatic semitone pitch classes.
          </p>

          <div className="h-36 flex items-end justify-between gap-1.5 pt-4 pb-2 px-2 bg-[#FAF7F2] rounded border border-[#E3DDD4]">
            {chroma.map((value, idx) => {
              const label = chroma_labels[idx] || `${idx}`;
              const heightPercent = Math.round((value / maxChroma) * 100);
              const isTop = idx === sortedIndices[0]?.idx;
              const isRunnerUp = idx === sortedIndices[1]?.idx;

              return (
                <div
                  key={label}
                  className="flex-1 flex flex-col items-center h-full justify-end group relative"
                >
                  {/* Tooltip on hover */}
                  <div className="opacity-0 group-hover:opacity-100 transition-opacity pointer-events-none absolute -top-7 px-1.5 py-0.5 rounded bg-[#27232B] text-[10px] font-mono text-white z-10 whitespace-nowrap shadow-xs">
                    {label}: {value.toFixed(3)}
                  </div>

                  <div
                    className={`w-full rounded-xs transition-all duration-200 ${
                      isTop
                        ? "bg-[#D77C63]"
                        : isRunnerUp
                        ? "bg-[#D6A75D]"
                        : "bg-[#CCD4D0] hover:bg-[#A6B5AD]"
                    }`}
                    style={{ height: `${Math.max(heightPercent, 4)}%` }}
                  />
                  <span
                    className={`mt-2 text-[10px] font-mono ${
                      isTop
                        ? "text-[#D77C63] font-bold"
                        : isRunnerUp
                        ? "text-[#9A7337] font-semibold"
                        : "text-[#8E8692]"
                    }`}
                  >
                    {label}
                  </span>
                </div>
              );
            })}
          </div>
        </div>

        <div className="mt-3 text-[10px] text-[#6B626E]">
          Normalized harmonic chroma vector computed via librosa chromagram.
        </div>
      </div>

      {/* Signal Dynamics & Tempo Profile */}
      <div className="editorial-panel p-5 flex flex-col justify-between">
        <div>
          <div className="flex items-center justify-between mb-3">
            <div className="flex items-center gap-2">
              <Gauge className="h-4 w-4 text-[#76556F]" />
              <h3 className="text-xs font-semibold uppercase tracking-wider text-[#27232B]">
                Signal Dynamics &amp; Metric Estimations
              </h3>
            </div>
            <span className="text-xs font-mono text-[#6B626E]">
              Dynamic Profile
            </span>
          </div>

          <p className="text-xs text-[#6B626E] mb-4">
            Frame-level Root Mean Square (RMS) loudness envelope and dynamic tempo tracking.
          </p>

          <div className="space-y-4">
            {/* Mean RMS Loudness */}
            <div className="p-3 bg-[#FAF7F2] rounded border border-[#E3DDD4]">
              <div className="flex justify-between items-baseline text-xs mb-1.5">
                <span className="text-[#6B626E] font-medium">Mean RMS Energy</span>
                <span className="font-mono text-sm font-semibold text-[#27232B] tabular-nums">
                  {rms_mean.toFixed(4)}
                </span>
              </div>
              <div className="h-1.5 w-full bg-[#EAE2D8] rounded-xs overflow-hidden">
                <div
                  className="h-full bg-[#76556F] rounded-xs"
                  style={{
                    width: `${Math.min(Math.max(rms_mean * 300, 4), 100)}%`,
                  }}
                />
              </div>
            </div>

            {/* RMS Energy Variance */}
            <div className="p-3 bg-[#FAF7F2] rounded border border-[#E3DDD4]">
              <div className="flex justify-between items-baseline text-xs mb-1.5">
                <span className="text-[#6B626E] font-medium">Energy Variance (RMS Std)</span>
                <span className="font-mono text-sm font-semibold text-[#27232B] tabular-nums">
                  {rms_std.toFixed(4)}
                </span>
              </div>
              <div className="h-1.5 w-full bg-[#EAE2D8] rounded-xs overflow-hidden">
                <div
                  className="h-full bg-[#6C7F75] rounded-xs"
                  style={{
                    width: `${Math.min(Math.max(rms_std * 350, 4), 100)}%`,
                  }}
                />
              </div>
            </div>

            {/* Estimated Tempo */}
            <div className="p-3 bg-[#FAF7F2] rounded border border-[#E3DDD4] flex items-center justify-between">
              <div>
                <span className="text-xs font-medium text-[#27232B] block">Estimated Tempo (BPM)</span>
                <span className="text-[10px] text-[#8E8692]">Dynamic beat onset tracking</span>
              </div>
              <div className="text-right">
                <span className="font-mono text-lg font-bold text-[#D77C63] tabular-nums">
                  {tempo_bpm_estimate.toFixed(1)}
                </span>
                <span className="text-[10px] font-mono text-[#6B626E] ml-1">BPM</span>
              </div>
            </div>
          </div>
        </div>

        <div className="mt-3 text-[10px] text-[#8E8692]">
          Tempo is an algorithmic onset estimate from librosa; actual tempo may vary in polymetric compositions.
        </div>
      </div>
    </div>
  );
}
