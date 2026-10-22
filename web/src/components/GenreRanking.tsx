"use client";

import React from "react";
import { AlertCircle, CheckCircle2, Info } from "lucide-react";
import { GenreResult } from "@/types/api";

interface GenreRankingProps {
  genre: GenreResult;
}

export default function GenreRanking({ genre }: GenreRankingProps) {
  const { top_label, confidence, is_uncertain, scores, model } = genre;

  // Compute margin between top 2 genres
  const sortedScores = [...scores].sort((a, b) => b.score - a.score);
  const topScore = sortedScores[0]?.score || 0;
  const secondScore = sortedScores[1]?.score || 0;
  const margin = topScore - secondScore;
  const runnerUpLabel = sortedScores[1]?.label || "runner-up";

  return (
    <div className="editorial-panel p-6">
      {/* Primary Classification Result Header */}
      <div className="flex flex-col sm:flex-row sm:items-start justify-between gap-4 pb-5 border-b border-[#E3DDD4]">
        <div>
          <div className="flex items-center gap-2 mb-1">
            <span className="text-[10px] uppercase tracking-wider font-semibold text-[#6B626E]">
              Primary Classification
            </span>
            <span className="text-[10px] font-mono px-2 py-0.5 rounded bg-[#FAF7F2] text-[#27232B] border border-[#E3DDD4]">
              {model || "GenreCNN"}
            </span>
          </div>
          <div className="flex items-baseline gap-3">
            <h2 className="text-3xl font-serif font-bold text-[#27232B]">
              {top_label}
            </h2>
            <span className="text-xl font-mono font-semibold text-[#D77C63] tabular-nums">
              {(confidence * 100).toFixed(1)}%
            </span>
          </div>
        </div>

        {/* Confidence or Calibrated Uncertainty Badge */}
        <div>
          {is_uncertain ? (
            <div className="flex items-start gap-2.5 px-3 py-2 rounded border border-[#EBD6B0] bg-[#FBF6EC] text-[#9A7337]">
              <AlertCircle className="h-4 w-4 shrink-0 mt-0.5 text-[#D6A75D]" />
              <div>
                <div className="text-xs font-semibold uppercase tracking-wider">
                  Uncertain / Hybrid Influences
                </div>
                <div className="text-[11px] text-[#7A5B2A] mt-0.5">
                  Margin: +{(margin * 100).toFixed(1)}% over {runnerUpLabel}
                </div>
              </div>
            </div>
          ) : (
            <div className="flex items-start gap-2.5 px-3 py-2 rounded border border-[#C6D4CC] bg-[#EEF3F0] text-[#42584D]">
              <CheckCircle2 className="h-4 w-4 shrink-0 mt-0.5 text-[#6C7F75]" />
              <div>
                <div className="text-xs font-semibold uppercase tracking-wider">
                  Confident Classification
                </div>
                <div className="text-[11px] text-[#556D60] mt-0.5">
                  Margin: +{(margin * 100).toFixed(1)}% over {runnerUpLabel}
                </div>
              </div>
            </div>
          )}
        </div>
      </div>

      {/* Uncertainty Notice / Decision Guidance */}
      {is_uncertain && (
        <div className="mt-4 p-3.5 rounded bg-[#FBF6EC] border border-[#EBD6B0] text-xs text-[#7A5B2A] flex items-start gap-2.5">
          <Info className="h-4 w-4 shrink-0 mt-0.5 text-[#D6A75D]" />
          <p className="leading-relaxed">
            <strong className="font-semibold text-[#9A7337]">Acoustic Decision Guidance:</strong>{" "}
            Softmax output represents relative model scores across the musical feature space rather than guaranteed probabilities.
            This track displays hybrid stylistic markers overlapping between{" "}
            <span className="font-medium underline decoration-[#D6A75D]/40">{top_label}</span> and{" "}
            <span className="font-medium underline decoration-[#D6A75D]/40">{runnerUpLabel}</span>.
          </p>
        </div>
      )}

      {/* Ranked Horizontal Bar Breakdown */}
      <div className="mt-5">
        <div className="flex items-center justify-between mb-3 text-xs">
          <span className="font-semibold uppercase tracking-wider text-[#6B626E] text-[10px]">
            Ranked Class Probabilities (8-Class FMA Distribution)
          </span>
          <span className="text-[11px] text-[#6B626E] font-mono">
            Sum: 100.0%
          </span>
        </div>

        <div className="space-y-2">
          {sortedScores.map((item, idx) => {
            const isTop = idx === 0;
            const isRunnerUp = idx === 1;
            const isThird = idx === 2;
            const pct = Math.round(item.score * 1000) / 10;

            const barColor = isTop
              ? is_uncertain
                ? "bg-[#D6A75D]"
                : "bg-[#76556F]"
              : isRunnerUp
              ? "bg-[#D77C63]"
              : isThird
              ? "bg-[#D6A75D]"
              : "bg-[#A6B5AD]";

            return (
              <div
                key={item.label}
                className={`p-2.5 rounded border transition-colors ${
                  isTop
                    ? "bg-[#FAF7F2] border-[#E3DDD4]"
                    : "bg-white border-[#E3DDD4]"
                }`}
              >
                <div className="flex items-center justify-between text-xs mb-1.5">
                  <div className="flex items-center gap-2">
                    <span
                      className={`h-4 w-4 rounded-xs flex items-center justify-center text-[10px] font-mono font-medium ${
                        isTop
                          ? "bg-[#76556F] text-white"
                          : isRunnerUp
                          ? "bg-[#FDF1EE] text-[#D77C63] border border-[#F1D0CD]"
                          : "bg-[#FAF7F2] text-[#6B626E]"
                      }`}
                    >
                      {idx + 1}
                    </span>
                    <span
                      className={`font-medium ${
                        isTop ? "text-[#76556F] font-semibold" : "text-[#27232B]"
                      }`}
                    >
                      {item.label}
                    </span>
                    {isTop && (
                      <span className="text-[9px] font-mono px-1.5 py-0.2 rounded bg-[#F0E8EE] text-[#76556F] font-medium">
                        Leading
                      </span>
                    )}
                  </div>
                  <div className="flex items-center gap-3">
                    <span className="font-mono text-[#8E8692] text-[11px] tabular-nums hidden sm:inline">
                      {item.score.toFixed(4)}
                    </span>
                    <span
                      className={`font-mono font-medium text-xs tabular-nums ${
                        isTop ? "text-[#76556F] font-semibold" : "text-[#27232B]"
                      }`}
                    >
                      {pct.toFixed(1)}%
                    </span>
                  </div>
                </div>

                {/* Horizontal Progress Bar */}
                <div className="h-1.5 w-full rounded-xs bg-[#EAE2D8] overflow-hidden">
                  <div
                    className={`h-full rounded-xs transition-all duration-300 ${barColor}`}
                    style={{ width: `${Math.max(pct, 1.0)}%` }}
                  />
                </div>
              </div>
            );
          })}
        </div>
      </div>
    </div>
  );
}
