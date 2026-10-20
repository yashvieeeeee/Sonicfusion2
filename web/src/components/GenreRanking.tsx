"use client";

import React from "react";
import {
  AlertTriangle,
  Award,
  CheckCircle,
  HelpCircle,
  Sparkles,
} from "lucide-react";
import { GenreResult } from "@/types/api";

interface GenreRankingProps {
  genre: GenreResult;
}

export default function GenreRanking({ genre }: GenreRankingProps) {
  const { top_label, confidence, is_uncertain, status, scores, model } = genre;

  // Compute margin between top 2 genres
  const sortedScores = [...scores].sort((a, b) => b.score - a.score);
  const topScore = sortedScores[0]?.score || 0;
  const secondScore = sortedScores[1]?.score || 0;
  const margin = topScore - secondScore;

  return (
    <div className="rounded-2xl border border-slate-800 bg-slate-900/90 p-5 sm:p-6 backdrop-blur-md">
      {/* Top Header */}
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4 pb-5 border-b border-slate-800">
        <div>
          <div className="flex items-center gap-2 mb-1">
            <span className="text-xs uppercase tracking-wider font-semibold text-slate-400">
              Primary Classification
            </span>
            <span className="text-[10px] font-mono px-2 py-0.5 rounded bg-slate-800 text-slate-300 border border-slate-700">
              {model}
            </span>
          </div>
          <div className="flex items-baseline gap-3">
            <h2 className="text-3xl sm:text-4xl font-extrabold tracking-tight text-white">
              {top_label}
            </h2>
            <span className="text-lg font-mono font-bold text-indigo-400">
              {(confidence * 100).toFixed(1)}%
            </span>
          </div>
        </div>

        {/* Confidence or Uncertainty Badge */}
        <div>
          {is_uncertain ? (
            <div className="flex items-center gap-2.5 px-3.5 py-2 rounded-xl bg-amber-950/60 border border-amber-600/60 text-amber-300 shadow-lg shadow-amber-950/40 glow-uncertain">
              <AlertTriangle className="h-5 w-5 text-amber-400 shrink-0 animate-pulse" />
              <div>
                <div className="text-xs font-bold uppercase tracking-wider">
                  Uncertain / Multiple Influences
                </div>
                <div className="text-[11px] text-amber-200/80">
                  Top score low ({(confidence * 100).toFixed(0)}%) or narrow margin ({(margin * 100).toFixed(1)}%)
                </div>
              </div>
            </div>
          ) : (
            <div className="flex items-center gap-2.5 px-3.5 py-2 rounded-xl bg-emerald-950/60 border border-emerald-600/60 text-emerald-300 shadow-lg shadow-emerald-950/40">
              <CheckCircle className="h-5 w-5 text-emerald-400 shrink-0" />
              <div>
                <div className="text-xs font-bold uppercase tracking-wider">
                  Confident Classification
                </div>
                <div className="text-[11px] text-emerald-200/80">
                  Margin: +{(margin * 100).toFixed(1)}% over runner-up
                </div>
              </div>
            </div>
          )}
        </div>
      </div>

      {/* Uncertainty Advisory Notice if applicable */}
      {is_uncertain && (
        <div className="mt-4 p-3.5 rounded-xl bg-amber-950/30 border border-amber-800/40 text-xs text-amber-200/90 flex items-start gap-3">
          <HelpCircle className="h-4 w-4 text-amber-400 shrink-0 mt-0.5" />
          <p>
            <strong className="text-amber-300">Model Guidance (PLAN §09):</strong>{" "}
            Softmax values reflect relative model scores rather than absolute calibrated ground truth. The audio may blend overlapping stylistic characteristics between{" "}
            <span className="underline decoration-amber-400/50">{top_label}</span> and{" "}
            <span className="underline decoration-amber-400/50">{sortedScores[1]?.label}</span>.
          </p>
        </div>
      )}

      {/* Ranked Score Breakdown */}
      <div className="mt-5">
        <h4 className="text-xs font-semibold uppercase tracking-wider text-slate-400 mb-3">
          Ranked FMA Genre Scores (Softmax Distribution)
        </h4>

        <div className="space-y-2.5">
          {sortedScores.map((item, idx) => {
            const isTop = idx === 0;
            const isRunnerUp = idx === 1;
            const pct = Math.round(item.score * 1000) / 10;

            return (
              <div
                key={item.label}
                className={`p-2.5 rounded-xl border transition-all ${
                  isTop
                    ? "bg-indigo-950/30 border-indigo-500/40 shadow-sm shadow-indigo-500/10"
                    : "bg-slate-950/40 border-slate-800/80 hover:border-slate-700"
                }`}
              >
                <div className="flex items-center justify-between text-xs mb-1.5">
                  <div className="flex items-center gap-2">
                    <span
                      className={`h-5 w-5 rounded-md flex items-center justify-center text-[10px] font-mono font-bold ${
                        isTop
                          ? "bg-indigo-500 text-white"
                          : isRunnerUp
                          ? "bg-slate-700 text-slate-300"
                          : "bg-slate-800 text-slate-500"
                      }`}
                    >
                      #{idx + 1}
                    </span>
                    <span
                      className={`font-semibold ${
                        isTop ? "text-white text-sm" : "text-slate-300"
                      }`}
                    >
                      {item.label}
                    </span>
                    {isTop && (
                      <span className="text-[10px] px-1.5 py-0.2 rounded bg-indigo-500/20 text-indigo-400 border border-indigo-500/30">
                        Top Prediction
                      </span>
                    )}
                  </div>
                  <div className="flex items-center gap-2">
                    <span className="font-mono text-slate-400 text-xs">
                      {item.score.toFixed(4)}
                    </span>
                    <span
                      className={`font-mono font-bold text-xs ${
                        isTop ? "text-indigo-400" : "text-slate-300"
                      }`}
                    >
                      {pct.toFixed(1)}%
                    </span>
                  </div>
                </div>

                {/* Progress bar */}
                <div className="h-2 w-full rounded-full bg-slate-900 overflow-hidden border border-slate-800/60">
                  <div
                    className={`h-full rounded-full transition-all duration-500 ${
                      isTop
                        ? is_uncertain
                          ? "bg-gradient-to-r from-amber-500 to-indigo-500"
                          : "bg-gradient-to-r from-indigo-500 to-cyan-400"
                        : isRunnerUp
                        ? "bg-slate-500"
                        : "bg-slate-700"
                    }`}
                    style={{ width: `${Math.max(pct, 1.5)}%` }}
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
