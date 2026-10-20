"use client";

import React, { useEffect, useState } from "react";
import { CheckCircle2, Cpu, Disc3, Loader2, Music, Waves } from "lucide-react";

interface StagedLoadingProps {
  filename: string;
}

interface Stage {
  id: number;
  title: string;
  desc: string;
  icon: React.ElementType;
}

const STAGES: Stage[] = [
  {
    id: 1,
    title: "Decoding Audio Stream",
    desc: "Validating audio bitstream and converting to 22,050 Hz standardized PCM array",
    icon: Disc3,
  },
  {
    id: 2,
    title: "Extracting Acoustic Signal Features",
    desc: "Computing librosa beat tempo (BPM), frame-level RMS energy, and 12-bin chroma",
    icon: Music,
  },
  {
    id: 3,
    title: "Running GenreCNN 2D Inference",
    desc: "Evaluating 10-second log-mel spectrogram windows through 3-block 2D CNN",
    icon: Cpu,
  },
  {
    id: 4,
    title: "Generating Visualizations & Matrices",
    desc: "Downsampling peak waveform envelope (1,000 pts) and log-mel matrix (128x300)",
    icon: Waves,
  },
];

export default function StagedLoading({ filename }: StagedLoadingProps) {
  const [activeStage, setActiveStage] = useState<number>(1);

  useEffect(() => {
    // Progress through staged display to reflect the pipeline phases while awaiting backend response
    const t1 = setTimeout(() => setActiveStage(2), 1200);
    const t2 = setTimeout(() => setActiveStage(3), 3200);
    const t3 = setTimeout(() => setActiveStage(4), 5800);

    return () => {
      clearTimeout(t1);
      clearTimeout(t2);
      clearTimeout(t3);
    };
  }, []);

  return (
    <div className="w-full max-w-xl mx-auto rounded-2xl border border-slate-800 bg-slate-900/90 p-6 sm:p-8 backdrop-blur-xl shadow-2xl">
      <div className="flex items-center gap-3 mb-6 pb-4 border-b border-slate-800">
        <div className="flex h-10 w-10 items-center justify-center rounded-xl bg-indigo-500/20 text-indigo-400 border border-indigo-500/30 animate-pulse">
          <Loader2 className="h-5 w-5 animate-spin" />
        </div>
        <div>
          <h3 className="text-base font-semibold text-white">
            Processing Audio Pipeline
          </h3>
          <p className="text-xs text-slate-400 font-mono truncate max-w-sm">
            {filename}
          </p>
        </div>
      </div>

      {/* Stage indicators */}
      <div className="space-y-4">
        {STAGES.map((stage) => {
          const Icon = stage.icon;
          const isDone = activeStage > stage.id;
          const isCurrent = activeStage === stage.id;
          const isPending = activeStage < stage.id;

          return (
            <div
              key={stage.id}
              className={`flex items-start gap-4 p-3.5 rounded-xl border transition-all duration-300 ${
                isCurrent
                  ? "bg-indigo-950/40 border-indigo-500/40 shadow-sm shadow-indigo-500/10"
                  : isDone
                  ? "bg-slate-950/40 border-slate-800 text-slate-400"
                  : "bg-slate-950/20 border-slate-900/60 opacity-50"
              }`}
            >
              <div
                className={`mt-0.5 flex h-8 w-8 shrink-0 items-center justify-center rounded-lg ${
                  isDone
                    ? "bg-emerald-950/70 text-emerald-400 border border-emerald-800/60"
                    : isCurrent
                    ? "bg-indigo-600 text-white shadow-md shadow-indigo-500/30"
                    : "bg-slate-800 text-slate-500"
                }`}
              >
                {isDone ? (
                  <CheckCircle2 className="h-4 w-4" />
                ) : isCurrent ? (
                  <Icon className="h-4 w-4 animate-bounce" />
                ) : (
                  <Icon className="h-4 w-4" />
                )}
              </div>

              <div className="flex-1 min-w-0">
                <div className="flex items-center justify-between">
                  <h4
                    className={`text-xs font-semibold uppercase tracking-wider ${
                      isCurrent
                        ? "text-indigo-300"
                        : isDone
                        ? "text-emerald-400"
                        : "text-slate-500"
                    }`}
                  >
                    Stage {stage.id}: {stage.title}
                  </h4>
                  {isCurrent && (
                    <span className="flex items-center gap-1 text-[11px] text-indigo-400 font-medium">
                      <span className="h-1.5 w-1.5 rounded-full bg-indigo-400 animate-ping" />
                      In progress...
                    </span>
                  )}
                  {isDone && (
                    <span className="text-[11px] text-emerald-400 font-medium">
                      Completed
                    </span>
                  )}
                </div>
                <p className="text-xs text-slate-400 mt-0.5">{stage.desc}</p>
              </div>
            </div>
          );
        })}
      </div>

      <div className="mt-6 pt-4 border-t border-slate-800/80 flex items-center justify-between text-[11px] text-slate-500">
        <span>Running real local CPU inference</span>
        <span>Standardized FMA small config</span>
      </div>
    </div>
  );
}
