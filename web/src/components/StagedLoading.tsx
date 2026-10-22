"use client";

import React, { useEffect, useState } from "react";
import { Check, Loader2 } from "lucide-react";

interface StagedLoadingProps {
  filename: string;
}

interface Stage {
  id: number;
  title: string;
  desc: string;
}

const STAGES: Stage[] = [
  {
    id: 1,
    title: "Audio Stream Decoding",
    desc: "Validating container bytes and converting to standardized 22,050 Hz PCM mono array",
  },
  {
    id: 2,
    title: "Acoustic Feature Extraction",
    desc: "Computing librosa beat tempo, frame-level RMS energy, and 12-bin chroma pitch profile",
  },
  {
    id: 3,
    title: "GenreCNN 2D Spectrogram Inference",
    desc: "Evaluating standardized 128-band log-mel windows through frozen 3-block 2D CNN",
  },
  {
    id: 4,
    title: "Visual Matrix Generation",
    desc: "Downsampling peak waveform envelope (1,000 pts) and log-mel power matrix (128x300)",
  },
];

export default function StagedLoading({ filename }: StagedLoadingProps) {
  const [activeStage, setActiveStage] = useState<number>(1);

  useEffect(() => {
    const t1 = setTimeout(() => setActiveStage(2), 1200);
    const t2 = setTimeout(() => setActiveStage(3), 3200);
    const t3 = setTimeout(() => setActiveStage(4), 5600);

    return () => {
      clearTimeout(t1);
      clearTimeout(t2);
      clearTimeout(t3);
    };
  }, []);

  return (
    <div className="w-full max-w-xl mx-auto glass-floating p-6 sm:p-8 rounded-xl">
      <div className="flex items-center gap-3 pb-4 mb-5 border-b border-[#E3DDD4]">
        <div className="flex h-9 w-9 items-center justify-center rounded-lg bg-[#F0E8EE] text-[#76556F]">
          <Loader2 className="h-4 w-4 animate-spin text-[#D77C63]" />
        </div>
        <div>
          <h3 className="text-sm font-semibold text-[#27232B]">
            Processing Audio Pipeline
          </h3>
          <p className="text-xs text-[#6B626E] font-mono truncate max-w-md">
            {filename}
          </p>
        </div>
      </div>

      {/* Stage indicators */}
      <div className="space-y-3">
        {STAGES.map((stage) => {
          const isDone = activeStage > stage.id;
          const isCurrent = activeStage === stage.id;

          return (
            <div
              key={stage.id}
              className={`flex items-start gap-3.5 p-3 rounded-lg border transition-all duration-300 ${
                isCurrent
                  ? "bg-white border-[#76556F]/40 shadow-xs"
                  : isDone
                  ? "bg-white/80 border-[#E3DDD4]"
                  : "bg-white/40 border-[#E3DDD4]/60 opacity-50"
              }`}
            >
              <div
                className={`mt-0.5 flex h-6 w-6 shrink-0 items-center justify-center rounded-md text-xs font-mono font-medium ${
                  isDone
                    ? "bg-[#76556F] text-white"
                    : isCurrent
                    ? "bg-[#FDF1EE] text-[#D77C63] border border-[#F1D0CD]"
                    : "bg-[#FAF7F2] text-[#8E8692]"
                }`}
              >
                {isDone ? <Check className="h-3.5 w-3.5" /> : stage.id}
              </div>

              <div className="flex-1 min-w-0">
                <div className="flex items-center justify-between">
                  <h4
                    className={`text-xs font-medium ${
                      isCurrent
                        ? "text-[#76556F] font-semibold"
                        : isDone
                        ? "text-[#27232B]"
                        : "text-[#6B626E]"
                    }`}
                  >
                    {stage.title}
                  </h4>
                  {isCurrent && (
                    <span className="text-[10px] font-mono text-[#D77C63] font-medium">
                      In Progress...
                    </span>
                  )}
                  {isDone && (
                    <span className="text-[10px] font-mono text-[#6C7F75]">
                      Completed
                    </span>
                  )}
                </div>
                <p className="text-[11px] text-[#6B626E] mt-0.5 leading-relaxed">
                  {stage.desc}
                </p>
              </div>
            </div>
          );
        })}
      </div>

      <div className="mt-5 pt-3 border-t border-[#E3DDD4] flex items-center justify-between text-[10px] font-mono text-[#8E8692]">
        <span>Standardized PyTorch CPU execution</span>
        <span>Sampling: 22,050 Hz Mono</span>
      </div>
    </div>
  );
}
