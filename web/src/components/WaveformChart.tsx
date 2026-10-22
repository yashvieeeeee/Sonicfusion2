"use client";

import React, { useRef, useEffect, useState } from "react";
import { Waves } from "lucide-react";

interface WaveformChartProps {
  waveform: number[];
  durationSeconds: number;
  currentTime?: number;
  onSeek?: (time: number) => void;
}

export default function WaveformChart({
  waveform,
  durationSeconds,
  currentTime = 0,
  onSeek,
}: WaveformChartProps) {
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const [hoverTime, setHoverTime] = useState<number | null>(null);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas || !waveform || waveform.length === 0) return;

    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    const dpr = window.devicePixelRatio || 1;
    const rect = canvas.getBoundingClientRect();

    canvas.width = rect.width * dpr;
    canvas.height = rect.height * dpr;
    ctx.scale(dpr, dpr);

    const width = rect.width;
    const height = rect.height;
    const midY = height / 2;

    ctx.clearRect(0, 0, width, height);

    // Warm canvas background
    ctx.fillStyle = "#FAF7F2";
    ctx.fillRect(0, 0, width, height);

    // Gridlines in soft border
    ctx.strokeStyle = "#E3DDD4";
    ctx.lineWidth = 1;
    ctx.setLineDash([3, 3]);

    // Center zero axis
    ctx.beginPath();
    ctx.moveTo(0, midY);
    ctx.lineTo(width, midY);
    ctx.stroke();

    // ±0.5 Amplitude guides
    ctx.beginPath();
    ctx.moveTo(0, midY - height * 0.25);
    ctx.lineTo(width, midY - height * 0.25);
    ctx.moveTo(0, midY + height * 0.25);
    ctx.lineTo(width, midY + height * 0.25);
    ctx.stroke();
    ctx.setLineDash([]);

    // Draw waveform envelope in dusty plum with subtle density
    const numPoints = waveform.length;
    const step = width / numPoints;

    ctx.fillStyle = "#76556F"; // Dusty plum

    for (let i = 0; i < numPoints; i++) {
      const x = i * step;
      const amp = Math.min(Math.abs(waveform[i]), 1.0);
      const barHeight = amp * (height * 0.86);
      const y = midY - barHeight / 2;

      ctx.fillRect(x, y, Math.max(step * 0.85, 1), Math.max(barHeight, 1.5));
    }

    // Playhead in muted coral
    if (durationSeconds > 0 && currentTime > 0) {
      const playX = (currentTime / durationSeconds) * width;
      ctx.strokeStyle = "#D77C63"; // Muted coral
      ctx.lineWidth = 1.75;
      ctx.beginPath();
      ctx.moveTo(playX, 0);
      ctx.lineTo(playX, height);
      ctx.stroke();

      // Precise playhead dot
      ctx.fillStyle = "#D77C63";
      ctx.beginPath();
      ctx.arc(playX, 4, 3.5, 0, Math.PI * 2);
      ctx.fill();
    }
  }, [waveform, durationSeconds, currentTime]);

  const handleMouseMove = (e: React.MouseEvent<HTMLCanvasElement>) => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const rect = canvas.getBoundingClientRect();
    const x = e.clientX - rect.left;
    const ratio = Math.max(0, Math.min(1, x / rect.width));
    setHoverTime(ratio * durationSeconds);
  };

  const handleClick = (e: React.MouseEvent<HTMLCanvasElement>) => {
    if (!onSeek) return;
    const canvas = canvasRef.current;
    if (!canvas) return;
    const rect = canvas.getBoundingClientRect();
    const x = e.clientX - rect.left;
    const ratio = Math.max(0, Math.min(1, x / rect.width));
    onSeek(ratio * durationSeconds);
  };

  return (
    <div className="editorial-panel p-5">
      <div className="flex items-center justify-between mb-3">
        <div className="flex items-center gap-2">
          <Waves className="h-4 w-4 text-[#76556F]" />
          <h3 className="text-xs font-semibold uppercase tracking-wider text-[#27232B]">
            Amplitude Waveform &amp; Peak Envelope
          </h3>
        </div>
        <div className="flex items-center gap-3 text-xs text-[#6B626E]">
          <span className="font-mono text-[11px] text-[#76556F] bg-[#F0E8EE] px-2 py-0.5 rounded">
            {waveform.length} Samples
          </span>
          {hoverTime !== null && (
            <span className="font-mono text-[#D77C63] text-[11px] tabular-nums">
              Hover: {hoverTime.toFixed(2)}s
            </span>
          )}
          <span className="font-mono text-[#27232B] text-[11px] tabular-nums font-medium">
            {currentTime.toFixed(1)}s / {durationSeconds.toFixed(1)}s
          </span>
        </div>
      </div>

      <div className="relative w-full h-32 bg-[#FAF7F2] rounded border border-[#E3DDD4] overflow-hidden cursor-crosshair">
        <canvas
          ref={canvasRef}
          className="w-full h-full block"
          onMouseMove={handleMouseMove}
          onMouseLeave={() => setHoverTime(null)}
          onClick={handleClick}
        />
        <div className="absolute left-2 top-1 text-[9px] font-mono text-[#8E8692] select-none">
          +1.0
        </div>
        <div className="absolute left-2 bottom-1 text-[9px] font-mono text-[#8E8692] select-none">
          -1.0
        </div>
        <div className="absolute right-2 bottom-1 text-[9px] font-mono text-[#8E8692] select-none">
          {durationSeconds.toFixed(1)}s
        </div>
      </div>

      <div className="flex items-center justify-between mt-2.5 text-[10px] text-[#6B626E]">
        <span>Standardized 22,050 Hz PCM envelope. Click waveform to seek playhead.</span>
        <span className="font-mono text-[#76556F]">FMA 30s Crop</span>
      </div>
    </div>
  );
}
