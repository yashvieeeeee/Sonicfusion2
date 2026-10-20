"use client";

import React, { useRef, useEffect, useState } from "react";
import { Maximize2, Waves } from "lucide-react";

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

    // Background grid lines
    ctx.strokeStyle = "rgba(51, 65, 85, 0.3)";
    ctx.lineWidth = 1;
    ctx.setLineDash([4, 4]);

    // Zero amplitude line
    ctx.beginPath();
    ctx.moveTo(0, midY);
    ctx.lineTo(width, midY);
    ctx.stroke();

    // +0.5 and -0.5 amplitude lines
    ctx.beginPath();
    ctx.moveTo(0, midY - height * 0.25);
    ctx.lineTo(width, midY - height * 0.25);
    ctx.moveTo(0, midY + height * 0.25);
    ctx.lineTo(width, midY + height * 0.25);
    ctx.stroke();
    ctx.setLineDash([]);

    // Draw waveform bars / envelope
    const numPoints = waveform.length;
    const step = width / numPoints;

    const gradient = ctx.createLinearGradient(0, 0, 0, height);
    gradient.addColorStop(0, "#06b6d4"); // cyan
    gradient.addColorStop(0.5, "#6366f1"); // indigo
    gradient.addColorStop(1, "#3b82f6"); // blue

    ctx.fillStyle = gradient;
    ctx.strokeStyle = "#818cf8";
    ctx.lineWidth = 1.5;

    for (let i = 0; i < numPoints; i++) {
      const x = i * step;
      const amp = Math.min(Math.abs(waveform[i]), 1.0);
      const barHeight = amp * (height * 0.88);
      const y = midY - barHeight / 2;

      ctx.fillRect(x, y, Math.max(step * 0.8, 1), Math.max(barHeight, 2));
    }

    // Draw playhead if playing
    if (durationSeconds > 0 && currentTime > 0) {
      const playX = (currentTime / durationSeconds) * width;
      ctx.strokeStyle = "#f43f5e"; // rose playhead
      ctx.lineWidth = 2;
      ctx.beginPath();
      ctx.moveTo(playX, 0);
      ctx.lineTo(playX, height);
      ctx.stroke();

      // Playhead handle
      ctx.fillStyle = "#f43f5e";
      ctx.beginPath();
      ctx.arc(playX, 6, 5, 0, Math.PI * 2);
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
    <div className="rounded-xl border border-slate-800 bg-slate-900/80 p-4">
      <div className="flex items-center justify-between mb-3">
        <div className="flex items-center gap-2">
          <Waves className="h-4 w-4 text-cyan-400" />
          <h4 className="text-xs font-semibold uppercase tracking-wider text-slate-300">
            Amplitude Waveform (Peak Envelope)
          </h4>
        </div>
        <div className="flex items-center gap-3 text-xs text-slate-400">
          <span>{waveform.length} sampled points</span>
          {hoverTime !== null && (
            <span className="text-cyan-400 font-mono">
              Hover: {hoverTime.toFixed(2)}s
            </span>
          )}
          <span className="font-mono text-slate-300">
            {currentTime.toFixed(1)}s / {durationSeconds.toFixed(1)}s
          </span>
        </div>
      </div>

      <div className="relative w-full h-36 bg-slate-950/70 rounded-lg overflow-hidden border border-slate-800/80 cursor-pointer">
        <canvas
          ref={canvasRef}
          className="w-full h-full block"
          onMouseMove={handleMouseMove}
          onMouseLeave={() => setHoverTime(null)}
          onClick={handleClick}
        />
        <div className="absolute left-2 top-1 text-[10px] font-mono text-slate-500 select-none">
          +1.0
        </div>
        <div className="absolute left-2 bottom-1 text-[10px] font-mono text-slate-500 select-none">
          -1.0
        </div>
        <div className="absolute right-2 bottom-1 text-[10px] font-mono text-slate-500 select-none">
          {durationSeconds.toFixed(1)}s
        </div>
      </div>
      <p className="text-[11px] text-slate-500 mt-2">
        Downsampled peak envelope per §11. Click on the waveform to seek playback.
      </p>
    </div>
  );
}
