"use client";

import React, { useRef, useEffect, useState } from "react";
import { Activity } from "lucide-react";

interface SpectrogramHeatmapProps {
  spectrogram: number[][]; // [freq_bins, time_bins]
  durationSeconds: number;
}

// Custom perceptually uniform sequential colormap matching the Atmospheric Laboratory palette:
// Deep Ink Plum -> Dark Dusty Plum -> Dusty Plum -> Muted Coral -> Soft Amber -> Warm Ivory Peak
function laboratorySpectrogramColor(norm: number): [number, number, number] {
  const t = Math.max(0, Math.min(1, norm));

  const stops = [
    { p: 0.0, r: 31, g: 26, b: 34 },    // Deep ink plum (#1F1A22)
    { p: 0.22, r: 77, g: 49, b: 71 },   // Dark dusty plum (#4D3147)
    { p: 0.45, r: 118, g: 85, b: 111 }, // Dusty plum (#76556F)
    { p: 0.70, r: 215, g: 124, b: 99 }, // Muted coral (#D77C63)
    { p: 0.88, r: 214, g: 167, b: 93 }, // Soft amber (#D6A75D)
    { p: 1.0, r: 255, g: 249, b: 238 }, // Warm pale ivory peak (#FFF9EE)
  ];

  for (let i = 0; i < stops.length - 1; i++) {
    const s0 = stops[i];
    const s1 = stops[i + 1];
    if (t >= s0.p && t <= s1.p) {
      const localT = (t - s0.p) / (s1.p - s0.p);
      const r = Math.round(s0.r + localT * (s1.r - s0.r));
      const g = Math.round(s0.g + localT * (s1.g - s0.g));
      const b = Math.round(s0.b + localT * (s1.b - s0.b));
      return [r, g, b];
    }
  }

  return [255, 249, 238];
}

export default function SpectrogramHeatmap({
  spectrogram,
  durationSeconds,
}: SpectrogramHeatmapProps) {
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const [hoverInfo, setHoverInfo] = useState<{
    timeSec: number;
    freqHz: number;
    dbValue: number;
    x: number;
    y: number;
  } | null>(null);

  const numFreq = spectrogram.length;
  const numTime = numFreq > 0 ? spectrogram[0].length : 0;

  // Compute min/max dB
  let minDb = -80;
  let maxDb = 0;
  if (numFreq > 0 && numTime > 0) {
    let localMin = Infinity;
    let localMax = -Infinity;
    for (let f = 0; f < numFreq; f++) {
      for (let t = 0; t < numTime; t++) {
        const v = spectrogram[f][t];
        if (v < localMin) localMin = v;
        if (v > localMax) localMax = v;
      }
    }
    if (localMin !== Infinity) minDb = Math.floor(localMin);
    if (localMax !== -Infinity) maxDb = Math.ceil(localMax);
  }

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas || numFreq === 0 || numTime === 0) return;

    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    const width = canvas.width;
    const height = canvas.height;
    const imgData = ctx.createImageData(width, height);
    const data = imgData.data;

    const range = Math.max(maxDb - minDb, 1.0);

    for (let py = 0; py < height; py++) {
      // Audio frequency 0 Hz is at the bottom, Nyquist at top
      const freqIdx = Math.floor(((height - 1 - py) / height) * numFreq);
      const safeFreq = Math.min(Math.max(freqIdx, 0), numFreq - 1);

      for (let px = 0; px < width; px++) {
        const timeIdx = Math.floor((px / width) * numTime);
        const safeTime = Math.min(Math.max(timeIdx, 0), numTime - 1);

        const db = spectrogram[safeFreq][safeTime];
        const norm = (db - minDb) / range;
        const [r, g, b] = laboratorySpectrogramColor(norm);

        const pIndex = (py * width + px) * 4;
        data[pIndex] = r;
        data[pIndex + 1] = g;
        data[pIndex + 2] = b;
        data[pIndex + 3] = 255;
      }
    }

    ctx.putImageData(imgData, 0, 0);
  }, [spectrogram, numFreq, numTime, minDb, maxDb]);

  const handleMouseMove = (e: React.MouseEvent<HTMLCanvasElement>) => {
    const canvas = canvasRef.current;
    if (!canvas || numFreq === 0 || numTime === 0) return;
    const rect = canvas.getBoundingClientRect();
    const px = e.clientX - rect.left;
    const py = e.clientY - rect.top;

    const timeRatio = Math.max(0, Math.min(1, px / rect.width));
    const freqRatio = Math.max(0, Math.min(1, 1 - py / rect.height));

    const timeIdx = Math.min(Math.floor(timeRatio * numTime), numTime - 1);
    const freqIdx = Math.min(Math.floor(freqRatio * numFreq), numFreq - 1);

    const db = spectrogram[freqIdx][timeIdx];
    const timeSec = timeRatio * durationSeconds;
    // Nyquist of 22,050 Hz is 11,025 Hz
    const freqHz = Math.round(freqRatio * 11025);

    setHoverInfo({
      timeSec,
      freqHz,
      dbValue: db,
      x: px,
      y: py,
    });
  };

  return (
    <div className="editorial-panel p-5">
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-2 mb-3">
        <div className="flex items-center gap-2">
          <Activity className="h-4 w-4 text-[#76556F]" />
          <h3 className="text-xs font-semibold uppercase tracking-wider text-[#27232B]">
            Log-Mel Power Spectrogram (128 Mel Bins &times; {numTime} Frames)
          </h3>
        </div>
        <div className="flex items-center gap-3 text-xs text-[#6B626E]">
          <span>Range: {minDb} to {maxDb} dB</span>
          <span className="font-mono text-[11px] text-[#76556F] bg-[#F0E8EE] px-2 py-0.5 rounded font-medium">
            Tensor [{numFreq}&times;{numTime}]
          </span>
        </div>
      </div>

      <div className="relative">
        <div className="flex items-stretch gap-2">
          {/* Y-axis frequency scale */}
          <div className="flex flex-col justify-between py-1 text-[10px] font-mono text-[#8E8692] text-right w-14 select-none">
            <span>11.0 kHz</span>
            <span>5.5 kHz</span>
            <span>2.2 kHz</span>
            <span>0 Hz</span>
          </div>

          {/* Canvas container */}
          <div className="relative flex-1 h-52 bg-[#1F1A22] rounded border border-[#E3DDD4] overflow-hidden cursor-crosshair">
            <canvas
              ref={canvasRef}
              width={numTime || 300}
              height={numFreq || 128}
              className="w-full h-full block"
              onMouseMove={handleMouseMove}
              onMouseLeave={() => setHoverInfo(null)}
            />

            {/* Selective glassmorphic floating inspection tooltip */}
            {hoverInfo && (
              <div
                className="pointer-events-none absolute z-20 px-2.5 py-1.5 rounded bg-white/90 backdrop-blur-md border border-[#E3DDD4] text-[11px] font-mono text-[#27232B] shadow-md"
                style={{
                  left: Math.min(hoverInfo.x + 12, 260),
                  top: Math.max(hoverInfo.y - 50, 8),
                }}
              >
                <div>t = {hoverInfo.timeSec.toFixed(2)}s</div>
                <div>f = {hoverInfo.freqHz} Hz</div>
                <div className="text-[#D77C63] font-semibold">
                  {hoverInfo.dbValue.toFixed(1)} dB
                </div>
              </div>
            )}
          </div>
        </div>

        {/* X-axis time scale */}
        <div className="flex justify-between pl-16 pr-2 mt-1.5 text-[10px] font-mono text-[#8E8692] select-none">
          <span>0.0s</span>
          <span>{(durationSeconds * 0.25).toFixed(1)}s</span>
          <span>{(durationSeconds * 0.5).toFixed(1)}s</span>
          <span>{(durationSeconds * 0.75).toFixed(1)}s</span>
          <span>{durationSeconds.toFixed(1)}s</span>
        </div>
      </div>

      {/* Color scale gradient indicator */}
      <div className="mt-4 pt-3 border-t border-[#E3DDD4] flex flex-col sm:flex-row items-center justify-between gap-2 text-[11px] text-[#6B626E]">
        <div className="flex items-center gap-2">
          <span>Spectral Power:</span>
          <div className="flex items-center gap-1.5">
            <span className="font-mono text-[10px]">{minDb} dB</span>
            <div
              className="h-2.5 w-44 rounded-xs border border-[#E3DDD4]"
              style={{
                background: "linear-gradient(to right, #1F1A22 0%, #4D3147 22%, #76556F 45%, #D77C63 70%, #D6A75D 88%, #FFF9EE 100%)",
              }}
            />
            <span className="font-mono text-[10px]">{maxDb} dB</span>
          </div>
        </div>
        <span className="text-[10px] text-[#8E8692]">
          STFT 2048 FFT &bull; 512 Hop &bull; 128 Mel Filters (CNN Input Tensor)
        </span>
      </div>
    </div>
  );
}
