"use client";

import React, { useRef, useEffect, useState } from "react";
import { Activity, Flame } from "lucide-react";

interface SpectrogramHeatmapProps {
  spectrogram: number[][]; // [freq_bins, time_bins]
  durationSeconds: number;
}

// Perceptually uniform Inferno color interpolation for dB values
function infernoColor(val: number): [number, number, number] {
  // val is normalized 0.0 to 1.0
  const t = Math.max(0, Math.min(1, val));
  // Polynomial approximation of inferno colormap
  const r = Math.min(255, Math.max(0, Math.round(255 * (0.0002 + 1.25 * t + 0.9 * t * t - 1.15 * t * t * t))));
  const g = Math.min(255, Math.max(0, Math.round(255 * (0.001 + 0.1 * t + 1.8 * t * t - 0.9 * t * t * t))));
  const b = Math.min(255, Math.max(0, Math.round(255 * (0.05 + 1.2 * t - 1.8 * t * t + 0.6 * t * t * t))));
  return [r, g, b];
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

    // Use actual dimensions
    const width = canvas.width;
    const height = canvas.height;
    const imgData = ctx.createImageData(width, height);
    const data = imgData.data;

    const range = Math.max(maxDb - minDb, 1.0);

    for (let py = 0; py < height; py++) {
      // In audio spectrograms, freq 0 is bottom, highest freq is top
      const freqIdx = Math.floor(((height - 1 - py) / height) * numFreq);
      const safeFreq = Math.min(Math.max(freqIdx, 0), numFreq - 1);

      for (let px = 0; px < width; px++) {
        const timeIdx = Math.floor((px / width) * numTime);
        const safeTime = Math.min(Math.max(timeIdx, 0), numTime - 1);

        const db = spectrogram[safeFreq][safeTime];
        const norm = (db - minDb) / range;
        const [r, g, b] = infernoColor(norm);

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
    // 0 to 11,025 Hz (Nyquist of 22,050 Hz)
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
    <div className="rounded-xl border border-slate-800 bg-slate-900/80 p-4">
      <div className="flex items-center justify-between mb-3">
        <div className="flex items-center gap-2">
          <Flame className="h-4 w-4 text-amber-400" />
          <h4 className="text-xs font-semibold uppercase tracking-wider text-slate-300">
            Log-Mel Spectrogram Heatmap (128 Mel Bins x {numTime} Frames)
          </h4>
        </div>
        <div className="flex items-center gap-4 text-xs text-slate-400">
          <span>Range: {minDb.toFixed(0)} dB to {maxDb.toFixed(0)} dB</span>
          <span className="font-mono text-indigo-400">
            Shape: [{numFreq}, {numTime}]
          </span>
        </div>
      </div>

      <div className="relative">
        {/* Y Axis frequency labels */}
        <div className="flex items-stretch gap-2">
          <div className="flex flex-col justify-between py-1 text-[10px] font-mono text-slate-500 text-right w-12 select-none">
            <span>11 kHz</span>
            <span>5.5 kHz</span>
            <span>2.2 kHz</span>
            <span>0 Hz</span>
          </div>

          {/* Canvas container */}
          <div className="relative flex-1 h-56 bg-black rounded-lg overflow-hidden border border-slate-800">
            <canvas
              ref={canvasRef}
              width={numTime || 300}
              height={numFreq || 128}
              className="w-full h-full block cursor-crosshair"
              onMouseMove={handleMouseMove}
              onMouseLeave={() => setHoverInfo(null)}
            />

            {hoverInfo && (
              <div
                className="pointer-events-none absolute z-20 px-2 py-1 rounded bg-slate-900/95 border border-indigo-500/50 text-[11px] font-mono text-white shadow-xl"
                style={{
                  left: Math.min(hoverInfo.x + 12, 280),
                  top: Math.max(hoverInfo.y - 45, 10),
                }}
              >
                <div>t = {hoverInfo.timeSec.toFixed(2)}s</div>
                <div>f = {hoverInfo.freqHz} Hz</div>
                <div className="text-amber-400">
                  {hoverInfo.dbValue.toFixed(1)} dB
                </div>
              </div>
            )}
          </div>
        </div>

        {/* X Axis time labels */}
        <div className="flex justify-between pl-14 pr-2 mt-1.5 text-[10px] font-mono text-slate-500 select-none">
          <span>0.0s</span>
          <span>{(durationSeconds * 0.25).toFixed(1)}s</span>
          <span>{(durationSeconds * 0.5).toFixed(1)}s</span>
          <span>{(durationSeconds * 0.75).toFixed(1)}s</span>
          <span>{durationSeconds.toFixed(1)}s</span>
        </div>
      </div>

      {/* Colorbar legend */}
      <div className="mt-4 flex items-center justify-between text-[11px] text-slate-400">
        <span>Low Energy ({minDb.toFixed(0)} dB)</span>
        <div className="h-2.5 w-60 rounded-full bg-gradient-to-r from-[#000004] via-[#bb3754] to-[#fcffa4] border border-slate-700/60" />
        <span>Peak Energy ({maxDb.toFixed(0)} dB)</span>
      </div>
      <p className="text-[11px] text-slate-500 mt-2">
        STFT log-mel power spectrogram (2,048 FFT, 512 hop, 128 bands). This representation serves as direct 2D image input into the GenreCNN classifier.
      </p>
    </div>
  );
}
