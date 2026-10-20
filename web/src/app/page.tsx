"use client";

import React, { useState, useRef } from "react";
import { useRouter } from "next/navigation";
import {
  AlertCircle,
  ArrowRight,
  Disc,
  FileAudio,
  Music,
  Radio,
  Sparkles,
  UploadCloud,
  Zap,
} from "lucide-react";
import AudioPlayer from "@/components/AudioPlayer";
import StagedLoading from "@/components/StagedLoading";
import { analyzeAudio, fetchSampleAudio, getModels } from "@/lib/api";
import { saveAnalysisResult } from "@/lib/storage";

interface SampleTrack {
  id: string;
  name: string;
  genre: string;
  duration: string;
  url: string;
}

const SAMPLE_TRACKS: SampleTrack[] = [
  {
    id: "000002",
    name: "FMA 000002",
    genre: "Hip-Hop / Electronic",
    duration: "30s",
    url: "/samples/sample_000002.mp3",
  },
  {
    id: "000005",
    name: "FMA 000005",
    genre: "Rock / Indie",
    duration: "30s",
    url: "/samples/sample_000005.mp3",
  },
];

export default function UploadPage() {
  const router = useRouter();
  const fileInputRef = useRef<HTMLInputElement | null>(null);

  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [audioUrl, setAudioUrl] = useState<string | null>(null);
  const [isDragging, setIsDragging] = useState<boolean>(false);
  const [isAnalyzing, setIsAnalyzing] = useState<boolean>(false);
  const [errorMessage, setErrorMessage] = useState<string | null>(null);
  const [loadingSample, setLoadingSample] = useState<string | null>(null);

  // Config-driven limits loaded dynamically from GET /api/models
  const [configLimits, setConfigLimits] = useState<{
    maxDurationSeconds: number;
    maxUploadMb: number;
    allowedFormats: string[];
  }>({
    maxDurationSeconds: 900,
    maxUploadMb: 30,
    allowedFormats: ["mp3", "wav", "m4a"],
  });

  React.useEffect(() => {
    let isMounted = true;
    getModels()
      .then((data) => {
        if (isMounted && data) {
          setConfigLimits({
            maxDurationSeconds: data.max_duration_seconds || 900,
            maxUploadMb: data.max_upload_mb || 30,
            allowedFormats: data.allowed_formats || ["mp3", "wav", "m4a"],
          });
        }
      })
      .catch(() => {
        // Fallback to defaults if backend unavailable at mount
      });
    return () => {
      isMounted = false;
    };
  }, []);

  const formatsString = configLimits.allowedFormats.map((f) => f.toUpperCase()).join(", ");
  const maxMinutes = Math.round(configLimits.maxDurationSeconds / 60);
  const acceptAttribute = [
    ...configLimits.allowedFormats.map((f) => `.${f.toLowerCase()}`),
    "audio/*",
  ].join(",");
  const helperText = `Supported formats: ${formatsString}; max ${maxMinutes} minutes; max ${configLimits.maxUploadMb} MB`;

  const handleFileSelect = (file: File) => {
    setErrorMessage(null);

    // Client-side size cap from config
    const maxSizeBytes = configLimits.maxUploadMb * 1024 * 1024;
    if (file.size > maxSizeBytes) {
      setErrorMessage(`File exceeds ${configLimits.maxUploadMb} MB limit. Please select a smaller audio file.`);
      return;
    }

    // Do NOT block on extension alone; backend verifies container by real content
    setSelectedFile(file);
    const objectUrl = URL.createObjectURL(file);
    setAudioUrl(objectUrl);
  };

  const handleDrop = (e: React.DragEvent<HTMLDivElement>) => {
    e.preventDefault();
    setIsDragging(false);
    if (e.dataTransfer.files && e.dataTransfer.files.length > 0) {
      handleFileSelect(e.dataTransfer.files[0]);
    }
  };

  const handleDragOver = (e: React.DragEvent<HTMLDivElement>) => {
    e.preventDefault();
    setIsDragging(true);
  };

  const handleDragLeave = () => {
    setIsDragging(false);
  };

  const handleSampleClick = async (sample: SampleTrack) => {
    setErrorMessage(null);
    setLoadingSample(sample.id);
    try {
      const file = await fetchSampleAudio(sample.url);
      setSelectedFile(file);
      setAudioUrl(sample.url);
    } catch (err: unknown) {
      const msg = err instanceof Error ? err.message : "Failed to load sample track";
      setErrorMessage(msg);
    } finally {
      setLoadingSample(null);
    }
  };

  const startAnalysis = async () => {
    if (!selectedFile) return;
    setIsAnalyzing(true);
    setErrorMessage(null);

    try {
      // Real API call to FastAPI backend POST /api/analyze
      const result = await analyzeAudio(selectedFile);
      saveAnalysisResult(result);
      router.push("/analysis");
    } catch (err: unknown) {
      const msg = err instanceof Error ? err.message : "Analysis failed unexpectedly.";
      setErrorMessage(msg);
      setIsAnalyzing(false);
    }
  };

  return (
    <div className="flex-1 py-8 sm:py-12 px-4 sm:px-6 lg:px-8 max-w-5xl mx-auto w-full">
      {/* Hero section */}
      <div className="text-center mb-8 sm:mb-12">
        <div className="inline-flex items-center gap-2 px-3 py-1 rounded-full bg-indigo-500/10 border border-indigo-500/20 text-indigo-400 text-xs font-semibold mb-4">
          <Sparkles className="h-3.5 w-3.5" />
          <span>FMA Small 8-Genre Recognition & Signal Analysis</span>
        </div>
        <h1 className="text-3xl sm:text-5xl font-extrabold tracking-tight text-white mb-3">
          Music Intelligence & Acoustic Insights
        </h1>
        <p className="max-w-2xl mx-auto text-sm sm:text-base text-slate-400 leading-relaxed">
          Upload any audio track to extract tempo (BPM), signal energy, 12-bin pitch chroma,
          generate high-resolution log-mel spectrograms, and classify musical genre using our
          trained 2D deep learning CNN.
        </p>
      </div>

      {isAnalyzing ? (
        <StagedLoading filename={selectedFile?.name || "audio.mp3"} />
      ) : (
        <div className="space-y-6">
          {/* Upload Dropzone */}
          <div
            onDrop={handleDrop}
            onDragOver={handleDragOver}
            onDragLeave={handleDragLeave}
            onClick={() => fileInputRef.current?.click()}
            className={`relative rounded-2xl border-2 border-dashed p-8 sm:p-12 text-center transition-all cursor-pointer ${
              isDragging
                ? "border-indigo-400 bg-indigo-950/30 scale-[1.01]"
                : "border-slate-800 bg-slate-900/50 hover:border-slate-700 hover:bg-slate-900/80"
            }`}
          >
            <input
              ref={fileInputRef}
              type="file"
              accept={acceptAttribute}
              className="hidden"
              onChange={(e) => {
                if (e.target.files && e.target.files.length > 0) {
                  handleFileSelect(e.target.files[0]);
                }
              }}
            />

            <div className="flex flex-col items-center gap-4">
              <div className="flex h-16 w-16 items-center justify-center rounded-2xl bg-indigo-500/10 text-indigo-400 border border-indigo-500/20">
                <UploadCloud className="h-8 w-8" />
              </div>
              <div>
                <h3 className="text-lg font-semibold text-white">
                  Drop your audio track here, or{" "}
                  <span className="text-indigo-400 underline decoration-indigo-400/40">
                    browse files
                  </span>
                </h3>
                <p className="text-xs text-slate-400 mt-1">
                  {helperText}
                </p>
              </div>
            </div>
          </div>

          {/* Quick-test with Sample FMA Small tracks */}
          <div className="rounded-xl border border-slate-800/80 bg-slate-900/40 p-4">
            <div className="flex items-center justify-between mb-3">
              <div className="flex items-center gap-2">
                <Music className="h-4 w-4 text-cyan-400" />
                <span className="text-xs font-semibold uppercase tracking-wider text-slate-300">
                  Or test instantly with FMA Small benchmark samples:
                </span>
              </div>
              <span className="text-[11px] text-slate-500 hidden sm:inline">
                Real audio from Free Music Archive
              </span>
            </div>

            <div className="grid grid-cols-1 sm:grid-cols-2 gap-3">
              {SAMPLE_TRACKS.map((sample) => (
                <button
                  key={sample.id}
                  onClick={() => handleSampleClick(sample)}
                  disabled={loadingSample !== null}
                  className="flex items-center justify-between p-3 rounded-lg border border-slate-800 bg-slate-950/60 hover:border-indigo-500/50 hover:bg-slate-800/50 transition-all text-left group"
                >
                  <div className="flex items-center gap-3">
                    <div className="flex h-8 w-8 items-center justify-center rounded-md bg-slate-800 text-slate-400 group-hover:bg-indigo-600 group-hover:text-white transition-colors">
                      <Disc className="h-4 w-4" />
                    </div>
                    <div>
                      <div className="text-xs font-semibold text-slate-200 group-hover:text-white">
                        {sample.name}
                      </div>
                      <div className="text-[10px] text-slate-400">
                        {sample.genre} &bull; {sample.duration}
                      </div>
                    </div>
                  </div>
                  <span className="text-xs font-medium text-indigo-400 group-hover:translate-x-0.5 transition-transform">
                    {loadingSample === sample.id ? "Loading..." : "Load Sample →"}
                  </span>
                </button>
              ))}
            </div>
          </div>

          {/* Error Banner */}
          {errorMessage && (
            <div className="p-4 rounded-xl bg-rose-950/60 border border-rose-600/60 text-rose-200 text-xs flex items-start gap-3">
              <AlertCircle className="h-5 w-5 text-rose-400 shrink-0 mt-0.5" />
              <div>
                <strong className="font-semibold text-rose-300">
                  Audio Processing Error:
                </strong>{" "}
                {errorMessage}
              </div>
            </div>
          )}

          {/* Selected File & Audio Preview Player */}
          {selectedFile && audioUrl && (
            <div className="rounded-2xl border border-indigo-500/30 bg-slate-900/90 p-5 sm:p-6 backdrop-blur-md shadow-xl glow-primary">
              <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4 mb-4 pb-4 border-b border-slate-800">
                <div className="flex items-center gap-3">
                  <div className="flex h-10 w-10 items-center justify-center rounded-xl bg-indigo-500/20 text-indigo-400 border border-indigo-500/30">
                    <FileAudio className="h-5 w-5" />
                  </div>
                  <div>
                    <h4 className="text-sm font-semibold text-white truncate max-w-sm">
                      {selectedFile.name}
                    </h4>
                    <p className="text-xs text-slate-400">
                      {(selectedFile.size / (1024 * 1024)).toFixed(2)} MB &bull; Audio track ready
                    </p>
                  </div>
                </div>

                {/* Primary CTA */}
                <button
                  onClick={startAnalysis}
                  className="flex items-center justify-center gap-2 px-6 py-3 rounded-xl bg-gradient-to-r from-indigo-600 via-indigo-500 to-cyan-500 text-white font-semibold text-sm shadow-lg shadow-indigo-500/30 hover:scale-[1.02] active:scale-[0.98] transition-all"
                >
                  <Zap className="h-4 w-4" />
                  <span>Analyze Audio</span>
                  <ArrowRight className="h-4 w-4 ml-1" />
                </button>
              </div>

              {/* Player Preview */}
              <div className="mt-2">
                <span className="text-[11px] font-semibold uppercase tracking-wider text-slate-400 block mb-2">
                  Preview Audio Playback:
                </span>
                <AudioPlayer audioUrl={audioUrl} filename={selectedFile.name} />
              </div>
            </div>
          )}
        </div>
      )}

      {/* Feature Highlights Grid */}
      <div className="mt-16 grid grid-cols-1 sm:grid-cols-3 gap-6 text-center sm:text-left">
        <div className="p-5 rounded-xl border border-slate-800/80 bg-slate-900/30">
          <div className="flex h-9 w-9 items-center justify-center rounded-lg bg-indigo-500/10 text-indigo-400 mb-3 mx-auto sm:mx-0">
            <Radio className="h-5 w-5" />
          </div>
          <h4 className="text-sm font-semibold text-white mb-1">
            2D CNN Deep Learning
          </h4>
          <p className="text-xs text-slate-400 leading-relaxed">
            Standardized 128-band log-mel spectrograms processed through 3-block convolutional neural network.
          </p>
        </div>

        <div className="p-5 rounded-xl border border-slate-800/80 bg-slate-900/30">
          <div className="flex h-9 w-9 items-center justify-center rounded-lg bg-cyan-500/10 text-cyan-400 mb-3 mx-auto sm:mx-0">
            <Music className="h-5 w-5" />
          </div>
          <h4 className="text-sm font-semibold text-white mb-1">
            Acoustic Signal Metrics
          </h4>
          <p className="text-xs text-slate-400 leading-relaxed">
            Algorithmic tempo (BPM), frame-level RMS energy profiles, and 12-bin pitch-class chroma vectors.
          </p>
        </div>

        <div className="p-5 rounded-xl border border-slate-800/80 bg-slate-900/30">
          <div className="flex h-9 w-9 items-center justify-center rounded-lg bg-amber-500/10 text-amber-400 mb-3 mx-auto sm:mx-0">
            <AlertCircle className="h-5 w-5" />
          </div>
          <h4 className="text-sm font-semibold text-white mb-1">
            Uncertainty Detection
          </h4>
          <p className="text-xs text-slate-400 leading-relaxed">
            Explicit &quot;uncertain / multiple influences&quot; warnings when confidence is low or margins are narrow.
          </p>
        </div>
      </div>
    </div>
  );
}
