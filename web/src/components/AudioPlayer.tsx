"use client";

import React, { useRef, useState, useEffect } from "react";
import { Pause, Play, RotateCcw, Volume2, VolumeX } from "lucide-react";

interface AudioPlayerProps {
  audioUrl: string | null;
  filename: string;
  onTimeUpdate?: (currentTime: number) => void;
  seekTime?: number | null;
}

export default function AudioPlayer({
  audioUrl,
  filename,
  onTimeUpdate,
  seekTime,
}: AudioPlayerProps) {
  const audioRef = useRef<HTMLAudioElement | null>(null);
  const [isPlaying, setIsPlaying] = useState<boolean>(false);
  const [currentTime, setCurrentTime] = useState<number>(0);
  const [duration, setDuration] = useState<number>(0);
  const [isMuted, setIsMuted] = useState<boolean>(false);

  useEffect(() => {
    if (seekTime !== null && seekTime !== undefined && audioRef.current) {
      audioRef.current.currentTime = seekTime;
      setCurrentTime(seekTime);
    }
  }, [seekTime]);

  useEffect(() => {
    setIsPlaying(false);
    setCurrentTime(0);
  }, [audioUrl]);

  const togglePlay = () => {
    if (!audioRef.current) return;
    if (isPlaying) {
      audioRef.current.pause();
      setIsPlaying(false);
    } else {
      audioRef.current
        .play()
        .then(() => setIsPlaying(true))
        .catch((err) => console.warn("Audio playback interrupted", err));
    }
  };

  const handleTimeUpdate = () => {
    if (!audioRef.current) return;
    const curr = audioRef.current.currentTime;
    setCurrentTime(curr);
    if (onTimeUpdate) {
      onTimeUpdate(curr);
    }
  };

  const handleLoadedMetadata = () => {
    if (!audioRef.current) return;
    setDuration(audioRef.current.duration || 0);
  };

  const handleSeekChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const val = parseFloat(e.target.value);
    if (audioRef.current) {
      audioRef.current.currentTime = val;
      setCurrentTime(val);
      if (onTimeUpdate) onTimeUpdate(val);
    }
  };

  const handleEnded = () => {
    setIsPlaying(false);
    setCurrentTime(0);
  };

  const toggleMute = () => {
    if (!audioRef.current) return;
    audioRef.current.muted = !isMuted;
    setIsMuted(!isMuted);
  };

  const restartAudio = () => {
    if (!audioRef.current) return;
    audioRef.current.currentTime = 0;
    setCurrentTime(0);
    audioRef.current.play().then(() => setIsPlaying(true));
  };

  const formatTime = (secs: number) => {
    if (isNaN(secs)) return "0:00";
    const m = Math.floor(secs / 60);
    const s = Math.floor(secs % 60);
    return `${m}:${s < 10 ? "0" : ""}${s}`;
  };

  if (!audioUrl) return null;

  return (
    <div className="glass-floating rounded-lg p-3.5 transition-all">
      <audio
        ref={audioRef}
        src={audioUrl}
        onTimeUpdate={handleTimeUpdate}
        onLoadedMetadata={handleLoadedMetadata}
        onEnded={handleEnded}
      />

      <div className="flex flex-col sm:flex-row items-center gap-3">
        {/* Play/Pause & Restart Controls */}
        <div className="flex items-center gap-1.5 shrink-0">
          <button
            onClick={togglePlay}
            className="flex h-9 w-9 items-center justify-center rounded-md bg-[#76556F] text-white hover:bg-[#583C52] active:scale-95 transition-all shadow-xs"
            title={isPlaying ? "Pause" : "Play"}
            aria-label={isPlaying ? "Pause audio" : "Play audio"}
          >
            {isPlaying ? (
              <Pause className="h-4 w-4" />
            ) : (
              <Play className="h-4 w-4 ml-0.5 fill-current" />
            )}
          </button>

          <button
            onClick={restartAudio}
            className="flex h-8 w-8 items-center justify-center rounded-md border border-[#E3DDD4] bg-[#FAF7F2] text-[#6B626E] hover:text-[#27232B] hover:bg-white transition-colors"
            title="Restart playback"
            aria-label="Restart audio"
          >
            <RotateCcw className="h-3.5 w-3.5" />
          </button>
        </div>

        {/* Track Title and Seekbar */}
        <div className="flex-1 w-full min-w-0">
          <div className="flex items-center justify-between text-xs mb-1">
            <span className="font-medium text-[#27232B] truncate max-w-sm">
              {filename}
            </span>
            <span className="font-mono text-[#6B626E] text-[11px] tabular-nums">
              {formatTime(currentTime)} / {formatTime(duration)}
            </span>
          </div>

          <input
            type="range"
            min={0}
            max={duration || 100}
            step={0.1}
            value={currentTime}
            onChange={handleSeekChange}
            aria-label="Seek audio"
            className="w-full h-1.5 bg-[#EAE2D8] rounded-xs appearance-none cursor-pointer accent-[#D77C63]"
          />
        </div>

        {/* Volume button */}
        <div className="flex items-center gap-2 shrink-0">
          <button
            onClick={toggleMute}
            className="flex h-8 w-8 items-center justify-center rounded-md border border-[#E3DDD4] bg-[#FAF7F2] text-[#6B626E] hover:text-[#27232B] hover:bg-white transition-colors"
            title={isMuted ? "Unmute" : "Mute"}
            aria-label={isMuted ? "Unmute audio" : "Mute audio"}
          >
            {isMuted ? (
              <VolumeX className="h-3.5 w-3.5 text-[#D77C63]" />
            ) : (
              <Volume2 className="h-3.5 w-3.5" />
            )}
          </button>
        </div>
      </div>
    </div>
  );
}
