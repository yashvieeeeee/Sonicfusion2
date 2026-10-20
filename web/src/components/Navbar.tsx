"use client";

import React, { useEffect, useState } from "react";
import Link from "next/link";
import { usePathname } from "next/navigation";
import {
  Activity,
  BarChart3,
  Cpu,
  FileAudio,
  Info,
  Radio,
  Sparkles,
} from "lucide-react";
import { checkHealth } from "@/lib/api";
import { HealthResponse } from "@/types/api";

export default function Navbar() {
  const pathname = usePathname();
  const [health, setHealth] = useState<HealthResponse | null>(null);
  const [isChecking, setIsChecking] = useState<boolean>(true);

  useEffect(() => {
    let mounted = true;
    async function loadHealth() {
      try {
        const data = await checkHealth();
        if (mounted) {
          setHealth(data);
          setIsChecking(false);
        }
      } catch (err) {
        if (mounted) {
          setHealth({
            status: "unavailable",
            model_loaded: false,
            version: "1.0",
            device: "cpu",
            detail: "Cannot connect to FastAPI backend at port 8000",
          });
          setIsChecking(false);
        }
      }
    }
    loadHealth();
    const interval = setInterval(loadHealth, 15000);
    return () => {
      mounted = false;
      clearInterval(interval);
    };
  }, []);

  const navLinks = [
    { href: "/", label: "Upload & Analyze", icon: FileAudio },
    { href: "/analysis", label: "Results", icon: Activity },
    { href: "/evaluation", label: "Model Evaluation", icon: BarChart3 },
    { href: "/about", label: "Model Card", icon: Info },
  ];

  return (
    <header className="sticky top-0 z-50 w-full border-b border-slate-800 bg-[#090d16]/85 backdrop-blur-md">
      <div className="mx-auto flex max-w-7xl items-center justify-between px-4 sm:px-6 lg:px-8 h-16">
        {/* Brand */}
        <div className="flex items-center gap-3">
          <Link href="/" className="flex items-center gap-2.5 group">
            <div className="flex h-9 w-9 items-center justify-center rounded-xl bg-gradient-to-tr from-indigo-600 via-indigo-500 to-cyan-400 text-white shadow-md shadow-indigo-500/25 group-hover:scale-105 transition-transform">
              <Radio className="h-5 w-5" />
            </div>
            <div>
              <div className="flex items-center gap-1.5">
                <span className="font-bold tracking-tight text-white text-base">
                  SonicScope
                </span>
                <span className="text-xs font-semibold px-1.5 py-0.5 rounded bg-indigo-500/20 text-indigo-400 border border-indigo-500/30">
                  AI
                </span>
              </div>
              <p className="text-[10px] text-slate-400 -mt-0.5 hidden sm:block">
                Music Intelligence & Signal Deep Learning
              </p>
            </div>
          </Link>
        </div>

        {/* Navigation links */}
        <nav className="flex items-center gap-1 sm:gap-2">
          {navLinks.map((item) => {
            const Icon = item.icon;
            const isActive = pathname === item.href;
            return (
              <Link
                key={item.href}
                href={item.href}
                className={`flex items-center gap-2 px-3 py-1.5 rounded-lg text-xs sm:text-sm font-medium transition-all ${
                  isActive
                    ? "bg-slate-800 text-indigo-400 border border-slate-700 shadow-sm"
                    : "text-slate-300 hover:text-white hover:bg-slate-800/60"
                }`}
              >
                <Icon className={`h-4 w-4 ${isActive ? "text-indigo-400" : "text-slate-400"}`} />
                <span>{item.label}</span>
              </Link>
            );
          })}
        </nav>

        {/* Backend Status Pill */}
        <div className="hidden md:flex items-center gap-2">
          <div
            className={`flex items-center gap-2 px-3 py-1 rounded-full text-xs font-medium border ${
              isChecking
                ? "bg-slate-800 text-slate-400 border-slate-700"
                : health?.model_loaded
                ? "bg-emerald-950/60 text-emerald-400 border-emerald-800/60"
                : "bg-rose-950/60 text-rose-400 border-rose-800/60"
            }`}
            title={health?.detail || "Backend health status"}
          >
            <span
              className={`h-2 w-2 rounded-full ${
                isChecking
                  ? "bg-slate-400 animate-pulse"
                  : health?.model_loaded
                  ? "bg-emerald-400 animate-pulse"
                  : "bg-rose-500"
              }`}
            />
            <span>
              {isChecking
                ? "Checking API..."
                : health?.model_loaded
                ? `FastAPI: Ready (${health.device.toUpperCase()})`
                : "API: Offline / Checkpoint Missing"}
            </span>
          </div>
        </div>
      </div>
    </header>
  );
}
