"use client";

import React, { useEffect, useState } from "react";
import Link from "next/link";
import { usePathname } from "next/navigation";
import { ChevronRight, Menu, Plus } from "lucide-react";
import { checkHealth } from "@/lib/api";
import { HealthResponse } from "@/types/api";

interface NavbarProps {
  onToggleSidebar?: () => void;
}

export default function Navbar({ onToggleSidebar }: NavbarProps) {
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
      } catch {
        if (mounted) {
          setHealth({
            status: "unavailable",
            model_loaded: false,
            version: "1.0",
            device: "cpu",
            detail: "Cannot connect to FastAPI backend on port 8000",
          });
          setIsChecking(false);
        }
      }
    }
    loadHealth();
    const interval = setInterval(loadHealth, 20000);
    return () => {
      mounted = false;
      clearInterval(interval);
    };
  }, []);

  const getPageTitle = () => {
    if (pathname === "/") return "Home";
    if (pathname === "/studio") return "Studio";
    if (pathname.startsWith("/analysis")) return "Audio Analysis";
    if (pathname.startsWith("/collection")) return "Collection Analytics";
    if (pathname.startsWith("/evaluation")) return "Model Evaluation";
    if (pathname.startsWith("/about")) return "Model Card";
    return "Workspace";
  };

  return (
    <header className="sticky top-0 z-30 flex h-16 w-full items-center justify-between border-b border-[#E3DDD4] bg-[#F5F0E8]/85 backdrop-blur-md px-4 sm:px-6 shadow-2xs">
      {/* Left: Mobile Sidebar Trigger & Breadcrumb */}
      <div className="flex items-center gap-3">
        {onToggleSidebar && (
          <button
            onClick={onToggleSidebar}
            className="p-1.5 text-[#6B626E] hover:text-[#27232B] md:hidden rounded border border-[#E3DDD4] bg-white/80 shadow-2xs cursor-pointer"
            aria-label="Toggle navigation menu"
          >
            <Menu className="h-4 w-4" />
          </button>
        )}

        <div className="flex items-center gap-2 text-xs">
          <Link
            href="/"
            className="text-[#6B626E] hover:text-[#76556F] transition-colors font-medium"
          >
            Sonic Fusion
          </Link>
          <ChevronRight className="h-3 w-3 text-[#A6B5AD]" />
          <span className="font-semibold text-[#27232B]">
            {getPageTitle()}
          </span>
        </div>
      </div>

      {/* Right: Actual System Health Status & Primary Action */}
      <div className="flex items-center gap-3">
        {/* Model Status Indicator */}
        <div
          className={`flex items-center gap-2 px-2.5 py-1 rounded-full border text-[11px] font-mono transition-colors shadow-2xs ${
            isChecking
              ? "bg-white/70 text-[#6B626E] border-[#E3DDD4]"
              : health?.model_loaded
              ? "bg-[#EEF3F0] text-[#42584D] border-[#D0DEC9]"
              : "bg-[#FDF1EE] text-[#D77C63] border-[#F1D0CD]"
          }`}
          title={health?.detail || "Backend health status"}
        >
          <span
            className={`h-1.5 w-1.5 rounded-full ${
              isChecking
                ? "bg-[#A6B5AD]"
                : health?.model_loaded
                ? "bg-[#6C7F75] animate-pulse"
                : "bg-[#D77C63]"
            }`}
          />
          <span className="hidden sm:inline">
            {isChecking
              ? "Checking Backend..."
              : health?.model_loaded
              ? `GenreCNN: Online (${health.device.toUpperCase()})`
              : "Backend Offline"}
          </span>
          <span className="sm:hidden">
            {isChecking
              ? "..."
              : health?.model_loaded
              ? "Online"
              : "Offline"}
          </span>
        </div>

        {/* Enter Studio / Ingest Audio Action */}
        <Link
          href="/studio"
          className="flex items-center gap-1.5 rounded-xl bg-[#76556F] px-3.5 py-1.5 text-xs font-semibold text-white shadow-2xs hover:bg-[#583C52] transition-colors cursor-pointer"
        >
          <Plus className="h-3.5 w-3.5 text-[#D77C63]" />
          <span>{pathname === "/" ? "Enter Studio" : "Studio Ingestion"}</span>
        </Link>
      </div>
    </header>
  );
}
