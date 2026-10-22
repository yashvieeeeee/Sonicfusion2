"use client";

import React from "react";
import Link from "next/link";
import { usePathname } from "next/navigation";
import {
  Activity,
  BarChart2,
  BookOpen,
  Database,
  FileAudio,
  GitCompare,
  Home,
  ShieldCheck,
  Upload,
  X,
} from "lucide-react";

interface SidebarProps {
  isOpen?: boolean;
  onClose?: () => void;
}

export default function Sidebar({ isOpen = false, onClose }: SidebarProps) {
  const pathname = usePathname();

  const navigation = [
    {
      name: "Home",
      href: "/",
      icon: Home,
      exact: true,
      description: "Landing page & discovery",
    },
    {
      name: "Studio Workspace",
      href: "/studio",
      icon: Upload,
      exact: true,
      description: "Audio ingestion & staging",
    },
    {
      name: "Audio Analysis",
      href: "/analysis",
      icon: Activity,
      exact: true,
      description: "Waveform, spectrogram & chroma",
    },
    {
      name: "Genre Intelligence",
      href: "/analysis#genre",
      icon: BarChart2,
      exact: false,
      description: "Class probabilities & uncertainty",
    },
    {
      name: "Similar Tracks",
      href: "/analysis#similarity",
      icon: GitCompare,
      exact: false,
      description: "Acoustic nearest neighbors",
    },
    {
      name: "Collection Analytics",
      href: "/collection",
      icon: Database,
      exact: true,
      description: "FMA 8,000 track corpus & EDA",
    },
    {
      name: "Model Evaluation",
      href: "/evaluation",
      icon: ShieldCheck,
      exact: true,
      description: "Held-out test benchmark & McNemar",
    },
    {
      name: "Model Card",
      href: "/about",
      icon: BookOpen,
      exact: true,
      description: "Model card & dataset provenance",
    },
  ];

  const isLinkActive = (item: typeof navigation[0]) => {
    if (item.exact) {
      return pathname === item.href;
    }
    return pathname.startsWith(item.href.split("#")[0]);
  };

  return (
    <>
      {/* Mobile Backdrop */}
      {isOpen && (
        <div
          className="fixed inset-0 z-40 bg-[#27232B]/25 backdrop-blur-xs md:hidden"
          onClick={onClose}
          aria-hidden="true"
        />
      )}

      {/* Sidebar Panel */}
      <aside
        className={`fixed inset-y-0 left-0 z-50 flex w-64 flex-col border-r border-[#E3DDD4] bg-[#F0EAE0]/70 backdrop-blur-md transition-transform duration-200 ease-in-out md:static md:translate-x-0 ${
          isOpen ? "translate-x-0" : "-translate-x-full"
        }`}
      >
        {/* Brand Header */}
        <div className="flex h-16 items-center justify-between border-b border-[#E3DDD4] px-5">
          <Link
            href="/"
            onClick={onClose}
            className="flex items-center gap-3 group"
          >
            {/* Custom minimalist audio-wave inspired brand mark with dusty plum and coral */}
            <div className="flex h-8 w-8 items-center justify-center rounded bg-[#27232B] shadow-2xs">
              <svg
                className="h-4 w-4"
                viewBox="0 0 16 16"
                fill="none"
                xmlns="http://www.w3.org/2000/svg"
              >
                <rect x="2" y="6" width="1.75" height="4" rx="0.75" fill="#A6B5AD" />
                <rect x="5" y="3" width="1.75" height="10" rx="0.75" fill="#76556F" />
                <rect x="8" y="1" width="1.75" height="14" rx="0.75" fill="#D77C63" />
                <rect x="11" y="4" width="1.75" height="8" rx="0.75" fill="#D6A75D" />
              </svg>
            </div>
            <div>
              <span className="font-serif text-sm font-semibold tracking-wide text-[#27232B]">
                Sonic Fusion
              </span>
              <p className="text-[10px] font-mono tracking-wider uppercase text-[#6B626E]">
                Music Intelligence
              </p>
            </div>
          </Link>

          {/* Mobile close button */}
          {onClose && (
            <button
              onClick={onClose}
              className="p-1 text-[#6B626E] hover:text-[#27232B] md:hidden"
              aria-label="Close sidebar"
            >
              <X className="h-5 w-5" />
            </button>
          )}
        </div>

        {/* Navigation Section */}
        <div className="flex-1 overflow-y-auto px-3 py-4">
          <div className="mb-2 px-3 text-[10px] font-semibold uppercase tracking-wider text-[#6B626E]">
            Research Lab
          </div>
          <nav className="space-y-0.5">
            {navigation.map((item) => {
              const active = isLinkActive(item);
              const Icon = item.icon;
              return (
                <Link
                  key={item.name}
                  href={item.href}
                  onClick={onClose}
                  className={`group flex items-center justify-between rounded-md px-3 py-2 text-xs font-medium transition-colors ${
                    active
                      ? "bg-[#F0E8EE] text-[#76556F] font-semibold shadow-2xs"
                      : "text-[#4D4552] hover:bg-[#EBE3D7] hover:text-[#27232B]"
                  }`}
                >
                  <div className="flex items-center gap-2.5">
                    <Icon
                      className={`h-4 w-4 shrink-0 transition-colors ${
                        active
                          ? "text-[#76556F]"
                          : "text-[#6B626E] group-hover:text-[#27232B]"
                      }`}
                    />
                    <span>{item.name}</span>
                  </div>
                  {active && (
                    <span className="h-1.5 w-1.5 rounded-full bg-[#D77C63]" />
                  )}
                </Link>
              );
            })}
          </nav>
        </div>

        {/* System & Model Provenance Footer */}
        <div className="border-t border-[#E3DDD4] bg-[#EBE3D7]/60 p-3.5">
          <div className="rounded border border-[#E3DDD4] bg-white/90 p-2.5 shadow-2xs">
            <div className="flex items-center justify-between text-[10px]">
              <span className="font-semibold text-[#27232B]">FMA Small Corpus</span>
              <span className="font-mono text-[#76556F] bg-[#F0E8EE] px-1.5 py-0.2 rounded font-medium">
                8,000 Tracks
              </span>
            </div>
            <p className="mt-1 text-[10px] text-[#6B626E] leading-tight">
              3-Block 2D GenreCNN &bull; RF Baseline &bull; CPU Inference
            </p>
          </div>
          <div className="mt-2 flex items-center justify-between text-[9px] text-[#8E8692]">
            <span className="font-mono">v1.2.0 &bull; Laboratory</span>
            <span>Zero Leakage</span>
          </div>
        </div>
      </aside>
    </>
  );
}
