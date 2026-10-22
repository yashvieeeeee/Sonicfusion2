"use client";

import React, { useState } from "react";
import { usePathname } from "next/navigation";
import Sidebar from "@/components/Sidebar";
import Navbar from "@/components/Navbar";

export default function AppShell({ children }: { children: React.ReactNode }) {
  const [sidebarOpen, setSidebarOpen] = useState(false);
  const pathname = usePathname();
  const isLanding = pathname === "/";

  return (
    <div className="flex min-h-screen bg-[#F5F0E8]">
      {/* Sidebar: Mobile drawer everywhere; on desktop active for Studio & Analytics routes */}
      {(!isLanding || sidebarOpen) && (
        <Sidebar isOpen={sidebarOpen} onClose={() => setSidebarOpen(false)} />
      )}

      {/* Main Content Area */}
      <div className="flex flex-1 flex-col min-w-0">
        <Navbar onToggleSidebar={() => setSidebarOpen(true)} />
        <main className="flex-1">{children}</main>
        
        {/* Workspace Footer on Studio / Analytics Routes */}
        {!isLanding && (
          <footer className="border-t border-[#E3DDD4] bg-[#FAF7F2]/80 py-4 text-center text-xs text-[#6B626E]">
            <div className="max-w-7xl mx-auto px-6 flex flex-col sm:flex-row items-center justify-between gap-2">
              <span className="font-mono text-[11px]">
                Sonic Fusion &bull; Academic Music Information Retrieval &amp; Spectrogram Deep Learning
              </span>
              <span className="text-[11px] font-mono">
                Trained on FMA Small &bull; Librosa 0.10 &bull; PyTorch CPU
              </span>
            </div>
          </footer>
        )}
      </div>
    </div>
  );
}
