import type { Metadata } from "next";
import Navbar from "@/components/Navbar";
import "./globals.css";

export const metadata: Metadata = {
  title: "SonicScope AI | Music Intelligence & Genre Recognition",
  description:
    "Acoustic feature extraction and log-mel spectrogram deep learning for music genre recognition on FMA Small.",
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en" className="dark h-full antialiased">
      <body className="min-h-full flex flex-col bg-[#090d16] text-slate-100 selection:bg-indigo-500 selection:text-white">
        <Navbar />
        <main className="flex-1 flex flex-col">{children}</main>
        <footer className="border-t border-slate-800/80 bg-[#070a12] py-6 text-center text-xs text-slate-500">
          <div className="max-w-7xl mx-auto px-4 flex flex-col sm:flex-row items-center justify-between gap-2">
            <span>
              SonicScope AI (SonicFusion 2) &bull; Academic ML & Spectrogram Deep Learning
            </span>
            <span>
              Trained on FMA Small &bull; Librosa 0.10 &bull; PyTorch CPU
            </span>
          </div>
        </footer>
      </body>
    </html>
  );
}
