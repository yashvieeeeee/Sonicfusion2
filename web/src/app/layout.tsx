import type { Metadata } from "next";
import AppShell from "@/components/AppShell";
import "./globals.css";

export const metadata: Metadata = {
  title: "Sonic Fusion | Music Intelligence & Data Analytics",
  description:
    "Acoustic feature extraction and log-mel spectrogram deep learning for music genre recognition on the Free Music Archive.",
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en" className="h-full antialiased">
      <body className="min-h-full flex flex-col bg-[#F5F0E8] text-[#27232B] selection:bg-[#F0E8EE] selection:text-[#76556F]">
        <AppShell>{children}</AppShell>
      </body>
    </html>
  );
}
