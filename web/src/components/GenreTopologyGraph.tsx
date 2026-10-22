"use client";

import React, { useState } from "react";
import {
  Compass,
  GitCompare,
  Info,
  Layers,
  Music,
  Radio,
  Sliders,
  Sparkles,
} from "lucide-react";

export interface GenreNode {
  genre: string;
  x: number; // percentage in SVG viewBox (15-85)
  y: number; // percentage in SVG viewBox (15-85)
  pca_coords: [number, number];
  color: string;
  bgLight: string;
  borderColor: string;
  cluster: string;
  stats: {
    spectral_centroid: number;
    rms: number;
    tempo: number;
  };
  description: string;
  top_similar: { target: string; similarity: number }[];
  all_similar: { target: string; similarity: number }[];
}

export interface GenreEdge {
  source: string;
  target: string;
  weight: number;
  strength: "high" | "moderate" | "subtle";
}

const GENRE_NODES: GenreNode[] = [
  {
    genre: "Electronic",
    x: 72.8,
    y: 53.0,
    pca_coords: [6.023, 0.114],
    color: "#76556F",
    bgLight: "#F0E8EE",
    borderColor: "#583C52",
    cluster: "Synthetic & High-Energy",
    stats: {
      spectral_centroid: 2158.78,
      rms: 0.2191,
      tempo: 118.91,
    },
    description: "Characterized by synthesized timbres, compressed dynamic range, and strong mid-to-high spectral centroid energy.",
    top_similar: [
      { target: "Hip-Hop", similarity: 0.834 },
      { target: "Pop", similarity: 0.03 },
      { target: "Rock", similarity: -0.019 },
    ],
    all_similar: [
      { target: "Hip-Hop", similarity: 0.834 },
      { target: "Pop", similarity: 0.03 },
      { target: "Rock", similarity: -0.019 },
      { target: "Experimental", similarity: -0.37 },
      { target: "International", similarity: -0.36 },
      { target: "Instrumental", similarity: -0.681 },
      { target: "Folk", similarity: -0.787 },
    ],
  },
  {
    genre: "Hip-Hop",
    x: 85.0,
    y: 85.0,
    pca_coords: [8.821, 3.413],
    color: "#D77C63",
    bgLight: "#FDF1EE",
    borderColor: "#C46D54",
    cluster: "Synthetic & High-Energy",
    stats: {
      spectral_centroid: 2276.64,
      rms: 0.2272,
      tempo: 113.74,
    },
    description: "Features the highest overall RMS loudness and spectral centroid among all genres, driven by heavy sub-bass and crisp percussion transients.",
    top_similar: [
      { target: "Electronic", similarity: 0.834 },
      { target: "Pop", similarity: 0.139 },
      { target: "International", similarity: -0.107 },
    ],
    all_similar: [
      { target: "Electronic", similarity: 0.834 },
      { target: "Pop", similarity: 0.139 },
      { target: "International", similarity: -0.107 },
      { target: "Rock", similarity: -0.323 },
      { target: "Experimental", similarity: -0.553 },
      { target: "Folk", similarity: -0.62 },
      { target: "Instrumental", similarity: -0.672 },
    ],
  },
  {
    genre: "Folk",
    x: 16.5,
    y: 82.1,
    pca_coords: [-6.25, 2.918],
    color: "#D6A75D",
    bgLight: "#FBF6EC",
    borderColor: "#B88A3E",
    cluster: "Organic & Acoustic",
    stats: {
      spectral_centroid: 1495.27,
      rms: 0.1511,
      tempo: 119.96,
    },
    description: "Acoustic-driven instrumentation with lower spectral centroid, dynamic headroom, and pronounced harmonic resonance in lower octaves.",
    top_similar: [
      { target: "Instrumental", similarity: 0.721 },
      { target: "International", similarity: 0.315 },
      { target: "Pop", similarity: -0.044 },
    ],
    all_similar: [
      { target: "Instrumental", similarity: 0.721 },
      { target: "International", similarity: 0.315 },
      { target: "Pop", similarity: -0.044 },
      { target: "Experimental", similarity: -0.101 },
      { target: "Rock", similarity: -0.369 },
      { target: "Hip-Hop", similarity: -0.62 },
      { target: "Electronic", similarity: -0.787 },
    ],
  },
  {
    genre: "Instrumental",
    x: 15.0,
    y: 74.9,
    pca_coords: [-6.578, 1.707],
    color: "#A6B5AD",
    bgLight: "#EEF3F0",
    borderColor: "#6C7F75",
    cluster: "Organic & Acoustic",
    stats: {
      spectral_centroid: 1250.43,
      rms: 0.1312,
      tempo: 113.81,
    },
    description: "Lowest mean spectral centroid (1,250 Hz) and lowest RMS loudness, reflecting classical, ambient, and acoustic chamber arrangements without vocals.",
    top_similar: [
      { target: "Folk", similarity: 0.721 },
      { target: "Experimental", similarity: 0.116 },
      { target: "International", similarity: 0.003 },
    ],
    all_similar: [
      { target: "Folk", similarity: 0.721 },
      { target: "Experimental", similarity: 0.116 },
      { target: "International", similarity: 0.003 },
      { target: "Rock", similarity: -0.253 },
      { target: "Pop", similarity: -0.416 },
      { target: "Hip-Hop", similarity: -0.672 },
      { target: "Electronic", similarity: -0.681 },
    ],
  },
  {
    genre: "Experimental",
    x: 39.2,
    y: 43.1,
    pca_coords: [-1.264, -2.372],
    color: "#8E6E87",
    bgLight: "#F5EFF4",
    borderColor: "#704F69",
    cluster: "Abrasive & Unconventional",
    stats: {
      spectral_centroid: 1840.58,
      rms: 0.1547,
      tempo: 119.82,
    },
    description: "High variance in spectral flux and non-harmonic noise bands, sharing substantial acoustic timbral overlap with distorted Rock.",
    top_similar: [
      { target: "Rock", similarity: 0.523 },
      { target: "Instrumental", similarity: 0.116 },
      { target: "International", similarity: -0.059 },
    ],
    all_similar: [
      { target: "Rock", similarity: 0.523 },
      { target: "Instrumental", similarity: 0.116 },
      { target: "International", similarity: -0.059 },
      { target: "Folk", similarity: -0.101 },
      { target: "Pop", similarity: -0.262 },
      { target: "Electronic", similarity: -0.37 },
      { target: "Hip-Hop", similarity: -0.553 },
    ],
  },
  {
    genre: "Rock",
    x: 45.6,
    y: 15.0,
    pca_coords: [0.16, -8.36],
    color: "#583C52",
    bgLight: "#ECE3EA",
    borderColor: "#3E2639",
    cluster: "Abrasive & Unconventional",
    stats: {
      spectral_centroid: 2137.82,
      rms: 0.2125,
      tempo: 121.76,
    },
    description: "Strong spectral energy driven by distorted electric guitars, mid-range drums, and steady 120+ BPM rhythmic drive.",
    top_similar: [
      { target: "Experimental", similarity: 0.523 },
      { target: "Pop", similarity: 0.063 },
      { target: "Electronic", similarity: -0.019 },
    ],
    all_similar: [
      { target: "Experimental", similarity: 0.523 },
      { target: "Pop", similarity: 0.063 },
      { target: "Electronic", similarity: -0.019 },
      { target: "Instrumental", similarity: -0.253 },
      { target: "Hip-Hop", similarity: -0.323 },
      { target: "Folk", similarity: -0.369 },
      { target: "International", similarity: -0.503 },
    ],
  },
  {
    genre: "Pop",
    x: 46.0,
    y: 64.6,
    pca_coords: [0.238, -0.011],
    color: "#C46D54",
    bgLight: "#FBF0EC",
    borderColor: "#A34F37",
    cluster: "Harmonic Center",
    stats: {
      spectral_centroid: 1966.67,
      rms: 0.191,
      tempo: 119.34,
    },
    description: "Acts as the central acoustic bridge in the feature space, exhibiting balanced vocal presence, moderate compression, and steady 119 BPM tempo.",
    top_similar: [
      { target: "International", similarity: 0.228 },
      { target: "Hip-Hop", similarity: 0.139 },
      { target: "Rock", similarity: 0.063 },
    ],
    all_similar: [
      { target: "International", similarity: 0.228 },
      { target: "Hip-Hop", similarity: 0.139 },
      { target: "Rock", similarity: 0.063 },
      { target: "Electronic", similarity: 0.03 },
      { target: "Folk", similarity: -0.044 },
      { target: "Experimental", similarity: -0.262 },
      { target: "Instrumental", similarity: -0.416 },
    ],
  },
  {
    genre: "International",
    x: 39.7,
    y: 80.1,
    pca_coords: [-1.15, 2.592],
    color: "#8CA095",
    bgLight: "#EBF1EE",
    borderColor: "#576B60",
    cluster: "Global & Poly-Rhythmic",
    stats: {
      spectral_centroid: 2072.84,
      rms: 0.155,
      tempo: 125.25,
    },
    description: "Features the highest average tempo (125.25 BPM) and diverse polyrhythms, linking acoustic folk traditions with modern upbeat pop structures.",
    top_similar: [
      { target: "Folk", similarity: 0.315 },
      { target: "Pop", similarity: 0.228 },
      { target: "Instrumental", similarity: 0.003 },
    ],
    all_similar: [
      { target: "Folk", similarity: 0.315 },
      { target: "Pop", similarity: 0.228 },
      { target: "Instrumental", similarity: 0.003 },
      { target: "Experimental", similarity: -0.059 },
      { target: "Hip-Hop", similarity: -0.107 },
      { target: "Electronic", similarity: -0.36 },
      { target: "Rock", similarity: -0.503 },
    ],
  },
];

const GENRE_EDGES: GenreEdge[] = [
  { source: "Electronic", target: "Hip-Hop", weight: 0.834, strength: "high" },
  { source: "Folk", target: "Instrumental", weight: 0.721, strength: "high" },
  { source: "Experimental", target: "Rock", weight: 0.523, strength: "high" },
  { source: "Folk", target: "International", weight: 0.315, strength: "moderate" },
  { source: "International", target: "Pop", weight: 0.228, strength: "subtle" },
  { source: "Hip-Hop", target: "Pop", weight: 0.139, strength: "subtle" },
  { source: "Experimental", target: "Instrumental", weight: 0.116, strength: "subtle" },
];

export default function GenreTopologyGraph() {
  const [selectedGenre, setSelectedGenre] = useState<string>("Electronic");
  const [hoveredGenre, setHoveredGenre] = useState<string | null>(null);

  const activeGenreName = hoveredGenre || selectedGenre;
  const activeNode = GENRE_NODES.find((n) => n.genre === activeGenreName) || GENRE_NODES[0];

  const nodeMap = new Map(GENRE_NODES.map((n) => [n.genre, n]));

  // Check if an edge is related to the currently active genre
  const isEdgeHighlighted = (edge: GenreEdge) => {
    return edge.source === activeGenreName || edge.target === activeGenreName;
  };

  return (
    <div className="editorial-panel p-6 sm:p-8 space-y-6">
      {/* Section Header */}
      <div className="flex flex-col sm:flex-row sm:items-end justify-between gap-4 pb-5 border-b border-[#E3DDD4]">
        <div>
          <div className="flex items-center gap-2 mb-1.5">
            <span className="text-[10px] font-mono uppercase tracking-wider font-semibold text-[#76556F] bg-[#F0E8EE] px-2 py-0.5 rounded">
              High-Dimensional Manifold Projection
            </span>
            <span className="text-[10px] font-mono text-[#6B626E]">
              PCA 2D Projection (83.7% Explained Variance)
            </span>
          </div>
          <h2 className="text-2xl sm:text-3xl font-serif font-bold text-[#27232B] tracking-tight">
            Genre Topology
          </h2>
          <p className="text-sm text-[#6B626E] mt-1 max-w-2xl leading-relaxed">
            Explore relationships, similarities, and acoustic patterns across music genres.
          </p>
        </div>

        {/* Legend Pills */}
        <div className="flex items-center gap-2 text-[11px] font-medium text-[#6B626E] flex-wrap">
          <span className="inline-flex items-center gap-1.5">
            <span className="h-2 w-4 rounded-full bg-[#76556F]" />
            <span>Strong Sim (&gt; 0.50)</span>
          </span>
          <span className="text-[#E3DDD4]">&bull;</span>
          <span className="inline-flex items-center gap-1.5">
            <span className="h-1.5 w-3 rounded-full bg-[#D77C63]" />
            <span>Moderate (0.25 - 0.50)</span>
          </span>
          <span className="text-[#E3DDD4]">&bull;</span>
          <span className="inline-flex items-center gap-1.5">
            <span className="h-1 w-3 rounded-full bg-[#A6B5AD]" />
            <span>Subtle (0.10 - 0.25)</span>
          </span>
        </div>
      </div>

      {/* Main Interactive Topology Area */}
      <div className="grid grid-cols-1 lg:grid-cols-12 gap-6 items-start">
        {/* Left: Interactive 2D Graph Canvas (7 cols) */}
        <div className="lg:col-span-7 relative rounded-2xl bg-[#FAF7F2] border border-[#E3DDD4] p-4 sm:p-6 overflow-hidden">
          {/* Subtle Grid Lines & Axes in Background */}
          <div className="absolute inset-0 pointer-events-none opacity-40">
            <div className="absolute top-1/2 left-0 right-0 h-px border-b border-dashed border-[#E3DDD4]" />
            <div className="absolute left-1/2 top-0 bottom-0 w-px border-r border-dashed border-[#E3DDD4]" />
          </div>

          <div className="relative aspect-4/3 w-full">
            <svg
              viewBox="0 0 100 100"
              className="w-full h-full overflow-visible select-none"
            >
              {/* Edges */}
              {GENRE_EDGES.map((edge, idx) => {
                const src = nodeMap.get(edge.source);
                const tgt = nodeMap.get(edge.target);
                if (!src || !tgt) return null;

                const highlighted = isEdgeHighlighted(edge);

                const strokeWidth =
                  edge.strength === "high"
                    ? highlighted ? 2.5 : 1.8
                    : edge.strength === "moderate"
                    ? highlighted ? 1.8 : 1.2
                    : highlighted ? 1.4 : 0.8;

                const strokeColor = highlighted
                  ? "#D77C63"
                  : edge.strength === "high"
                  ? "rgba(118, 85, 111, 0.45)"
                  : edge.strength === "moderate"
                  ? "rgba(214, 167, 93, 0.4)"
                  : "rgba(166, 181, 173, 0.35)";

                return (
                  <g key={idx}>
                    <line
                      x1={src.x}
                      y1={src.y}
                      x2={tgt.x}
                      y2={tgt.y}
                      stroke={strokeColor}
                      strokeWidth={strokeWidth}
                      strokeDasharray={edge.strength === "subtle" ? "1.5,1.5" : undefined}
                      className="transition-all duration-300"
                    />
                    {/* Weight badge on highlighted edge */}
                    {highlighted && (
                      <text
                        x={(src.x + tgt.x) / 2}
                        y={(src.y + tgt.y) / 2 - 1.5}
                        fontSize="2.4"
                        fontFamily="monospace"
                        fill="#27232B"
                        textAnchor="middle"
                        className="bg-white px-1 font-bold pointer-events-none"
                      >
                        {edge.weight > 0 ? `+${edge.weight.toFixed(2)}` : edge.weight.toFixed(2)}
                      </text>
                    )}
                  </g>
                );
              })}

              {/* Nodes */}
              {GENRE_NODES.map((node) => {
                const isSelected = selectedGenre === node.genre;
                const isHovered = hoveredGenre === node.genre;
                const isActive = isSelected || isHovered;

                const nodeRadius = isActive ? 5.2 : 4.4;

                return (
                  <g
                    key={node.genre}
                    className="cursor-pointer transition-transform duration-200"
                    onClick={() => setSelectedGenre(node.genre)}
                    onMouseEnter={() => setHoveredGenre(node.genre)}
                    onMouseLeave={() => setHoveredGenre(null)}
                  >
                    {/* Concentric Halo on Selected / Hovered Node */}
                    {isActive && (
                      <circle
                        cx={node.x}
                        cy={node.y}
                        r={nodeRadius + 2.4}
                        fill="none"
                        stroke={node.color}
                        strokeWidth="0.8"
                        strokeOpacity="0.5"
                        className="animate-pulse"
                      />
                    )}

                    {/* Node Background */}
                    <circle
                      cx={node.x}
                      cy={node.y}
                      r={nodeRadius}
                      fill={isActive ? node.color : "#FFFFFF"}
                      stroke={node.color}
                      strokeWidth={isActive ? "1.6" : "1.2"}
                      filter="drop-shadow(0px 1px 2px rgba(39,35,43,0.12))"
                      className="transition-all duration-200"
                    />

                    {/* Node Icon / Initial */}
                    <text
                      x={node.x}
                      y={node.y + 1.2}
                      textAnchor="middle"
                      fontSize="3.2"
                      fontWeight="bold"
                      fill={isActive ? "#FFFFFF" : node.color}
                      className="pointer-events-none font-sans"
                    >
                      {node.genre.slice(0, 1)}
                    </text>

                    {/* Label Tag below node */}
                    <text
                      x={node.x}
                      y={node.y + nodeRadius + 3.4}
                      textAnchor="middle"
                      fontSize="2.8"
                      fontWeight={isActive ? "bold" : "600"}
                      fill={isActive ? "#27232B" : "#6B626E"}
                      className="pointer-events-none font-sans tracking-tight"
                    >
                      {node.genre}
                    </text>
                  </g>
                );
              })}
            </svg>
          </div>

          {/* Graph Footer Caption */}
          <div className="pt-3 border-t border-[#E3DDD4] flex items-center justify-between text-[11px] text-[#8E8692]">
            <span>PC1 (X): Timbral Energy vs. Acoustic Organicism</span>
            <span>PC2 (Y): Harmonic Resonance &amp; Tempo</span>
          </div>
        </div>

        {/* Right: Selected Genre Inspector Card (5 cols) */}
        <div className="lg:col-span-5 space-y-4">
          <div className="p-5 sm:p-6 rounded-2xl bg-white border border-[#E3DDD4] shadow-sm space-y-4">
            {/* Inspector Title */}
            <div className="flex items-center justify-between pb-3 border-b border-[#E3DDD4]">
              <div className="flex items-center gap-2">
                <div
                  className="h-3.5 w-3.5 rounded-full"
                  style={{ backgroundColor: activeNode.color }}
                />
                <h3 className="text-lg font-serif font-bold text-[#27232B]">
                  {activeNode.genre}
                </h3>
              </div>
              <span className="text-[10px] font-mono px-2 py-0.5 rounded-full bg-[#FAF7F2] border border-[#E3DDD4] text-[#6B626E] font-semibold">
                {activeNode.cluster}
              </span>
            </div>

            {/* Description */}
            <p className="text-xs text-[#6B626E] leading-relaxed">
              {activeNode.description}
            </p>

            {/* Extracted Acoustic Parameter Statistics */}
            <div className="space-y-2 pt-1">
              <span className="text-[10px] font-mono uppercase tracking-wider text-[#8E8692] font-semibold block">
                Acoustic Centroid Profile (Audited Dataset):
              </span>

              <div className="grid grid-cols-3 gap-2 text-center text-xs">
                <div className="p-2.5 rounded-xl bg-[#FAF7F2] border border-[#E3DDD4]/80">
                  <span className="text-[10px] text-[#8E8692] block">Centroid</span>
                  <span className="font-mono text-xs font-bold text-[#27232B]">
                    {activeNode.stats.spectral_centroid.toFixed(0)} Hz
                  </span>
                </div>
                <div className="p-2.5 rounded-xl bg-[#FAF7F2] border border-[#E3DDD4]/80">
                  <span className="text-[10px] text-[#8E8692] block">RMS Loudness</span>
                  <span className="font-mono text-xs font-bold text-[#76556F]">
                    {activeNode.stats.rms.toFixed(3)}
                  </span>
                </div>
                <div className="p-2.5 rounded-xl bg-[#FAF7F2] border border-[#E3DDD4]/80">
                  <span className="text-[10px] text-[#8E8692] block">Mean Tempo</span>
                  <span className="font-mono text-xs font-bold text-[#D77C63]">
                    {activeNode.stats.tempo.toFixed(1)} BPM
                  </span>
                </div>
              </div>
            </div>

            {/* Top Similar Genres Ranking */}
            <div className="pt-2 space-y-2">
              <span className="text-[10px] font-mono uppercase tracking-wider text-[#8E8692] font-semibold block">
                Acoustic Similarity over 52 Features:
              </span>

              <div className="space-y-2">
                {activeNode.top_similar.map((sim, i) => {
                  const targetNode = nodeMap.get(sim.target);
                  const isPositive = sim.similarity > 0;
                  const pct = Math.round(Math.abs(sim.similarity) * 100);

                  return (
                    <div
                      key={sim.target}
                      onClick={() => setSelectedGenre(sim.target)}
                      className="p-2.5 rounded-xl border border-[#E3DDD4] bg-[#FAF7F2] hover:bg-white hover:border-[#76556F] transition-all cursor-pointer group"
                    >
                      <div className="flex items-center justify-between text-xs mb-1">
                        <div className="flex items-center gap-2">
                          <span className="text-[10px] font-mono font-bold text-[#8E8692]">
                            #{i + 1}
                          </span>
                          <span className="font-semibold text-[#27232B] group-hover:text-[#76556F] transition-colors">
                            {sim.target}
                          </span>
                        </div>
                        <span className="font-mono text-[11px] font-bold text-[#27232B]">
                          {isPositive ? `+${sim.similarity.toFixed(3)}` : sim.similarity.toFixed(3)}
                        </span>
                      </div>

                      {/* Bar indicator */}
                      <div className="h-1.5 w-full bg-white rounded-full border border-[#E3DDD4] overflow-hidden">
                        <div
                          className="h-full rounded-full transition-all duration-300"
                          style={{
                            width: `${Math.min(100, Math.max(8, pct))}%`,
                            backgroundColor: isPositive
                              ? targetNode?.color || "#76556F"
                              : "#A6B5AD",
                          }}
                        />
                      </div>
                    </div>
                  );
                })}
              </div>
            </div>

            {/* PCA Coordinates Chip */}
            <div className="pt-3 border-t border-[#E3DDD4] flex items-center justify-between text-[11px] font-mono text-[#8E8692]">
              <span>Projection Vector:</span>
              <span>
                PC1: {activeNode.pca_coords[0]} &bull; PC2: {activeNode.pca_coords[1]}
              </span>
            </div>
          </div>
        </div>
      </div>

      {/* Scientific Methodology Note */}
      <div className="p-4 rounded-xl bg-[#FAF7F2] border border-[#E3DDD4] flex items-start gap-3 text-xs text-[#6B626E] leading-relaxed">
        <Info className="h-4 w-4 text-[#76556F] shrink-0 mt-0.5" />
        <p>
          <strong className="text-[#27232B]">Methodology &amp; Mathematical Provenance:</strong> Node coordinates are calculated using Principal Component Analysis (PCA: 83.7% aggregate variance explained across PC1 and PC2) over standardized genre feature centroids extracted from all 7,994 decodable tracks in the Free Music Archive (FMA Small). Edges represent pairwise cosine similarities over the 52 handcrafted audio features (timbral MFCCs, spectral centroid, spectral contrast, spectral rolloff, chroma, and tempo onset dynamics).
        </p>
      </div>
    </div>
  );
}
