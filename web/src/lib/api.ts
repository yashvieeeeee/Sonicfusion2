import {
  AnalysisResponse,
  HealthResponse,
  MetricsResponse,
  ModelsResponse,
} from "@/types/api";

/**
 * Resolves the base URL for API requests based on execution context:
 * - Server-side (SSR / Server Functions): reads the internal Vercel service binding BACKEND_URL.
 * - Client-side (browser on Vercel): uses relative URL ("") since /api/(.*) rewrites to the app service.
 * - Localhost dev fallback: points to http://localhost:8000 when running standalone without Vercel rewrites.
 */
function getApiBaseUrl(): string {
  if (process.env.NEXT_PUBLIC_API_BASE_URL) {
    return process.env.NEXT_PUBLIC_API_BASE_URL;
  }
  if (typeof window === "undefined") {
    return process.env.BACKEND_URL || "http://localhost:8000";
  }
  if (
    typeof window !== "undefined" &&
    (window.location.hostname === "localhost" || window.location.hostname === "127.0.0.1") &&
    window.location.port === "3000"
  ) {
    return "http://localhost:8000";
  }
  return "";
}

export async function checkHealth(): Promise<HealthResponse> {
  const res = await fetch(`${getApiBaseUrl()}/api/health`, {
    cache: "no-store",
  });
  if (!res.ok) {
    throw new Error(`Health check failed with status: ${res.status}`);
  }
  return res.json();
}

export async function getModels(): Promise<ModelsResponse> {
  const res = await fetch(`${getApiBaseUrl()}/api/models`, {
    cache: "no-store",
  });
  if (!res.ok) {
    throw new Error(`Failed to fetch models: ${res.status}`);
  }
  return res.json();
}

export async function getMetrics(): Promise<MetricsResponse> {
  const res = await fetch(`${getApiBaseUrl()}/api/metrics`, {
    cache: "no-store",
  });
  if (!res.ok) {
    throw new Error(`Failed to fetch metrics: ${res.status}`);
  }
  return res.json();
}

export async function analyzeAudio(file: File): Promise<AnalysisResponse> {
  const formData = new FormData();
  formData.append("file", file);

  const res = await fetch(`${getApiBaseUrl()}/api/analyze`, {
    method: "POST",
    body: formData,
  });

  if (!res.ok) {
    let errorDetail = `Analysis failed (${res.status})`;
    try {
      const errorJson = await res.json();
      if (errorJson.detail) {
        errorDetail = errorJson.detail;
      }
    } catch {
      // ignore
    }
    throw new Error(errorDetail);
  }

  return res.json();
}

export async function fetchSampleAudio(samplePath: string): Promise<File> {
  const res = await fetch(samplePath);
  if (!res.ok) {
    throw new Error(`Failed to load sample audio from ${samplePath}`);
  }
  const blob = await res.blob();
  const filename = samplePath.split("/").pop() || "sample.mp3";
  return new File([blob], filename, { type: "audio/mpeg" });
}

export function getFigureUrl(filename: string): string {
  return `${getApiBaseUrl()}/api/figures/${filename}`;
}

