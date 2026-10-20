import { AnalysisResponse } from "@/types/api";

const STORAGE_KEY = "sonicscope_last_analysis";
const AUDIO_NAME_KEY = "sonicscope_audio_name";

export function saveAnalysisResult(result: AnalysisResponse): void {
  if (typeof window !== "undefined") {
    try {
      sessionStorage.setItem(STORAGE_KEY, JSON.stringify(result));
      sessionStorage.setItem(AUDIO_NAME_KEY, result.filename);
    } catch (e) {
      console.warn("Could not save analysis to sessionStorage", e);
    }
  }
}

export function getStoredAnalysisResult(): AnalysisResponse | null {
  if (typeof window !== "undefined") {
    try {
      const item = sessionStorage.getItem(STORAGE_KEY);
      if (item) {
        const parsed = JSON.parse(item);
        if (parsed && Array.isArray(parsed.similar_tracks)) {
          parsed.similar_tracks = parsed.similar_tracks.map((t: any) => ({
            track_id: t.track_id,
            title: t.title,
            artist: t.artist,
            genre: t.genre,
            feature_similarity_score:
              t.feature_similarity_score ??
              t.similarity_score ??
              (typeof t.similarity_percentage === "number" ? t.similarity_percentage / 100 : 0.0),
            explanation: t.explanation,
          }));
        }
        return parsed as AnalysisResponse;
      }
    } catch (e) {
      console.warn("Could not load analysis from sessionStorage", e);
    }
  }
  return null;
}

export function clearStoredAnalysisResult(): void {
  if (typeof window !== "undefined") {
    sessionStorage.removeItem(STORAGE_KEY);
    sessionStorage.removeItem(AUDIO_NAME_KEY);
  }
}
