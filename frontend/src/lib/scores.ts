import { supabase } from "./supabase";
import { renderPreviewSvg } from "../hooks/useVerovio";
import { tabXml, type Tab } from "../types";

export interface ScoreRow {
  id: string;
  user_id: string;
  name: string;
  created_at: string;
  updated_at: string;
}

const BUCKET = "scores";

// Portrait page (Verovio units are 1/10 mm) matching the 300x400 dashboard card.
const PAGE_W = 2100;
const PAGE_H = 2800;
const PNG_W = 600;
const PNG_H = 800;
const PADDING = 40;

const xmlPath = (userId: string, id: string) => `${userId}/${id}.musicxml`;
const pngPath = (userId: string, id: string) => `${userId}/${id}.png`;

async function svgToPng(svg: string, width: number, height: number): Promise<Blob | null> {
  try {
    const img = new Image();
    img.src = "data:image/svg+xml;charset=utf-8," + encodeURIComponent(svg);
    await img.decode();
    const canvas = document.createElement("canvas");
    canvas.width = width;
    canvas.height = height;
    canvas.getContext("2d")!.drawImage(img, PADDING, PADDING, width - PADDING * 2, height - PADDING * 2);
    return await new Promise((resolve) => canvas.toBlob(resolve, "image/png"));
  } catch (e) {
    console.error("Error rasterising preview:", e);
    return null;
  }
}

/**
 * Saves every tab and a freshly rendered preview to the `scores` bucket, creating the row when
 * `scoreId` is null. Returns the score id, or null if the save failed.
 * Leaves the shared Verovio toolkit laid out for the preview page — redraw the score afterwards.
 */
export async function saveScore(
  userId: string,
  tabs: Tab[],
  scoreId: string | null
): Promise<string | null> {
  let id = scoreId;

  if (id) {
    const { error } = await supabase
      .from("scores")
      .update({ updated_at: new Date().toISOString() })
      .eq("id", id);
    if (error) {
      console.error("Error updating score:", error);
      return null;
    }
  } else {
    const { data, error } = await supabase
      .from("scores")
      .insert([{ user_id: userId, name: "Untitled score" }])
      .select("id")
      .single();
    if (error) {
      console.error("Error creating score:", error);
      return null;
    }
    id = data.id;
  }

  const mainXml = tabXml(tabs.find((t) => t.name === "Main"));
  const png = mainXml ? await svgToPng(renderPreviewSvg(mainXml, PAGE_W, PAGE_H, 10), PNG_W, PNG_H) : null;

  const [xmlUpload, pngUpload] = await Promise.all([
    supabase.storage
      .from(BUCKET)
      .upload(xmlPath(userId, id!), new Blob([JSON.stringify(tabs)], { type: "application/xml" }), { upsert: true }),
    png
      ? supabase.storage
          .from(BUCKET)
          .upload(pngPath(userId, id!), png, { upsert: true, contentType: "image/png" })
      : Promise.resolve({ error: null }),
  ]);

  if (xmlUpload.error) {
    console.error("Error uploading score:", xmlUpload.error);
    return null;
  }
  if (pngUpload.error) console.error("Error uploading preview:", pngUpload.error);

  return id;
}

export async function listScores(userId: string): Promise<ScoreRow[]> {
  const { data, error } = await supabase
    .from("scores")
    .select("*")
    .eq("user_id", userId)
    .order("updated_at", { ascending: false });

  if (error) {
    console.error("Error listing scores:", error);
    return [];
  }

  return data;
}

export async function getScore(scoreId: string): Promise<ScoreRow | null> {
  const { data, error } = await supabase
    .from("scores")
    .select("*")
    .eq("id", scoreId)
    .single();

  if (error) {
    console.error("Error listing scores:", error);
    return null;
  }

  return data;
}

export async function setScoreName(scoreId: string, name: string): Promise<String | null> {
  const { error } = await supabase
    .from("scores")
    .update({ name })
    .eq("id", scoreId);
  if (error) {
    console.error("Error updating score:", error);
    return null;
  }
  return 'Success'
}

export async function fetchScoreXml(userId: string, id: string): Promise<string | null> {
  const { data, error } = await supabase.storage.from(BUCKET).download(xmlPath(userId, id));

  if (error) {
    console.error("Error fetching score:", error);
    return null;
  }

  return data.text();
}

export function formatDate(date: string): string {
  const formatter = new Intl.DateTimeFormat('en-US', {
    dateStyle: 'short', // Automatically outputs M/D/YY or M/D/YYYY depending on platform
    timeStyle: 'short'  // Automatically outputs hh:mm AM/PM
  });

  return formatter.format(Date.parse(date))
}

/** Public preview URL, busted on `updatedAt` since saves overwrite the same object path. */
export function previewUrl(userId: string, id: string, updatedAt: string): string {
  const { data } = supabase.storage.from(BUCKET).getPublicUrl(pngPath(userId, id));
  return `${data.publicUrl}?t=${Date.parse(updatedAt)}`;
}
