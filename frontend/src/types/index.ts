
// Profile Type (from Supabase)
export interface Profile {
  id: string;
  username: string | null;
  avatar_url: string | null;
  created_at: string;
}

// templates
export interface TemplateItem {
  measure?: number;
  pos?: number;
}

export interface TemplateMaskedBars {
  kind: "masked";
  measure_start: number;
  mask_measure_length: number;
  slots_per_measure: number;
}

export interface InstrumentTemplateItem extends TemplateItem {
  kind: "instrument";
  instrument?: string;
  voice_idx: number;
}

export interface ClefTemplateItem extends TemplateItem {
  kind: "clef";
  name?: string;
  voice_idx: number;
}

export interface KeyTemplateItem extends TemplateItem {
  kind: "key";
  key_tonic?: string;
  key_mode?: string;
}

export interface TimeTemplateItem extends TemplateItem {
  kind: "time";
  numerator?: string;
  denominator?: string;
}

export interface TempoTemplateItem extends TemplateItem {
  kind: "tempo";
  tempo?: number;
}

export interface NoteTemplateItem extends TemplateItem {
  kind: "note";
  duration?: number;
  pitch?: number;
  voice_idx?: number;
}

export interface ChordTemplateItem extends TemplateItem {
  kind: "chord";
  root?: string; // an absolute note name (G, Bb) or a roman numeral in the key in effect (ii, bVII)
  quality?: "maj7" | "maj" | "min" | "dom7" | "min7" | "dim7" | "dim" | "m7b5" | "sus2" | "7sus" | "7aug" | "aug";
}

// Discriminated union of every template item kind — mirrors the anyOf branches
// of the `template` field in claudeTemplates.ts's create_template_from_scratch tool.
export type AnyTemplateItem =
  | TemplateMaskedBars
  | InstrumentTemplateItem
  | ClefTemplateItem
  | KeyTemplateItem
  | TimeTemplateItem
  | TempoTemplateItem
  | NoteTemplateItem
  | ChordTemplateItem;

// tabs

/**
 * One editor tab. A tab's kind is read off `content`: a string is engraved MusicXML,
 * an array is a template (empty while the template has no items yet), and null is an
 * untouched score tab showing the empty state.
 */
export interface Tab {
  name: string;
  description: string;
  content: string | null;
  template: AnyTemplateItem[] | null;
}

export const tabXml = (t?: Tab | null) => (typeof t?.content === "string" ? t.content : null);
export const tabTemplate = (t?: Tab | null) => (Array.isArray(t?.template) ? t.template : null);

// chats

export interface ThoughtProcessEvent {
  title: string;
  input: any;
  tokensIn: number;
  tokensOut: number;
  loading?: boolean;
  isRecap?: boolean;
  /** The roadmap step. Its `tokensIn` is the whole context — intro, chat history, memory. */
  isRoadmap?: boolean;
  /** How long this step's thinking requests took; only set when /reasoning is on. */
  reasonedSeconds?: number;
}

export interface ThoughtProcessTask {
  task: string;
  isPrompt?: boolean;
}

/** One prompt and the recap the model closed it with — the reduction fed back as history. */
export interface ChatItem {
  prompt: string;
  output: string;
}

/**
 * One conversation in the chat panel: `thoughts` is what the panel renders (newest first),
 * `history` the reduced prompt/recap pairs that go back to the model as context.
 */
export interface ChatSession {
  name: string;
  thoughts: (ThoughtProcessEvent | ThoughtProcessTask)[];
  history: ChatItem[];
}