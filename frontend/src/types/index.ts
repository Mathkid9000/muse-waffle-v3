
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
  degree?: string; // roman numeral
  rootNumeral?: string; // numeral of the chord degree
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