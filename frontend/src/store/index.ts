import { create } from 'zustand'

export interface MeasureSelection {
  start: number
  end: number
  /** 0-indexed part/staff indices to target; null means all staves. */
  parts: number[] | null
}

export interface ScoreAppData {
  numInstruments: number
}

interface AppState {
  currentXml: string | null
  scoreData: ScoreAppData | null
  loading: boolean
  error: string | null
  zoom: number
  selection: MeasureSelection | null
  streamBar: number | null
  totalBars: number | null
  streamMode: 'infill' | 'generate' | 'append' | null

  setXml: (xml: string) => void
  setScoreData: (data: ScoreAppData | null) => void
  setLoading: (v: boolean) => void
  setError: (msg: string | null) => void
  setZoom: (v: number) => void
  setSelection: (s: MeasureSelection | null) => void
  setStreamProgress: (bar: number | null, total: number | null) => void
  setStreamMode: (mode: 'infill' | 'generate' | 'append' | null) => void
}

export const useStore = create<AppState>((set) => ({
  currentXml: null,
  scoreData: null,
  loading: false,
  error: null,
  zoom: 100,
  selection: null,
  streamBar: null,
  totalBars: null,
  streamMode: null,

  setXml: (xml) => set({ currentXml: xml, error: null }),
  setScoreData: (data) => set({ scoreData: data }),
  setLoading: (v) => set({ loading: v }),
  setError: (msg) => set({ error: msg, loading: false }),
  setZoom: (v) => set({ zoom: Math.max(25, Math.min(300, v)) }),
  setSelection: (s) => set({ selection: s }),
  setStreamProgress: (bar, total) => set({ streamBar: bar, totalBars: total }),
  setStreamMode: (mode) => set({ streamMode: mode }),
}))
