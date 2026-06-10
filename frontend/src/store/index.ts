import { create } from 'zustand'

export interface MeasureSelection {
  start: number
  end: number
}

interface AppState {
  currentXml: string | null
  loading: boolean
  error: string | null
  zoom: number
  selection: MeasureSelection | null
  streamBar: number | null
  totalBars: number | null

  setXml: (xml: string) => void
  setLoading: (v: boolean) => void
  setError: (msg: string | null) => void
  setZoom: (v: number) => void
  setSelection: (s: MeasureSelection | null) => void
  setStreamProgress: (bar: number | null, total: number | null) => void
}

export const useStore = create<AppState>((set) => ({
  currentXml: null,
  loading: false,
  error: null,
  zoom: 100,
  selection: null,
  streamBar: null,
  totalBars: null,

  setXml: (xml) => set({ currentXml: xml, error: null }),
  setLoading: (v) => set({ loading: v }),
  setError: (msg) => set({ error: msg, loading: false }),
  setZoom: (v) => set({ zoom: Math.max(25, Math.min(300, v)) }),
  setSelection: (s) => set({ selection: s }),
  setStreamProgress: (bar, total) => set({ streamBar: bar, totalBars: total }),
}))
