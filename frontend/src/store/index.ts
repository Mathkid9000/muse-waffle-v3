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
  /** Camera zoom as a percentage (25–300). scale = zoom / 100. */
  zoom: number
  /** Camera translation in screen pixels (transform-origin is the viewport top-left). */
  panX: number
  panY: number
  /** Viewport (outer container) size in pixels, kept in sync for center-anchored zoom. */
  viewportW: number
  viewportH: number
  selection: MeasureSelection | null
  streamBar: number | null
  totalBars: number | null
  streamMode: 'infill' | 'generate' | 'append' | null

  setXml: (xml: string) => void
  setScoreData: (data: ScoreAppData | null) => void
  setLoading: (v: boolean) => void
  setError: (msg: string | null) => void
  setZoom: (v: number) => void
  setPan: (x: number, y: number) => void
  /** Zoom to `newZoom`% while keeping the screen point (ox, oy) fixed under the cursor. */
  zoomTo: (newZoom: number, ox: number, oy: number) => void
  setViewport: (w: number, h: number) => void
  resetCamera: () => void
  setSelection: (s: MeasureSelection | null) => void
  setStreamProgress: (bar: number | null, total: number | null) => void
  setStreamMode: (mode: 'infill' | 'generate' | 'append' | null) => void
}

const ZOOM_MIN = 25
const ZOOM_MAX = 300
const clampZoom = (v: number) => Math.max(ZOOM_MIN, Math.min(ZOOM_MAX, v))

export const useStore = create<AppState>((set) => ({
  currentXml: null,
  scoreData: null,
  loading: false,
  error: null,
  zoom: 100,
  panX: 0,
  panY: 0,
  viewportW: 0,
  viewportH: 0,
  selection: null,
  streamBar: null,
  totalBars: null,
  streamMode: null,

  setXml: (xml) => set({ currentXml: xml, error: null }),
  setScoreData: (data) => set({ scoreData: data }),
  setLoading: (v) => set({ loading: v }),
  setError: (msg) => set({ error: msg, loading: false }),
  setZoom: (v) => set({ zoom: clampZoom(v) }),
  setPan: (x, y) => set({ panX: x, panY: y }),
  zoomTo: (newZoom, ox, oy) =>
    set((s) => {
      const z = clampZoom(newZoom)
      const ratio = z / s.zoom
      // Keep the world point under (ox, oy) fixed: pan' = origin - (origin - pan) * ratio
      return {
        zoom: z,
        panX: ox - (ox - s.panX) * ratio,
        panY: oy - (oy - s.panY) * ratio,
      }
    }),
  setViewport: (w, h) => set({ viewportW: w, viewportH: h }),
  resetCamera: () => set({ zoom: 100, panX: 0, panY: 0 }),
  setSelection: (s) => set({ selection: s }),
  setStreamProgress: (bar, total) => set({ streamBar: bar, totalBars: total }),
  setStreamMode: (mode) => set({ streamMode: mode }),
}))
