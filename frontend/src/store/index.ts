import { create } from 'zustand'
import { createJSONStorage, persist, type StateStorage } from 'zustand/middleware'
import { tabXml, type Tab } from '../types'

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
  /** Every open tab, in bar order. A tab owns its own content — there is no parallel xml map. */
  tabs: Tab[]
  activeTab: string
  getTab: (name?: string) => Tab | undefined
  getXml: (key: string) => string | null
  /** Row id of the saved score being edited; null until the first save. */
  scoreId: string | null
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
  /** Bumped to force ScoreViewer to re-render the SVG from the current XML. */
  redrawNonce: number

  setTabs: (tabs: Tab[]) => void
  setActiveTab: (tab: string) => void
  /** Adds a tab, de-duplicating its name; returns the name it actually got. */
  addNewTab: (tab: string, init?: Partial<Tab>) => string
  deleteTab: (tab: string) => void
  /** Patches one tab's fields (description, content) in place. */
  updateTab: (tab: string, patch: Partial<Tab>) => void
  /** Renames a tab, de-duplicating; returns the name it actually got. */
  renameTab: (tab: string, newName: string) => string

  setXml: (xml: string, key?: string) => void
  setScoreId: (id: string | null) => void
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
  /** Force a full re-render of the score SVG. */
  redrawScore: () => void
  /** Drop the open document entirely (starting a new score, leaving a saved one). */
  resetScore: () => void
}

const ZOOM_MIN = 5
const ZOOM_MAX = 300
const clampZoom = (v: number) => Math.max(ZOOM_MIN, Math.min(ZOOM_MAX, v))

const PERSIST_KEY = 'muse-waffle:editor'
const WRITE_DELAY_MS = 400

const mainTab = (): Tab[] => [{ name: 'Main', description: '', content: null }]

/** `name`, or `name (n)` with the lowest n that no tab in `taken` already uses. */
function uniqueName(name: string, taken: string[]): string {
  const base = name.replace(/ *\([0-9]+\)$/, '')
  let candidate = name
  for (let i = 1; i < 100 && taken.includes(candidate); i++) candidate = `${base} (${i})`
  return candidate
}

/**
 * sessionStorage with write-behind. A score's MusicXML runs to hundreds of kilobytes and
 * setXml fires on every streaming update, so serialising on each one would stall generation;
 * writes are coalesced into at most one every WRITE_DELAY_MS instead. Anything still pending
 * is flushed when the page is hidden (tab switch, reload, close) — exactly the moments the
 * value has to already be on disk.
 */
function createWriteBehindStorage(): StateStorage {
  const pending = new Map<string, string>()
  let timer: ReturnType<typeof setTimeout> | null = null

  const flush = () => {
    if (timer !== null) {
      clearTimeout(timer)
      timer = null
    }
    for (const [name, value] of pending) {
      // Quota overruns (a very large score) just mean no persistence for that write.
      try {
        sessionStorage.setItem(name, value)
      } catch (err) {
        console.warn('[store] could not persist editor state', err)
      }
    }
    pending.clear()
  }

  window.addEventListener('pagehide', flush)
  document.addEventListener('visibilitychange', () => {
    if (document.visibilityState === 'hidden') flush()
  })

  return {
    getItem: (name) => {
      if (pending.has(name)) return pending.get(name)!
      try {
        return sessionStorage.getItem(name)
      } catch {
        return null
      }
    },
    setItem: (name, value) => {
      pending.set(name, value)
      if (timer === null) timer = setTimeout(flush, WRITE_DELAY_MS)
    },
    removeItem: (name) => {
      pending.delete(name)
      try {
        sessionStorage.removeItem(name)
      } catch { /* storage unavailable */ }
    },
  }
}

export const useStore = create<AppState>()(persist((set, get) => ({
  tabs: mainTab(),
  activeTab: 'Main',
  scoreId: null,
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
  redrawNonce: 0,

  getTab: (name) => get().tabs.find((t) => t.name === (name ?? get().activeTab)),
  getXml: (key) => tabXml(get().getTab(key)),
  addNewTab: (tab, init) => {
    const name = uniqueName(tab, get().tabs.map((t) => t.name))
    set((s) => ({ tabs: [...s.tabs, { name, description: '', content: null, ...init }] }))
    return name
  },
  deleteTab: (tab) => set((s) => ({ tabs: s.tabs.filter((t) => t.name !== tab) })),
  updateTab: (tab, patch) =>
    set((s) => ({ tabs: s.tabs.map((t) => (t.name === tab ? { ...t, ...patch } : t)) })),
  renameTab: (tab, newName) => {
    const name = uniqueName(newName, get().tabs.filter((t) => t.name !== tab).map((t) => t.name))
    get().updateTab(tab, { name })
    return name
  },

  setTabs: (tabs) => set({ tabs: tabs }),
  setActiveTab: (tab) => set({ activeTab: tab }),
  // Writing xml to a tab that isn't open yet creates it, so a generated score can land
  // straight in its own tab.
  setXml: (xml, key) =>
    set((s) => {
      const name = key ?? s.activeTab
      const tabs = s.tabs.some((t) => t.name === name)
        ? s.tabs.map((t) => (t.name === name ? { ...t, content: xml } : t))
        : [...s.tabs, { name, description: '', content: xml }]
      return { tabs, error: null }
    }),
  setScoreId: (id) => set({ scoreId: id }),
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
  redrawScore: () => { set((s) => ({ redrawNonce: s.redrawNonce + 1 })) },
  resetScore: () =>
    set({
      tabs: mainTab(),
      activeTab: 'Main',
      scoreId: null,
      scoreData: null,
      selection: null,
      error: null,
      streamBar: null,
      totalBars: null,
      streamMode: null,
    }),
}), {
  name: PERSIST_KEY,
  version: 2,
  storage: createJSONStorage(createWriteBehindStorage),
  // Only the document survives a reload. The camera, the selection and the streaming
  // progress are per-view state, and rehydrating a half-finished stream would leave the
  // UI stuck mid-generation with no request behind it.
  partialize: (s) => ({ tabs: s.tabs, activeTab: s.activeTab, scoreId: s.scoreId, scoreData: s.scoreData }),
  // v1 kept tab names and their xml in two parallel structures; fold them into Tab[].
  migrate: (persisted, version) => {
    if (version >= 2) return persisted as Partial<AppState>
    const old = persisted as { tabs?: string[]; currentXml?: Record<string, string | null> }
    return {
      ...(persisted as object),
      tabs: (old.tabs ?? ['Main']).map((name) => ({
        name,
        description: '',
        content: old.currentXml?.[name] ?? null,
      })),
    } as Partial<AppState>
  },
}))
