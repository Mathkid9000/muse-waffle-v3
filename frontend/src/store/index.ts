import { create } from 'zustand'
import { createJSONStorage, persist, type StateStorage } from 'zustand/middleware'
import { tabTemplate, tabXml, type AnyTemplateItem, type ChatSession, type Tab } from '../types'

export interface MeasureSelection {
  start: number
  end: number
  /** 0-indexed part/staff indices to target; null means all staves. */
  parts: number[] | null
}

export interface ScoreAppData {
  numInstruments: number
}

/** The document as it stood before an undoable action. Tabs are immutable, so this is just a reference. */
export interface Snapshot {
  tabs: Tab[]
  activeTab: string
  label: string
}

interface AppState {
  /** Every open tab, in bar order. A tab owns its own content — there is no parallel xml map. */
  tabs: Tab[]
  activeTab: string
  /** Every chat in the panel. Like tabs, a chat owns its own thoughts and history. */
  chats: ChatSession[]
  activeChat: string
  getTab: (name?: string) => Tab | undefined
  getChat: (name?: string) => ChatSession | undefined
  getXml: (key: string) => string | null
  getTemplate: (key: string) => AnyTemplateItem[] | null
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
  /** Show the open template as its raw JSON instead of the readable item list. */
  textView: boolean
  /** Undo/redo stacks, newest last. Session-only — never persisted. */
  past: Snapshot[]
  future: Snapshot[]

  /** Records the current document before an undoable action; clears redo. */
  checkpoint: (label: string) => void
  undo: () => void
  redo: () => void

  setTabs: (tabs: Tab[]) => void
  setActiveTab: (tab: string) => void
  /** Adds a tab, de-duplicating its name; returns the name it actually got. */
  addNewTab: (tab: string, init?: Partial<Tab>) => string
  deleteTab: (tab: string) => void
  /** Patches one tab's fields (description, content) in place. */
  updateTab: (tab: string, patch: Partial<Tab>) => void
  /** Renames a tab, de-duplicating; returns the name it actually got. */
  renameTab: (tab: string, newName: string) => string

  /** Replaces the whole chat set (loading a score) and activates its first chat. */
  setChats: (chats: ChatSession[]) => void
  setActiveChat: (chat: string) => void
  /** Adds an empty chat, de-duplicating its name; returns the name it actually got. */
  addNewChat: (chat?: string) => string
  /** Deletes a chat, never leaving the panel empty. */
  deleteChat: (chat: string) => void
  /** Patches one chat's fields (thoughts, history) in place. */
  updateChat: (chat: string, patch: Partial<ChatSession>) => void
  renameChat: (chat: string, newName: string) => string

  setXml: (xml: string | null, key?: string) => void
  setTemplate: (xml: AnyTemplateItem[], key?: string) => void
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
  toggleTextView: () => void
  /** Drop the open document entirely (starting a new score, leaving a saved one). */
  resetScore: () => void
}

const ZOOM_MIN = 5
const ZOOM_MAX = 300
const clampZoom = (v: number) => Math.max(ZOOM_MIN, Math.min(ZOOM_MAX, v))

const PERSIST_KEY = 'muse-waffle:editor'
const WRITE_DELAY_MS = 400
const HISTORY_MAX = 50

const mainTab = (): Tab[] => [{ name: 'Main', description: '', content: null, template: null }]
const mainChat = (): ChatSession[] => [{ name: 'Chat', thoughts: [], history: [] }]

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
  chats: mainChat(),
  activeChat: mainChat()[0].name,
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
  textView: false,
  past: [],
  future: [],

  // Checkpoints are taken once per logical action (a whole Claude run, a tab delete) rather
  // than on every set, so streaming setXml calls don't each become an undo step.
  checkpoint: (label) =>
    set((s) => ({ past: [...s.past, { tabs: s.tabs, activeTab: s.activeTab, label }].slice(-HISTORY_MAX), future: [] })),
  undo: () =>
    set((s) => {
      const snap = s.past.at(-1)
      if (!snap) return {}
      return {
        past: s.past.slice(0, -1),
        future: [...s.future, { tabs: s.tabs, activeTab: s.activeTab, label: snap.label }],
        tabs: snap.tabs,
        activeTab: snap.tabs.some((t) => t.name === snap.activeTab) ? snap.activeTab : 'Main',
        redrawNonce: s.redrawNonce + 1,
      }
    }),
  redo: () =>
    set((s) => {
      const snap = s.future.at(-1)
      if (!snap) return {}
      return {
        future: s.future.slice(0, -1),
        past: [...s.past, { tabs: s.tabs, activeTab: s.activeTab, label: snap.label }],
        tabs: snap.tabs,
        activeTab: snap.tabs.some((t) => t.name === snap.activeTab) ? snap.activeTab : 'Main',
        redrawNonce: s.redrawNonce + 1,
      }
    }),

  getTab: (name) => get().tabs.find((t) => t.name === (name ?? get().activeTab)),
  getChat: (name) => get().chats.find((c) => c.name === (name ?? get().activeChat)),
  getXml: (key) => tabXml(get().getTab(key)),
  getTemplate: (key) => { 
    // check to see if template is still on content just in case
    const state = get()
    const tab = state.getTab(key)
    if (tab?.content && Array.isArray(tab?.content)) {
      state.setTemplate(tab.content, key)
      state.setXml('', key)
    }
    return tabTemplate(tab)
  },
  addNewTab: (tab, init) => {
    const name = uniqueName(tab, get().tabs.map((t) => t.name))
    set((s) => ({ tabs: [...s.tabs, { name, description: '', content: null, template: null, ...init }] }))
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

  addNewChat: (chat) => {
    const name = uniqueName(chat ?? 'Chat', get().chats.map((c) => c.name))
    set((s) => ({ chats: [...s.chats, { name, thoughts: [], history: [] }], activeChat: name }))
    return name
  },
  deleteChat: (chat) =>
    set((s) => {
      const chats = s.chats.filter((c) => c.name !== chat)
      // The panel always has somewhere to type, so removing the last chat opens a fresh one.
      const next = chats.length ? chats : mainChat()
      return { chats: next, activeChat: s.activeChat === chat ? next[0].name : s.activeChat }
    }),
  updateChat: (chat, patch) =>
    set((s) => ({ chats: s.chats.map((c) => (c.name === chat ? { ...c, ...patch } : c)) })),
  renameChat: (chat, newName) => {
    const name = uniqueName(newName, get().chats.filter((c) => c.name !== chat).map((c) => c.name))
    set((s) => ({
      chats: s.chats.map((c) => (c.name === chat ? { ...c, name } : c)),
      activeChat: s.activeChat === chat ? name : s.activeChat,
    }))
    return name
  },

  // Loading a score replaces the document, so the old one's history can't apply to it.
  setTabs: (tabs) => set({ tabs: tabs, past: [], future: [] }),
  setActiveTab: (tab) => set({ activeTab: tab }),
  setChats: (chats) => {
    const next = chats.length ? chats : mainChat()
    set({ chats: next, activeChat: next[0].name })
  },
  setActiveChat: (chat) => set({ activeChat: chat }),
  // Writing xml to a tab that isn't open yet creates it, so a generated score can land
  // straight in its own tab.
  setXml: (xml, key) =>
    set((s) => {
      const name = key ?? s.activeTab
      const tabs = s.tabs.some((t) => t.name === name)
        ? s.tabs.map((t) => (t.name === name ? { ...t, content: xml } : t))
        : [...s.tabs, { name, description: '', content: xml, template: null }]
      return { tabs, error: null }
    }),
  setTemplate: (template, key) =>
    set((s) => {
      const name = key ?? s.activeTab
      const tabs = s.tabs.some((t) => t.name === name)
        ? s.tabs.map((t) => (t.name === name ? { ...t, template: template } : t))
        : [...s.tabs, { name, description: '', template: template, content: null }]
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
  redrawScore: () => { 
    const state = get()
    if (state.getTemplate(state.activeTab) && state.getXml(state.activeTab))
      state.setXml(null, state.activeTab)
    state.resetCamera();
    set((s) => ({ redrawNonce: s.redrawNonce + 1 }))
  },
  toggleTextView: () => set((s) => ({ textView: !s.textView })),
  resetScore: () =>
    set({
      tabs: mainTab(),
      activeTab: 'Main',
      chats: mainChat(),
      activeChat: mainChat()[0].name,
      scoreId: null,
      scoreData: null,
      selection: null,
      error: null,
      streamBar: null,
      totalBars: null,
      streamMode: null,
      past: [],
      future: [],
    }),
}), {
  name: PERSIST_KEY,
  version: 2,
  storage: createJSONStorage(createWriteBehindStorage),
  // Only the document survives a reload. The camera, the selection and the streaming
  // progress are per-view state, and rehydrating a half-finished stream would leave the
  // UI stuck mid-generation with no request behind it.
  partialize: (s) => ({
    tabs: s.tabs,
    activeTab: s.activeTab,
    chats: s.chats,
    activeChat: s.activeChat,
    scoreId: s.scoreId,
    scoreData: s.scoreData,
  }),
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
