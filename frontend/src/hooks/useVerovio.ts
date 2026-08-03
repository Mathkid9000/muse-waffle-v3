import { useEffect, useRef, useState } from 'react'
import createVerovioModule from 'verovio/wasm'
import { VerovioToolkit } from 'verovio/esm'

let shared: VerovioToolkit | null = null
let loadingPromise: Promise<VerovioToolkit> | null = null

const PAGE_HEIGHT_PER_INSTRUMENT = 160
const PAGE_MARGIN = 140

async function loadVerovio(): Promise<VerovioToolkit> {
  if (shared) return shared
  if (loadingPromise) return loadingPromise
  loadingPromise = createVerovioModule().then((mod) => {
    shared = new VerovioToolkit(mod)
    return shared
  })
  return loadingPromise
}

function applyOptions(tk: VerovioToolkit, pageWidth: number, scale: number, instr_num: number): void {
  tk.setOptions({
    pageWidth,
    pageHeight: PAGE_HEIGHT_PER_INSTRUMENT * instr_num + PAGE_MARGIN * 2,
    breaks: 'auto',
    scale,
    footer: 'none',
    header: 'none',
    spacingLinear: 0.2,
  })
}

function loadScore(tk: VerovioToolkit, data: string): boolean {
  return data.startsWith('__MXL__')
    ? tk.loadZipDataBase64(data.slice(7))
    : tk.loadData(data)
}

/**
 * Staves in the score currently loaded into `tk`, read off its first rendered measure. Safe to call
 * at any page height: Verovio never splits a system across pages, so page 1's first measure always
 * carries every staff even when the page is far too short for them.
 */
function loadedStaffCount(tk: VerovioToolkit): number {
  const doc = new DOMParser().parseFromString(tk.renderToSVG(1), 'image/svg+xml')
  return doc.querySelector('g.measure')?.querySelectorAll('g.staff').length ?? 0
}

/**
 * Loads `data` and sizes the page to the staves it turns out to hold — the count is read back off
 * the loaded score rather than passed in, so the page always fits what was just loaded however the
 * staff count got there (a generation, an added instrument, a deleted staff, a fresh file).
 * Returns the page count, or 0 when the data could not be loaded.
 */
function loadFitted(tk: VerovioToolkit, data: string, pageWidth: number, scale: number): number {
  applyOptions(tk, pageWidth, scale, 1)
  if (!loadScore(tk, data)) return 0
  const staves = loadedStaffCount(tk)
  if (staves > 1) {
    applyOptions(tk, pageWidth, scale, staves)
    tk.redoLayout()
  }
  return tk.getPageCount()
}

/** Returns the shared VerovioToolkit once WASM has loaded, or null before that. */
export function getSharedToolkit(): VerovioToolkit | null {
  return shared
}

/**
 * Loads the score at a page sized to fit it, and returns the page count.
 * Returns 0 if the toolkit is not ready or the data fails to load.
 * After this call, getSharedToolkit()!.renderToSVG(page) can be called directly.
 */
export function loadAndGetPageCount(data: string, pageWidth: number, scale: number): number {
  return shared ? loadFitted(shared, data, pageWidth, scale) : 0
}

/**
 * Counts staves in the score's first measure — used as the instrument count when
 * opening a file, since each instrument in this app occupies exactly one staff.
 */
export async function countInstruments(data: string): Promise<number> {
  const tk = await loadVerovio()
  applyOptions(tk, 1000, 40, 1)
  if (!loadScore(tk, data)) return 1
  return Math.max(loadedStaffCount(tk), 1)
}

/** Returns base64-encoded MIDI for the currently-loaded score. */
export function getCurrentMidi(): string {
  return shared?.renderToMIDI() ?? ''
}

/**
 * Returns an array of measure start times in seconds, indexed by measure number.
 * measureStartTimes[0] is unused; measureStartTimes[1] = measure 1 start (always 0),
 * measureStartTimes[2] = measure 2 start, etc.
 */
export function getMeasureStartTimes(): number[] {
  if (!shared) return []
  try {
    const entries = shared.renderToTimemap({ includeMeasures: true }) as Array<{ tstamp?: number; on?: string[] }>
    const noteToTstamp = new Map<string, number>()
    for (const e of entries) {
      if (e.on && e.tstamp !== undefined) {
        for (const id of e.on) noteToTstamp.set(id, e.tstamp)
      }
    }
    if (noteToTstamp.size === 0) return []

    const pageCount = shared.getPageCount()
    const parser = new DOMParser()
    const times: number[] = [0] // index 0 unused

    for (let page = 1; page <= pageCount; page++) {
      const doc = parser.parseFromString(shared.renderToSVG(page), 'image/svg+xml')
      for (const measure of doc.querySelectorAll('g.measure')) {
        let minTstamp = Infinity
        for (const el of measure.querySelectorAll('[id]')) {
          const t = noteToTstamp.get(el.getAttribute('id')!)
          if (t !== undefined && t < minTstamp) minTstamp = t
        }
        if (minTstamp !== Infinity) times.push(minTstamp / 1000)
      }
    }

    return times.length > 1 ? times : []
  } catch(e) {
    console.error('getMeasureStartTimes error:', e)
    return []
  }
}

export function useVerovio() {
  const [ready, setReady] = useState(false)
  const tkRef = useRef<VerovioToolkit | null>(null)

  useEffect(() => {
    loadVerovio().then((tk) => {
      tkRef.current = tk
      setReady(true)
    })
  }, [])

  function renderXml(data: string, pageWidth: number, scale: number): string[] {
    const tk = tkRef.current
    if (!tk) return []
    const pageCount = loadFitted(tk, data, pageWidth, scale)
    return Array.from({ length: pageCount }, (_, i) => tk.renderToSVG(i + 1))
  }

  return { ready, renderXml }
}
