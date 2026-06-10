import { useEffect, useRef, useState } from 'react'
import createVerovioModule from 'verovio/wasm'
import { VerovioToolkit } from 'verovio/esm'

let shared: VerovioToolkit | null = null
let loadingPromise: Promise<VerovioToolkit> | null = null

const PAGE_HEIGHT = 800

async function loadVerovio(): Promise<VerovioToolkit> {
  if (shared) return shared
  if (loadingPromise) return loadingPromise
  loadingPromise = createVerovioModule().then((mod) => {
    shared = new VerovioToolkit(mod)
    return shared
  })
  return loadingPromise
}

function applyOptions(tk: VerovioToolkit, pageWidth: number, scale: number): void {
  tk.setOptions({
    pageWidth,
    pageHeight: PAGE_HEIGHT,
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

/** Returns the shared VerovioToolkit once WASM has loaded, or null before that. */
export function getSharedToolkit(): VerovioToolkit | null {
  return shared
}

/**
 * Applies layout options, loads the score, and returns the page count.
 * Returns 0 if the toolkit is not ready or the data fails to load.
 * After this call, getSharedToolkit()!.renderToSVG(page) can be called directly.
 */
export function loadAndGetPageCount(data: string, pageWidth: number, scale: number): number {
  if (!shared) return 0
  applyOptions(shared, pageWidth, scale)
  if (!loadScore(shared, data)) return 0
  return shared.getPageCount()
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
    applyOptions(tk, pageWidth, scale)
    if (!loadScore(tk, data)) return []
    const pageCount = tk.getPageCount()
    return Array.from({ length: pageCount }, (_, i) => tk.renderToSVG(i + 1))
  }

  return { ready, renderXml }
}
