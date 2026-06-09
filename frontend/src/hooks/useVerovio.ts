import { useEffect, useRef, useState } from 'react'
import createVerovioModule from 'verovio/wasm'
import { VerovioToolkit } from 'verovio/esm'

let shared: VerovioToolkit | null = null
let loadingPromise: Promise<VerovioToolkit> | null = null

async function loadVerovio(): Promise<VerovioToolkit> {
  if (shared) return shared
  if (loadingPromise) return loadingPromise
  loadingPromise = createVerovioModule().then((mod) => {
    shared = new VerovioToolkit(mod)
    return shared
  })
  return loadingPromise
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
    const raw = shared.renderToTimemap(JSON.stringify({ includeMeasures: true }))
    const entries = JSON.parse(raw) as Array<{ tstamp?: number; measureOn?: string }>
    const times: number[] = [0] // index 0 unused
    for (const e of entries) {
      if (e.measureOn !== undefined && e.tstamp !== undefined) {
        times.push(e.tstamp / 1000)
      }
    }
    return times.length > 1 ? times : []
  } catch {
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

    tk.setOptions({
      pageWidth,
      pageHeight: 2970,
      breaks: 'auto',
      scale,
      footer: 'none',
      header: 'none',
      spacingLinear: 0.2,
    })

    // MXL files are stored with a prefix so we know to use the ZIP loader
    const loaded = data.startsWith('__MXL__')
      ? tk.loadZipDataBase64(data.slice(7))
      : tk.loadData(data)

    if (!loaded) return []

    const pageCount = tk.getPageCount()
    return Array.from({ length: pageCount }, (_, i) => tk.renderToSVG(i + 1))
  }

  return { ready, renderXml }
}
