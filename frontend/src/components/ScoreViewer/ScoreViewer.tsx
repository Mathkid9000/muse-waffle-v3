import { useEffect, useMemo, useState, useRef, Activity } from 'react'
import { createRoot } from 'react-dom/client'
import { useVerovio, getSharedToolkit, loadAndGetPageCount } from '../../hooks/useVerovio'
import { useStore } from '../../store'
import styles from './ScoreViewer.module.css'
import AppendBar from './AppendBar'
import InscorePrompt from './InscorePrompt'
import { promptAnthropic } from '../../api/claudeTools'
import { applyClaudeEdit } from '../../api/applyClaudeEdit'
import { promptTemplateAnthropic } from '../../api/claudeTemplates'
import { tabTemplate, tabXml } from '../../types'
import { templateToXml } from '../../api/templates'

const MEASURE_SEL = 'g.measure'
const STAFF_SEL = 'g.staff'
const PAGE_SEL = '[data-page-index]'
const MARGIN_SEL = 'g.page-margin'
// Verovio draws the staff lines as the direct <path> children of a g.staff.
const STAFF_LINE_SEL = ':scope > path'

// Layout constants — must match .pages padding / gap in the CSS module.
const PAD = 20
const GAP = 20
// White paper around the engraving, on every side — must match .page padding in the CSS module.
const PAGE_PAD = 100
// Padding around a selection band, in unzoomed layout pixels.
const SEL_PAD_PX = 5
// Fallback page size (Verovio's own px units) for pages not yet rendered.
const DEFAULT_SIZE: Size = { w: 1000, h: 700 }

const svgParser = new DOMParser()

/** Size of a rendered page, in the px units Verovio writes on the SVG (shared across pages). */
interface Size {
  w: number
  h: number
}

interface PageInfo {
  count: number
  size: Size
}

interface PageMeta {
  measureCounts: number[]
  /** Number of measures in all pages before this one (cumulative offset). */
  offsets: number[]
  /** Engraved size of each page — its paper is this plus PAGE_PAD on every side. */
  sizes: Size[]
  /** Tallest engraving; the page holding it is the one scaled to fill the viewport height. */
  maxH: number
  total: number
}

/**
 * The engraved size of a rendered Verovio page. Since the editor renders with
 * adjustPageWidth/Height and no page margin (see useVerovio), this is the score itself —
 * the box the paper has to match.
 */
function svgSize(doc: Document): Size {
  const svg = doc.documentElement
  const w = parseFloat(svg.getAttribute('width') || '')
  const h = parseFloat(svg.getAttribute('height') || '')
  if (w > 0 && h > 0) return { w, h }
  const vb = svg.getAttribute('viewBox')
  if (vb) {
    const p = vb.split(/[\s,]+/).map(Number)
    if (p.length === 4 && p[2] > 0 && p[3] > 0) return { w: p[2], h: p[3] }
  }
  return DEFAULT_SIZE
}

/** An axis-aligned box, in whatever space its source coordinates came from. */
interface Box {
  x0: number
  y0: number
  x1: number
  y1: number
}

function unionBox(a: Box | null, b: Box | null): Box | null {
  if (!a) return b
  if (!b) return a
  return {
    x0: Math.min(a.x0, b.x0),
    y0: Math.min(a.y0, b.y0),
    x1: Math.max(a.x1, b.x1),
    y1: Math.max(a.y1, b.y1),
  }
}

/**
 * The staff rectangle of one measure — bottom staff line to top staff line, spanning exactly
 * the measure's width — in the page's user space.
 *
 * Only the staff lines are measured. Everything that overflows the staff (notes, stems,
 * ledger lines) is nested deeper inside g.staff, and slurs, ties and hairpins are siblings of
 * g.staff, so none of them can stretch this box the way g.measure.getBBox() does.
 * Returns null when nothing is measurable (e.g. the page is not rendered).
 */
function staffLinesBox(staffEl: Element): Box | null {
  let box: Box | null = null
  for (const line of staffEl.querySelectorAll<SVGGraphicsElement>(STAFF_LINE_SEL)) {
    try {
      const b = line.getBBox()
      box = unionBox(box, { x0: b.x, y0: b.y, x1: b.x + b.width, y1: b.y + b.height })
    } catch { /* not rendered */ }
  }
  return box
}

/** Same box as staffLinesBox, but in client (screen) pixels. */
function staffLinesClientBox(staffEl: Element): Box | null {
  let box: Box | null = null
  for (const line of staffEl.querySelectorAll<SVGGraphicsElement>(STAFF_LINE_SEL)) {
    const r = line.getBoundingClientRect()
    if (r.width === 0 && r.height === 0) continue
    box = unionBox(box, { x0: r.left, y0: r.top, x1: r.right, y1: r.bottom })
  }
  return box
}

/**
 * SVG user units per unzoomed layout pixel, so a pixel-sized pad has that size at 100% zoom
 * and then scales with the camera like the engraving does. getScreenCTM() folds the camera
 * scale in, so it is divided back out. Returns 0 when the element is not rendered.
 */
function userUnitsPerPx(el: SVGGraphicsElement, zoom: number): number {
  const ctm = el.getScreenCTM()
  if (!ctm || !ctm.a) return 0
  return zoom / 100 / ctm.a
}

/** Page index (0-based) containing 1-based global measure `m`, or -1. */
function pageOfMeasure(meta: PageMeta, m: number): number {
  for (let i = 0; i < meta.measureCounts.length; i++) {
    if (meta.offsets[i] < m && m <= meta.offsets[i] + meta.measureCounts[i]) return i
  }
  return -1
}

export default function ScoreViewer() {
  const { ready, renderXml } = useVerovio()
  const setXml = useStore((s) => s.setXml)
  const state = useStore.getState()
  const tab = useStore((s) => s.tabs.find((t) => t.name === s.activeTab))
  const xml = state.getXml(state.activeTab)
  const template = state.getTemplate(state.activeTab)
  const scoreData = useStore((s) => s.scoreData)
  const zoom = useStore((s) => s.zoom)
  const panX = useStore((s) => s.panX)
  const panY = useStore((s) => s.panY)
  const viewportW = useStore((s) => s.viewportW)
  const viewportH = useStore((s) => s.viewportH)
  const selection = useStore((s) => s.selection)
  const setSelection = useStore((s) => s.setSelection)
  const streamMode = useStore((s) => s.streamMode)
  const redrawNonce = useStore((s) => s.redrawNonce)

  const [pages, setPages] = useState<string[]>([])
  const outerRef = useRef<HTMLDivElement>(null)
  const innerRef = useRef<HTMLDivElement>(null)
  const marqueeRef = useRef<HTMLDivElement>(null)
  const dragStartClientRef = useRef<{ x: number; y: number } | null>(null)
  const rectCacheRef = useRef<Array<{ measure: number; staff: number; rect: DOMRect }>>([])
  const staffCountRef = useRef(0)
  const lastSelKeyRef = useRef<string>('')
  const zoomRef = useRef(zoom)
  zoomRef.current = zoom
  // True while a Ctrl+right-drag pan gesture is in progress (transform is updated
  // imperatively during the drag, then committed to the store on release).
  const panningRef = useRef(false)

  const cachedPagesRef = useRef<string[]>([])
  // Per-SVG metadata cache keyed by the SVG string, so unchanged pages (which keep
  // their string identity across streaming updates) are never reparsed.
  const metaCacheRef = useRef<Map<string, PageInfo>>(new Map())

  // The prompt is a single DOM node + React root created once and reused for the
  // lifetime of ScoreViewer. Each redraw just moves this same node to its new
  // position instead of destroying/recreating it, so InscorePrompt's internal state
  // (e.g. the typed-in text) survives redraws.
  const promptHostRef = useRef<HTMLDivElement | null>(null)
  useEffect(() => resetPromptHost(), [])

  const resetPromptHost = () => {
    const host = document.createElement('div')
    // Read the active tab at call time — this root is created once and outlives tab switches.
    createRoot(host).render(<InscorePrompt getXml={() => useStore.getState().getXml(useStore.getState().activeTab)} getSelection={() => useStore.getState().selection!} setXml={(xml) => setXml(xml, useStore.getState().activeTab)} />)
    promptHostRef.current = host
    return () => host.remove()
  }

  // Per-page measure counts / offsets / engraved sizes. Selection maps a page's local
  // measure indices to global measure numbers via offsets; placeholders size from sizes.
  const pageMeta = useMemo<PageMeta>(() => {
    const cache = metaCacheRef.current
    const measureCounts: number[] = []
    const sizes: Size[] = []
    const offsets: number[] = []
    let acc = 0
    let maxH = 0
    let lastSize = DEFAULT_SIZE
    for (const svg of pages) {
      offsets.push(acc)
      if (!svg) {
        measureCounts.push(0)
        sizes.push(lastSize)
        maxH = Math.max(maxH, lastSize.h)
        continue
      }
      let info = cache.get(svg)
      if (!info) {
        const doc = svgParser.parseFromString(svg, 'image/svg+xml')
        info = { count: doc.querySelectorAll(MEASURE_SEL).length, size: svgSize(doc) }
        cache.set(svg, info)
      }
      measureCounts.push(info.count)
      sizes.push(info.size)
      maxH = Math.max(maxH, info.size.h)
      lastSize = info.size
      acc += info.count
    }
    return { measureCounts, offsets, sizes, maxH: maxH || DEFAULT_SIZE.h, total: acc }
  }, [pages])

  // Which pages currently intersect the viewport (plus a one-viewport buffer). Computed
  // analytically from the camera transform + page layout, since the pan/zoom camera uses
  // CSS transforms (an IntersectionObserver wouldn't reliably fire on transform changes).
  const visiblePages = useMemo<Set<number>>(() => {
    const set = new Set<number>()
    const n = pageMeta.sizes.length
    if (n === 0) return set
    if (viewportW === 0 || viewportH === 0) {
      for (let i = 0; i < n; i++) set.add(i)
      return set
    }
    const z = zoom / 100
    // Every page is drawn at the one scale that makes the tallest engraving fill the viewport
    // height, so pages of different sizes stay in proportion to each other (see the CSS module).
    const k = Math.max(1, viewportH - 2 * PAD - 2 * PAGE_PAD) / pageMeta.maxH
    const widths = pageMeta.sizes.map((s) => s.w * k + 2 * PAGE_PAD)
    // Pages lay out left-to-right (flex-start) starting after the container padding.
    let x = PAD
    // The tallest page spans the full content band and the rest are centred inside it, so
    // vertical visibility is shared.
    const sTop = panY + z * PAD
    const sBot = panY + z * (viewportH - PAD)
    const vOK = sBot >= -viewportH && sTop <= 2 * viewportH
    const bufX = viewportW * 0.5
    for (let i = 0; i < n; i++) {
      const sL = panX + z * x
      const sR = panX + z * (x + widths[i])
      if (vOK && sR >= -bufX && sL <= viewportW + bufX) set.add(i)
      x += widths[i] + GAP
    }
    return set
  }, [pageMeta, panX, panY, zoom, viewportW, viewportH])

  useEffect(() => {
    if (!xml && template) {
      buildXmlFromTemplate()
    }

    if (!ready || !xml) {
      cachedPagesRef.current = []
      metaCacheRef.current.clear()
      setPages([])
      return
    }

    const w = outerRef.current?.clientWidth || 1000
    const { streamMode, selection: sel } = useStore.getState()
    const tk = getSharedToolkit()

    if (!tk || !streamMode || cachedPagesRef.current.length === 0) {
      // Full render: initial load, not streaming, or cache empty (first streaming event)
      const allPages = renderXml(xml, w * 2.5, 40)
      cachedPagesRef.current = allPages
      setPages(allPages)
      return
    }

    // Incremental: loadAndGetPageCount applies options + loads data into the shared tk
    const prevCount = cachedPagesRef.current.length
    const newCount = loadAndGetPageCount(xml, w * 2.5, 40)
    if (newCount === 0) return

    let startPage: number
    let endPage: number
    if (streamMode === 'infill') {
      const gapStart = sel?.start ?? 1
      const gapPageIdx = pageOfMeasure(pageMeta, gapStart)
      startPage = (gapPageIdx < 0 ? prevCount - 1 : gapPageIdx) + 1  // 0-indexed page -> 1-indexed
      endPage = startPage
    } else {
      // generate/append: re-render from old last page through new last page
      startPage = Math.max(1, prevCount)
      endPage = newCount
    }

    const updated = [...cachedPagesRef.current]
    while (updated.length < newCount) updated.push('')
    for (let p = startPage; p <= endPage; p++) {
      updated[p - 1] = tk.renderToSVG(p)
    }
    cachedPagesRef.current = updated
    setPages([...updated])
    // the page height comes from the score itself now, so nothing outside these has to be watched
  }, [ready, xml, template, streamMode, redrawNonce])

  // Inject selection highlight rects directly into the SVG DOM.
  // One band per selected staff, drawn from the staff's own staff lines (see staffLinesBox)
  // and padded by SEL_PAD_PX, so a slur, tie or hairpin leaving the measure never widens it.
  // Consecutive measures on the same system are merged into a single band, otherwise the
  // padding of neighbouring measures would overlap and the translucent fill would show a
  // seam at every barline.
  // Only mounted (visible) pages carry measure elements; measures on off-screen pages
  // are skipped and re-applied by this effect once those pages scroll into view.
  useEffect(() => {
    const inner = innerRef.current
    if (!inner) return
    inner.querySelectorAll('.mw-sel').forEach((el) => el.remove())
    promptHostRef.current?.remove()
    if (!selection) return

    const ns = 'http://www.w3.org/2000/svg'
    // One highlight layer per coordinate-space host (the page margin group), inserted first
    // so the bands paint behind the engraving. Keeping the rects out of the measure groups
    // leaves measure/staff geometry untouched for the hit-test and the drag rect cache.
    const layers = new Map<Element, SVGGraphicsElement>()

    function layerFor(host: Element): SVGGraphicsElement {
      const existing = layers.get(host)
      if (existing) return existing
      const layer = document.createElementNS(ns, 'g')
      layer.setAttribute('class', 'mw-sel')
      layer.setAttribute('pointer-events', 'none')
      host.insertBefore(layer, host.firstChild)
      layers.set(host, layer)
      return layer
    }

    function addRect(layer: SVGGraphicsElement, box: Box, pad: number) {
      const rect = document.createElementNS(ns, 'rect')
      rect.setAttribute('x', String(box.x0 - pad))
      rect.setAttribute('y', String(box.y0 - pad))
      rect.setAttribute('width', String(box.x1 - box.x0 + 2 * pad))
      rect.setAttribute('height', String(box.y1 - box.y0 + 2 * pad))
      rect.setAttribute('fill', 'var(--user-selection)')
      rect.setAttribute('stroke-width', '4')
      rect.setAttribute('class', 'mw-sel')
      rect.setAttribute('pointer-events', 'none')
      layer.appendChild(rect)
    }

    // Resolve each selected measure to its mounted group, dropping the ones whose page is
    // not currently rendered.
    interface Cell {
      measure: number
      pageEl: HTMLElement
      measureEl: SVGGraphicsElement
      /** Coordinate-space host shared by every measure of the page. */
      host: Element
      /** Measures of one system are horizontally contiguous; a new system breaks the run. */
      system: Element | null
    }
    const cells: Cell[] = []
    for (let m = selection.start; m <= selection.end; m++) {
      const pageIdx = pageOfMeasure(pageMeta, m)
      if (pageIdx < 0) continue
      const pageEl = inner.querySelector<HTMLElement>(`[data-page-index="${pageIdx}"]`)
      if (!pageEl) continue
      const measures = pageEl.querySelectorAll<SVGGraphicsElement>(MEASURE_SEL)
      if (measures.length === 0) continue // page not mounted
      const measureEl = measures[m - pageMeta.offsets[pageIdx] - 1]
      if (!measureEl) continue
      const host = pageEl.querySelector(MARGIN_SEL) ?? measureEl.parentElement
      if (!host) continue
      cells.push({ measure: m, pageEl, measureEl, host, system: measureEl.closest('g.system') })
    }

    /** Paints one run and returns the top-left of its first band, for the prompt host. */
    function paintRun(run: Cell[]): { pageEl: HTMLElement; layer: SVGGraphicsElement; x: number; y: number } | null {
      const layer = layerFor(run[0].host)
      const pad = SEL_PAD_PX * userUnitsPerPx(layer, zoomRef.current)
      const staffCount = run[0].measureEl.querySelectorAll(STAFF_SEL).length
      const parts = selection!.parts ?? Array.from({ length: staffCount }, (_, i) => i)
      let first: { pageEl: HTMLElement; layer: SVGGraphicsElement; x: number; y: number } | null = null
      for (const partIdx of parts) {
        let box: Box | null = null
        for (const cell of run) {
          const staffEl = cell.measureEl.querySelectorAll<SVGGraphicsElement>(STAFF_SEL)[partIdx]
          if (staffEl) box = unionBox(box, staffLinesBox(staffEl))
        }
        if (!box) continue
        addRect(layer, box, pad)
        first ??= { pageEl: run[0].pageEl, layer, x: box.x0 - pad, y: box.y0 - pad }
      }
      return first
    }

    // Split the measures into runs that can share one band: consecutive, same page, same system.
    const runs: Cell[][] = []
    for (const cell of cells) {
      const run = runs[runs.length - 1]
      const prev = run?.[run.length - 1]
      if (prev && prev.measure === cell.measure - 1 && prev.host === cell.host && prev.system === cell.system) {
        run.push(cell)
      } else {
        runs.push([cell])
      }
    }

    let anchor: { pageEl: HTMLElement; layer: SVGGraphicsElement; x: number; y: number } | null = null
    for (const r of runs) {
      const first = paintRun(r)
      if (!anchor && r.some((c) => c.measure === selection.start)) anchor = first
    }

    // "type something" prompt, anchored to the top-left corner of the selection.
    const prompt = promptHostRef.current
    if (prompt && anchor) {
      const { pageEl, layer, x, y } = anchor
      const svg = pageEl.querySelector('svg') as SVGSVGElement | null
      const ctm = layer.getScreenCTM()
      if (svg && ctm) {
        const pt = svg.createSVGPoint()
        pt.x = x
        pt.y = y
        const screenPt = pt.matrixTransform(ctm)
        const pageRect = pageEl.getBoundingClientRect()
        const z = zoomRef.current / 100
        prompt.style.cssText = `position: absolute; top: ${(screenPt.y - pageRect.top) / z - 10}px; left: ${(screenPt.x - pageRect.left) / z}px`
      }
      pageEl.appendChild(prompt)
    }
  }, [selection, pages, visiblePages, pageMeta])

  // Wheel: Ctrl/Cmd + wheel zooms toward the cursor; plain wheel pans the camera.
  // Registered manually (not via React's onWheel) so the listener is non-passive and
  // preventDefault() actually suppresses native page scroll / browser zoom.
  useEffect(() => {
    const outer = outerRef.current
    if (!outer) return
    function onWheel(e: WheelEvent) {
      e.preventDefault()
      const rect = outer!.getBoundingClientRect()
      const ox = e.clientX - rect.left
      const oy = e.clientY - rect.top
      const { zoomTo, setPan, panX, panY } = useStore.getState()
      if (e.ctrlKey || e.metaKey) {
        const prev = zoomRef.current
        const next = Math.round(prev * (e.deltaY < 0 ? 1.1 : 1 / 1.1))
        if (next !== prev) zoomTo(next, ox, oy)
      } else {
        setPan(panX - e.deltaX, panY - e.deltaY)
      }
    }
    outer.addEventListener('wheel', onWheel, { passive: false })
    return () => outer.removeEventListener('wheel', onWheel)
  }, [])

  // Keep the store's viewport size in sync so the toolbar can zoom around the center.
  useEffect(() => {
    const outer = outerRef.current
    if (!outer) return
    const ro = new ResizeObserver(() => useStore.getState().setViewport(outer.clientWidth, outer.clientHeight))
    ro.observe(outer)
    return () => ro.disconnect()
  }, [])

  // Returns the 1-based global measure index and the 0-based staff index for a mouse event.
  // staffIdx is null when the click doesn't land on a specific staff (e.g. barlines).
  // Scopes to the page div under the pointer, then maps the local measure index to a
  // global measure number via that page's offset.
  // 1. Tries closest() — works when the click lands on a note/rest/barline child.
  // 2. Falls back to geometric hit-test — works when clicking empty staff space.
  function measureAndStaffAt(e: React.MouseEvent): { measure: number; staffIdx: number | null } | null {
    const target = e.target
    if (!(target instanceof Element)) return null

    const pageEl = target.closest<HTMLElement>(PAGE_SEL)
    if (!pageEl) return null
    const offset = pageMeta.offsets[Number(pageEl.dataset.pageIndex)] ?? 0
    const measures = pageEl.querySelectorAll<SVGGraphicsElement>(MEASURE_SEL)
    if (measures.length === 0) return null

    // Fast path: clicked element is inside a measure group
    const directMeasure = target.closest<Element>(MEASURE_SEL)
    if (directMeasure && pageEl.contains(directMeasure)) {
      const local = Array.from(measures).indexOf(directMeasure as SVGGraphicsElement)
      if (local < 0) return null
      const directStaff = target.closest<Element>(STAFF_SEL)
      let staffIdx: number | null = null
      if (directStaff && directMeasure.contains(directStaff)) {
        const staffEls = Array.from(directMeasure.querySelectorAll(STAFF_SEL))
        const si = staffEls.indexOf(directStaff as Element)
        if (si >= 0) staffIdx = si
      }
      return { measure: offset + local + 1, staffIdx }
    }

    // Geometric fallback: map pointer into SVG coordinates and test bounding boxes.
    const svg = pageEl.querySelector('svg') as SVGSVGElement | null
    if (!svg) return null
    const svgRect = svg.getBoundingClientRect()
    const vb = (svg.querySelector('.definition-scale') as SVGSVGElement)?.viewBox?.baseVal
    if (!vb || svgRect.width === 0 || svgRect.height === 0) return null

    const scaleX = vb.width / svgRect.width
    const scaleY = vb.height / svgRect.height
    const x = (e.clientX - svgRect.left) * scaleX + vb.x
    const y = (e.clientY - svgRect.top) * scaleY + vb.y

    // Test against the staff rectangles, padded to match the drawn highlight, so the clickable
    // area of a measure is exactly the area that lights up — a slur or hairpin overflowing
    // into the next bar no longer lets a measure claim its neighbour's space.
    const pad = SEL_PAD_PX * userUnitsPerPx(measures[0], zoomRef.current)
    for (let i = 0; i < measures.length; i++) {
      const staffBoxes = Array.from(measures[i].querySelectorAll<SVGGraphicsElement>(STAFF_SEL), staffLinesBox)
      const mBox = staffBoxes.reduce<Box | null>((acc, b) => unionBox(acc, b), null)
      if (!mBox) continue
      if (x < mBox.x0 - pad || x > mBox.x1 + pad || y < mBox.y0 - pad || y > mBox.y1 + pad) continue
      // Found the measure — try to narrow down to a specific staff
      let staffIdx: number | null = null
      for (let s = 0; s < staffBoxes.length; s++) {
        const b = staffBoxes[s]
        if (b && y >= b.y0 - pad && y <= b.y1 + pad) {
          staffIdx = s
          break
        }
      }
      return { measure: offset + i + 1, staffIdx }
    }

    return null
  }

  // Cache a client-space rect for every staff in every mounted measure, so the drag
  // marquee can hit-test without thrashing layout on each mousemove. Off-screen pages
  // carry no measures and are simply skipped.
  function buildRectCache() {
    const inner = innerRef.current
    const cache: Array<{ measure: number; staff: number; rect: DOMRect }> = []
    if (!inner) {
      rectCacheRef.current = cache
      staffCountRef.current = 0
      return
    }
    let staffCount = 0
    inner.querySelectorAll<HTMLElement>(PAGE_SEL).forEach((pageEl) => {
      const offset = pageMeta.offsets[Number(pageEl.dataset.pageIndex)] ?? 0
      pageEl.querySelectorAll<SVGGraphicsElement>(MEASURE_SEL).forEach((m, mLocal) => {
        const staffEls = m.querySelectorAll<SVGGraphicsElement>(STAFF_SEL)
        if (staffCount === 0 && staffEls.length) staffCount = staffEls.length
        staffEls.forEach((s, sIdx) => {
          // Staff lines only: a high note or a ledger line sticking out of the staff must not
          // make the drag marquee grab a staff it never touched.
          const b = staffLinesClientBox(s)
          if (!b) return
          const rect = new DOMRect(b.x0, b.y0, b.x1 - b.x0, b.y1 - b.y0)
          cache.push({ measure: offset + mLocal + 1, staff: sIdx, rect })
        })
      })
    })
    rectCacheRef.current = cache
    staffCountRef.current = staffCount || (scoreData?.numInstruments ?? 1)
  }

  function hideMarquee() {
    if (marqueeRef.current) marqueeRef.current.style.display = 'none'
  }

  function onMouseDown(e: React.MouseEvent) {
    // Ctrl + right button: pan the camera instead of selecting.
    if (e.button === 1) {
      startPan(e)
      return
    }
    // Only the left button drives selection.
    if (e.button !== 0) return
    e.preventDefault()
    dragStartClientRef.current = { x: e.clientX, y: e.clientY }
    lastSelKeyRef.current = ''
    buildRectCache()
    hideMarquee()

    // Immediate single-click feedback (drag may override this on mousemove).
    const hit = measureAndStaffAt(e)
    if (hit === null) {
      setSelection(null)
      return
    }
    const staffCount = staffCountRef.current
    const parts = staffCount > 1 && hit.staffIdx !== null ? [hit.staffIdx] : null
    setSelection({ start: hit.measure, end: hit.measure, parts })
  }

  function onMouseMove(e: React.MouseEvent) {
    const start = dragStartClientRef.current
    if (!start || !(e.buttons & 1)) return

    const boxLeft = Math.min(start.x, e.clientX)
    const boxRight = Math.max(start.x, e.clientX)
    const boxTop = Math.min(start.y, e.clientY)
    const boxBottom = Math.max(start.y, e.clientY)

    // Below the threshold this is still a click, not a drag — leave the
    // mousedown selection in place and don't draw the marquee.
    if (boxRight - boxLeft < 4 && boxBottom - boxTop < 4) return

    // Draw the marquee imperatively (no React re-render) so it tracks the
    // cursor smoothly without churning the SVG highlight on every event.
    const outer = outerRef.current
    const mq = marqueeRef.current
    if (outer && mq) {
      const o = outer.getBoundingClientRect()
      mq.style.display = 'block'
      mq.style.left = `${boxLeft - o.left}px`
      mq.style.top = `${boxTop - o.top}px`
      mq.style.width = `${boxRight - boxLeft}px`
      mq.style.height = `${boxBottom - boxTop}px`
    }

    // Hit-test cached rects (cheap arithmetic, no layout). Only write the
    // selection when the covered measure/staff set actually changes, so the
    // highlight effect doesn't reflow the SVG while dragging within a measure.
    let start1 = Infinity
    let end1 = -Infinity
    const staffSet = new Set<number>()
    for (const { measure, staff, rect } of rectCacheRef.current) {
      const overlaps = rect.right >= boxLeft && rect.left <= boxRight && rect.bottom >= boxTop && rect.top <= boxBottom
      if (overlaps) {
        if (measure < start1) start1 = measure
        if (measure > end1) end1 = measure
        staffSet.add(staff)
      }
    }

    if (start1 === Infinity) {
      if (lastSelKeyRef.current !== 'none') {
        lastSelKeyRef.current = 'none'
        setSelection(null)
      }
      return
    }
    const staffCount = staffCountRef.current
    const parts = staffSet.size === 0 || staffSet.size === staffCount ? null : [...staffSet].sort((a, b) => a - b)
    const key = `${start1}-${end1}-${parts ? parts.join(',') : 'all'}`
    if (key !== lastSelKeyRef.current) {
      lastSelKeyRef.current = key
      setSelection({ start: start1, end: end1, parts })
    }
  }

  function onMouseUp() {
    dragStartClientRef.current = null
    rectCacheRef.current = []
    hideMarquee()
  }

  // Ctrl + right-drag pans the camera. The transform is updated imperatively on the
  // inner element during the gesture (no React re-render per move), then committed to
  // the store on release. Window listeners ensure the gesture ends even if the cursor
  // leaves the viewport.
  function startPan(e: React.MouseEvent) {
    e.preventDefault()
    panningRef.current = true
    const { panX: px0, panY: py0 } = useStore.getState()
    const sx = e.clientX
    const sy = e.clientY
    let last = { x: px0, y: py0 }
    if (outerRef.current) outerRef.current.style.cursor = 'grabbing'

    const onMove = (ev: MouseEvent) => {
      last = { x: px0 + (ev.clientX - sx), y: py0 + (ev.clientY - sy) }
      if (innerRef.current) {
        innerRef.current.style.transform = `translate(${last.x}px, ${last.y}px) scale(${zoomRef.current / 100})`
      }
    }
    const onUp = () => {
      window.removeEventListener('mousemove', onMove)
      window.removeEventListener('mouseup', onUp)
      panningRef.current = false
      if (outerRef.current) outerRef.current.style.cursor = ''
      useStore.getState().setPan(last.x, last.y)
    }
    window.addEventListener('mousemove', onMove)
    window.addEventListener('mouseup', onUp)
  }

 async function handleBlankScoreInput(e: React.KeyboardEvent) {
    if (e.key !== 'Enter') return

    console.log(e.target.value)
    const prompt = e.target.value
    const outputXml = await promptTemplateAnthropic(prompt, useStore.getState().getXml('Main'))
    console.log(outputXml)
    if (outputXml) {
      setXml(outputXml as string, 'Main')
      useStore.getState().resetCamera()
    }
    // const response = await promptAnthropic(prompt, '', 
    //   (thought) => {console.log(thought)}, 
    //   (title) => {console.log(title)}, null)

    // console.log(response)
    // const edits = Array.isArray(response[0]) ? response[0] : []
    // for (const { tool, input } of edits) {
    //   applyClaudeEdit(tool, input, getXml(activeTab) ?? '', (xml) => setXml(xml, activeTab))
    // }
  }

  async function buildXmlFromTemplate() {
    const tabName = tab?.name
    if (!template || !tabName) return;
    const xml = await templateToXml(template)
    setXml(xml, tabName)
    console.log(xml)
  }

  return (
    <div
      ref={outerRef}
      className={styles.outer}
      onMouseDown={onMouseDown}
      onMouseMove={onMouseMove}
      onMouseUp={onMouseUp}
      onContextMenu={(e) => { if (e.ctrlKey || e.metaKey) e.preventDefault() }}
    >
      {template && !xml ? (
        <div className={styles.emptyState} ref={innerRef} style={{ transform: `translate(${panX}px, ${panY}px) scale(${zoom / 100})` }}>
          <div className={`${styles.paper} ${styles.templatePaper}`}>
            <h2 className={styles.templateName}>{tab!.name}</h2>
            <p className={styles.paperHint}>{tab!.description || 'Template'}</p>
            {template.length === 0 && <p className={styles.paperHint}>No items yet.</p>}
            <ol className={styles.templateItems}>
              {template.map((item, i) => {
                const { kind, ...fields } = item
                return <li key={i}><b>{kind}</b> {JSON.stringify(fields)}</li>
              })}
            </ol>
          </div>
        </div>
      ) : !xml ? (
        <div className={styles.emptyState}>
          <div className={styles.paper}>
            <div style={{ display: 'block' }}>
              <p className={styles.paperHint}>{tab?.description || 'Open a MusicXML file or generate a score'}</p>
              <input className={styles.paperInput} placeholder={'Start a score...'} onKeyDown={handleBlankScoreInput} onMouseDown={(e) => e.stopPropagation()}></input>
            </div>
          </div>
        </div>
      ) : !ready || pages.length === 0 ? (
        <div className={styles.loading}>Rendering score…</div>
      ) : (
        <div
          ref={innerRef}
          className={styles.pages}
          style={{ transform: `translate(${panX}px, ${panY}px) scale(${zoom / 100})` }}
        >
          {pages.map((svg, i) => {
            // The paper is the engraving plus PAGE_PAD on every side. The tallest page fills the
            // viewport height; the others take the same share of it that their engraving takes of
            // the tallest one, so every page is drawn at the same scale.
            const size = pageMeta.sizes[i] ?? DEFAULT_SIZE
            const pad = 2 * PAGE_PAD
            return (
              <div
                key={i}
                className={styles.page}
                data-page-index={i}
                style={{ height: `calc((100% - ${pad}px) * ${size.h / pageMeta.maxH} + ${pad}px)` }}
              >
                <div
                  className={styles.sheet}
                  style={{ aspectRatio: `${size.w} / ${size.h}` }}
                  {...(visiblePages.has(i) ? { dangerouslySetInnerHTML: { __html: svg } } : {})}
                />
              </div>
            )
          })}
          <AppendBar />
        </div>
      )}
      <div ref={marqueeRef} className={styles.marquee} style={{ display: 'none' }} />
    </div>
  )
}
