import { useEffect, useMemo, useState, useRef } from 'react'
import { useVerovio, getSharedToolkit, loadAndGetPageCount } from '../../hooks/useVerovio'
import { useStore } from '../../store'
import styles from './ScoreViewer.module.css'
import AppendBar from './AppendBar'

const MEASURE_SEL = 'g.measure'
const STAFF_SEL = 'g.staff'
const PAGE_SEL = '[data-page-index]'

// Layout constants — must match .pages padding / gap in the CSS module.
const PAD = 20
const GAP = 20
// Fallback page aspect ratio (width / height) for pages not yet rendered.
const DEFAULT_ASPECT = 1.4

const svgParser = new DOMParser()

interface PageInfo {
  count: number
  aspect: number
}

interface PageMeta {
  measureCounts: number[]
  /** Number of measures in all pages before this one (cumulative offset). */
  offsets: number[]
  /** width / height of each page's rendered SVG, for placeholder sizing. */
  aspects: number[]
  total: number
}

/** width / height of a rendered Verovio page, from its root <svg> attrs or viewBox. */
function svgAspect(doc: Document): number {
  const svg = doc.documentElement
  const w = parseFloat(svg.getAttribute('width') || '')
  const h = parseFloat(svg.getAttribute('height') || '')
  if (w > 0 && h > 0) return w / h
  const vb = svg.getAttribute('viewBox')
  if (vb) {
    const p = vb.split(/[\s,]+/).map(Number)
    if (p.length === 4 && p[3] > 0) return p[2] / p[3]
  }
  return DEFAULT_ASPECT
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
  const currentXml = useStore((s) => s.currentXml)
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

  // Per-page measure counts / offsets / aspect ratios. Selection maps a page's local
  // measure indices to global measure numbers via offsets; placeholders size from aspects.
  const pageMeta = useMemo<PageMeta>(() => {
    const cache = metaCacheRef.current
    const measureCounts: number[] = []
    const aspects: number[] = []
    const offsets: number[] = []
    let acc = 0
    let lastAspect = DEFAULT_ASPECT
    for (const svg of pages) {
      offsets.push(acc)
      if (!svg) {
        measureCounts.push(0)
        aspects.push(lastAspect)
        continue
      }
      let info = cache.get(svg)
      if (!info) {
        const doc = svgParser.parseFromString(svg, 'image/svg+xml')
        info = { count: doc.querySelectorAll(MEASURE_SEL).length, aspect: svgAspect(doc) }
        cache.set(svg, info)
      }
      measureCounts.push(info.count)
      aspects.push(info.aspect)
      lastAspect = info.aspect
      acc += info.count
    }
    return { measureCounts, offsets, aspects, total: acc }
  }, [pages])

  // Which pages currently intersect the viewport (plus a one-viewport buffer). Computed
  // analytically from the camera transform + page layout, since the pan/zoom camera uses
  // CSS transforms (an IntersectionObserver wouldn't reliably fire on transform changes).
  const visiblePages = useMemo<Set<number>>(() => {
    const set = new Set<number>()
    const n = pageMeta.aspects.length
    if (n === 0) return set
    if (viewportW === 0 || viewportH === 0) {
      for (let i = 0; i < n; i++) set.add(i)
      return set
    }
    const z = zoom / 100
    const contentH = Math.max(1, viewportH - 2 * PAD)
    const widths = pageMeta.aspects.map((a) => contentH * a)
    // Pages lay out left-to-right (flex-start) starting after the container padding.
    let x = PAD
    // Pages are full-height and vertically centered, so vertical visibility is shared.
    const sTop = panY + z * PAD
    const sBot = panY + z * (PAD + contentH)
    const vOK = sBot >= -viewportH && sTop <= 2 * viewportH
    const bufX = viewportW
    for (let i = 0; i < n; i++) {
      const sL = panX + z * x
      const sR = panX + z * (x + widths[i])
      if (vOK && sR >= -bufX && sL <= viewportW + bufX) set.add(i)
      x += widths[i] + GAP
    }
    return set
  }, [pageMeta, panX, panY, zoom, viewportW, viewportH])

  useEffect(() => {
    if (!ready || !currentXml) {
      cachedPagesRef.current = []
      metaCacheRef.current.clear()
      setPages([])
      return
    }

    const w = outerRef.current?.clientWidth ?? 1000
    const { streamMode, selection: sel } = useStore.getState()
    const tk = getSharedToolkit()

    if (!tk || !streamMode || cachedPagesRef.current.length === 0) {
      // Full render: initial load, not streaming, or cache empty (first streaming event)
      const allPages = renderXml(currentXml, w * 2.5, 40, scoreData?.numInstruments ?? 1)
      cachedPagesRef.current = allPages
      setPages(allPages)
      return
    }

    // Incremental: loadAndGetPageCount applies options + loads data into the shared tk
    const prevCount = cachedPagesRef.current.length
    const newCount = loadAndGetPageCount(currentXml, w * 2.5, 40, scoreData?.numInstruments ?? 1)
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
  }, [ready, currentXml, streamMode, redrawNonce])

  // Inject selection highlight rects directly into the SVG DOM.
  // When selection.parts is set, highlight only those staff rows within each measure.
  // Only mounted (visible) pages carry measure elements; measures on off-screen pages
  // are skipped and re-applied by this effect once those pages scroll into view.
  useEffect(() => {
    const inner = innerRef.current
    if (!inner) return
    inner.querySelectorAll('.mw-sel').forEach((el) => el.remove())
    if (!selection) return

    const ns = 'http://www.w3.org/2000/svg'

    function addRect(parent: SVGGraphicsElement, bb: DOMRect | { x: number; y: number; width: number; height: number }) {
      const rect = document.createElementNS(ns, 'rect')
      rect.setAttribute('x', String(bb.x))
      rect.setAttribute('y', String(bb.y))
      rect.setAttribute('width', String(bb.width))
      rect.setAttribute('height', String(bb.height))
      rect.setAttribute('fill', 'rgba(26,115,232,0.18)')
      rect.setAttribute('stroke', 'rgba(26,115,232,0.7)')
      rect.setAttribute('stroke-width', '4')
      rect.setAttribute('class', 'mw-sel')
      rect.setAttribute('pointer-events', 'none')
      parent.insertBefore(rect, parent.firstChild)
    }

    for (let m = selection.start; m <= selection.end; m++) {
      const pageIdx = pageOfMeasure(pageMeta, m)
      if (pageIdx < 0) continue
      const pageEl = inner.querySelector<HTMLElement>(`[data-page-index="${pageIdx}"]`)
      if (!pageEl) continue
      const measures = pageEl.querySelectorAll<SVGGraphicsElement>(MEASURE_SEL)
      if (measures.length === 0) continue // page not mounted
      const g = measures[m - pageMeta.offsets[pageIdx] - 1]
      if (!g) continue
      try {
        if (!selection.parts) {
          // Highlight entire measure
          addRect(g, g.getBBox())
        } else {
          // Highlight only selected staves within this measure
          const staffEls = g.querySelectorAll<SVGGraphicsElement>(STAFF_SEL)
          for (const partIdx of selection.parts) {
            const staffEl = staffEls[partIdx]
            if (!staffEl) continue
            try {
              addRect(g, staffEl.getBBox())
            } catch { /* off-screen */ }
          }
        }
      } catch { /* getBBox unavailable off-screen */ }
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

    for (let i = 0; i < measures.length; i++) {
      try {
        const mBB = measures[i].getBBox()
        if (x >= mBB.x && x <= mBB.x + mBB.width && y >= mBB.y && y <= mBB.y + mBB.height) {
          // Found the measure — try to narrow down to a specific staff
          const staffEls = measures[i].querySelectorAll<SVGGraphicsElement>(STAFF_SEL)
          let staffIdx: number | null = null
          for (let s = 0; s < staffEls.length; s++) {
            try {
              const sBB = staffEls[s].getBBox()
              if (y >= sBB.y && y <= sBB.y + sBB.height) {
                staffIdx = s
                break
              }
            } catch { /* off-screen */ }
          }
          return { measure: offset + i + 1, staffIdx }
        }
      } catch { }
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
          cache.push({ measure: offset + mLocal + 1, staff: sIdx, rect: s.getBoundingClientRect() })
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

  return (
    <div
      ref={outerRef}
      className={styles.outer}
      onMouseDown={onMouseDown}
      onMouseMove={onMouseMove}
      onMouseUp={onMouseUp}
      onContextMenu={(e) => { if (e.ctrlKey || e.metaKey) e.preventDefault() }}
    >
      {!currentXml ? (
        <div className={styles.emptyState}>
          <div className={styles.paper}>
            <p className={styles.paperHint}>Open a MusicXML file or generate a score</p>
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
          {pages.map((svg, i) => (
            <div
              key={i}
              className={styles.page}
              data-page-index={i}
              style={{ aspectRatio: String(pageMeta.aspects[i] ?? DEFAULT_ASPECT) }}
              {...(visiblePages.has(i) ? { dangerouslySetInnerHTML: { __html: svg } } : {})}
            />
          ))}
          <AppendBar />
        </div>
      )}
      <div ref={marqueeRef} className={styles.marquee} style={{ display: 'none' }} />
    </div>
  )
}
