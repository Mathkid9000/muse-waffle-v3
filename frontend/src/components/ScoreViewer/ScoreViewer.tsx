import { useEffect, useState, useRef } from 'react'
import { useVerovio } from '../../hooks/useVerovio'
import { useStore } from '../../store'
import styles from './ScoreViewer.module.css'
import AppendBar from './AppendBar'

// Matches Verovio measure elements by class (standard) or by the "m-" ID prefix (fallback).
// querySelectorAll deduplicates, so elements matching both selectors appear exactly once.
const MEASURE_SEL = 'g.measure'

export default function ScoreViewer() {
  const { ready, renderXml } = useVerovio()
  const currentXml = useStore((s) => s.currentXml)
  const zoom = useStore((s) => s.zoom)
  const setZoom = useStore((s) => s.setZoom)
  const selection = useStore((s) => s.selection)
  const setSelection = useStore((s) => s.setSelection)

  const [pages, setPages] = useState<string[]>([])
  const outerRef = useRef<HTMLDivElement>(null)
  const innerRef = useRef<HTMLDivElement>(null)
  const dragStartRef = useRef<number | null>(null)
  const zoomRef = useRef(zoom)
  zoomRef.current = zoom

  useEffect(() => {
    if (!ready || !currentXml) { setPages([]); return }
    const w = outerRef.current?.clientWidth ?? 1000
    setPages(renderXml(currentXml, w * 2.5, 40))
  }, [ready, currentXml])

  // Inject selection highlight rects directly into the SVG DOM.
  // No data-measure stamp needed: we query measures by selector at runtime.
  useEffect(() => {
    const inner = innerRef.current
    if (!inner) return
    inner.querySelectorAll('.mw-sel').forEach((el) => el.remove())
    if (!selection) return

    const measures = inner.querySelectorAll<SVGGraphicsElement>(MEASURE_SEL)
    const ns = 'http://www.w3.org/2000/svg'

    for (let m = selection.start; m <= selection.end; m++) {
      const g = measures[m - 1]
      if (!g) continue
      try {
        const bb = g.getBBox()
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
        g.insertBefore(rect, g.firstChild)
      } catch { /* getBBox unavailable off-screen */ }
    }
  }, [selection, pages])

  // Returns the 1-based measure index for a mouse event.
  // 1. Tries closest() — works when the click lands on a note/rest/barline child.
  // 2. Falls back to geometric hit-test — works when clicking empty staff space.
  function measureAt(e: React.MouseEvent): number | null {
    const target = e.target
    if (!(target instanceof Element)) return null
    const inner = innerRef.current
    if (!inner) return null

    const allMeasures = inner.querySelectorAll<SVGGraphicsElement>(MEASURE_SEL)
    if (allMeasures.length === 0) return null

    // Fast path: clicked element is inside a measure group
    const direct = target.closest<Element>(MEASURE_SEL)
    if (direct) {
      const idx = Array.from(allMeasures).indexOf(direct as SVGGraphicsElement)
      return idx >= 0 ? idx + 1 : null
    }

    // Geometric fallback: map pointer into SVG coordinates and test bounding boxes.
    // getScreenCTM() is unreliable when CSS zoom is applied; instead derive the transform
    // from getBoundingClientRect() (zoom-aware) and the SVG viewBox.
    let svg = target.closest('svg') as SVGSVGElement | null
    if (!svg && target instanceof SVGSVGElement) svg = target
    if (!svg) return null


    const svgRect = svg.getBoundingClientRect()
    const vb = (svg.querySelector(".definition-scale") as SVGSVGElement).viewBox.baseVal
    if (!vb || svgRect.width === 0 || svgRect.height === 0) return null

    const scaleX = vb.width / svgRect.width
    const scaleY = vb.height / svgRect.height
    const x = (e.clientX - svgRect.left) * scaleX + vb.x
    const y = (e.clientY - svgRect.top) * scaleY + vb.y

    for (let i = 0; i < allMeasures.length; i++) {
      if (!svg.contains(allMeasures[i])) continue
      try {
        const bb = allMeasures[i].getBBox()
        if (x >= bb.x && x <= bb.x + bb.width && y >= bb.y && y <= bb.y + bb.height) {
          return i + 1
        }
      } catch { }
    }

    return null
  }

  function onMouseDown(e: React.MouseEvent) {
    const m = measureAt(e)
    if (m === null) {
      setSelection(null)
      return
    }
    e.preventDefault()
    dragStartRef.current = m
    setSelection({ start: m, end: m })
  }

  function onMouseMove(e: React.MouseEvent) {
    if (dragStartRef.current === null || !(e.buttons & 1)) return
    const m = measureAt(e)
    if (m === null) return
    setSelection({
      start: Math.min(dragStartRef.current, m),
      end: Math.max(dragStartRef.current, m),
    })
  }

  function onMouseUp() {
    dragStartRef.current = null
  }

  function onWheel(e: React.WheelEvent) {
    if (!e.ctrlKey && !e.metaKey) return
    e.preventDefault()
    const outer = outerRef.current
    if (!outer) return

    const rect = outer.getBoundingClientRect()
    const cy = e.clientY - rect.top
    const prevZoom = zoomRef.current
    const delta = e.deltaY < 0 ? 10 : -10
    const newZoom = Math.max(25, Math.min(300, prevZoom + delta))
    if (newZoom === prevZoom) return

    const newScrollTop = (outer.scrollTop + cy) * (newZoom / prevZoom) - cy
    setZoom(newZoom)
    requestAnimationFrame(() => {
      if (outer) outer.scrollTop = Math.max(0, newScrollTop)
    })
  }

  return (
    <div
      ref={outerRef}
      className={styles.outer}
      onWheel={onWheel}
      onMouseDown={onMouseDown}
      onMouseMove={onMouseMove}
      onMouseUp={onMouseUp}
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
          style={{ zoom: zoom / 100 }}
        >
          {pages.map((svg, i) => (
            <div key={i} className={styles.page} dangerouslySetInnerHTML={{ __html: svg }} />
          ))}
          <AppendBar />
        </div>
      )}
    </div>
  )
}
