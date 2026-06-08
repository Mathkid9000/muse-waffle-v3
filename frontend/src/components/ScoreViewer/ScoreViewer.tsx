import { useEffect, useState, useRef } from 'react'
import { useVerovio } from '../../hooks/useVerovio'
import { useStore } from '../../store'
import styles from './ScoreViewer.module.css'

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

  // Re-render Verovio only when XML changes; zoom is handled purely via CSS
  useEffect(() => {
    if (!ready || !currentXml) {
      setPages([])
      return
    }
    const w = outerRef.current?.clientWidth ?? 1000
    setPages(renderXml(currentXml, w * 2.5, 40))
  }, [ready, currentXml])

  // Stamp sequential data-measure attributes by querying the live DOM.
  // This is more reliable than regex on the SVG string.
  useEffect(() => {
    const inner = innerRef.current
    if (!inner) return
    inner.querySelectorAll<Element>('g.measure').forEach((g, i) =>
      g.setAttribute('data-measure', String(i + 1))
    )
  }, [pages])

  // Inject/refresh selection highlight rects after measure attributes are set
  useEffect(() => {
    const inner = innerRef.current
    if (!inner) return
    inner.querySelectorAll('.mw-sel').forEach((el) => el.remove())
    if (!selection) return
    const ns = 'http://www.w3.org/2000/svg'
    for (let m = selection.start; m <= selection.end; m++) {
      const g = inner.querySelector(`[data-measure="${m}"]`) as SVGGraphicsElement | null
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

  function measureAt(target: EventTarget | null): number | null {
    if (!(target instanceof Element)) return null
    const g = target.closest('[data-measure]')
    if (!g) return null
    const n = parseInt(g.getAttribute('data-measure') ?? '', 10)
    return isNaN(n) ? null : n
  }

  function onMouseDown(e: React.MouseEvent) {
    const m = measureAt(e.target)
    if (m === null) return
    e.preventDefault()
    dragStartRef.current = m
    setSelection({ start: m, end: m })
  }

  function onMouseMove(e: React.MouseEvent) {
    if (dragStartRef.current === null || !(e.buttons & 1)) return
    const m = measureAt(e.target)
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

    // Keep the content point under the cursor stationary after zoom.
    // scrollTop is in "zoomed layout pixels"; dividing by zoom gives natural coords.
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
        </div>
      )}
    </div>
  )
}
