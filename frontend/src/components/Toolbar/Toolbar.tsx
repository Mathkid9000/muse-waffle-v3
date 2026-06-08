import { useRef } from 'react'
import { useStore } from '../../store'
import styles from './Toolbar.module.css'

export default function Toolbar() {
  const fileInputRef = useRef<HTMLInputElement>(null)
  const currentXml = useStore((s) => s.currentXml)
  const zoom = useStore((s) => s.zoom)
  const setXml = useStore((s) => s.setXml)
  const setError = useStore((s) => s.setError)
  const setZoom = useStore((s) => s.setZoom)

  function handleFileChange(e: React.ChangeEvent<HTMLInputElement>) {
    const file = e.target.files?.[0]
    if (!file) return
    const reader = new FileReader()
    reader.onload = (ev) => {
      if (typeof ev.target?.result === 'string') setXml(ev.target.result)
    }
    reader.onerror = () => setError('Failed to read file.')
    reader.readAsText(file)
    e.target.value = ''
  }

  function handleDownload() {
    if (!currentXml) return
    const blob = new Blob([currentXml], { type: 'application/xml' })
    const url = URL.createObjectURL(blob)
    const a = document.createElement('a')
    a.href = url
    a.download = 'score.musicxml'
    a.click()
    URL.revokeObjectURL(url)
  }

  return (
    <header className={styles.toolbar}>
      <span className={styles.brand}>Muse Waffle</span>
      <div className={styles.actions}>
        <button className={styles.btn} onClick={() => fileInputRef.current?.click()}>
          Open MusicXML
        </button>
        <button className={styles.btn} onClick={handleDownload} disabled={!currentXml}>
          Download
        </button>
        <div className={styles.zoom}>
          <button className={styles.zoomBtn} onClick={() => setZoom(zoom - 10)} disabled={zoom <= 25} title="Zoom out (Ctrl+Scroll)">−</button>
          <span className={styles.zoomLabel} title="Double-click to reset" onDoubleClick={() => setZoom(100)} style={{ cursor: 'default' }}>{zoom}%</span>
          <button className={styles.zoomBtn} onClick={() => setZoom(zoom + 10)} disabled={zoom >= 300} title="Zoom in (Ctrl+Scroll)">+</button>
        </div>
      </div>
      <input
        ref={fileInputRef}
        type="file"
        accept=".xml,.musicxml,.mxl"
        style={{ display: 'none' }}
        onChange={handleFileChange}
      />
    </header>
  )
}
