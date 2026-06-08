import { useState } from 'react'
import { usePlayback } from '../../hooks/usePlayback'
import { useStore } from '../../store'
import styles from './PlaybackBar.module.css'

export default function PlaybackBar() {
  const currentXml = useStore((s) => s.currentXml)
  const selection = useStore((s) => s.selection)
  const { playing, play, stop } = usePlayback()
  const [fromMeasure, setFromMeasure] = useState(1)

  // Sync fromMeasure with score selection when it changes
  const selStart = selection?.start ?? null
  if (selStart !== null && selStart !== fromMeasure && !playing) {
    setFromMeasure(selStart)
  }

  if (!currentXml) return null

  return (
    <div className={styles.bar}>
      <button
        className={playing ? styles.stopBtn : styles.playBtn}
        onClick={() => (playing ? stop() : play(fromMeasure))}
        title={playing ? 'Stop' : 'Play'}
      >
        {playing ? '■' : '▶'}
      </button>
      <label className={styles.label}>
        From measure
        <input
          type="number"
          min="1"
          value={fromMeasure}
          onChange={(e) => setFromMeasure(Math.max(1, +e.target.value))}
          disabled={playing}
          className={styles.input}
        />
      </label>
      {playing && <span className={styles.playing}>Playing…</span>}
    </div>
  )
}
