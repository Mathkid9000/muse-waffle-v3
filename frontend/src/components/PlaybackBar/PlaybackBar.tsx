import { useState } from 'react'
import { usePlayback } from '../../hooks/usePlayback'
import { useStore } from '../../store'
import styles from './PlaybackBar.module.css'

export default function PlaybackBar() {
  const currentXml = useStore((s) => s.currentXml)
  const selection = useStore((s) => s.selection)
  const { playing, loading, play, stop, volume, setVolume } = usePlayback()
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
        disabled={loading}
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
      <label className={styles.label}>
        Vol
        <input
          type="range"
          min="0"
          max="1"
          step="0.01"
          value={volume}
          onChange={(e) => setVolume(+e.target.value)}
          className={styles.volumeSlider}
        />
      </label>
      {loading && <span className={styles.loading}>Loading…</span>}
      {playing && <span className={styles.playing}>Playing…</span>}
    </div>
  )
}
