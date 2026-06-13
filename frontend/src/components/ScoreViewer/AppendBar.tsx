import { useState } from 'react'
import { useStore } from '../../store'
import { api } from '../../api/client'
import styles from './AppendBar.module.css'

export default function AppendBar() {
  const currentXml = useStore((s) => s.currentXml)
  const loading = useStore((s) => s.loading)
  const setXml = useStore((s) => s.setXml)
  const setLoading = useStore((s) => s.setLoading)
  const setError = useStore((s) => s.setError)

  const [expanded, setExpanded] = useState(false)
  const [numBars, setNumBars] = useState(4)
  const [temperature, setTemperature] = useState(0.9)
  const [topK, setTopK] = useState(50)
  const [device, setDevice] = useState('cpu')

  async function handleAppend() {
    if (!currentXml) return
    setLoading(true)
    try {
      for await (const evt of api.streamAppend({
        xml: currentXml,
        num_bars: numBars,
        temperature,
        top_k: topK,
        device,
      })) {
        if (evt.xml) setXml(evt.xml)
        if (evt.done) break
      }
      setExpanded(false)
    } catch (e: unknown) {
      setError(e instanceof Error ? e.message : String(e))
    } finally {
      setLoading(false)
    }
  }

  return (
    <div className={styles.appendBar}>
      {!expanded ? (
        <button
          className={styles.trigger}
          onClick={() => setExpanded(true)}
          disabled={loading}
          title="Generate new measures at the end of this score"
        >
          + Continue score
        </button>
      ) : (
        <div className={styles.form}>
          <div className={styles.formRow}>
            <label className={styles.formLabel}>
              Bars
              <input
                type="number"
                min={1}
                max={32}
                value={numBars}
                onChange={(e) => setNumBars(Math.max(1, Math.min(32, +e.target.value)))}
                className={styles.numInput}
              />
            </label>
            <label className={styles.formLabel}>
              Temp
              <input
                type="number"
                min={0.1}
                max={2}
                step={0.05}
                value={temperature}
                onChange={(e) => setTemperature(+e.target.value)}
                className={styles.numInput}
              />
            </label>
            <label className={styles.formLabel}>
              Top-K
              <input
                type="number"
                min={1}
                max={200}
                value={topK}
                onChange={(e) => setTopK(+e.target.value)}
                className={styles.numInput}
              />
            </label>
            <label className={styles.formLabel}>
              Device
              <select
                value={device}
                onChange={(e) => setDevice(e.target.value)}
                className={styles.select}
              >
                <option value="cpu">CPU</option>
                <option value="cuda">CUDA</option>
              </select>
            </label>
          </div>
          <div className={styles.formActions}>
            <button
              className={styles.cancelBtn}
              onClick={() => setExpanded(false)}
              disabled={loading}
            >
              Cancel
            </button>
            <button
              className={styles.generateBtn}
              onClick={handleAppend}
              disabled={loading || !currentXml}
            >
              {loading ? 'Generating…' : `Generate ${numBars} bars`}
            </button>
          </div>
        </div>
      )}
    </div>
  )
}
