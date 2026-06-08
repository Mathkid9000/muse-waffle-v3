import { useState } from 'react'
import { useStore } from '../../store'
import { api } from '../../api/client'
import styles from './AIPanel.module.css'

const INSTR_NAMES = [
  'Other', 'Piano', 'Harpsichord', 'Organ', 'Accordion', 'Guitar', 'Bass Guitar',
  'Violin', 'Viola', 'Cello', 'Double Bass', 'Harp', 'Flute', 'Oboe',
  'Clarinet', 'Bassoon', 'Saxophone', 'Horn', 'Trumpet', 'Trombone', 'Tuba',
  'Percussion', 'Timpani', 'Celesta', 'Soprano', 'Mezzo-Soprano', 'Alto',
  'Tenor', 'Baritone', 'Bass Voice', 'Voice', 'Choir',
]

const KEY_LABELS = [
  '6♭', '5♭', '4♭', '3♭', '2♭', '1♭', 'C', '1♯', '2♯', '3♯', '4♯', '5♯', '6♯',
]

const TIME_SIG_LABELS = ['2/4', '3/4', '4/4', '6/8', '3/8', '12/8']

const TEMPO_LABELS = [
  '≈20', '≈40', '≈52', '≈63', '≈72', '≈84', '≈96', '≈108',
  '≈116', '≈126', '≈138', '≈152', '≈168', '≈192', '≈216', '≈256',
]

type Tab = 'infill' | 'generate'

export default function AIPanel() {
  const [tab, setTab] = useState<Tab>('infill')
  const currentXml = useStore((s) => s.currentXml)
  const loading = useStore((s) => s.loading)
  const error = useStore((s) => s.error)
  const selection = useStore((s) => s.selection)
  const setSelection = useStore((s) => s.setSelection)
  const setXml = useStore((s) => s.setXml)
  const setLoading = useStore((s) => s.setLoading)
  const setError = useStore((s) => s.setError)

  // Infill range — kept in sync with score selection
  const startMeasure = selection?.start ?? 1
  const endMeasure = selection?.end ?? 1
  function setStartMeasure(v: number) { setSelection({ start: v, end: Math.max(v, endMeasure) }) }
  function setEndMeasure(v: number) { setSelection({ start: Math.min(startMeasure, v), end: v }) }
  const [temperature, setTemperature] = useState(0.9)
  const [topK, setTopK] = useState(50)
  const [device, setDevice] = useState('cpu')

  // Generate params
  const [selectedInstrs, setSelectedInstrs] = useState<number[]>([1])
  const [key, setKey] = useState(6)
  const [timeSig, setTimeSig] = useState(2)
  const [tempo, setTempo] = useState(5)
  const [numBars, setNumBars] = useState(16)
  const [genTemp, setGenTemp] = useState(0.9)
  const [genTopK, setGenTopK] = useState(50)
  const [genDevice, setGenDevice] = useState('cpu')

  async function handleInfill() {
    if (!currentXml) return
    if (startMeasure > endMeasure) {
      setError('Start measure must be ≤ end measure.')
      return
    }
    setLoading(true)
    try {
      const res = await api.infill({
        xml: currentXml,
        start_measure: startMeasure,
        end_measure: endMeasure,
        temperature,
        top_k: topK,
        device,
      })
      setXml(res.xml)
    } catch (e: unknown) {
      setError(e instanceof Error ? e.message : String(e))
    } finally {
      setLoading(false)
    }
  }

  async function handleGenerate() {
    if (selectedInstrs.length === 0) {
      setError('Select at least one instrument.')
      return
    }
    setLoading(true)
    try {
      const res = await api.generate({
        instruments: selectedInstrs,
        key,
        time_sig: timeSig,
        tempo,
        num_bars: numBars,
        temperature: genTemp,
        top_k: genTopK,
        device: genDevice,
      })
      setXml(res.xml)
    } catch (e: unknown) {
      setError(e instanceof Error ? e.message : String(e))
    } finally {
      setLoading(false)
    }
  }

  function toggleInstr(idx: number) {
    setSelectedInstrs((prev) =>
      prev.includes(idx) ? prev.filter((i) => i !== idx) : [...prev, idx]
    )
  }

  return (
    <div className={styles.panel}>
      <div className={styles.tabs}>
        <button
          className={tab === 'infill' ? styles.activeTab : styles.tab}
          onClick={() => setTab('infill')}
        >
          AI Fill
        </button>
        <button
          className={tab === 'generate' ? styles.activeTab : styles.tab}
          onClick={() => setTab('generate')}
        >
          Generate New
        </button>
      </div>

      {error && (
        <div className={styles.error} onClick={() => setError(null)}>
          {error}
        </div>
      )}

      {tab === 'infill' && (
        <div className={styles.section}>
          <p className={styles.hint}>
            Enter the measure range to fill with the AI model.
          </p>
          <div className={styles.row}>
            <label className={styles.label}>
              From
              <input
                type="number" min="1" value={startMeasure}
                onChange={(e) => setStartMeasure(Math.max(1, +e.target.value))}
              />
            </label>
            <label className={styles.label}>
              To
              <input
                type="number" min="1" value={endMeasure}
                onChange={(e) => setEndMeasure(Math.max(1, +e.target.value))}
              />
            </label>
          </div>
          <label className={styles.label}>Temperature
            <input type="range" min="0.1" max="2" step="0.05"
              value={temperature} onChange={(e) => setTemperature(+e.target.value)} />
            <span>{temperature.toFixed(2)}</span>
          </label>
          <label className={styles.label}>Top-K
            <input type="range" min="1" max="200" step="1"
              value={topK} onChange={(e) => setTopK(+e.target.value)} />
            <span>{topK}</span>
          </label>
          <label className={styles.label}>Device
            <select value={device} onChange={(e) => setDevice(e.target.value)}>
              <option value="cpu">CPU</option>
              <option value="cuda">CUDA</option>
            </select>
          </label>
          <button
            className={styles.runBtn}
            onClick={handleInfill}
            disabled={loading || !currentXml}
          >
            {loading ? 'Running…' : 'Fill Measures'}
          </button>
        </div>
      )}

      {tab === 'generate' && (
        <div className={styles.section}>
          <fieldset className={styles.fieldset}>
            <legend>Instruments</legend>
            <div className={styles.instrGrid}>
              {INSTR_NAMES.map((name, idx) => (
                <label key={idx} className={styles.instrLabel}>
                  <input
                    type="checkbox"
                    checked={selectedInstrs.includes(idx)}
                    onChange={() => toggleInstr(idx)}
                  />
                  {name}
                </label>
              ))}
            </div>
          </fieldset>
          <label className={styles.label}>Key
            <select value={key} onChange={(e) => setKey(+e.target.value)}>
              {KEY_LABELS.map((k, i) => <option key={i} value={i}>{k}</option>)}
            </select>
          </label>
          <label className={styles.label}>Time Signature
            <select value={timeSig} onChange={(e) => setTimeSig(+e.target.value)}>
              {TIME_SIG_LABELS.map((t, i) => <option key={i} value={i}>{t}</option>)}
            </select>
          </label>
          <label className={styles.label}>Tempo (BPM)
            <select value={tempo} onChange={(e) => setTempo(+e.target.value)}>
              {TEMPO_LABELS.map((t, i) => <option key={i} value={i}>{t}</option>)}
            </select>
          </label>
          <label className={styles.label}>Bars
            <input type="number" min="1" max="128" value={numBars}
              onChange={(e) => setNumBars(+e.target.value)} />
          </label>
          <label className={styles.label}>Temperature
            <input type="range" min="0.1" max="2" step="0.05"
              value={genTemp} onChange={(e) => setGenTemp(+e.target.value)} />
            <span>{genTemp.toFixed(2)}</span>
          </label>
          <label className={styles.label}>Top-K
            <input type="range" min="1" max="200" step="1"
              value={genTopK} onChange={(e) => setGenTopK(+e.target.value)} />
            <span>{genTopK}</span>
          </label>
          <label className={styles.label}>Device
            <select value={genDevice} onChange={(e) => setGenDevice(e.target.value)}>
              <option value="cpu">CPU</option>
              <option value="cuda">CUDA</option>
            </select>
          </label>
          <button
            className={styles.runBtn}
            onClick={handleGenerate}
            disabled={loading}
          >
            {loading ? 'Generating…' : 'Generate Score'}
          </button>
        </div>
      )}
    </div>
  )
}
