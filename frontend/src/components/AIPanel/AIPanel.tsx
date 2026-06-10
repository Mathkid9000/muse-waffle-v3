import { useState } from 'react'
import { useStore } from '../../store'
import { api } from '../../api/client'
import styles from './AIPanel.module.css'

type Tab = 'infill' | 'generate' | 'append'

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
  const streamBar = useStore((s) => s.streamBar)
  const totalBars = useStore((s) => s.totalBars)
  const setStreamProgress = useStore((s) => s.setStreamProgress)

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

  // Append params
  const [appendBars, setAppendBars] = useState(4)
  const [appendTemp, setAppendTemp] = useState(0.9)
  const [appendTopK, setAppendTopK] = useState(50)
  const [appendDevice, setAppendDevice] = useState('cpu')

  async function handleInfill() {
    if (!currentXml) return
    if (startMeasure > endMeasure) {
      setError('Start measure must be ≤ end measure.')
      return
    }
    setLoading(true)
    setStreamProgress(0, endMeasure - startMeasure + 1)
    try {
      for await (const evt of api.streamInfill({
        xml: currentXml,
        start_measure: startMeasure,
        end_measure: endMeasure,
        temperature,
        top_k: topK,
        device,
      })) {
        if (evt.xml) setXml(evt.xml)
        if (evt.bar != null && evt.total != null) setStreamProgress(evt.bar, evt.total)
        if (evt.done) break
      }
    } catch (e: unknown) {
      setError(e instanceof Error ? e.message : String(e))
    } finally {
      setLoading(false)
      setStreamProgress(null, null)
    }
  }

  async function handleGenerate() {
    if (selectedInstrs.length === 0) {
      setError('Select at least one instrument.')
      return
    }
    setLoading(true)
    setStreamProgress(0, numBars)
    try {
      for await (const evt of api.streamGenerate({
        instruments: selectedInstrs,
        key,
        time_sig: timeSig,
        tempo,
        num_bars: numBars,
        temperature: genTemp,
        top_k: genTopK,
        device: genDevice,
      })) {
        if (evt.xml) setXml(evt.xml)
        if (evt.bar != null && evt.total != null) setStreamProgress(evt.bar, evt.total)
        if (evt.done) break
      }
    } catch (e: unknown) {
      setError(e instanceof Error ? e.message : String(e))
    } finally {
      setLoading(false)
      setStreamProgress(null, null)
    }
  }

  async function handleAppend() {
    if (!currentXml) return
    setLoading(true)
    setStreamProgress(0, appendBars)
    try {
      for await (const evt of api.streamAppend({
        xml: currentXml,
        num_bars: appendBars,
        temperature: appendTemp,
        top_k: appendTopK,
        device: appendDevice,
      })) {
        if (evt.xml) setXml(evt.xml)
        if (evt.bar != null && evt.total != null) setStreamProgress(evt.bar, evt.total)
        if (evt.done) break
      }
    } catch (e: unknown) {
      setError(e instanceof Error ? e.message : String(e))
    } finally {
      setLoading(false)
      setStreamProgress(null, null)
    }
  }

  function toggleInstr(idx: number) {
    setSelectedInstrs((prev) =>
      prev.includes(idx) ? prev.filter((i) => i !== idx) : [...prev, idx]
    )
  }

  function progressLabel(verb: string) {
    if (!loading) return null
    if (streamBar != null && totalBars != null) {
      return `${verb} bar ${streamBar} of ${totalBars}…`
    }
    return `${verb}…`
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
        <button
          className={tab === 'append' ? styles.activeTab : styles.tab}
          onClick={() => setTab('append')}
        >
          Append
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
            {progressLabel('Filling') ?? 'Fill Measures'}
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
            {progressLabel('Generating') ?? 'Generate Score'}
          </button>
        </div>
      )}

      {tab === 'append' && (
        <div className={styles.section}>
          <p className={styles.hint}>
            Generate new measures at the end of the current score.
          </p>
          <label className={styles.label}>Bars to add
            <input type="number" min="1" max="64" value={appendBars}
              onChange={(e) => setAppendBars(Math.max(1, +e.target.value))} />
          </label>
          <label className={styles.label}>Temperature
            <input type="range" min="0.1" max="2" step="0.05"
              value={appendTemp} onChange={(e) => setAppendTemp(+e.target.value)} />
            <span>{appendTemp.toFixed(2)}</span>
          </label>
          <label className={styles.label}>Top-K
            <input type="range" min="1" max="200" step="1"
              value={appendTopK} onChange={(e) => setAppendTopK(+e.target.value)} />
            <span>{appendTopK}</span>
          </label>
          <label className={styles.label}>Device
            <select value={appendDevice} onChange={(e) => setAppendDevice(e.target.value)}>
              <option value="cpu">CPU</option>
              <option value="cuda">CUDA</option>
            </select>
          </label>
          <button
            className={styles.runBtn}
            onClick={handleAppend}
            disabled={loading || !currentXml}
          >
            {progressLabel('Appending') ?? 'Append Measures'}
          </button>
        </div>
      )}
    </div>
  )
}
