import { useState } from 'react'
import { useStore } from '../../store'
import { api } from '../../api/client'
import styles from './AIPanel.module.css'
import { promptAnthropic } from '../../api/claudeTools'
import type {
  EditClefRequest,
  EditTimeSigRequest,
  EditKeySigRequest,
  EditTempoRequest,
  EditInstrRequest,
  EditRmStaffRequest,
  EditRmMeasureRequest,
} from '../../api/client'

type Tab = 'infill' | 'generate' | 'append'



// Claude tool string enums → backend indices. Order must match the enums in
// claudeTools.ts (and the backend index conventions). TIME_SIG_LABELS above already
// mirrors the time-signature enum, so it doubles as the lookup for new_time.

interface ThoughtProcessEvent {
  title: string
  input: string
}

export default function AIPanel() {
  const [tab, setTab] = useState<Tab>('infill')
  const currentXml = useStore((s) => s.currentXml)
  const scoreData = useStore((s) => s.scoreData)
  const loading = useStore((s) => s.loading)
  const error = useStore((s) => s.error)
  const selection = useStore((s) => s.selection)
  const setSelection = useStore((s) => s.setSelection)
  const setXml = useStore((s) => s.setXml)
  const setScoreData = useStore((s) => s.setScoreData)
  const setLoading = useStore((s) => s.setLoading)
  const setError = useStore((s) => s.setError)
  const streamBar = useStore((s) => s.streamBar)
  const totalBars = useStore((s) => s.totalBars)
  const setStreamProgress = useStore((s) => s.setStreamProgress)
  const setStreamMode = useStore((s) => s.setStreamMode)

  // Infill range — kept in sync with score selection
  const startMeasure = selection?.start ?? 1
  const endMeasure = selection?.end ?? 1
  const selectedParts = selection?.parts ?? null
  function setStartMeasure(v: number) { setSelection({ start: v, end: Math.max(v, endMeasure), parts: selectedParts }) }
  function setEndMeasure(v: number) { setSelection({ start: Math.min(startMeasure, v), end: v, parts: selectedParts }) }
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

  // Claude prompting
  const [promptStatus, setPromptStatus] = useState('')
  const [promptThoughtProcess, setPromptThoughtProcess] = useState<ThoughtProcessEvent[]>([])

  async function handleInfill() {
    if (!currentXml) return
    if (startMeasure > endMeasure) {
      setError('Start measure must be ≤ end measure.')
      return
    }
    setLoading(true)
    setStreamMode('infill')
    setStreamProgress(0, endMeasure - startMeasure + 1)
    try {
      for await (const evt of api.streamInfill({
        xml: currentXml,
        start_measure: startMeasure,
        end_measure: endMeasure,
        parts: selectedParts,
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
      setStreamMode(null)
      setStreamProgress(null, null)
    }
  }

  async function handleGenerate() {
    if (selectedInstrs.length === 0) {
      setError('Select at least one instrument.')
      return
    }
    setLoading(true)
    setStreamMode('generate')
    setStreamProgress(0, numBars)

    setScoreData({  numInstruments: 0 })
    let newScoreData = {  numInstruments: 0 }

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
        if (evt.instr_num) {
          newScoreData.numInstruments = evt.instr_num
          console.log("NUM INSTRUMENT:", newScoreData)
          setScoreData(newScoreData)
        }
        if (evt.bar != null && evt.total != null) setStreamProgress(evt.bar, evt.total)
        if (evt.done) break
      }
    } catch (e: unknown) {
      setError(e instanceof Error ? e.message : String(e))
    } finally {
      setLoading(false)
      setStreamMode(null)
      setStreamProgress(null, null)
    }
  }

  async function handleAppend() {
    if (!currentXml) return
    setLoading(true)
    setStreamMode('append')
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
      setStreamMode(null)
      setStreamProgress(null, null)
    }
  }

  async function handleEdit() {
    setLoading(true)
    setPromptStatus('Prompting...')

    setPromptThoughtProcess([])

    try {
      const promptText = document.querySelector("#edit-input-span")?.textContent
      console.log(promptText)

      if (!promptText) return

      // Each update must be a NEW array so React re-renders — mutating the existing
      // array in place and re-setting the same reference is a no-op to React, which is
      // why logs previously only appeared once some later state change forced a render.
      setPromptThoughtProcess([
        { title: "---------------------------", input: "" },
        { title: "Beginning prompt...", input: promptText },
      ])
      const response = await promptAnthropic(promptText, currentXml!, (title: string, input: string) => {
        setPromptThoughtProcess((prev) => [...prev, { title, input }])
      }, selection)
      if (response.length === 0) {
        setPromptStatus('Nothing was returned')
        return
      }

      const edits = response[0]
      const usage = [ response[1], response[2] ]
      console.log('Prompt usage: ', usage)

      setPromptStatus(`Found ${edits.length} edits...`)

      // Apply each edit in sequence, threading the updated XML through so later
      // edits build on earlier ones.
      let xml = currentXml!
      let applied = 0
      for (const { tool, input } of edits) {
        const args = input as Record<string, unknown>
        switch (tool) {
          case "edit_clef": {
            const new_clef = CLEF_ENUM.indexOf(String(args.new_clef))
            if (new_clef < 0) break // "none"/unknown — backend has no clef-removal case
            const req: EditClefRequest = {
              xml,
              measure_num: Number(args.measure_num),
              staff_num: Number(args.staff_num),
              new_clef,
            }
            xml = (await api.editClef(req)).xml
            applied++
            break
          }
          case "edit_time_sig": {
            const new_time = TIME_SIG_LABELS.indexOf(String(args.new_time))
            if (new_time < 0) break
            const req: EditTimeSigRequest = {
              xml,
              measure_num: Number(args.measure_num),
              new_time,
            }
            xml = (await api.editTimeSig(req)).xml
            applied++
            break
          }
          case "edit_key_sig": {
            const req: EditKeySigRequest = {
              xml,
              measure_num: Number(args.measure_num),
              new_tonic: String(args.tonic),
              new_mode: String(args.mode),
            }
            xml = (await api.editKeySig(req)).xml
            applied++
            break
          }
          case "edit_tempo": {
            const req: EditTempoRequest = {
              xml,
              measure_num: Number(args.measure_num),
              new_tempo: Number(args.new_tempo),
            }
            xml = (await api.editTempo(req)).xml
            applied++
            break
          }
          case "edit_add_instrument": {
            const instr = INSTR_ENUM.indexOf(String(args.new_instrument))
            if (instr < 0) break
            const req: EditInstrRequest = { xml, instr }
            xml = (await api.editInstr(req)).xml
            applied++
            break
          }
          case "edit_delete_staff": {
            const req: EditRmStaffRequest = {
              xml,
              staff_num: Number(args.staff_num),
            }
            xml = (await api.editRmStaff(req)).xml
            applied++
            break
          }
          case "edit_delete_measures": {
            const req: EditRmMeasureRequest = {
              xml,
              measure_start_num: Number(args.measure_start_num),
              measure_end_num: Number(args.measure_end_num),
            }
            xml = (await api.editRmMeasure(req)).xml
            applied++
            break
          }
        }
      }
      if (applied > 0) setXml(xml)
    } catch (e: unknown) {
      setError(e instanceof Error ? e.message : String(e))
    } finally {
      setLoading(false)
      setPromptStatus('')
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
    return `${verb}...`
  }

  return (
    <>
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
              Click a staff in the score to select it, then drag to extend the measure range.
            </p>
            <p className={styles.hint}>
              {selectedParts
                ? `Staves: ${selectedParts.map((p) => p + 1).join(', ')}`
                : 'Staves: all'}
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
      <div className={styles.panel} style={ { borderTop: "1px solid grey" } }>
        <div className={styles.section}>
          <span id="edit-input-span" contentEditable style={ { border: "1px solid grey" } }>
            ...
          </span>
          <button
            className={styles.runBtn}
            onClick={handleEdit}
            disabled={loading || !currentXml}
          >
            {promptStatus === '' ? 'Edit' : promptStatus}
          </button>
          <div style={{ marginLeft: "5px" }}>
            {
              promptThoughtProcess.toReversed().map(event => (
                <div style={{ marginBottom: "10px" }}>
                  <span style={{ fontSize: "14px", display: "block", fontWeight: 600 }}>{event.title}</span>
                  <span style={{ fontSize: "11px", marginLeft: "20px", display: "block", lineHeight: "15px", whiteSpace: "pre-line" }}>{event.input}</span>
                </div>
              ))
            }
          </div>
        </div>
      </div>
    </>
  )
}
