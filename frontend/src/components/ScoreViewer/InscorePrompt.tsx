import sendImage from '../../assets/send.svg'
import styles from './InscorePrompt.module.css'
import { useState } from 'react'
import { promptAnthropic } from '../../api/claudeTools'
import type { MeasureSelection } from '../../store'
import { applyClaudeEdit } from '../../api/applyClaudeEdit'

/*

const promptContainer = document.createElement('div')
promptContainer.className = styles.promptContainer
promptContainer.addEventListener('mousedown', (e) => e.stopPropagation())

const promptInput = document.createElement('textarea')
promptInput.placeholder = "Type something..."
promptInput.className = styles.promptInput
promptInput.value = inscorePrompt ?? ''
promptInput.rows = 1
promptInput.onchange = (e) => {
    setInscorePrompt(e.target!.value)
}
promptContainer.appendChild(promptInput)

const promptButton = document.createElement('button')
promptButton.className = styles.promptButton
promptContainer.appendChild(promptButton)

const promptButtonImg = document.createElement('img')
promptButtonImg.src = sendImage
promptButtonImg.className = styles.promptButtonImg
promptButton.appendChild(promptButtonImg)
promptButton.onclick = (e) => {
    const prompt = inscorePrompt
    setInscorePrompt(null)
    console.log(prompt)
}

*/

/*

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

*/

const TEST_OBJ = JSON.parse(`
[
    {
        "title": "-----------------",
        "input": "",
        "tokensIn": 0,
        "tokensOut": 0
    },
    {
        "title": "Beginning prompt...",
        "input": "Change key to F#minor",
        "tokensIn": 0,
        "tokensOut": 0
    },
    {
        "title": "Selected 1 tools...",
        "input": " * edit_key_sig",
        "tokensIn": 839,
        "tokensOut": 39
    },
    {
        "title": "Queried for key_signature...",
        "input": "Key signature by measure (1-based; listed at m1 and every change):whole score: m1=2 flats, m4=3 flats",
        "tokensIn": 541,
        "tokensOut": 30.5
    },
    {
        "title": "Queried for selection...",
        "input": "The user has not selected anything on the score.",
        "tokensIn": 541,
        "tokensOut": 30.5
    },
    {
        "title": "Found 1 edits...",
        "input": " * edit_key_sig",
        "tokensIn": 1202,
        "tokensOut": 118
    },
    {
        "title": "Thinking complete...",
        "input": "3123 in / 218 out",
        "tokensIn": 0,
        "tokensOut": 0
    },
    {
        "title": "Edit made: edit_key_sig (1/1)",
        "input": "measure_num: 1, tonic: F#, mode: minor",
        "tokensIn": 0,
        "tokensOut": 0
    }
]
    `) as ThoughtProcessEvent[]

interface ThoughtProcessEvent {
  title: string
  input: string
  tokensIn: number
  tokensOut: number
}

interface InscorePromptProps {
    getXml: () => string
    getSelection: () => MeasureSelection
    setXml: (xml: string) => void
}

export default function InscorePrompt({ getXml, getSelection, setXml }: InscorePromptProps) {
    const [ inscorePrompt, setInscorePrompt ] = useState<string | null>(null)
    const [ promptState, setPromptState ] = useState<('input' | 'eval' | 'done')>('input')
    const [ promptThoughtProcess, setPromptThoughtProcess ] = useState<ThoughtProcessEvent[]>([]) 
    const [ promptStatus, setPromptStatus ] = useState<string>('') 

    function handleSubmit(prompt: string | null) {
        if (!prompt) return

        console.log(prompt)
        setInscorePrompt(null)
        setPromptState('eval')

        // console.log(getXml(), getSelection())
        //promptWithThoughProcess(prompt)
        console.log(TEST_OBJ)
        setPromptThoughtProcess(TEST_OBJ)
    }

    async function promptWithThoughProcess(prompt: string) {
        const currentXml = getXml()
        const selection = getSelection()

        // init thoughts
        setPromptThoughtProcess([
            { title: '-----------------', input: '', tokensIn: 0, tokensOut: 0 },
            { title: 'Beginning prompt...', input: prompt, tokensIn: 0, tokensOut: 0 }
        ])

        let mirror = [
            { title: '-----------------', input: '', tokensIn: 0, tokensOut: 0 },
            { title: 'Beginning prompt...', input: prompt, tokensIn: 0, tokensOut: 0 }
        ]

        // trigger prompting
        const response = await promptAnthropic(prompt, currentXml!, (title: string, input: string, tokensIn: number, tokensOut: number) => {
            setPromptThoughtProcess((prev) => [...prev, { title, input, tokensIn, tokensOut }])
            mirror.push({ title, input, tokensIn, tokensOut })
        }, selection)
        if (response.length === 0) {
            setPromptStatus('Nothing was returned')
            setPromptState('done')
            return
        }

        const edits = response[0]
        const usage = [ response[1], response[2] ]
        
        let xml = currentXml
        let applied = 0
        for (const { tool, input } of edits) {
            xml = await applyClaudeEdit(tool, input, xml)
            setPromptThoughtProcess((prev) => [...prev, { title: `Edit made: ${tool} (${applied + 1}/${edits.length})`, input, tokensIn: 0, tokensOut: 0 }])
            mirror.push({ title: `Edit made: ${tool} (${applied + 1}/${edits.length})`, input, tokensIn: 0, tokensOut: 0 })
            applied += 1
            console.log(promptThoughtProcess)
        }
        if (applied > 0) setXml(xml)
        setPromptStatus('Complete')
        console.log(mirror)
    }

    return (
        <div className={styles.promptContainer} onMouseDown={(e) => e.stopPropagation()}>
            {promptState == 'input' ? (
                <>
                    <textarea className={styles.promptInput}
                        placeholder='Type something...'
                        rows={1}
                        value={inscorePrompt ?? ''}
                        onChange={(e) => setInscorePrompt(e.target!.value)} />
                    <button className={styles.promptButton} onClick={(_) => handleSubmit(inscorePrompt)}>
                        <img src={sendImage} className={styles.promptButtonImg} />
                    </button>
                </>
            ) : (
                <>
                    <div className={styles.thoughtPanel}>
                        {
                            promptThoughtProcess.map(event => (
                                <div className={styles.thoughtContainer}>
                                    <span className={styles.thoughtTitle}>{event.title}</span>
                                    <span className={styles.thoughtText}>
                                        {event.input}
                                    </span>
                                </div>
                            ))
                        }
                    </div>
                </>
            )}
        </div>
    )
}