import sendImage from '../../assets/send.svg'
import arrowImage from '../../assets/arrow.svg'
import undoImage from '../../assets/undo.svg'
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

const TEST_OBJ = [
    {
        "task": "Selecting tools"
    },
    {
        "title": "Selected 1 tools",
        "input": [
            "edit_key_sig"
        ],
        "tokensIn": 903,
        "tokensOut": 61
    },
    {
        "task": "Querying next steps"
    },
    {
        "title": "Found 1 edits",
        "input": [
            {
                "tool": "edit_key_sig",
                "measure_num": 1,
                "tonic": "C",
                "mode": "minor"
            },
			{
                "tool": "anotehr",
                "measure_num": 1,
                "tonic": "C",
                "mode": "minor"
            }
        ],
        "tokensIn": 1076,
        "tokensOut": 93
    },
    {
        "task": "Querying next steps"
    },
    {
        "title": "Edit made: edit_key_sig (1/1)",
        "input": '',
        "tokensIn": 0,
        "tokensOut": 0
    }
] as (ThoughtProcessEvent | ThoughtProcessTask)[]

export interface ThoughtProcessEvent {
	title: string
	input: any
	tokensIn: number
	tokensOut: number
}

export interface ThoughtProcessTask {
	task: string
}

interface InscorePromptProps {
    getXml: () => string
    getSelection: () => MeasureSelection
    setXml: (xml: string) => void
}

export default function InscorePrompt({ getXml, getSelection, setXml }: InscorePromptProps) {
    const [ inscorePrompt, setInscorePrompt ] = useState<string | null>(null)
    const [ promptState, setPromptState ] = useState<('input' | 'eval' | 'done')>('input')
    const [ promptThoughtProcess, setPromptThoughtProcess ] = useState<(ThoughtProcessEvent | ThoughtProcessTask)[]>([]) 
    const [ promptStatus, setPromptStatus ] = useState<string>('') 
    const [ currentPrompt, setCurrentPrompt ] = useState<string>('') 
    const [ expandedThoughts, setExpandedThoughts ] = useState<boolean[]>([]) 
    const [ thoughtUsage, setThoughtUsage ] = useState<number>(0) 
    const [ originalXml, setOriginalXml ] = useState<string | null>(null) 

	function addNewThought(thoughtObj: ThoughtProcessEvent | ThoughtProcessTask) {
		setPromptThoughtProcess(prev => [...prev, thoughtObj])
		setExpandedThoughts(prev => [...prev, false])
	}

    async function handleSubmit(prompt: string | null) {
        if (!prompt) return

        console.log(prompt)
		setCurrentPrompt(prompt)
        setInscorePrompt(null)
        setPromptState('eval')
		setThoughtUsage(0)

        // console.log(getXml(), getSelection())
		setOriginalXml(getXml())
        promptWithThoughProcess(prompt).catch((e: unknown) => {
            addNewThought({ title: 'Prompt failed', input: e instanceof Error ? e.message : String(e), tokensIn: 0, tokensOut: 0 })
            setPromptState('done')
        })
        // console.log(TEST_OBJ)
        // testProcess()

		// setTimeout(() => {
		// 	console.log('complete')
		// 	setPromptState('done')
		// }, 2000 * (TEST_OBJ.length + 1))
    }

	async function testProcess() {
		const delay = (ms: number) => new Promise(resolve => setTimeout(resolve, ms));

		setPromptThoughtProcess([])
		TEST_OBJ.forEach(async (obj, i) => {
			await delay(1000 * (i + 1))
			addNewThought(obj)
		})
	}

    async function promptWithThoughProcess(prompt: string) {
        const currentXml = getXml()
        const selection = getSelection()
		let totalUsage = 0

        // trigger prompting
        const response = await promptAnthropic(prompt, currentXml!, (thought: ThoughtProcessEvent | ThoughtProcessTask) => {
            addNewThought(thought)
			if (Object.keys(thought).includes('title')) {
				thought = thought as ThoughtProcessEvent
				totalUsage += thought.tokensIn + thought.tokensOut
				setThoughtUsage(totalUsage)
			}
        }, (title: string) => {
			setCurrentPrompt(title)
		}, selection)
        if (response.length === 0) {
            setPromptStatus('Nothing was returned')
            setPromptState('done')
            return
        }

        const edits = response[0]
		totalUsage = response[1] + response[2]
		setThoughtUsage(totalUsage)
        
        let xml = currentXml
        let applied = 0
        try {
            for (const { tool, input } of edits) {
                addNewThought({ task: `Applying edit (${applied + 1}/${edits.length})` })
                xml = await applyClaudeEdit(tool, input, xml, setXml)
                addNewThought({ title: `Edit (${applied + 1}/${edits.length}) made: ${tool}`, input: '', tokensIn: 0, tokensOut: 0 })
                applied += 1
            }
            setPromptStatus('Complete')
        } catch (e: unknown) {
            // without this the panel just sits on "Applying edit" forever and the reason is lost
            const message = e instanceof Error ? e.message : String(e)
            addNewThought({ title: `Edit ${applied + 1} failed`, input: message, tokensIn: 0, tokensOut: 0 })
            setPromptStatus(message)
        }
        if (applied > 0) setXml(xml)

		setPromptState('done')
    }

	function handleExpandThought(idx: number) {
		const shallowArray = [...expandedThoughts]
		shallowArray[idx] = !expandedThoughts[idx]
		setExpandedThoughts(shallowArray)
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
                <div style={{ display: 'block' }}>
					{ promptState === 'done' && originalXml && (
						<button className={styles.undoThoughtButton} onClick={() => setXml(originalXml)}>
							<img src={undoImage} className={styles.undoThoughtButtonImage}/>
						</button>
					)}
					<div className={styles.thoughtPanelTitle}>{currentPrompt} <p className={styles.thoughtPanelTokens}>{thoughtUsage} toks</p></div>
                    <div className={styles.thoughtPanel}>
                        {
                            promptThoughtProcess.map((event, idx) => {
                                const isLastAndNotLoading = idx === promptThoughtProcess.length - 1 && promptState !== 'done'
								const arrowClassList = expandedThoughts[idx] ? [styles.thoughtContainerArrow, styles.expanded] : [styles.thoughtContainerArrow]
								const containerClassList = Object.keys(event).includes('input') && (event as ThoughtProcessEvent).input !== '' ? [styles.thoughtContainer, styles.hasContent] : [styles.thoughtContainer]

								if (Object.keys(event).includes('task')) {
									if (!isLastAndNotLoading) return null

									return (
										<div className={styles.thoughtContainer}>
											<span className={[styles.thoughtTitle, styles.loading].join(' ')}>
												{(event as ThoughtProcessTask).task}
											</span>
										</div>
									)
								}

								event = event as ThoughtProcessEvent

                                return (
									<div className={containerClassList.join(' ')} onClick={() => { if (event.input !== '') handleExpandThought(idx) } }>
										<span className={styles.thoughtTitle}>
											{ event.input !== '' && (
												<img src={arrowImage} className={arrowClassList.join(' ')}/>
											)}
											{event.title}
											{ expandedThoughts[idx] ? (
												Array.isArray(event.input) ? (
													<ul className={styles.thoughtText}>
														{
															event.input.map((text, subIdx) => 
																typeof text === 'object' ? (
																	<>
																		{
																			Object.entries(text).map((entry) => (
																				<li className={[ styles.thoughtTextListItem, styles.codeText ].join(' ')}>{entry[0]} = "{String(entry[1])}"</li>
																			))
																		}
																		{ subIdx < event.input.length - 1 && (
																			<p style={{ marginTop: "20px" }}></p>	
																		)} 
																	</>
																) :
																(
																	<li className={styles.thoughtTextListItem}>{text}</li>
																)
															)
														}
													</ul>
												) : ( typeof event.input === 'object' ? (
													<ul className={styles.thoughtText}>
														{
															Object.entries(event.input).map((entry) => (
																<li className={[ styles.thoughtTextListItem, styles.codeText ].join(' ')}>{entry[0]} = "{String(entry[1])}"</li>
															))
														}													
													</ul>
												) : (
													<span className={styles.thoughtText}>
														{event.input}												
													</span>
												)))
											: (
												<></>
											)}
										</span>
									</div>
								)
                            })
                        }
                    </div>
                </div>
            )}
        </div>
    )
}