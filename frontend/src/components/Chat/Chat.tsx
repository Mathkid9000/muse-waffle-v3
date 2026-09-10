import { useState, useRef, useEffect } from 'react'
import sendImage from '../../assets/send.svg'
import arrowImage from '../../assets/arrow.svg'
import undoImage from '../../assets/undo.svg'

import { useStore } from '../../store'
import { api } from '../../api/client'
import styles from './Chat.module.css'
import { promptAnthropic } from '../../api/claudeTools'
import { tabXml, type ThoughtProcessEvent, type ThoughtProcessTask } from '../../types'
import { promptTemplateAnthropic } from '../../api/claudeTemplates'

// Claude tool string enums → backend indices. Order must match the enums in
// claudeTools.ts (and the backend index conventions). TIME_SIG_LABELS above already
// mirrors the time-signature enum, so it doubles as the lookup for new_time.

const TEST_OBJ = [
    {
        "task": "sdfhsdfjhsdkjf Do this tajxnlzkd.",
        "isPrompt": true
    },
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
        "task": "Generating roadmap"
    },
    {
        "title": "Roadmap",
        "input": "ROADMAP (complete these steps one by one):\n# Step 1 (tool=generate_score):\nGenerate the first A section with 8 measures to establish the main melodic and harmonic material. Save this as 'A_section'.\n\n# Step 2 (tool=create_template_from_score):\nCreate a template from the A_section by masking certain musical elements (like some notes and rhythms) to allow for variation. Save this as 'A_template'.\n\n# Step 3 (tool=fill_template):\nFill the A_template to create a second variation of the A section. Save this as 'A_section_2'.\n\n# Step 4 (tool=generate_score):\nGenerate the B section (bridge) with 8 measures that contrasts with the A section in melody, harmony, or rhythm. Save this as 'B_section'.\n\n# Step 5 (tool=write_score):\nCombine all sections into the final song by writing: A_section (measures 1-8), A_section_2 (measures 9-16), B_section (measures 17-24), and A_section (measures 25-32) to create the AABA form. Save the result as 'OUTPUT'."
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
    },
    {// "Write a chorale (soprano, alto, tenor, and bass) of 8 bars. Then, create a template from it but remove all pitches in the bass and save this template in place."
        "title": "sdfsdf",
        "input": "Generated an 8-bar chorale with 4 staves (soprano, alto, tenor, and bass) and saved it to 'chorale_8bar'. Created a template from this score and saved it as 'chorale_template'. Then edited the template to remove all pitches in the bass staff while preserving the rhythm and other elements, saving the edited version back to 'chorale_template'.",
        "tokensIn": 0,
        "tokensOut": 0,
        "isRecap": true
    }
] as (ThoughtProcessEvent | ThoughtProcessTask)[]

export default function Chat() {
    const [ inputPrompt, setInputPrompt ] = useState("")
    const [ loading, setLoading ] = useState(false)
    const [ expandedThoughts, setExpandedThoughts ] = useState<number>(-1) 
    const [ thoughtUsage, setThoughtUsage ] = useState<number>(0) 
    const [ panelWidth, setPanelWidth ] = useState(400)
    // Chats live in the store so they persist with the project (see store/index and lib/scores).
    const chats = useStore((s) => s.chats)
    const activeChat = useStore((s) => s.activeChat)
    const setActiveChat = useStore((s) => s.setActiveChat)
    const addNewChat = useStore((s) => s.addNewChat)
    const deleteChat = useStore((s) => s.deleteChat)
    const promptThoughtProcess = chats.find((c) => c.name === activeChat)?.thoughts ?? []
    const isResizing = useRef(false)
    const resizeStartX = useRef(0)
    const resizeStartWidth = useRef(400)

    useEffect(() => {
        function onMouseMove(e: MouseEvent) {
            if (!isResizing.current) return
            const deltaX = resizeStartX.current - e.clientX
            setPanelWidth(Math.min(Math.max(resizeStartWidth.current + deltaX, 250), 800))
        }
        function onMouseUp() {
            isResizing.current = false
            document.body.style.cursor = ''
            document.body.style.userSelect = ''
        }
        window.addEventListener('mousemove', onMouseMove)
        window.addEventListener('mouseup', onMouseUp)
        return () => {
            window.removeEventListener('mousemove', onMouseMove)
            window.removeEventListener('mouseup', onMouseUp)
        }
    }, [])

    function handleResizeStart(e: React.MouseEvent) {
        e.preventDefault()
        isResizing.current = true
        resizeStartX.current = e.clientX
        resizeStartWidth.current = panelWidth
        document.body.style.cursor = 'ew-resize'
        document.body.style.userSelect = 'none'
    }

    function handleSubmit() {
        const prompt = inputPrompt
        // The chat the prompt was sent from: everything it produces lands there even if the
        // user switches chats while it runs.
        const chatName = activeChat
        setInputPrompt("")
        setLoading(true)
        setThoughtUsage(0)

        promptWithThoughtProcess(prompt, chatName).catch((e: unknown) => {
            addNewThought({ title: 'Prompt failed', input: e instanceof Error ? e.message : String(e), tokensIn: 0, tokensOut: 0 }, chatName)
            setLoading(false)
        })
    }

    function addNewThought(thoughtObj: ThoughtProcessEvent | ThoughtProcessTask, chatName: string) {
        const { getChat, updateChat } = useStore.getState()
        updateChat(chatName, { thoughts: [thoughtObj, ...(getChat(chatName)?.thoughts ?? [])] })
        if (chatName === useStore.getState().activeChat) setExpandedThoughts(prev => prev === -1 ? -1 : prev + 1)
    }

    function handleSwitchChat(name: string) {
        // The expanded index points into the chat being left, so it can't carry over.
        setExpandedThoughts(-1)
        setActiveChat(name)
    }

    async function promptWithThoughtProcess(prompt: string, chatName: string) {
        let usage = 0;
        // const delay = (ms: number) => new Promise(resolve => setTimeout(resolve, ms));

		// // setPromptThoughtProcess([])
		// TEST_OBJ.forEach(async (obj, i) => {
		// 	await delay(1000 * (i + 1))
		// 	addNewThought(obj,'Chat')
		// })

        // await delay(1000 * (TEST_OBJ.length + 1))
        

        addNewThought({ task: prompt, isPrompt: true }, chatName)
        const history = useStore.getState().getChat(chatName)?.history ?? []
		const recap = await promptTemplateAnthropic(history, prompt, useStore.getState().tabs, (thought) => {
            addNewThought(thought, chatName)
            if (Object.keys(thought).includes('title')) {
                thought = thought as ThoughtProcessEvent
                usage += thought.tokensIn + thought.tokensOut
                setThoughtUsage(usage)
            }
        })

        // A run that ends without a recap (no roadmap, no tool) has nothing to reduce, and
        // pushing its empty result would feed the next prompt a blank turn.
        if (typeof recap === 'string') {
            const { getChat, updateChat } = useStore.getState()
            updateChat(chatName, { history: [...(getChat(chatName)?.history ?? []), { prompt, output: recap }] })
        }

        setLoading(false);
    }

	function handleExpandThought(idx: number) {
		// const shallowArray = [...expandedThoughts]
		// shallowArray[idx] = !expandedThoughts[idx]
		setExpandedThoughts(expandedThoughts === idx ? -1 : idx)
	}

    return (
        <>
            <div className={styles.panel} style={ { borderTop: "1px solid grey", width: panelWidth } }>
                <div className={styles.resizeHandle} onMouseDown={handleResizeStart} />
                <div className={styles.section}>
                    <div className={styles.chatTabs}>
                        {chats.map(c => (
                            <button key={c.name}
                                className={c.name === activeChat ? `${styles.chatTab} ${styles.activeChatTab}` : styles.chatTab}
                                onClick={() => handleSwitchChat(c.name)}>
                                {c.name}
                                {c.name === activeChat && chats.length > 1 && (
                                    <span className={styles.chatTabClose}
                                        onClick={(e) => { e.stopPropagation(); setExpandedThoughts(-1); deleteChat(c.name) }}>×</span>
                                )}
                            </button>
                        ))}
                        <button className={styles.chatTab} disabled={loading} onClick={() => handleSwitchChat(addNewChat())}>+</button>
                    </div>
                    <div className={styles.thoughtPanel}>
                    {
                        promptThoughtProcess.map((event: ThoughtProcessEvent | ThoughtProcessTask, idx) => {
                            if (Object.keys(event).includes('isPrompt') && (event as ThoughtProcessTask).isPrompt) {
                                return (
                                    <div className={styles.userPrompt}>
                                        {(event as ThoughtProcessTask).task}
                                    </div>
                                )
                            }

                            const isLastAndNotLoading = idx === 0 && loading
                            const arrowClassList = expandedThoughts === idx ? [styles.thoughtContainerArrow, styles.expanded] : [styles.thoughtContainerArrow]
                            const containerClassList = Object.keys(event).includes('input') && (event as ThoughtProcessEvent).input !== '' ? [styles.thoughtContainer, styles.hasContent] : [styles.thoughtContainer]
                            if (expandedThoughts === idx) containerClassList.push(styles.expanded)

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

                            if (Object.keys(event).includes('isRecap') && (event as ThoughtProcessEvent).isRecap) {
                                return (
                                    <span className={styles.recap}>
                                        {(event as ThoughtProcessEvent).input}
                                    </span>
                                )
                            }

                            event = event as ThoughtProcessEvent

                            return (
                                <div className={containerClassList.join(' ')} onClick={() => { if (event.input !== '') handleExpandThought(idx) } }>
                                    <span className={[styles.thoughtTitle, event.loading && idx === 0 ? styles.loading : ''].join(' ')}>
                                        { event.input !== '' && (
                                            <img src={arrowImage} className={arrowClassList.join(' ')}/>
                                        )}
                                        {event.title}
                                        { expandedThoughts === idx ? (
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
                                                    {event.input.split('\n').flatMap((line: string, i: number, arr: any) => {
                                                        if (line.includes('ROADMAP (complete these steps one by one):')) return;
                                                        const isH2 = line.startsWith('## ')
                                                        const isH1 = !isH2 && line.startsWith('# ')
                                                        let text = isH1 ? line.slice(2) : isH2 ? line.slice(3) : line
                                                        text = text.replace(/\s\(tool=[A-Za-z_]+\)/i, '')
                                                        const node = isH1 || isH2 ? (
                                                            <strong key={i} style={{ fontStyle: 'normal', fontSize: isH1 ? '15px' : '12.5px', fontWeight: 600, letterSpacing: '1px' }}>{text}</strong>
                                                        ) : text
                                                        return i < arr.length - 1 ? [node, <br key={`${i}-br`} />] : [node]
                                                    })}
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
                    <div className={`${styles.promptContainer} ` + (loading ? styles.promptLoading : '')}>
                        <input className={styles.promptInput}
                            placeholder={loading ? '. . . .' : 'Type something...'}
                            // rows={1}
                            disabled={loading}
                            value={inputPrompt}
                            onChange={(e) => setInputPrompt(e.target!.value)} />
                        <button className={styles.promptButton} disabled={loading || inputPrompt.length === 0} onClick={(_) => handleSubmit()}>
                            <img src={sendImage} className={styles.promptButtonImg} />
                        </button>
                    </div>
                </div>
            </div>
        </>
    )
}
