import Anthropic from '@anthropic-ai/sdk';
import { anthropic, type ScoreSelection } from "./claudeTools";
import type { ToolUseBlock } from "@anthropic-ai/sdk/resources";
import type { AnyTemplateItem, ChatItem, Tab, ThoughtProcessEvent, ThoughtProcessTask } from '../types';
import { api } from './client';
import { applyClaudeEdit } from './applyClaudeEdit';
import { fillTemplate, scoreToTemplate } from './templates';
import { useStore } from '../store';
import { describeTab, describeTemplateMeasures, getPromptContext, templateItemInMeasures } from './readContext';
import { chordProblems, repairChords } from './templateGuard';
import { activeModel, reasoningOn } from './chatCommands';

// A roadmap step takes one turn, and a long arrangement is edited a few measures at a time, so a
// run can legitimately take dozens. The last turn is not a step: it is spent forcing the recap.
const MAX_TURNS = 40;

// thinking shares max_tokens with the answer, so a request that thinks is given this much more room
const THINKING_HEADROOM = 4096

// Time spent in thinking requests since the last takeReasonedSeconds(), for the chat's
// "Reasoned for {x}s..." line. The API reports no thinking time, so this is the request's duration.
let reasonedMs = 0

/**
 * A request with the model's built-in thinking when /reasoning is on. No tool choice is forced
 * anywhere here, so the model can think in the same call that picks its tool.
 */
async function createThinking(request: Anthropic.Messages.MessageCreateParamsNonStreaming) {
    if (!reasoningOn()) return anthropic.messages.create(request)
    const start = performance.now()
    try {
        return await anthropic.messages.create({
            ...request,
            max_tokens: request.max_tokens + THINKING_HEADROOM,
            // Haiku 4.5 predates adaptive thinking and takes a fixed budget instead
            thinking: request.model.startsWith('claude-haiku-4-5')
                ? { type: 'enabled', budget_tokens: THINKING_HEADROOM / 2 }
                : { type: 'adaptive' },
        })
    } finally {
        reasonedMs += performance.now() - start
    }
}

function takeReasonedSeconds(): number | undefined {
    const seconds = reasonedMs > 0 ? Math.round(reasonedMs / 100) / 10 : undefined
    reasonedMs = 0
    return seconds
}

export async function promptTemplateAnthropic(chatHistory: ChatItem[], prompt: string, tabs: Tab[], pushCallback: (thought: ThoughtProcessEvent | ThoughtProcessTask) => void, pushChatRename: (title: string) => void) {
    // propogate memory with tab data
    const memory: Record<string, string | AnyTemplateItem[]> = {}
    tabs.forEach(t => {
        if (!t.template && !t.content) return
        memory[t.name] = (t.template ?? t.content)! // TODO: add description  + info likje instruments, bar dur,  etc..
    })
    
    pushCallback({ task: "Generating a procedure" })

    // step 1: roadmap
    const activeName = useStore.getState().activeTab
    const activeIsTemplate = tabs.some(t => t.name === activeName && t.template)
    const context = PROMPT_INTRO + "\n\n" + getHistoryContext(chatHistory) + "\n\n" + 
        getMemoryContext(memory) + "\n\nThe user's current selection:\nTab: " + activeName
        + (activeName !== 'Main' ? (activeIsTemplate ? " (a TEMPLATE; removing instruments/voices, editing clefs, tempos, key signatures or any other change to it must use edit_template, never any of the other edits)" : " (a score)") : "") + "\nSelection: " + describeSelection(useStore.getState().selection)
    console.log(context)
    reasonedMs = 0
    const roadmapResponse = await createThinking({
        model: activeModel(),
        max_tokens: 1024,
        tools: [ withoutScoreTools(ROADMAP_TOOL, activeIsTemplate, memory) ] as unknown as Anthropic.Messages.ToolUnion[],
        system: context,
        messages: [{ role: 'user', content: prompt }],
        // tool_choice: { type: "tool", name: "roadmap_tool" }
    });

    const roadmapTool = roadmapResponse.content.find(
        (i): i is ToolUseBlock => i.type === "tool_use" && i.name === "roadmap_tool"
    );
    if (!roadmapTool) {
        console.log('Found no roadmap. Terminating...')
        // no roadmap means the model answered in prose (a question, a refusal): show it
        const prose = textOf(roadmapResponse)
        pushCallback({ title: '', input: prose || `The model returned no plan (stop reason: ${roadmapResponse.stop_reason}).`,
            tokensIn: roadmapResponse.usage.input_tokens, tokensOut: roadmapResponse.usage.output_tokens, isRecap: true })
        return
    }
    
    const chatName = (roadmapTool.input as any)['chat_name'] as string | null
    if (chatName) {
        pushChatRename(chatName)
    }

    const roadmap = "ROADMAP (complete these steps one by one):\n" + (roadmapTool.input as any).steps.map((s: { tool: string, intention: string }, idx: number) => 
        `# Step ${idx + 1} (tool=${s.tool}):\n${s.intention}`
    ).join('\n\n')

    pushCallback({ title: "Generated a procedure", input: roadmap, tokensIn: roadmapResponse.usage.input_tokens, tokensOut: roadmapResponse.usage.output_tokens, isRoadmap: true, reasonedSeconds: takeReasonedSeconds() })

    const messages: Anthropic.Messages.MessageParam[] = [{ role: 'user', content: prompt }, { role: 'assistant', content: roadmap },
        // Sonnet/Opus reject a conversation that ends on an assistant turn (no prefill): a 400.
        { role: 'user', content: 'Select the first step of the roadmap.' }];
    let usageIn = roadmapResponse.usage.input_tokens;
    let usageOut = roadmapResponse.usage.output_tokens;

    // Step 2: carry out roadmap
    // what each finished step did, so the closing recap knows what actually happened
    const executed: string[] = []
    for (let turn = 0; turn < MAX_TURNS; turn++) {
        console.log('TURN:', turn, 'MESSAGES:', messages, 'USAGE IN:', usageIn, 'USAGE OUT:', usageOut)
        pushCallback({ task: "Querying next steps" })

        // Out of turns, the model is not asked for a next step: it is made to close the run, so
        // that a run always ends with a recap (and the next prompt knows what is unfinished).
        const outOfSteps = turn === MAX_TURNS - 1
        let selectionUsage = { input_tokens: 0, output_tokens: 0 }
        let toolTriggerContent: { tool?: string, intention?: string, next_steps?: string } | undefined
        if (outOfSteps) {
            console.warn(`Reached the ${MAX_TURNS} turn limit. Forcing a recap...`)
            pushCallback({ title: 'Reached the step limit', input: `Stopped after ${MAX_TURNS - 1} steps and wrote a recap of what was done. Send another prompt to carry on.`,
                tokensIn: usageIn, tokensOut: usageOut })
            toolTriggerContent = {
                tool: 'complete',
                intention: `The step limit was reached before the roadmap finished. Recap what was actually completed (see the steps carried out so far) and state plainly what is still unfinished, so a follow-up prompt can pick it up.`,
            }
        }
        else {
            const toolSelectionMessage = await anthropic.messages.create({
                model: activeModel(),
                max_tokens: 1024,
                tools: withoutScoreTools(TOOL_INTROS, activeIsTemplate, memory) as unknown as Anthropic.Messages.ToolUnion[],
                system: getMemoryContext(memory),
                messages: messages,
                // tool_choice: { type: "tool", name: "trigger_tool" }
            });
            usageIn += toolSelectionMessage.usage.input_tokens
            usageOut += toolSelectionMessage.usage.output_tokens
            selectionUsage = toolSelectionMessage.usage

            toolTriggerContent = toolSelectionMessage.content.find(
                (i): i is ToolUseBlock => i.type === "tool_use" && i.name === "trigger_tool"
            )?.input as typeof toolTriggerContent
            if (!toolTriggerContent?.tool) {
                // The model answered in prose instead of picking a tool (tool_choice can't be forced
                // on every model). Show what it said: it is usually a question or a refusal.
                console.log('Found no tool. Terminating...', toolSelectionMessage)
                pushCallback({ title: '', input: textOf(toolSelectionMessage) || `The model did not pick a next step (stop reason: ${toolSelectionMessage.stop_reason}).`,
                    tokensIn: usageIn, tokensOut: usageOut, isRecap: true })
                return
            }
        }

        const selectedTool = toolTriggerContent.tool!
        const intention = toolTriggerContent.intention ?? ''
        const next_steps = toolTriggerContent.next_steps
        const descriptorIndex = Math.floor(Math.random() * DESCRIPTORS.length)
        const context = getPromptContext(selectedTool, tabs)

        if (selectedTool !== 'complete') {
            pushCallback({ title: (TOOL_DESCRIPTORS[selectedTool] ?? selectedTool).replace('<>',  DESCRIPTORS[descriptorIndex]),
                loading: true, 
                input: (context ? (context + '\n\n') : '') + intention, tokensIn: selectionUsage.input_tokens, tokensOut: selectionUsage.output_tokens })
        }

        // if (selectedTool === 'edit_template') { console.log(memory); return; }
        // the model writing a template is the one that has to know what the filler does with it
        const writesTemplate = selectedTool === 'create_template_from_scratch'
        // the recap is written from this prompt alone, so it is told what the steps so far did
        const progress = selectedTool === 'complete' && executed.length
            ? `\n\nSteps carried out so far:\n${executed.map((s, i) => `${i + 1}. ${s}`).join('\n')}` : ''
        const toolPrompt = `${prompt}\n\n${roadmap}${progress}\n\nDo this now: ${intention}`
        const toolSystem = (writesTemplate ? TEMPLATE_CONTRACT + '\n\n' : '') + (context ? (context + '\n\n') : '') + getMemoryContext(memory)
        const toolRequest = {
            model: activeModel(),
            // a whole template is returned in one go: a long multi-voice one runs to hundreds of items
            max_tokens: writesTemplate ? 32000 : 1024,
            tools: [ TOOLS[selectedTool as keyof typeof TOOLS] ] as unknown as Anthropic.Messages.ToolUnion[],
            system: toolSystem,
            // tool_choice: { type: "tool", name: selectedTool },
        } as const
        let toolMessage: Anthropic.Messages.Message
        try {
            // the closing "complete" step only writes the recap, so it doesn't think
            toolMessage = await (selectedTool === 'complete' ? anthropic.messages.create : createThinking)({ ...toolRequest, messages: [{ role: 'user', content: toolPrompt }] });
        } catch (e) {
            // A long template can outlast the proxy's timeout. End the run with the reason shown,
            // rather than throw past the chat and leave it loading forever.
            console.error(`${selectedTool} request failed:`, e)
            pushCallback({ title: `${TOOL_DESCRIPTORS[selectedTool] ?? selectedTool} failed`.replace('<>', 'a step'),
                input: e instanceof Error ? e.message : String(e), tokensIn: usageIn, tokensOut: usageOut })
            return
        }
        usageIn += toolMessage.usage.input_tokens
        usageOut += toolMessage.usage.output_tokens

        if (writesTemplate) {
            // A named chord has to arrive as a chord item, or the filler never hears about it. Check
            // the template against what the user asked for, give the model one correction, and as
            // a last resort fix it up in code rather than fill a template that ignores the request.
            const userText = `${prompt}\n${intention}`
            const written = () => toolMessage.content.find((b): b is ToolUseBlock => b.type === 'tool_use')
            const templateOf = () => (written()?.input as Record<string, unknown> | undefined)?.['template'] as AnyTemplateItem[] | undefined
            const problems = chordProblems(userText, templateOf() ?? [])
            const first = written()
            if (problems.length && first) {
                console.warn('Template does not carry the requested chords — asking for a correction:', problems)
                try {
                    toolMessage = await createThinking({ ...toolRequest, messages: [
                        { role: 'user', content: toolPrompt },
                        { role: 'assistant', content: toolMessage.content },
                        { role: 'user', content: [{ type: 'tool_result', tool_use_id: first.id, is_error: true,
                            content: 'The template was not accepted:\n- ' + problems.join('\n- ') + '\nCall the tool again with a corrected template.' }] },
                    ] })
                    usageIn += toolMessage.usage.input_tokens
                    usageOut += toolMessage.usage.output_tokens
                } catch (e) {
                    // the first template is still in hand: repair it in code instead of losing the run
                    console.warn('Correction request failed — keeping the first template:', e)
                }
                const retried = templateOf()
                if (retried && chordProblems(userText, retried).length) repairChords(userText, retried)
            }
        }

        const toolUses = toolMessage.content.filter((b): b is ToolUseBlock => b.type === "tool_use");
        // A model that answers in prose instead of calling the tool is shown, not skipped over:
        // carrying on would tell the next step this one was done.
        const prose = textOf(toolMessage)
        if (selectedTool === "complete") {
            // prose here is the recap itself, written without the tool
            const written = toolUses[0]?.input as Record<string, unknown> | undefined
            const recap = typeof written?.['recap'] === 'string' && written['recap'] ? written['recap'] as string : prose
            if (!recap) {
                pushCallback({ title: 'Recap failed', input: `The model returned no recap (stop reason: ${toolMessage.stop_reason}).`, tokensIn: usageIn, tokensOut: usageOut })
                return
            }
            pushCallback({ title: '', input: recap, tokensIn: selectionUsage.input_tokens + toolMessage.usage.input_tokens,
                tokensOut: selectionUsage.output_tokens + toolMessage.usage.output_tokens, isRecap: true
            })
            return recap
        }
        if (toolUses.length === 0) {
            console.warn(`${selectedTool} returned no tool call:`, toolMessage)
            pushCallback({ title: '', input: prose || `The model made no ${selectedTool} call (stop reason: ${toolMessage.stop_reason}).`,
                tokensIn: usageIn, tokensOut: usageOut, isRecap: true })
            return
        }
        // console.log('Found Tool Uses:', toolUses, "in message:", toolMessage, getMemoryContext(memory))

        const results: Anthropic.Messages.ToolResultBlockParam[] = []
        for (const tool of toolUses) {
            let readResult: string | undefined
            try {
                readResult = await processCommand({ tool: tool.name, input: tool.input as Record<string, unknown> }, memory, intention)
            } catch (e) {
                // An API error (a 422/500 from the backend) ends the whole run: carrying on would
                // tell the model the step was "done" and have it build on a result that never arrived.
                console.error(`${tool.name} failed:`, e)
                pushCallback({ title: `${TOOL_DESCRIPTORS[selectedTool] ?? selectedTool} failed`.replace('<>', 'a step'),
                    input: e instanceof Error ? e.message : String(e), tokensIn: usageIn, tokensOut: usageOut })
                return
            }
            if (readResult) {
                results.push({
                    type: "tool_result",
                    tool_use_id: tool.id,
                    content: `${tool.name} done:\n${readResult}\nMemory holds: ${Object.keys(memory).join(', ') || 'nothing yet'}`,
                })
            }
            else {
                results.push({
                    type: "tool_result",
                    tool_use_id: tool.id,
                    content: `${tool.name} done. Memory holds: ${Object.keys(memory).join(', ') || 'nothing yet'}`,
                })
            }
        }
        // Thinking blocks are only valid in the conversation that produced them, and this step ran
        // as its own request, so they are left out of the roadmap conversation.
        messages.push({ role: 'assistant', content: toolMessage.content.filter((b) => b.type !== 'thinking' && b.type !== 'redacted_thinking') });
        // the next-steps note rides in the user turn: ending on an assistant turn is a 400 on newer models
        const nextNote = `Done. Plan from here: ${String(next_steps ?? '').trim() || 'continue the roadmap'}\nSelect the next tool.`
        messages.push({ role: 'user', content: [...results, { type: 'text', text: nextNote }] });

        executed.push(`${selectedTool}: ${intention}`)

        pushCallback({ title: (TOOL_DESCRIPTORS[selectedTool] ?? selectedTool).replace('ing', 'ed').replace('<>',  DESCRIPTORS_PAST[descriptorIndex]),
            input: toolUses.length === 1 ? toolUses[0].input as Record<string, any> : toolUses.map(u => u.input as Record<string, any>),
            tokensIn: selectionUsage.input_tokens,
            tokensOut: selectionUsage.output_tokens, reasonedSeconds: takeReasonedSeconds() })
    }
    console.log('PROMPTING COMPLETE', memory, usageIn, usageOut)
}

/** The prose a response carries: its text blocks, joined. Empty when it is only tool calls. */
function textOf(message: Anthropic.Messages.Message): string {
    return message.content.flatMap((b) => b.type === 'text' ? [b.text.trim()] : []).filter(Boolean).join('\n\n')
}

const SCORE_ONLY_TOOLS = [ "edit_clef", "edit_time_sig", "edit_key_sig", "edit_tempo", "edit_add_instrument", "edit_delete_staff", "edit_delete_measures" ]

// Drops the score-only tools from every tool-name enum while the active tab is a template and
// memory holds no score they could act on, so they can't be picked at all.
function withoutScoreTools<T>(tools: T, activeIsTemplate: boolean, memory: Record<string, string | AnyTemplateItem[]>): T {
    if (!activeIsTemplate || Object.values(memory).some(v => typeof v === 'string')) return tools
    return JSON.parse(JSON.stringify(tools, (k, v) => k === 'enum' && Array.isArray(v) ? v.filter(x => !SCORE_ONLY_TOOLS.includes(x)) : v))
}

function describeSelection(sel?: ScoreSelection | null): string {
    if (!sel) return "The user has not selected anything on the tab.";
    const parts: string[] = [];
    if (sel.start != null && sel.end != null) {
        parts.push(sel.start === sel.end ? `measure ${sel.start}` : `measures ${sel.start} to ${sel.end} (1-based, inclusive)`);
    }
    parts.push(sel.parts && sel.parts.length > 0
        ? `staff_num ${sel.parts.join(", ")} (0-based)`
        : "all staves");
    return `The user's current selection on the tab: ${parts.join(", ")}.`;
}

function getMemoryContext(memory: Record<string, string | AnyTemplateItem[]>) {
    if (Object.keys(memory).length == 0) return "There are currently no scores or templates saved in memory." 
    return "Current scores and templates saved in memory that may be used:\n" + 
        Object.keys(memory).map(k => k === 'Main' ? `${k} (likely a null score that cannot be edited if no edits have been made. Double check first that it has content before editing)` : `${k} (${typeof memory[k] === "string" ? 'score' : 'template'})`).join(', ')
}

async function processCommand(command: { tool: string, input: Record<string, unknown> }, memory: Record<string, string | AnyTemplateItem[]>, intention: string) {
    switch (command.tool) {
        case "edit_clef":
        case "edit_time_sig":
        case "edit_key_sig":
        case "edit_tempo":
        case "edit_add_instrument":
        case "edit_delete_staff":
        case "edit_delete_measures": {
            const { in_key, out_key } = getScoreInputOutputNames(command.input)
            const stored = memory[in_key]
            if (Array.isArray(stored) && command.tool === "edit_add_instrument") {
                // the target is a template, not a score: add an instrument item for a new voice
                const voices = stored.flatMap(i => 'voice_idx' in i && typeof i.voice_idx === 'number' ? [i.voice_idx] : [])
                const item: AnyTemplateItem = {
                    kind: "instrument", instrument: String(command.input['new_instrument']),
                    voice_idx: voices.length ? Math.max(...voices) + 1 : 0, measure: 1, pos: 0,
                }
                saveToMemory(memory, out_key, [...stored, item])
                break
            }
            const xml = stored
            if (typeof xml !== 'string' || !xml) {
                console.warn(command, "No score stored under", in_key)
                break
            }
            const newXml = await applyClaudeEdit(command.tool, command.input, xml)
            saveToMemory(memory, out_key, newXml)
            break
        }
        case "generate_score": {
            const { in_key, out_key } = getScoreInputOutputNames(command.input)
            if (in_key && in_key in memory) {
                const xml = memory[in_key] as string
                const newXml = await applyClaudeEdit(command.tool, command.input, xml)
                saveToMemory(memory, out_key, newXml)
            }
            else {
                const newXml = await applyClaudeEdit(command.tool, command.input, '')
                saveToMemory(memory, out_key, newXml)
            }
            console.log("FINISHED GENERATE:",memory)
            break
        }
        case "create_template_from_scratch": {
            const { in_key: _, out_key } = getScoreInputOutputNames(command.input)
            const template = command.input['template'] as AnyTemplateItem[]
            saveToMemory(memory, out_key, template)
            break
        }
        case "create_template_from_score": {
            const { in_key, out_key } = getScoreInputOutputNames(command.input)
            const xml = memory[in_key] as string
            const template = await scoreToTemplate(xml)
            saveToMemory(memory, out_key, template)
            break
        }
        case "fill_template": {
            const { in_key, out_key } = getScoreInputOutputNames(command.input)
            const template = memory[in_key] as AnyTemplateItem[]
            if (!template) {
                console.warn(command, "No template stored under", in_key)
                break
            }
            const newXml = await fillTemplate(template)
            saveToMemory(memory, out_key, newXml)

            break
        }
        case "edit_template": {
            const { in_key } = getScoreInputOutputNames(command.input)
            const template = memory[in_key]
            // Without this the stage 2 model is handed "undefined" as its template and, having
            // nothing to edit, invents a fresh one-item template instead.
            if (!Array.isArray(template)) {
                console.warn(command, "No template stored under", in_key, "— nothing to edit. Memory holds:", Object.keys(memory))
                return `Nothing was edited: no template named "${in_key}" is in memory.`
            }

            // With edit_measures, stage 2 only sees (and returns) the items in those measures; the
            // rest is spliced back around its answer. Omitted, or matching nothing, the whole
            // template is sent as before.
            const measures = new Set((command.input['edit_measures'] ?? []) as number[])
            const inMeasures = (i: AnyTemplateItem) => templateItemInMeasures(i, measures)
            const scoped = template.filter(inMeasures)
            // Stage 1 may say the edit rewrites those measures wholesale and doesn't depend on what
            // is there. Their items are then left out of stage 2's context (a real saving on a long
            // template) and replaced by what it returns. Needs measures to say what is replaced.
            const blind = command.input['needs_context'] === false && measures.size > 0
            const shown = blind ? [] : scoped.length ? scoped : template
            const scopedRun = blind || scoped.length > 0

            // the setup the scoped items sit in, shown read-only so it is not returned (and duplicated)
            const setup = scopedRun ? template.filter((i) => !inMeasures(i)
                && ['instrument', 'clef', 'time', 'key', 'tempo'].includes(i.kind)) : []

            const measureList = [...measures].join(', ')
            const stage2Task = (setup.length ? `Context — the template's setup items, for reference only. Do NOT return these:\n${JSON.stringify(setup)}\n\n` : '')
                + (blind
                    ? `Template "${in_key}" — measures ${measureList}. Whatever is in these measures now is NOT shown: everything in them is replaced by the items you return, and the rest of the template is kept as is.\n\n`
                        + `Write the items for these measures so that: ${intention}\n\n`
                        + `Return only the new items for measures ${measureList}, in order, and stay within these measures.`
                    : `Template "${in_key}" — ${shown.length} items`
                        + (scoped.length ? ` (only the items in measures ${measureList}; the rest of the template is kept as is, so stay within these measures)` : '')
                        + `:\n${JSON.stringify(shown)}\n\n`
                        + `Edit this template so that: ${intention}\n\n`
                        + `Return the complete edited template: all ${shown.length} items in their original order, with your edits applied.`)
            const messageStage2 = await createThinking({
                model: activeModel(),
                // The whole edited template comes back in this response, so the budget has to fit
                // every item of the input template, not just the ones being changed.
                max_tokens: 16384,
                tools: [ EDIT_TEMPLATE_STAGE2_TOOL ] as unknown as Anthropic.Messages.ToolUnion[],
                system: EDIT_TEMPLATE_STAGE2_INTRO,
                // A single self-contained request: the roadmap conversation ends on an assistant
                // turn and never carries this step's intention, so stage 2 gets its own turn.
                messages: [{ role: 'user', content: stage2Task }],
                // tool_choice: { type: "tool", name: 'edit_template' }
            });
            console.log('EDIT TEMPLATE STAGE 2:', messageStage2)
            if (messageStage2.stop_reason === 'max_tokens') {
                console.warn('Stage 2 hit the output limit — the edited template is truncated:', messageStage2)
            }
            const toolUses = messageStage2.content.filter((b): b is ToolUseBlock => b.type === "tool_use");
            if (toolUses.length == 0) {
                // Prose instead of an edit: throw it so the run ends with what the model said shown,
                // rather than reporting an edit that never happened.
                console.warn('NO EDITS FOUND:', template, messageStage2)
                throw new Error(textOf(messageStage2) || `The model returned no edit (stop reason: ${messageStage2.stop_reason}).`)
            }
            const input = toolUses[0].input as Record<string, unknown>
            const { out_key } = getScoreInputOutputNames(input)
            const newTemplate = input['template'] as AnyTemplateItem[]
            if (!Array.isArray(newTemplate) || newTemplate.length === 0) {
                console.warn('Stage 2 returned no template — keeping', in_key, 'as it was:', input)
                return `Nothing was edited: the edit returned no items, so "${in_key}" is unchanged.`
            }
            let merged: AnyTemplateItem[]
            if (scopedRun) {
                // the new items go where the replaced ones started; with none there, before the
                // first later measure (or at the end)
                const first = scoped.length ? template.indexOf(scoped[0])
                    : template.findIndex((i) => i.kind !== 'masked' && typeof i.measure === 'number' && i.measure > Math.min(...measures))
                const at = first === -1 ? template.length : first
                merged = [...template.slice(0, at).filter((i) => !inMeasures(i)), ...newTemplate,
                    ...template.slice(at).filter((i) => !inMeasures(i))]
            }
            else merged = newTemplate
            saveToMemory(memory, out_key || `${in_key}_edited`, keepInstruments(template, merged))
            break
        }
        case "write_score": {
            // a list of writes, pasted one after another in the order given
            const writes = (command.input['writes'] ?? [ command.input ]) as Record<string, unknown>[]
            for (const write of writes) {
                const { in_key, out_key } = getScoreInputOutputNames(write)
                const source = memory[in_key] as string
                if (!source) {
                    console.warn(command, "No score stored under", in_key)
                    continue
                }
                const newXml = (await api.editWriteScore({
                    xml: (memory[out_key] as string) ?? '',
                    source_xml: source,
                    measure_start_num: write['write_start_measure'] as number,
                })).xml

                saveToMemory(memory, out_key, newXml)
            }
            break
        }
        case "read_tab": {
            // Whether each tab is a score or a template, and its basics. Blank reads the active tab.
            const state = useStore.getState()
            const names = ((command.input['memory_names'] ?? []) as string[]).filter(Boolean)
            const out = (names.length ? names : [state.activeTab]).map((name) => {
                const known = name in memory
                return known || name === state.activeTab
                    ? describeTab(name, memory[name])
                    : `# Tab "${name}"\nNo tab by that name is in memory. Memory holds: ${Object.keys(memory).join(', ') || 'nothing yet'}.`
            })
            if (command.input['include_selection']) out.push(`Active tab: ${state.activeTab}\n${describeSelection(state.selection)}`)
            return out.join('\n\n')
        }
        case "read_template": {
            const { in_key } = getScoreInputOutputNames(command.input)
            const template = memory[in_key]
            if (!Array.isArray(template)) {
                return typeof template === 'string'
                    ? `"${in_key}" is a score, not a template: use read_tab for its basics.`
                    : `No template named "${in_key}" is in memory. Memory holds: ${Object.keys(memory).join(', ') || 'nothing yet'}.`
            }
            return describeTemplateMeasures(in_key, template, (command.input['measures'] ?? []) as number[])
        }
        default: {
            console.warn(command, "No implementation yet")
        }
    }

    const open = (command.input['open'] ?? '') as string
    console.log(command.input, open)
    if (open !== '') {
        console.log(useStore.getState().tabs)
        useStore.getState().setActiveTab(open);
    }
}

/**
 * fill_template fails outright ("the template names no instrument to write for") when nothing
 * says what instrument to write, so an edit is never allowed to drop every instrument item.
 */
function keepInstruments(source: AnyTemplateItem[], edited: AnyTemplateItem[]): AnyTemplateItem[] {
    if (edited.some((i) => i.kind === 'instrument')) return edited
    const instruments = source.filter((i) => i.kind === 'instrument')
    if (instruments.length === 0) return edited
    console.warn('edit_template dropped every instrument item — re-adding', instruments)
    return [...instruments, ...edited]
}

function saveToMemory(memory: Record<string, string | AnyTemplateItem[]>, key: string, value: string | AnyTemplateItem[]) {
    memory[key] = value
    const appState = useStore.getState()

    if (!appState.tabs.some(t => t.name === key)) {
        appState.addNewTab(key)
    }
    else { console.log(key) }

    if (typeof value == 'string')
        appState.updateTab(key, { content: value })
    else appState.updateTab(key, { template: value })
}

function getScoreInputOutputNames(input: Record<string, unknown>) {
    let input_name = ""
    let output_name = ""
    if ('memory_name' in input) input_name = input['memory_name'] as string
    if ('memory_output_name' in input) output_name = input['memory_output_name'] as string
    return { in_key: input_name, out_key: output_name }
}

function getHistoryContext(history: ChatItem[]) {
    if (history.length === 0) return "" 
    // a run that ended without a recap is just its prompt
    return "History of Prompts:\n" + history.map(h => "User prompt: " + h.prompt + (h.output ? "\nOutput: " + h.output : "")).join('\n')
}

const TOOL_DESCRIPTORS: Record<string, string> = {
    "edit_clef": "<> clefs",
    "edit_time_sig": "<> time signatures",
    "edit_key_sig": "<> key signatures",
    "edit_tempo": "<> tempos",
    "edit_add_instrument": "<> instruments",
    "edit_delete_staff": "<> instruments",
    "edit_delete_measures": "<> measures",
    "generate_score": "Generating music",
    "create_template_from_scratch": "Creating a blank template",
    "create_template_from_score": "Converting a new template",
    "edit_template": "<> a template",
    "fill_template": "Filling a template",
    "write_score": "<> the score",
    "read_tab": "Reading the score",
    "read_template": "Reading a template"
}

const DESCRIPTORS = [ 'Modifying', 'Editing', 'Shifting', 'Revising', 'Adjusting', 'Shaping', 'Composing', 'Arranging', 'Writing' ]
const DESCRIPTORS_PAST = [ 'Modified', 'Edited', 'Shifted', 'Revised', 'Adjusted', 'Shaped', 'Composed', 'Arranged', 'Wrote' ]

const PROMPT_INTRO = "Your job is to classify which edit types are required next to achieve the user's goal and in what order."
    + "For generation tasks with content details or repetition (ex. \"Generate a score in AABA form\" or \"Write a song with this chord progression: D, Em, A7\"), always use templates. A template is score abstraction where some elements are intentionally masked to be filled in by a creative model. Actions like creating a template from a score don't change anything about the score, it just modifies the type of the object so that it can be edited using other actions. When a template is filled in, it merely takes the blank/masked elements of the template and fills them in automatically."
    + "Using templates is your only way of biasing what is generated by the creative model. For example, if the user wants you to write string pads over a chord, insert whole notes with masked pitches and the given chord progression."
    + "If nothing matches an edit type, return no tool. You also have the ability to store/retrieve templates and score segments to/from memory with a name identifier so that they can be reused (Note: operations do not happen in place automatically)."
    + "Do not worry about missing details like which measures/staves/values are meant. "
    + "If the user gives no location on where something should happen, assume the user means their current selection and tab. Consider the type of the current tab (score or template) to decide what specific commands should be used. Mention this in your roadmap."
    + "Finally, unless specified, the operation should typically be done in place, and you should draw the user's attention to your actions by opening the score or template."

const ROADMAP_TOOL = {
    name: "roadmap_tool",
    description: "Create a roadmap plan for how to achieve the user's task using the given tools",
    input_schema: {
        type: "object",
        properties: {
            steps: {
                type: "array",
                items: {
                    type: "object",
                    properties: {
                        tool: {
                            type: "string",
                            enum: [ "edit_clef", "edit_time_sig", "edit_key_sig", "edit_tempo", "edit_add_instrument", "edit_delete_staff", "edit_delete_measures",
                                "generate_score",
                                "create_template_from_scratch", "create_template_from_score", "edit_template",
                                "fill_template",
                                "write_score",
                                "read_tab",
                                "read_template"
                            ],
                            description: "Each distinct type of edit that can be made. Tool descriptions:"
                                + "\ngenerate_score: \"Uses a transformer model only to append more creative measures to the score. Your only controls are the key, time_sig, tempo so do not use if there are detailed content requirements. Do not include key, time_signature, or tempo when these attributes should remain the same.\""
                                + "\ncreate_template_from_scratch: \"Create a template from scratch. Only to be used when the user has very specific intentions.\""
                                + "\ncreate_template_from_score: \"Create a template from an existing score in memory. To be used in combination with generate_score when the user does not have specific intentions (e.g. melodic ideas, rhythmic ideas, chord progs, etc...)\""
                                + "\nedit_template: \"Edit a template already stored in memory (e.g. to strip it down/mask some of its musical elements). Use this to alter a template so that there is more variance when it is filled.\""
                                + "\nfill_template: \"Fills a template's masked elements, essentially converting it from a template to a score and storing it to a new name on memory.\""
                                + "\nwrite_score: \"Write a score onto another score. Essentially a copy and paste. Accepts multiple writes to perfrom at once to save tokens. Also lets you open a score if the user requires it.\""
                                + "\nread_tab: \"Reads the basics of a score or template: whether it is a score or a template, its measure count, instruments, clef, key, time signature and tempo. May also be used to read the user's current selection (e.g. 'where I am currently highlighting' or 'my current tab').\""
                                + "\nread_template: \"Reads the items of a template (notes, chords, etc...) in the given measures. Use it before editing a template when you need to see what is in specific measures. A template's content cannot be seen any other way.\""
                                + "\n\nedit_clef, edit_time_sig, edit_key_sig, edit_tempo, edit_add_instrument, edit_delete_staff, and edit_delete_measures are all ONLY usable for editing scores, not templates."
                        },
                        intention: {
                            type: "string",
                            description: "A brief description of what this step should do on a per step basis with any needed implementation details including the names of what will be read and stored to memory (e.g. generate a score with 16 measures and save to '16_bar_score')."
                        },
                        // macro_intention: {
                        //     type: "string",
                        //     description: "A brief description of what this step should achieve when looking at the big picture (e.g. generate the foundation of an A section to be converted into a template)"
                        // }
                    }
                },
                description: "The steps that comprise the plan to complete the user's request. These should be in order of implementation. Tools may be used more than once (e.g. generate score, convert score to template, edit template, etc...). If the user is asking for any edit to be made on a template, always use edit_template over any other edit like edit_delete_staff."
            },
            chat_name: {
                type: "string",
                description: "A very brief title for the name of this chat. Take into consideration previous chats. Omit to keep old name or no name at all"
            }
        }
    }
}

const TOOL_INTROS = [
    {
        name: "trigger_tool",
        description: "Select the next tool on the roadmap to execute.",
        input_schema: {
            type: "object",
            properties: {
                tool: {
                    type: "string",
                    enum: [ "edit_clef", "edit_time_sig", "edit_key_sig", "edit_tempo", "edit_add_instrument", "edit_delete_staff", "edit_delete_measures",
                        "generate_score",
                        "create_template_from_scratch", "create_template_from_score", "edit_template",
                        "fill_template",
                        "write_score",
                        "read_tab",
                        "read_template",
                        "complete"
                    ],
                    description: "The tool to execute, as provided by the current roadmap step. Use complete when the user's prompt has been completed. edit_clef, edit_time_sig, edit_key_sig, edit_tempo, edit_add_instrument, edit_delete_staff, and edit_delete_measures are all ONLY usable for editing scores, not templates."
                },
                intention: {
                    type: "string",
                    description: "A brief description of what this step should do on a per step basis with any needed implementation details including the names of what will be read and stored to memory (e.g. generate a score with 16 measures and save to '16_bar_score')"
                },
                next_steps: {
                    type: "string",
                    description: "A brief outline of what is complete and what you should do after this step using the roadmap at the beginning and the steps that have been completed so far."
                },
            },
            required: [ "tool" ]
        }
    },
]

const TEMPLATE_ITEM_LOCATION = {
    measure: {
        type: "integer",
        description: "The 1-based measure this item belongs to. Omit to be masked and regenerated."
    },
    pos: {
        type: "number",
        description: "The beat position (or rhythm) within the measure this item belongs to where 0 is the first beat of the measure, 1 is after a quarter note duration, and 0.5 is after an eighth note duration, etc... Omit to mask this note's rhythmic placement, to leave this item blank."
    }
}

const TEMPLATE_ITEM_SCHEMAS = [
    {
        type: "object",
        description: "Assigns a completely masked sequence of measures. This is the catalyst for completely new content. May also be used on a sub measure level by setting measure length to 1.",
        properties: {
            kind: { type: "string", enum: [ "masked" ] },
            measure_start: {
                type: "number",
                description: "The beginning of a sequence of completely masked measures (1-based)."
            },
            mask_measure_length: {
                type: "number",
                description: "The amount of measures to have completely masked.",
            },
            slots_per_measure: {
                type: "number",
                description: "The maximum number of new elements filled in per masked measure. Not all slots will be used"
            }
        },
        required: [ "kind", "measure_start", "mask_measure_length", "slots_per_measure" ]
    },
    {
        type: "object",
        description: "Assigns/blanks an instrument for a voice. Note: you should use multiple if the instrument has two staffs (ex. piano should have one for treble and the other for bass)",
        properties: {
            kind: { type: "string", enum: [ "instrument" ] },
            ...TEMPLATE_ITEM_LOCATION,
            instrument: {
                type: "string",
                enum: [ "other", "piano", "harpsichord", "organ", "accordion", "guitar", "bass_guitar", "violin", "viola", "cello", "double_bass", "harp", "flute", "oboe", "clarinet", "bassoon", "saxophone", "trumpet", "horn", "trombone", "tuba", "percussion", "timpani", "celesta", "voice_soprano", "voice_mezzo", "voice_alto", "voice_tenor", "voice_baritone", "voice_bass", "voice", "choir" ],
                description: "The instrument to assign. Omit to leave this item blank."
            },
            voice_idx: { type: "integer", description: "The index of the voice this item belongs to." }
        },
        required: [ "kind", "voice_idx" ]
    },
    {
        type: "object",
        description: "Assigns/blanks a clef for a voice.",
        properties: {
            kind: { type: "string", enum: [ "clef" ] },
            ...TEMPLATE_ITEM_LOCATION,
            name: {
                type: "string",
                enum: [ "treble", "bass", "alto", "tenor", "treble8vb", "treble8va", "bass8vb", "none" ],
                description: "The clef to assign. Omit to leave this item blank."
            },
            voice_idx: { type: "integer", description: "The index of the voice this item belongs to." }
        },
        required: [ "kind", "voice_idx" ]
    },
    {
        type: "object",
        description: "Assigns/blanks a key signature.",
        properties: {
            kind: { type: "string", enum: [ "key" ] },
            ...TEMPLATE_ITEM_LOCATION,
            key_tonic: { type: "string", description: "The tonic note of the key. Ex: C, Bb, E, F#. Omit to leave this item blank." },
            key_mode: { type: "string", description: "The mode of the key. Ex: major, minor, mixolydian. Omit to leave this item blank." }
        },
        required: [ "kind" ]
    },
    {
        type: "object",
        description: "Assigns/blanks a time signature.",
        properties: {
            kind: { type: "string", enum: [ "time" ] },
            ...TEMPLATE_ITEM_LOCATION,
            numerator: { type: "string", description: "The time signature's numerator. Omit to leave this item blank." },
            denominator: { type: "string", description: "The time signature's denominator. Omit to leave this item blank." }
        },
        required: [ "kind" ]
    },
    {
        type: "object",
        description: "Assigns/blanks a tempo.",
        properties: {
            kind: { type: "string", enum: [ "tempo" ] },
            ...TEMPLATE_ITEM_LOCATION,
            tempo: { type: "integer", description: "The tempo in beats per minute. Omit to leave this item blank." }
        },
        required: [ "kind" ]
    },
    {
        type: "object",
        description: "Assigns/blanks a single note.",
        properties: {
            kind: { type: "string", enum: [ "note" ] },
            ...TEMPLATE_ITEM_LOCATION,
            duration: { type: "number", description: "The note's duration where 1 is a quarter note, 0.5 is an eighth note, etc... Omit to leave this item blank." },
            pitch: { type: "integer", description: "The note's pitch. Omit to leave this item blank." },
            voice_idx: { type: "integer", description: "The index of the voice this note belongs to. Omit to leave this item blank." }
        },
        required: [ "kind" ]
    },
    {
        type: "object",
        description: "Assigns/blanks a chord.",
        properties: {
            kind: { type: "string", enum: [ "chord" ] },
            ...TEMPLATE_ITEM_LOCATION,
            root: { type: "string", description: "The chord's root only, without its quality. Either an absolute note name (G, Bb, F#) or a roman numeral relative to the key in effect (ii, V, bVII) — Ex: For Gm7, the root is \"G\". iim7 has root \"ii\". Omit to leave this item blank." },
            quality: {
                type: "string",
                enum: [ "maj7", "maj", "min", "dom7", "min7", "dim7", "dim", "m7b5", "sus2", "7sus", "7aug", "aug" ],
                description: "The chord's quality. Omit to leave this item blank."
            }
        },
        required: [ "kind" ]
    }
]

// What the model writing a template must know about whatever reads it. The roadmap model is told a
// version of this in PROMPT_INTRO, but it is the model writing the items that has to act on it.
const TEMPLATE_CONTRACT = "HOW A TEMPLATE IS USED:"
    + "\n- A separate creative model fills the template. It sees ONLY the items in the template -- not the user's request, not this conversation. Anything the user wants that is not encoded as an item is lost."
    + "\n- Items you keep complete are fixed and copied through as written. A field you omit is masked: the filler invents it."
    + "\n- A `chord` item (root + quality + measure + pos) is how you tell the filler what harmony to write. The filler chooses the pitches that realise it."
    + "\n- A chord item alone writes nothing: the filler needs somewhere to put notes. For a pad or sustained harmony, give EVERY instrument voice one `note` item per measure (kind 'note', voice_idx, measure, pos 0, duration = the whole measure, e.g. 4 in 4/4) with `pitch` OMITTED. That is a whole note whose pitch the filler chooses to fit the chord."
    + "\n- So encode the user's intent as structure, not as the answer: every chord the user names is a chord item with root, quality, measure and pos all set; the notes that realise it are pitch-less note items. Do NOT spell a requested chord out as note pitches -- that fixes those exact notes and leaves nothing to generate. Use a `masked` block only when the rhythm itself is open."
    + "\n- Never omit measure/pos/root/quality from a chord the user asked for; an omitted field masks it, and the chord is then not constrained at all."
    + "\n\nEXAMPLE -- \"a string pad over Abmaj7 for one measure\" with violin (voice 0), viola (voice 1) and cello (voice 2), key Ab, 4/4:"
    + "\n  CORRECT: instrument + clef items for the 3 voices, time 4/4, key Ab major,"
    + "\n    {kind:'chord', root:'Ab', quality:'maj7', measure:1, pos:0},"
    + "\n    {kind:'note', voice_idx:0, measure:1, pos:0, duration:4}, {kind:'note', voice_idx:1, measure:1, pos:0, duration:4}, {kind:'note', voice_idx:2, measure:1, pos:0, duration:4}   <- no pitch"
    + "\n  WRONG 1: note items with pitch 56, 60, 63, 67 hardcoded and no chord item. The chord is baked in and invisible to the filler."
    + "\n  WRONG 2: a chord item and instruments but no note items. The filler has no slot to write into, so nothing is generated."

const EDIT_TEMPLATE_STAGE2_INTRO = "You are editing one existing template in place. A template is an ordered list of items describing a piece of music; an element is masked by keeping its item and omitting the field being masked (e.g. drop `pitch` from a note item to mask that note's pitch), which is what lets the filler model regenerate it."
    + "\nA separate creative model fills the masked fields and sees ONLY the items of this template, so anything the user wants must be encoded as an item: name harmony with `chord` items (root, quality, measure, pos all set) and leave the pitches to be generated; never spell a requested chord out as hardcoded note pitches."
    + "\nRules:"
    + "\n- Return the COMPLETE edited template: every item of the input template, in its original order, including the items you did not touch. Any item you leave out is deleted from the template."
    + "\n- You are modifying the template you were given. Never return a brand new template, and never collapse it into a single fully-masked block, unless the request explicitly asks to replace all of its content."
    + "\n- Keep the setup items (instrument, clef, key, time, tempo) and their values. Filling fails if no item names an instrument."
    + "\n- To mask a musical element, omit the field(s) you want regenerated from its item. To remove content entirely, drop the item."

const EDIT_TEMPLATE_STAGE2_TOOL=  {
    name: "edit_template",
    description: "Return the given template with the requested edits applied (e.g. mask musical elements like rhythm, pitch, chords, etc...). The template you return replaces the one you were given, so it must contain every item you want to keep.",
    input_schema: {
        type: "object",
        properties: {
            memory_output_name: {
                type: "string",
                description: "The new name of the edited template to be stored in memory."
            },
            template: {
                type: "array",
                items: { anyOf: TEMPLATE_ITEM_SCHEMAS },
                description: "The COMPLETE edited template: every item of the template you were given, in the same order, with your edits applied — not just the items you changed. Each item's `kind` selects which fields apply. Omit a non-required field to mask it, to be filled in later. It is recommended to not edit elements like tempos, instruments, keys, and time signatures unless specified."
            },
        },
        required: [ "memory_output_name", "template" ]
    }
}

const TOOLS = {
    "edit_clef": {
        name: "edit_clef",
        description: "Edit the clef of a staff in the score by providing the measure number and new clef",
        input_schema: {
            type: "object",
            properties: {
                memory_name: {
                    type: "string",
                    description: "The name of the score stored in memory."
                },
                memory_output_name: {
                    type: "string",
                    description: "The new name of the score to be stored in memory. Use the same as the input name to make the operation in place."
                },
                measure_num: {
                    type: "integer",
                    description: "The measure to insert the new clef into. Begins at 1."
                },
                staff_num: {
                    type: "integer",
                    description: "The index of the staff to insert the new clef into. Begins at 0."
                },
                new_clef: {
                    type: "string",
                    enum: [ "treble", "bass", "alto", "tenor", "treble8vb", "treble8va", "bass8vb", "none" ],
                    description: "The new clef to insert. None to remove clef."
                }
            }
        }
    },
    "edit_time_sig": {
        name: "edit_time_sig",
        description: "Edit the time signature of the score by providing the measure number and new time",
        input_schema: {
            type: "object",
            properties: {
                memory_name: {
                    type: "string",
                    description: "The name of the score stored in memory."
                },
                memory_output_name: {
                    type: "string",
                    description: "The new name of the score to be stored in memory. Use the same as the input name to make the operation in place."
                },
                measure_num: {
                    type: "integer",
                    description: "The measure to insert the new time signature into. Begins at 1."
                },
                new_time: {
                    type: "string",
                    enum: [ "2/4", "3/4", "4/4", "6/8", "3/8", "12/8" ],
                    description: "The new time signature to insert"
                }
            }
        }
    },
    "edit_key_sig": {
        name: "edit_key_sig",
        description: "Edit the key signature of the score by providing the measure number and new key",
        input_schema: {
            type: "object",
            properties: {
                memory_name: {
                    type: "string",
                    description: "The name of the score stored in memory."
                },
                memory_output_name: {
                    type: "string",
                    description: "The new name of the score to be stored in memory. Use the same as the input name to make the operation in place."
                },
                measure_num: {
                    type: "integer",
                    description: "The measure to insert the new key signature into. Begins at 1."
                },
                tonic: {
                    type: "string",
                    description: "The tonic note of the new key. Ex: C, Bb, E, F#"
                },
                mode: {
                    type: "string",
                    description: "The mode of the new key. Ex: major, minor, mixolydian"
                }
            }
        }
    },
    "edit_tempo": {
        name: "edit_tempo",
        description: "Edit the tempo of the score by providing the measure number and new tempo",
        input_schema: {
            type: "object",
            properties: {
                memory_name: {
                    type: "string",
                    description: "The name of the score stored in memory."
                },
                memory_output_name: {
                    type: "string",
                    description: "The new name of the score to be stored in memory. Use the same as the input name to make the operation in place."
                },
                measure_num: {
                    type: "integer",
                    description: "The measure to insert the new tempo into. Begins at 1."
                },
                new_tempo: {
                    type: "integer",
                    description: "The new tempo in beats per minute"
                }
            }
        }
    },
    "edit_add_instrument": {
        name: "edit_add_instrument",
        description: "Add an instrument to the score by providing the new instrument",
        input_schema: {
            type: "object",
            properties: {
                memory_name: {
                    type: "string",
                    description: "The name of the score stored in memory."
                },
                memory_output_name: {
                    type: "string",
                    description: "The new name of the score to be stored in memory. Use the same as the input name to make the operation in place."
                },
                new_instrument: {
                    type: "string",
                    enum: [ "other", "piano",  "harpsichord", "organ", "accordion", "guitar", "bass_guitar", "violin", "viola", "cello", "double_bass", "harp", "flute", "oboe", "clarinet", "bassoon", "saxophone", "trumpet", "horn", "trombone", "tuba", "percussion", "timpani", "celesta", "voice_soprano", "voice_mezzo", "voice_alto", "voice_tenor", "voice_baritone", "voice_bass", "voice", "choir" ],
                    description: "The new instrument to add"
                }
            }
        }
    },
    "edit_delete_staff": {
        name: "edit_delete_staff",
        description: "Remove a staff on the score",
        input_schema: {
            type: "object",
            properties: {
                memory_name: {
                    type: "string",
                    description: "The name of the score stored in memory."
                },
                memory_output_name: {
                    type: "string",
                    description: "The new name of the score to be stored in memory. Use the same as the input name to make the operation in place."
                },
                staff_num: {
                    type: "integer",
                    description: "The index of the staff to delete. Begins at 0."
                }
            }
        }
    },
    "edit_delete_measures": {
        name: "edit_delete_measures",
        description: "Remove a range of measures on the score (inclusive)",
        input_schema: {
            type: "object",
            properties: {
                memory_name: {
                    type: "string",
                    description: "The name of the score stored in memory."
                },
                memory_output_name: {
                    type: "string",
                    description: "The new name of the score to be stored in memory. Use the same as the input name to make the operation in place."
                },
                measure_start_num: {
                    type: "integer",
                    description: "The first measure of the range to be deleted"
                },
                measure_end_num: {
                    type: "integer",
                    description: "The last measure of the range to be deleted"
                }
            }
        }
    },
    "generate_score": {
        name: "generate_score",
        description: "Uses a transformer model to generate a new score if no current score exists. Else, inserts into the given location. Do not include key, time_signature, or tempo when these attributes should remain the same.",
        input_schema: {
            type: "object",
            properties: {
                memory_name: {
                    type: "string",
                    description: "The name of the existing score stored in memory (if there is one)."
                },
                memory_output_name: {
                    type: "string",
                    description: "The new name of the score to be stored in memory. Use the same as the input name to make the operation in place."
                },
                measure_start_num: {
                    type: "integer",
                    description: "The first measure where the newly generated material will be inserted."
                },
                generation_staffs: {
                    type: "array",
                    items: {
                        type: "object",
                        properties: {
                            staff_num: {
                                type: "integer",
                                description: "The index of the staff"
                            },
                            instrument: {
                                type: "string",
                                enum: [ "other", "piano",  "harpsichord", "organ", "accordion", "guitar", "bass_guitar", "violin", "viola", "cello", "double_bass", "harp", "flute", "oboe", "clarinet", "bassoon", "saxophone", "trumpet", "horn", "trombone", "tuba", "percussion", "timpani", "celesta", "voice_soprano", "voice_mezzo", "voice_alto", "voice_tenor", "voice_baritone", "voice_bass", "voice", "choir" ],
                                description: "The name of the instrument to generate. Only include if the instruments aren't already identified by the staff_num (e.g. no current score exists)"
                            }
                        },
                        required: [ "staff_num" ]
                    },
                    description: "List the staffs to be generated. Provide an empty array if no staffs exist and the user hasn't given a list of staffs or instruments. If an instrument uses multiple staffs (like piano or organ which typically have treble and bass staffs), add that instrument multiple times."
                },
                generation_length: {
                    type: "integer",
                    description: "The number of measures to generate (multiple staffs still counts as one measure)"
                },
                overwrite_current: {
                    type: "boolean",
                    description: "True if the generation should overwrite the current measures located from measure_start_num to measure_start_num + generation_length. False means it will be inserted, creating new measures at the given location."
                },
                tonic: {
                    type: [ "string", "null" ],
                    enum: [ "Cb", "Gb", "Db", "Ab", "Eb", "Bb", "F", "C", "G", "D", "A", "E", "B", "F#", "C#", null ],
                    description: "The tonic of the key the generation is in, spelled as the mode's own tonic (A minor is 'A', not 'C'). Null to carry on in whatever key is in effect before measure_start_num."
                },
                mode: {
                    type: [ "string", "null" ],
                    enum: [ "major", "minor", "dorian", "phrygian", "lydian", "mixolydian", "locrian", null ],
                    description: "The mode of that key. Give it alongside tonic; the backend derives the key signature from the pair. Null is read as major."
                },
                time_signature: {
                    type: [ "string", "null" ],
                    enum: [ "2/4", "3/4", "4/4", "6/8", "3/8", "12/8", null ],
                    description: "The time signature the generation is in. Null to carry on in whatever time signature is in effect before measure_start_num."
                },
                tempo: {
                    type: [ "integer", "null" ],
                    description: "The tempo of the generation in beats per minute. Null to carry on at whatever tempo is in effect before measure_start_num."
                }
            },
            required: [ "measure_start_num", "generation_staffs", "generation_length", "overwrite_current", "tonic", "mode", "time_signature", "tempo" ]
        }
    },
    "create_template_from_scratch": {
        name: "create_template_from_scratch",
        description: "Create a template from scratch. Only to be used when the user has very specific intentions.",
        input_schema: {
            type: "object",
            properties: {
                // written before `template` on purpose: the model reasons here, then builds the items
                plan: {
                    type: "string",
                    description: "Scratchpad, written first. (1) List each concrete musical requirement in the user's request (chords, instruments, length, key, rhythm...). (2) For each, name the template item that will carry it -- the filler sees only the items. Chords are `chord` items, never hardcoded pitches. (3) State which items are left masked for the filler to write."
                },
                memory_output_name: {
                    type: "string",
                    description: "The new name of the template to be stored in memory."
                },
                template: {
                    type: "array",
                    items: { anyOf: TEMPLATE_ITEM_SCHEMAS },
                    description: "The ordered list of items making up the template. Each item's `kind` selects which fields apply. Omit a non-required field to leave that item blank, to be filled in later. Every chord the user names must appear here as a chord item with root, quality, measure and pos all set."
                },
                open: {
                    type: "string",
                    description: "Enter an existing tab or written-to tab to draw the user's attention by opening it. Omit to not open the tab."
                }
            },
            required: [ "template", "memory_output_name" ]
        }
    },
    "create_template_from_score": {
        name: "create_template_from_score",
        description: "Create a template from an existing score in memory.",
        input_schema: {
            type: "object",
            properties: {
                memory_name: {
                    type: "string",
                    description: "The name of the score already stored in memory to be converted."
                },
                memory_output_name: {
                    type: "string",
                    description: "The new name of the template to be stored in memory."
                },
                open: {
                    type: "string",
                    description: "Enter an existing tab or written-to tab to draw the user's attention by opening it. Omit to not open the tab."
                }
            }
        }
    },
    "edit_template": {
        name: "edit_template",
        description: "Edit a template already stored in memory. Note: this command is 2 stages. This is only the first stage 1. After giving the input template, you will be given the template stored in memory as json, and will then make edits to it.",
        input_schema: {
            type: "object",
            properties: {
                memory_name: {
                    type: "string",
                    description: "The name of the template already stored in memory that will be edited."
                },
                edit_measures: {
                    type: "array",
                    items: {
                        type: "number",
                        description: "A measure that will be edited."
                    },
                    description: "The measures to edit. Only edits within the measures of this array will be allowed. Can be omitted to edit everything.",
                },
                needs_context: {
                    type: "boolean",
                    description: "Whether stage 2 needs to see what is currently in edit_measures. Set false when the edit replaces those measures wholesale and does not depend on their current content (e.g. 'rewrite measures 9-16 as whole-note string pads over these chords', with the details in your intention): the existing items are then not sent to stage 2, which saves tokens, and are replaced by what it writes. Needs edit_measures. Leave true (the default) for any edit that keeps or modifies what is there (e.g. masking its pitches).",
                },
                open: {
                    type: "string",
                    description: "Enter an existing tab or written-to tab to draw the user's attention by opening it. Omit to not open the tab."
                }
            },
            required: [ "memory_name", "open" ]
        }
    },
    "fill_template": {
        name: "fill_template",
        description: "Fills a template, essentially converting it from a template to a score and storing it to a new name on memory.",
        input_schema: {
            type: "object",
            properties: {
                memory_name: {
                    type: "string",
                    description: "The name of the template already stored in memory that will be filled and converted."
                },
                memory_output_name: {
                    type: "string",
                    description: "The name of the filled score to be stored in memory. Must be different than the memory_name"
                },
                open: {
                    type: "string",
                    description: "Enter an existing tab or written-to tab to draw the user's attention by opening it. Omit to not open the tab."
                }
            }
        }
    },
    "write_score": {
        name: "write_score",
        description: "Write a score onto another score. Essentially a copy and paste. This action can perform multiple writes at once, saving tokens.",
        input_schema: {
            type: "object",
            properties: {
                writes: {
                    type: "array",
                    items: {
                        type: "object",
                        properties: {
                            memory_name: {
                                type: "string",
                                description: "The name of the score already stored in memory that will be read/copied."
                            },
                            memory_output_name: {
                                type: "string",
                                description: "The name of the score that will be written to."
                            },
                            write_start_measure: {
                                type: "number",
                                description: "The 1-based measure where the input score will be written on the output score."
                            }
                        },
                        required: [ "memory_name", "memory_output_name", "write_start_measure" ],
                        description: "A singular write command. Use multiple to make multiple edits. Edits made in the order provided."
                    },
                    description: "The write commands, carried out in the order given."
                },
                open: {
                    type: "string",
                    description: "Enter an existing tab or written-to tab to draw the user's attention by opening it. Omit to not open the tab."
                }
            },
            required: [ "writes" ]
        }
    },
    "read_tab": {
        name: "read_tab",
        description: "Reads the basics of the given scores/templates: whether each is a score or a template, then its measure count, instruments, clef, key signature, time signature and tempo. To see a template's items, use read_template.",
        input_schema: {
            type: "object",
            properties: {
                memory_names: {
                    type: "array",
                    items: { type: "string" },
                    description: "The names of the scores/templates in memory to read. Leave empty for the user's active tab."
                },
                include_selection: {
                    type: "boolean",
                    description: "True to also read the user's active tab and the measure(s) and staff/staves they have selected on it."
                }
            }
        }
    },
    "read_template": {
        name: "read_template",
        description: "Reads the items of a template that are in the given measures: notes, chords, and any other items, as stored. A field an item lacks is masked.",
        input_schema: {
            type: "object",
            properties: {
                memory_name: {
                    type: "string",
                    description: "The name of the template stored in memory to read."
                },
                measures: {
                    type: "array",
                    items: { type: "integer" },
                    description: "The 1-based measures whose items should be returned. Keep it to the measures you need."
                }
            },
            required: [ "memory_name", "measures" ]
        }
    },
    "complete": {
        name: "complete",
        description: "The final step: marks the user's task as complete",
        input_schema: {
            type: "object",
            properties: {
                recap: {
                    type: "string",
                    description: "Briefly summarize what was done to complete the user's prompt. A future model will use this as context for future user prompts, so ensure to include crucial details. For example, you should include what edits were made, the steps executed to complete the task, and what scores/templates in memory were modified/created and what they were for."
                }
            }
        }
    }
}